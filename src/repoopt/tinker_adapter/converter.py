"""ATIF-v1.8 to Qwen/Tinker Datum SFT conversion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal

import tinker
from pydantic import BaseModel, Field
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import Renderer, TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from repoopt.store.base import ArtifactStore, StoragePaths, sha256_bytes
from repoopt.tinker_adapter.atif_parser import (
    ATIFValidationError,
    ParsedTrajectory,
    parse_atif_trajectory,
)


@lru_cache(maxsize=4)
def _load_renderer(model_name: str, renderer_name: str) -> tuple[Any, Renderer]:
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    return tokenizer, renderer


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class DatumProvenance(BaseModel):
    """Provenance record for one converted Tinker Datum."""

    example_id: str
    experiment_id: str
    task_id: str
    attempt_id: str
    window_index: int
    compacted_from_prior: bool
    target_turn_index: int
    total_tokens: int
    trained_tokens: int
    masked_prompt_tokens: int
    weight_sum: float


class ConversionReport(BaseModel):
    """Summary report for ATIF-to-Tinker dataset conversion."""

    schema_version: str = "repoopt-conversion-v1"
    created_at: str
    model_name: str
    renderer_name: str
    max_length: int | None
    reduction: str
    trajectories_processed: int = 0
    trajectories_rejected: int = 0
    datums_emitted: int = 0
    datums_skipped_truncated: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
    examples: list[DatumProvenance] = Field(default_factory=list)


@dataclass
class TrajectoryConversionResult:
    """Converted Tinker Datums and per-example provenance for a single trajectory."""

    datums: list[tinker.Datum]
    provenance: list[DatumProvenance]
    warnings: list[str] = field(default_factory=list)
    skipped_truncated: int = 0


def convert_parsed_trajectory_to_datums(
    parsed: ParsedTrajectory,
    *,
    experiment_id: str = "local",
    task_id: str = "unknown",
    attempt_id: str = "attempt-0",
    model_name: str = "Qwen/Qwen3.5-9B",
    renderer_name: str | None = None,
    max_length: int | None = 32768,
    reduction: Literal["none", "mean"] = "mean",
) -> TrajectoryConversionResult:
    """Convert a ParsedTrajectory into prefix-consistent Qwen/Tinker Datum instances."""
    resolved_renderer = renderer_name or model_info.get_recommended_renderer_name(model_name)
    _, renderer = _load_renderer(model_name, resolved_renderer)

    datums: list[tinker.Datum] = []
    provenance: list[DatumProvenance] = []
    warnings: list[str] = list(parsed.warnings)
    skipped_truncated = 0

    for window in parsed.windows:
        warnings.extend(w for w in window.warnings if w not in warnings)
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=window.tools,
            system_prompt=window.system_prompt,
        )
        full_messages = prefix + window.messages
        prefix_len = len(prefix)

        for turn_offset, msg_idx in enumerate(window.trainable_assistant_indices):
            end_idx = prefix_len + msg_idx + 1
            sub_messages = full_messages[:end_idx]
            model_input, weights = renderer.build_supervised_example(
                sub_messages,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )

            prompt_tokens = renderer.build_generation_prompt(sub_messages[:-1]).to_ints()
            # Ensure prompt tokens strictly carry zero training loss before target turn
            if len(weights) >= len(prompt_tokens) and float(weights[: len(prompt_tokens)].sum()) != 0.0:
                raise ATIFValidationError(
                    f"Non-zero loss weight detected inside prompt prefix at window={window.window_index}, turn={msg_idx}"
                )

            datum = datum_from_model_input_weights(
                model_input,
                weights,
                max_length=max_length,
                reduction=reduction,
            )
            target_tokens = datum.loss_fn_inputs["target_tokens"].data
            loss_weights = datum.loss_fn_inputs["weights"].data

            if len(target_tokens) != len(loss_weights) or len(target_tokens) != datum.model_input.length:
                raise ATIFValidationError(
                    f"Datum length mismatch: input={datum.model_input.length}, "
                    f"targets={len(target_tokens)}, weights={len(loss_weights)}"
                )

            trained_count = sum(1 for w in loss_weights if float(w) > 0.0)
            weight_sum = float(sum(float(w) for w in loss_weights))
            if trained_count == 0 or weight_sum <= 0.0:
                skipped_truncated += 1
                warnings.append(
                    f"Skipped window={window.window_index} turn={turn_offset}: zero trained tokens after max_length={max_length}"
                )
                continue

            example_id = f"{task_id}__{attempt_id}__w{window.window_index:02d}__t{turn_offset:03d}"
            datums.append(datum)
            provenance.append(
                DatumProvenance(
                    example_id=example_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    attempt_id=attempt_id,
                    window_index=window.window_index,
                    compacted_from_prior=window.compacted_from_prior,
                    target_turn_index=msg_idx,
                    total_tokens=len(target_tokens),
                    trained_tokens=trained_count,
                    masked_prompt_tokens=len(target_tokens) - trained_count,
                    weight_sum=round(weight_sum, 6),
                )
            )

    return TrajectoryConversionResult(
        datums=datums,
        provenance=provenance,
        warnings=warnings,
        skipped_truncated=skipped_truncated,
    )


def convert_trajectory_dict_to_datums(
    trajectory_data: dict[str, Any],
    *,
    raw_provider_events: list[dict[str, Any]] | None = None,
    experiment_id: str = "local",
    task_id: str = "unknown",
    attempt_id: str = "attempt-0",
    model_name: str = "Qwen/Qwen3.5-9B",
    renderer_name: str | None = None,
    max_length: int | None = 32768,
    reduction: Literal["none", "mean"] = "mean",
) -> TrajectoryConversionResult:
    """Validate ATIF JSON, merge raw provider events, and convert into Tinker Datums."""
    parsed = parse_atif_trajectory(
        trajectory_data,
        raw_provider_events=raw_provider_events,
    )
    return convert_parsed_trajectory_to_datums(
        parsed,
        experiment_id=experiment_id,
        task_id=task_id,
        attempt_id=attempt_id,
        model_name=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        reduction=reduction,
    )


def export_attempts_to_tinker_dataset(
    store: ArtifactStore,
    experiment_id: str,
    attempts: list[tuple[str, str]],
    *,
    model_name: str = "Qwen/Qwen3.5-9B",
    renderer_name: str | None = None,
    max_length: int | None = 32768,
    reduction: Literal["none", "mean"] = "mean",
    require_verified_speedup: bool = False,
) -> tuple[list[tinker.Datum], ConversionReport]:
    """Convert stored attempts into Tinker Datums and persist examples + report under Section 7 layout."""
    resolved_renderer = renderer_name or model_info.get_recommended_renderer_name(model_name)
    report = ConversionReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        renderer_name=resolved_renderer,
        max_length=max_length,
        reduction=reduction,
    )
    all_datums: list[tinker.Datum] = []
    manifest_examples: list[dict[str, Any]] = []

    for task_id, attempt_id in attempts:
        eval_key = StoragePaths.attempt_evaluation(experiment_id, task_id, attempt_id)
        if require_verified_speedup:
            if not store.exists(eval_key):
                report.warnings.append(f"Skipped {task_id}/{attempt_id}: missing evaluation.json")
                continue
            eval_data = store.get_json(eval_key)
            if eval_data.get("label") != "VERIFIED_SPEEDUP":
                report.warnings.append(
                    f"Skipped {task_id}/{attempt_id}: label={eval_data.get('label')!r} != VERIFIED_SPEEDUP"
                )
                continue

        traj_key = StoragePaths.attempt_trajectory(experiment_id, task_id, attempt_id)
        if not store.exists(traj_key):
            report.trajectories_rejected += 1
            report.errors.append({"task_id": task_id, "attempt_id": attempt_id, "error": "Missing trajectory.json"})
            continue

        raw_prefix = f"{StoragePaths.attempt_prefix(experiment_id, task_id, attempt_id)}/raw-provider-events/"
        raw_keys = store.list_prefix(raw_prefix)
        raw_events = [store.get_json(k) for k in raw_keys] if raw_keys else None

        try:
            traj_data = store.get_json(traj_key)
            converted = convert_trajectory_dict_to_datums(
                traj_data,
                raw_provider_events=raw_events,
                experiment_id=experiment_id,
                task_id=task_id,
                attempt_id=attempt_id,
                model_name=model_name,
                renderer_name=resolved_renderer,
                max_length=max_length,
                reduction=reduction,
            )
        except Exception as exc:
            report.trajectories_rejected += 1
            report.errors.append({"task_id": task_id, "attempt_id": attempt_id, "error": str(exc)})
            continue

        report.trajectories_processed += 1
        report.datums_skipped_truncated += converted.skipped_truncated
        report.warnings.extend(converted.warnings)

        for datum, prov in zip(converted.datums, converted.provenance):
            ex_key = StoragePaths.training_example(experiment_id, model_name, f"{prov.example_id}.json")
            datum_payload = _to_jsonable(datum)
            receipt = store.put_json(ex_key, datum_payload)
            all_datums.append(datum)
            report.datums_emitted += 1
            report.examples.append(prov)
            manifest_examples.append(
                {
                    "example_id": prov.example_id,
                    "key": ex_key,
                    "sha256": receipt.sha256,
                    "size_bytes": receipt.size_bytes,
                    "total_tokens": prov.total_tokens,
                    "trained_tokens": prov.trained_tokens,
                }
            )

    report_key = StoragePaths.training_conversion_report(experiment_id, model_name)
    report_receipt = store.put_json(report_key, report.model_dump(mode="json"))

    manifest_key = StoragePaths.training_manifest(experiment_id, model_name)
    store.put_json(
        manifest_key,
        {
            "schema_version": "repoopt-training-manifest-v1",
            "experiment_id": experiment_id,
            "model_name": model_name,
            "renderer_name": resolved_renderer,
            "created_at": report.created_at,
            "conversion_report_key": report_key,
            "conversion_report_sha256": report_receipt.sha256,
            "num_examples": len(manifest_examples),
            "examples": manifest_examples,
        },
    )

    return all_datums, report

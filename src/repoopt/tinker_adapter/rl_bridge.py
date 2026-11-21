"""Tinker Cookbook RL bridge matching the Pi 4-tool surface and RepoOpt Harbor verifiers."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tomllib
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.harbor_rl.harbor_tools import HarborReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.sandbox import SandboxInterface, SandboxResult
from tinker_cookbook.tool_use import ToolResult, build_agent_tool_env, simple_tool_result, tool
from tinker_cookbook.tool_use.agent_tool_message_env import RewardFn

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 16384

REPOOPT_RL_SYSTEM_PROMPT = (
    "You are an autonomous scientific software optimization engineer working in /workspace. "
    "Use `read`, `write`, `edit`, and `bash` to inspect, profile, optimize, and benchmark code. "
    "You can run `repoopt-bench` via `bash` to check correctness and paired speedup against /opt/baseline."
)


class RepoOptPiTools:
    """Exposes Pi's exact 4-tool interface (read, write, edit, bash) over SandboxInterface."""

    def __init__(self, sandbox: SandboxInterface, *, command_timeout: int = 120) -> None:
        self._sandbox = sandbox
        self._command_timeout = command_timeout

    @tool
    async def read(
        self,
        path: Annotated[str, "Path to the file to read."],
        offset: Annotated[int | None, "Optional 1-indexed starting line."] = None,
        limit: Annotated[int | None, "Optional maximum number of lines to read."] = None,
    ) -> ToolResult:
        """Read a file from the sandbox environment."""
        res = await self._sandbox.read_file(path, timeout=self._command_timeout)
        if res.exit_code != 0:
            return simple_tool_result(
                json.dumps({"error": res.stderr or f"Failed to read {path}", "exit_code": res.exit_code})
            )
        text = res.stdout
        if offset is not None or limit is not None:
            lines = text.splitlines()
            start = max(0, (offset or 1) - 1)
            end = start + limit if limit is not None else len(lines)
            text = "\n".join(lines[start:end])
        return simple_tool_result(text[:MAX_OUTPUT_CHARS])

    @tool
    async def write(
        self,
        path: Annotated[str, "Destination file path."],
        content: Annotated[str, "Full content to write to the file."],
    ) -> ToolResult:
        """Write or overwrite a file inside the sandbox."""
        res = await self._sandbox.write_file(path, content, timeout=self._command_timeout)
        return simple_tool_result(
            json.dumps({"exit_code": res.exit_code, "path": path, "stderr": res.stderr[:1000]})
        )

    @tool
    async def edit(
        self,
        path: Annotated[str, "Path to the existing file to edit."],
        oldText: Annotated[str, "Exact text snippet to replace."],
        newText: Annotated[str, "Replacement text."],
    ) -> ToolResult:
        """Replace an exact snippet inside an existing file."""
        cur = await self._sandbox.read_file(path, timeout=self._command_timeout)
        if cur.exit_code != 0:
            return simple_tool_result(
                json.dumps({"error": cur.stderr or f"Cannot read {path}", "exit_code": cur.exit_code})
            )
        count = cur.stdout.count(oldText)
        if count == 0:
            return simple_tool_result(json.dumps({"error": "oldText not found in file", "exit_code": 1}))
        if count > 1:
            return simple_tool_result(
                json.dumps({"error": f"oldText matched {count} times; provide more context", "exit_code": 1})
            )
        updated = cur.stdout.replace(oldText, newText, 1)
        wr = await self._sandbox.write_file(path, updated, timeout=self._command_timeout)
        return simple_tool_result(
            json.dumps({"exit_code": wr.exit_code, "path": path, "stderr": wr.stderr[:1000]})
        )

    @tool
    async def bash(
        self,
        command: Annotated[str, "The bash command to execute."],
        timeout: Annotated[int | None, "Optional timeout in seconds."] = None,
    ) -> ToolResult:
        """Execute a bash command in the sandbox environment."""
        eff_timeout = timeout or self._command_timeout
        res = await self._sandbox.run_command(
            command,
            workdir="/workspace",
            timeout=eff_timeout,
            max_output_bytes=MAX_OUTPUT_CHARS,
        )
        return simple_tool_result(
            json.dumps(
                {
                    "exit_code": res.exit_code,
                    "stdout": res.stdout[:MAX_OUTPUT_CHARS],
                    "stderr": res.stderr[:MAX_OUTPUT_CHARS],
                }
            )
        )

    def as_tools(self) -> list[Any]:
        """Return the 4 bound tools in Pi schema order."""
        return [self.read, self.write, self.edit, self.bash]


class DockerSandbox(SandboxInterface):
    """Local Docker implementation of Tinker SandboxInterface for Harbor RL rollouts."""

    def __init__(self, image_tag: str, *, container_id: str, workdir: str = "/workspace") -> None:
        self._image_tag = image_tag
        self._container_id = container_id
        self._workdir = workdir

    @property
    def sandbox_id(self) -> str:
        return self._container_id

    @classmethod
    async def create_from_env_dir(
        cls,
        env_dir: Path,
        *,
        image_tag: str | None = None,
        cpus: int = 2,
        memory_mb: int = 4096,
    ) -> "DockerSandbox":
        tag = image_tag or f"repoopt-rl-{env_dir.parent.name.lower()}"

        def _build_and_start() -> str:
            subprocess.run(
                ["docker", "build", "-t", tag, str(env_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            name = f"repoopt-sb-{uuid.uuid4().hex[:12]}"
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    name,
                    f"--cpus={cpus}",
                    f"--memory={memory_mb}m",
                    "--network",
                    "none",
                    "-w",
                    "/workspace",
                    tag,
                    "sleep",
                    "infinity",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return proc.stdout.strip() or name

        cid = await asyncio.to_thread(_build_and_start)
        return cls(tag, container_id=cid)

    async def send_heartbeat(self, timeout: int = 30) -> None:
        del timeout

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        cwd = workdir or self._workdir
        limit = max_output_bytes or 131072

        def _exec() -> SandboxResult:
            try:
                proc = subprocess.run(
                    ["docker", "exec", "-w", cwd, self._container_id, "bash", "-lc", command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                return SandboxResult(
                    exit_code=proc.returncode,
                    stdout=(proc.stdout or "")[:limit],
                    stderr=(proc.stderr or "")[:limit],
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    exit_code=124,
                    stdout=(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))[:limit],
                    stderr=f"Command timed out after {timeout}s",
                )

        return await asyncio.to_thread(_exec)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        cmd = f"cat -- {json.dumps(path)}"
        res = await self.run_command(cmd, workdir="/workspace", timeout=timeout, max_output_bytes=max_bytes)
        return res

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        data = content.encode("utf-8") if isinstance(content, str) else content

        def _write() -> SandboxResult:
            parent = str(Path(path).parent)
            subprocess.run(
                ["docker", "exec", self._container_id, "mkdir", "-p", parent],
                check=False,
                capture_output=True,
                timeout=timeout,
            )
            proc = subprocess.run(
                ["docker", "exec", "-i", self._container_id, "tee", path],
                input=data,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode == 0 and executable:
                subprocess.run(
                    ["docker", "exec", self._container_id, "chmod", "+x", path],
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
            return SandboxResult(
                exit_code=proc.returncode,
                stdout="",
                stderr=(proc.stderr or b"").decode("utf-8", "replace"),
            )

        return await asyncio.to_thread(_write)

    async def cleanup(self) -> None:
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "rm", "-f", self._container_id],
            capture_output=True,
            check=False,
        )


@dataclass(frozen=True)
class RepoOptRLTask:
    """Loaded RepoOpt Harbor task ready for Tinker RL training."""

    task_name: str
    instruction: str
    task_dir: Path
    config: dict[str, Any] = field(default_factory=dict)


def load_repoopt_rl_task(task_dir: Path) -> RepoOptRLTask:
    """Load task.toml and instruction.md from a local Harbor task directory."""
    resolved = task_dir.resolve()
    return RepoOptRLTask(
        task_name=resolved.name,
        instruction=(resolved / "instruction.md").read_text(encoding="utf-8"),
        task_dir=resolved,
        config=tomllib.loads((resolved / "task.toml").read_text(encoding="utf-8")),
    )


SandboxFactory = Callable[[Path, int], Awaitable[SandboxInterface]]


async def docker_sandbox_factory(env_dir: Path, timeout: int) -> SandboxInterface:
    """Default local Docker sandbox factory for RepoOpt RL rollouts."""
    del timeout
    return await DockerSandbox.create_from_env_dir(env_dir)


def _initial_messages(
    task: RepoOptRLTask,
    renderer: Renderer,
    pi_tools: RepoOptPiTools,
) -> list[Message]:
    specs = [t.to_spec() for t in pi_tools.as_tools()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=specs,
        system_prompt=REPOOPT_RL_SYSTEM_PROMPT,
    )
    return prefix + [{"role": "user", "content": task.instruction}]


class RepoOptEnvGroupBuilder(EnvGroupBuilder):
    """Creates Tinker RL environments using Pi's 4-tool schema and RepoOpt Harbor verifiers."""

    def __init__(
        self,
        task: RepoOptRLTask,
        *,
        model_name: str = "Qwen/Qwen3.5-9B",
        renderer_name: str | None = "qwen3_5",
        max_turns: int = 30,
        group_size: int = 4,
        sandbox_timeout: int = 900,
        command_timeout: int = 120,
        grader_timeout: int = 300,
        max_trajectory_tokens: int = 32768,
        max_generation_tokens: int | None = 4096,
        context_overflow_reward: float = -0.1,
        sandbox_factory: SandboxFactory | None = None,
        reward_fn: RewardFn | None = None,
    ) -> None:
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.sandbox_timeout = sandbox_timeout
        self.command_timeout = command_timeout
        self.grader_timeout = grader_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.sandbox_factory = sandbox_factory or docker_sandbox_factory
        self.reward_fn = reward_fn
        self._sandboxes: list[SandboxInterface] = []

    async def make_envs(self) -> Sequence[Env]:
        self._sandboxes = []
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        resolved_renderer = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
        renderer = get_renderer(resolved_renderer, tokenizer)

        env_dir = self.task.task_dir / "environment"
        tests_dir = self.task.task_dir / "tests"

        envs: list[Env] = []
        for _ in range(self.group_size):
            sandbox = await self.sandbox_factory(env_dir, self.sandbox_timeout)
            self._sandboxes.append(sandbox)
            pi_tools = RepoOptPiTools(sandbox, command_timeout=self.command_timeout)
            reward_fn = self.reward_fn or HarborReward(
                tests_dir=tests_dir,
                sandbox=sandbox,
                grader_timeout=self.grader_timeout,
            )
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=pi_tools.as_tools(),
                    initial_messages=_initial_messages(self.task, renderer, pi_tools),
                    reward_fn=reward_fn,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=self.context_overflow_reward,
                )
            )
        return envs

    async def cleanup(self) -> None:
        for sb in self._sandboxes:
            try:
                await sb.cleanup()
            except Exception as exc:
                logger.warning("Failed to cleanup sandbox %s: %s", sb.sandbox_id, exc)
        self._sandboxes.clear()

    def logging_tags(self) -> list[str]:
        return ["repoopt", self.task.task_name]


class RepoOptRLDataset(RLDataset):
    """RLDataset yielding batches of RepoOptEnvGroupBuilders."""

    def __init__(self, builders: list[RepoOptEnvGroupBuilder], batch_size: int) -> None:
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return (len(self.builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class RepoOptRLDatasetBuilder(RLDatasetBuilder):
    """Tinker Cookbook dataset builder over local RepoOpt Harbor optimization tasks."""

    task_dirs: list[Path]
    batch_size: int = 2
    group_size: int = 4
    model_name: str = "Qwen/Qwen3.5-9B"
    renderer_name: str | None = "qwen3_5"
    max_turns: int = 30
    sandbox_timeout: int = 900
    command_timeout: int = 120
    grader_timeout: int = 300
    max_trajectory_tokens: int = 32768
    max_generation_tokens: int | None = 4096
    context_overflow_reward: float = -0.1
    sandbox_factory: SandboxFactory | None = None
    reward_fn: RewardFn | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tasks = [load_repoopt_rl_task(p) for p in self.task_dirs]
        train_builders = [
            RepoOptEnvGroupBuilder(
                task=t,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                grader_timeout=self.grader_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                sandbox_factory=self.sandbox_factory,
                reward_fn=self.reward_fn,
            )
            for t in tasks
        ]
        eval_builders = [
            RepoOptEnvGroupBuilder(
                task=t,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=1,
                sandbox_timeout=self.sandbox_timeout,
                command_timeout=self.command_timeout,
                grader_timeout=self.grader_timeout,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
                sandbox_factory=self.sandbox_factory,
                reward_fn=self.reward_fn,
            )
            for t in tasks
        ]
        return RepoOptRLDataset(train_builders, self.batch_size), RepoOptRLDataset(eval_builders, self.batch_size)

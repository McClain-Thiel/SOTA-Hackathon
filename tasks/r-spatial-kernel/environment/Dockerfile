FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    NO_PROXY=host.docker.internal,localhost,127.0.0.1 \
    no_proxy=host.docker.internal,localhost,127.0.0.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    gcc \
    g++ \
    libopenblas-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    python3-numpy \
    python3-scipy \
    python3-pytest \
    cython3 \
    r-base-core \
    r-base-dev \
    r-cran-rcpp \
    r-cran-testthat \
    r-cran-jsonlite \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash \
    && . /root/.nvm/nvm.sh \
    && nvm install 22 \
    && nvm alias default 22 \
    && npm install -g --ignore-scripts @earendil-works/pi-coding-agent@0.85.1 \
    && npm cache clean --force \
    && pi --version

COPY repo /opt/baseline_src
RUN mkdir -p /opt/baseline_rlib \
    && R CMD INSTALL --no-multiarch --library=/opt/baseline_rlib /opt/baseline_src

COPY bench_config.json /opt/bench/bench_config.json
COPY dev_cases.R /opt/bench/dev_cases.R
COPY repoopt_bench.py /usr/local/bin/repoopt-bench
RUN chmod 0755 /usr/local/bin/repoopt-bench

RUN cp -a /opt/baseline_src /workspace \
    && git config --global user.email "repoopt@local" \
    && git config --global user.name "RepoOpt" \
    && git -C /workspace init \
    && git -C /workspace add -A \
    && git -C /workspace commit -m "baseline"

WORKDIR /workspace
CMD ["sleep", "infinity"]

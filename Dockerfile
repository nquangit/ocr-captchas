# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["python3"]
CMD ["web.py"]

FROM builder as dev-envs

RUN <<EOF
apk update
apk add git
EOF

RUN <<EOF
addgroup -S docker
adduser -S --shell /bin/bash --ingroup docker vscode
EOF
# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /
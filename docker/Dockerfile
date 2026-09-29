# ForgeAI sandbox image — used by app/safety/sandbox.py to isolate
# run_tests / run_linter execution from the host machine (Phase 7).
#
# Contains only what the allowlisted commands in app/tools/testing.py
# need: pytest, ruff, mypy. A repository whose OWN tests import extra
# third-party packages (e.g. this project's google-genai, chromadb) needs
# them added below, then the image rebuilt:
#
#   docker build -t forgeai-sandbox:latest docker/
#
# Deliberately built WITHOUT any repo's source baked in — the repo is
# mounted at /workspace at container *run* time instead (see sandbox.py),
# so one image is reusable across any Python repository ForgeAI touches.
# Network access is only ever available at build time, here, to install
# packages; every actual test/lint run happens with --network none.

FROM python:3.13-slim

RUN pip install --no-cache-dir pytest ruff mypy

# Never run test code as root, even inside a disposable container.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox

WORKDIR /workspace
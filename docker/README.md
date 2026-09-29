# ForgeAI Sandbox

`app/tools/testing.py` (`run_tests`, `run_linter`) executes inside this
Docker image by default, instead of directly on your machine. This is
Phase 7's fix for the class of vulnerability found during Phase 6 testing:
a test file that spawns arbitrary processes now runs isolated, with
`--network none`, no credentials, no filesystem access beyond the mounted
repo, and capped CPU/memory/process count.

## First-time setup

Nothing to do manually — the first `run_tests`/`run_linter` call builds
the image automatically (`docker build -t forgeai-sandbox:latest docker/`)
and reuses it on every later call. Docker Desktop just needs to be running.

## If a repository needs extra packages

The image only ships `pytest`, `ruff`, and `mypy` (see `Dockerfile`). A
repository whose own tests import third-party packages (this project's
`google-genai`, `chromadb`, etc.) needs those installed in the image too:

```dockerfile
RUN pip install --no-cache-dir pytest ruff mypy google-genai chromadb rank-bm25 GitPython
```

Then rebuild:

```powershell
docker build -t forgeai-sandbox:latest docker/
```

Network access is only ever available at build time, here — every actual
test/lint *run* still gets `--network none`.

## Opting out (not recommended)

Set `FORGEAI_SANDBOX=off` to run `run_tests`/`run_linter` directly on the
host instead, bypassing all isolation. Only useful for local debugging
when Docker isn't available.
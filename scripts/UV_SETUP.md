# Using `uv` for dependency management

We use `uv` to install and manage dependencies for the entire project.

## Prerequisites
- Install `uv` (see https://docs.astral.sh/uv/):  
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or on macOS with Homebrew: brew install uv
  ```

## Create/activate a virtual environment
```bash
uv venv .venv
source .venv/bin/activate
```

## Install all runtime dependencies
```bash
uv sync
```
This reads `pyproject.toml` and installs the full runtime dependency set (covers all tasks).

## Install optional groups
- Development tools: `uv sync --group dev`
- Docs toolchain: `uv sync --group docs`

You can combine groups, e.g. `uv sync --group dev --group docs`.

## Notes
- `uv pip` also works if you need to install an extra quickly, e.g. `uv pip install <pkg>`.
- The lockfile is not committed; re-running `uv sync` will resolve versions as needed. If you need a deterministic lock, run `uv lock` to generate `uv.lock`.

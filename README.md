# ML projects with python

_WIP_

## Requirements

The only tool you need to install is uv:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Getting started

This project uses a `.python-version` file to specify the Python version. uv will automatically use this file to determine which Python version to use.

### Python version management

When you clone this repository, uv will automatically detect the required Python version from the `.python-version` file. If you don't have that version installed, uv will download and install it for you.

```bash
# Check the Python version specified in .python-version
cat .python-version

# Create a virtual environment with the correct Python version
uv venv
```

The virtual environment will be created in the `.venv` directory.

### Installing dependencies

To install all project dependencies at once:

```bash
# Install dependencies from the lockfile
uv sync
```

This will create/update the virtual environment and install all dependencies according to the lockfile.

### Running commands in the project

You can run commands in the context of your project using `uv run`:

```bash
# Run a Python script in your project environment
uv run my_script.py

# Run a command in your project environment
uv run -- python -m pytest
```

Alternatively, you can activate the virtual environment:

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

After running the dependencies, make sure to set up pre-commit hooks:

```bash
pre-commit install
```

## VS-code setup

There's a `.vscode` folder in this repo with some settings that apply into the workspace. You'll need at least the `Python` and `PyLance` VSCode extensions. You can install them directly through the VSCode interface or by hitting `ctrl/cmd + shift + P > show extensions`.

For formatting and linting, you will need to install the following extensions:

1. [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
2. [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker)

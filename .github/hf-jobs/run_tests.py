# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run TorchSim's pytest suite inside a Hugging Face Job on GPU.

Invoked by the ``.github/workflows/gpu-*.yml`` workflows via ``hf jobs uv run``.

The script is deliberately dependency-free at the UV level: it shells out to
``uv pip install`` once inside the job to install PyTorch (CUDA wheels) and
TorchSim itself. This keeps the script portable across HF Jobs images and
avoids ever-so-slightly fragile UV source-resolution behaviour for torch.

Environment variables (all required unless noted):
    TORCHSIM_SHA        Commit SHA to test. Passed from the GH Action.
    TORCHSIM_REPO       Git URL (default: upstream TorchSim).
    PYTEST_ARGS         Extra pytest args, whitespace-separated.
                        "" for smoke, "-m physical_validation" for validation.
    TORCH_CUDA_INDEX    PyTorch wheel index (default: cu124). Override if the
                        HF Jobs base image ships a different CUDA runtime.
"""

from __future__ import annotations

import os
import subprocess
import sys


SHA = os.environ["TORCHSIM_SHA"]
REPO = os.environ.get(
    "TORCHSIM_REPO", "https://github.com/torchsim/torch-sim.git"
)
PYTEST_ARGS = os.environ.get("PYTEST_ARGS", "").split()
CUDA_INDEX = os.environ.get(
    "TORCH_CUDA_INDEX", "https://download.pytorch.org/whl/cu124"
)


def run(cmd: list[str]) -> None:
    """subprocess.run with visible command + check=True."""
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


print(f"JOB_ID={os.environ.get('JOB_ID', '?')}", flush=True)
print(f"ACCELERATOR={os.environ.get('ACCELERATOR', '?')}", flush=True)
print(f"Testing commit: {SHA}", flush=True)

# --- Clone the exact commit ------------------------------------------------
run(["git", "clone", "--filter=blob:none", REPO, "torch-sim"])
run(["git", "-C", "torch-sim", "checkout", SHA])
os.chdir("torch-sim")

# --- Install torch (CUDA) + TorchSim ---------------------------------------
# `hf jobs uv run` launches this script inside a UV-managed ephemeral venv,
# so `--system` would target the Docker image's interpreter — which the
# script's `sys.executable` cannot see. Use `--python sys.executable` to
# install into the interpreter the script is actually running under.
#
# Single install call, with the CUDA wheel index as the primary source and
# PyPI as the fallback. UV's default `first-index` strategy then pulls
# torch from CUDA_INDEX (the only index that has it) and everything else
# from PyPI. Splitting this into two steps causes pip to re-resolve torch
# and replace the CUDA wheel with the latest PyPI (CPU) build.
#
# setuptools<82 pin mirrors .github/workflows/test.yml — 82+ removed
# pkg_resources, which several model deps still need.
run([
    "uv", "pip", "install", "--python", sys.executable,
    "--index-url", CUDA_INDEX,
    "--extra-index-url", "https://pypi.org/simple",
    "torch",
    "-e", ".[test]",
    "setuptools>=70,<82",
])

# --- Fail fast if CUDA is missing ------------------------------------------
# A clearer error than "every test that uses DEVICE silently runs on CPU".
run([
    sys.executable, "-c",
    "import torch;"
    "assert torch.cuda.is_available(), 'CUDA is not available inside HF Job';"
    "print('torch:', torch.__version__, '| GPU:', torch.cuda.get_device_name())"
])

# --- Run the tests ---------------------------------------------------------
# PYTEST_ARGS selects scope:
#   smoke                 ""                           (default set per pyproject.toml)
#   physical validation   "-m physical_validation"     (opt-in; excluded by default)
pytest_cmd = [
    sys.executable, "-m", "pytest",
    "-vv", "-ra", "-rs",
    *PYTEST_ARGS,
]
rc = subprocess.call(pytest_cmd)
print(f"\npytest exited with code {rc}", flush=True)
sys.exit(rc)

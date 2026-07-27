#!/usr/bin/env bash
# ===========================================================================
# setup_vastai.sh
# Reproduce the mmdet3d detection stack (local venv_mmdet3d) on a Vast.ai GPU box.
#
# WHY THIS IS A SCRIPT AND NOT A requirements.txt:
#   Different components install by different mechanisms:
#     - torch/torchvision : use the image's CUDA build (do NOT reinstall)
#     - mmcv, mmdet       : via `mim` (resolves CUDA/torch-matched build)
#     - mmdet3d           : git clone + checkout pinned commit + install
#     - everything else   : plain pip (requirements-box.txt)
#
# USAGE:
#   1. Start the box, SSH in.
#   2. tmux new -s setup        # so a dropped SSH doesn't kill the install
#   3. bash setup_vastai.sh
#
# The script STOPS on the first error (set -e). Read each STOP/CHECK note.
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG — the versions/commit that match your working local stack.
# ---------------------------------------------------------------------------
MMDET3D_COMMIT="fe25f7a51d36e3702f961e198894580d83c4387b"
MMCV_VERSION="2.1.0"
MMDET_VERSION="3.2.0"
MMENGINE_VERSION="0.10.7"
NUMPY_VERSION="1.26.4"
EXPECTED_TORCH_MAJOR_MINOR="2.4"   # your local torch is 2.4.1; prefer a matching image

# ===========================================================================
# STEP 0 — SANITY: confirm the box is what we expect BEFORE installing anything.
# ===========================================================================
echo "=== STEP 0: environment sanity check ==="
python --version
echo "--- nvidia-smi ---"
nvidia-smi || { echo "!! nvidia-smi failed — is this actually a GPU box?"; exit 1; }
echo "--- torch / CUDA ---"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version (torch):", torch.version.cuda)
PY

cat <<'NOTE'

  >>> STOP AND READ <<<
  Confirm ALL of the following before continuing:
    1. "cuda available: True"   (if False, this box/image is wrong — stop here)
    2. torch version is ~2.4.x  (matches the local stack; other versions may
       force a different mmcv and are a version-matching risk)
  If torch is a very different version (2.3, 2.5, etc.), consider destroying
  this instance and picking an image with CUDA torch 2.4.x. It is cheaper to
  swap the image now than to fight mmcv compatibility later.

  Press ENTER to continue if the checks look right, or Ctrl-C to abort.
NOTE
read -r _

# ===========================================================================
# STEP 1 — numpy first (ABI-critical: numpy>=2 breaks the mmcv/numba ABI).
# ===========================================================================
echo "=== STEP 1: pin numpy ${NUMPY_VERSION} (ABI-critical) ==="
pip install "numpy==${NUMPY_VERSION}"

# ===========================================================================
# STEP 2 — mm-stack via mim (mmcv must be CUDA/torch-matched, not a plain wheel).
# ===========================================================================
echo "=== STEP 2: openmim + mmcv/mmdet/mmengine ==="
pip install openmim
mim install "mmcv==${MMCV_VERSION}"
mim install "mmdet==${MMDET_VERSION}"
pip install "mmengine==${MMENGINE_VERSION}"

# --- mmcv fallback (uncomment and fill CUDA/torch if `mim install mmcv` picks a
#     mismatched build). Replace cuXXX and torchY.Z with the box's real values,
#     e.g. cu121 / torch2.4 — read them from the STEP 0 output:
# pip install "mmcv==${MMCV_VERSION}" -f \
#   https://download.openmmlab.com/mmcv/dist/cuXXX/torchY.Z/index.html

# ===========================================================================
# STEP 3 — mmdet3d at the exact pinned commit (behavioral reproducibility).
#   Using the SAME commit as local guarantees the config names, NuScenesMetric
#   behavior, create_data.py flags, and file paths behave identically to what
#   you already debugged. Non-editable install (--no-build-isolation) is the
#   path that worked locally; editable failed with a PEP 660 error.
# ===========================================================================
echo "=== STEP 3: mmdet3d @ ${MMDET3D_COMMIT} ==="
if [ ! -d mmdetection3d ]; then
  git clone https://github.com/open-mmlab/mmdetection3d.git
fi
cd mmdetection3d
git checkout "${MMDET3D_COMMIT}"
pip install . --no-build-isolation
cd ..

# ===========================================================================
# STEP 4 — plain pip dependencies (nuScenes + IO + numerical).
#   NOTE: install ONLY opencv-python-headless (NOT opencv-python) on a headless
#   box. requirements-box.txt already excludes the non-headless build.
# ===========================================================================
echo "=== STEP 4: plain deps ==="
pip install -r requirements-box.txt

# ===========================================================================
# STEP 5 — VERIFY the install before spending GPU time on data/eval.
# ===========================================================================
echo "=== STEP 5: verification ==="
echo "--- pip check (dependency conflicts) ---"
pip check || echo "!! pip check reported issues — review above before proceeding"

echo "--- numpy still 1.26.x? (a later install may have bumped it) ---"
python -c "import numpy; print('numpy', numpy.__version__); assert numpy.__version__.startswith('1.26'), 'numpy drifted off 1.26 — mmcv/numba ABI at risk'"

echo "--- mmdet3d imports and CUDA is live? ---"
python - <<'PY'
import torch
from mmdet3d.apis import init_model   # noqa: F401
print("mmdet3d import: OK")
print("cuda available:", torch.cuda.is_available())
PY

cat <<'NOTE'

=== SETUP COMPLETE ===
Next steps (NOT part of this script):
  1. Download nuScenes to the box (inside tmux), then run create_data.py.
  2. SMOKE-TEST on mini first (81 frames) before any trainval run:
       reuse your v1.0-mini eval command to confirm the whole pipeline works
       on the box, cheaply, before paying for a full trainval pass.
  3. Only then launch the trainval eval / voxel / BEVFusion runs.

Reminder: voxel CenterPoint + BEVFusion need the CUDA ops that this GPU build
provides but your CPU laptop could not — this is the environment where they
finally run.
NOTE

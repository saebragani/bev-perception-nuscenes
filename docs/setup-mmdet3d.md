# mmdetection3d environment setup (known-good recipe)

Environment for Milestones 2–4 (CenterPoint, BEVFusion). This is the **verified working recipe**, reconstructed after resolving a long chain of version conflicts. Rebuilds identically on Vast.ai — only the torch line changes (CPU locally → CUDA on the GPU box).

**The pins below are load-bearing. Do not "upgrade" them — several are exact for non-obvious reasons documented inline.**

## The known-good version set

| package | version | note |
|---|---|---|
| Python | 3.10.13 | 3.12 breaks (removed `pkg_resources`/`ImpImporter`) |
| torch / torchvision | 2.4.1 / 0.19.1 (CPU) | mmcv publishes wheels for torch 2.4 |
| mmengine | 0.10.7 | via MIM |
| **mmcv** | **2.1.0 (compiled from source)** | no torch-2.4 wheel exists; must compile |
| **mmdet** | **3.2.0** | mmdet3d 1.4.0 requires mmdet `<3.3.0` |
| mmdet3d | 1.4.0 | from source |
| numpy | 1.26.4 | `<2` hard constraint; creeps up constantly |
| opencv-python-headless | 4.10.0.84 (`<4.11`) | full/newer opencv forces numpy≥2 |
| plyfile | `<1.1` | plyfile ≥1.1 requires numpy≥2 |
| setuptools | `>=64,<80` | ≥64 for PEP 660, <80 keeps `pkg_resources`, <82 for torch |

**Why mmcv must be compiled, not wheeled:** the mmdet3d 1.4.0 stack pins `mmcv<2.2.0` (enforced by a runtime `assert`). The only mmcv with a torch-2.4 *wheel* is 2.2.0 — just past that ceiling. So the compatible mmcv (2.1.0) has no wheel and must be built from source. There is no wheels-only path; the compile is unavoidable on this stack.

## Prerequisites

```bash
gcc --version && g++ --version    # need a C++ toolchain for the mmcv compile
# if missing: sudo apt install build-essential
```

## Steps

```bash
# 1. Python 3.10 for this directory (commit the resulting .python-version)
pyenv local 3.10.13

# 2. Fresh venv
rm -rf venv_mmdet3d
python -m venv venv_mmdet3d
source venv_mmdet3d/bin/activate
python --version                                  # MUST read 3.10.13

# 3. Build tooling: >=64 (PEP 660), <80 (keeps pkg_resources, under torch's <82 cap)
pip install -U pip
pip install "setuptools>=64,<80" wheel

# 4. PyTorch CPU (local). On Vast.ai: install the CUDA build matching the box instead.
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cpu

# 5. mmengine via MIM
pip install -U openmim
mim install mmengine

# 6. mmcv 2.1.0 — COMPILE from source (no torch-2.4 wheel; --no-build-isolation
#    so it uses the venv's torch + working setuptools instead of the empty sandbox).
#    Takes several minutes on CPU and prints a lot of compiler output — normal.
pip install --no-build-isolation "mmcv==2.1.0"

# 7. mmdet 3.2.0 (NOT 3.3.0 — mmdet3d 1.4.0 caps mmdet <3.3.0)
mim install "mmdet==3.2.0"
pip install "numpy<2"                              # mmdet drags numpy up — knock it back

# 8. mmdet3d from source, NON-editable (-e hits a PEP 660 backend error;
#    --no-build-isolation because its setup.py imports torch at build time).
#    You run tools/ scripts from inside this repo dir, so editable isn't needed.
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v . --no-build-isolation
cd ..
pip install "numpy<2"

# 9. nuScenes devkit (mmdet3d's converter imports it)
pip install nuscenes-devkit
pip install "numpy<2"

# 10. Fix opencv: the devkit/mmdet pull a too-new full opencv-python (needs numpy≥2).
#     Remove it, clear any leftover phantom cv2 dir, install headless <4.11.
pip uninstall -y opencv-python
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "$SITE/cv2"                                 # phantom empty cv2/ dir breaks imports if left
pip install --force-reinstall --no-deps "opencv-python-headless<4.11"

# 11. plyfile (<1.1 avoids its numpy≥2 requirement)
pip install "plyfile<1.1"

# 12. Final numpy pin
pip install "numpy<2"

# 13. Jupyter kernel hook (run Jupyter Lab from another env; just bridge to this one)
pip install ipykernel
python -m ipykernel install --user --name mmdet3d --display-name "mmdet3d"
```

## Verify (all must pass)

```bash
python -c "import cv2, mmcv, mmdet, mmdet3d; print(cv2.__version__, '|', mmcv.__version__, mmdet.__version__, mmdet3d.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "from nuscenes import NuScenes; print('devkit ok')"
python -c "import cv2; print('cv2 file:', cv2.__file__)"     # must be a real path, NOT None
```

Target: `4.10.0 | 2.1.0 3.2.0 1.4.0`, numpy 1.26.4, devkit ok, and `cv2.__file__` a real `.../cv2/__init__.py` path.

## Freeze

```bash
pip freeze > docs/known-good-env-mmdet3d.txt
```

Reproduce on Vast.ai by re-running this recipe with the CUDA torch (step 4) — not `pip install -r` of the CPU freeze, since torch differs by machine. On the GPU box, mmcv 2.1.0 may have a matching CUDA wheel (skipping the compile) — check before compiling.

## Gotchas / ignorable warnings

- **`cv2.__file__` is `None`** → a hollow namespace-package `cv2/` dir was left behind by an opencv uninstall. Fix: `rm -rf "$SITE/cv2"` then `pip install --force-reinstall --no-deps "opencv-python-headless<4.11"`. Step 10 already does this.
- **`pip check` says `opencv-python ... not installed`** (from nuscenes-devkit/mmengine/lyft-sdk) → harmless naming quirk. Those want *a* cv2; headless provides it. Ignore. Confirm with `import cv2; cv2.__version__`.
- **`openxlab requires setuptools~=60.2.0`** → over-tight, outdated pin. Ignore; downgrading setuptools breaks other things.
- **numpy keeps returning to 2.x** → any step printing `Uninstalling numpy` is the flag. Re-pin `"numpy<2"` and re-verify. It's a reflex, not a one-time fix.
- **CPU-only limitation** → sparse-conv/spconv ops are CUDA-only, so **voxel CenterPoint inference will not run on this local CPU env.** Use local for install + `tools/create_data.py` + config validation; run inference on the Vast.ai GPU env.
- **`mim resources not found: .../mmdet3d/.mim`** → mmdet3d installed from source doesn't copy its configs into `.mim`. Harmless when running `tools/` from inside the repo (configs resolve by relative path).

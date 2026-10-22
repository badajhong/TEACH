# VAIC Setup on NVIDIA DGX Spark

This guide installs VAIC natively on DGX Spark (`aarch64`) without conda. It
uses one isolated Python 3.11 virtual environment for Isaac Sim, Isaac Lab,
CUDA PyTorch, and VAIC.

The required stack is:

- Isaac Sim `5.1.0`
- Isaac Lab `v2.3.2`
- Python `3.11`
- PyTorch `2.9.0` from the CUDA 13.0 (`cu130`) index
- TorchRL `0.8.0` and TensorDict `0.8.1` on `aarch64`

The installation order matters. The Isaac Sim package metadata initially
installs PyTorch 2.7, whose PyPI ARM wheel is CPU-only. NVIDIA's Isaac Lab
instructions then explicitly replace it with PyTorch 2.9 from the `cu130`
index for DGX Spark.

Official references:

- [Isaac Lab v2.3.2 pip installation](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html)
- [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

## 1. Check the Host

```bash
uname -m
cat /etc/os-release
nvidia-smi
ldd --version
python3.11 --version
```

Confirm that:

- `uname -m` reports `aarch64`.
- Python is `3.11`.
- `nvidia-smi` sees the NVIDIA GB10 and a CUDA 13 capable driver.
- The system has `glibc >= 2.35`.

Isaac Sim 5.1 officially supports its `aarch64` build only on DGX Spark.

## 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
  python3.11-venv python3.11-dev \
  build-essential cmake libgomp1 \
  libgl1-mesa-dev libx11-dev libxcursor-dev \
  libxi-dev libxinerama-dev libxrandr-dev
```

## 3. Start Without Conda

If the prompt contains only `(base)`, deactivate conda before activating the
VAIC environment:

```bash
conda deactivate
```

If it already shows both `(vaic-dgx-spark)` and `(base)`, reset them in this
order:

```bash
deactivate
conda deactivate
source ~/.venvs/vaic-dgx-spark/bin/activate
```

The prompt should not contain `(base)` and `(vaic-dgx-spark)` at the same time.
To disable automatic base activation in future terminals:

```bash
conda config --set auto_activate_base false
```

Create the environment with standard `venv`:

```bash
python3.11 -m venv ~/.venvs/vaic-dgx-spark
source ~/.venvs/vaic-dgx-spark/bin/activate
python -m pip install --upgrade pip setuptools "wheel<0.46"

which python
python --version
python -c "import platform; print(platform.machine())"
```

An equivalent `uv` creation command is:

```bash
uv venv --python 3.11 --seed ~/.venvs/vaic-dgx-spark
source ~/.venvs/vaic-dgx-spark/bin/activate
```

Choose either `venv` or `uv venv`, not both. Use this activation command in
each new terminal:

```bash
source ~/.venvs/vaic-dgx-spark/bin/activate
```

## 4. Install Isaac Sim 5.1

```bash
python -m pip install \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
```

Do not verify PyTorch yet. At this point, pip may have installed the generic
CPU-only ARM build required by Isaac Sim's package metadata.

## 5. Install CUDA PyTorch for DGX Spark

This is the required aarch64 override from the Isaac Lab v2.3.2 instructions:

```bash
python -m pip install --upgrade \
  torch==2.9.0 \
  torchvision==0.24.0 \
  torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Verify that this is a CUDA build and that a real tensor operation reaches the
GB10:

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda"))'
```

Expected key values are `2.9.0+cu130`, CUDA `13.0`, and `True`. Do not continue
if the version ends in `+cpu` or CUDA availability is `False`.

PyTorch 2.9 may print a compute-capability warning for GB10 (`12.1` versus the
wheel's advertised `12.0` maximum). This is the version NVIDIA pins for Isaac
Lab v2.3.2 on Spark; confirm functionality with the CUDA tensor operation above.

## 6. Configure the ARM OpenMP Preload

Use the system OpenMP library, as required by the Isaac Lab aarch64 notes:

```bash
unset LD_PRELOAD
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
export OMNI_KIT_ACCEPT_EULA=YES
```

Run the real Isaac Sim compatibility experience:

```bash
isaacsim isaacsim.exp.compatibility_check \
  --/app/quitAfter=10 \
  --no-window
```

The final line should report `System checking result: PASSED`.

Do not test `import isaacsim` with `python -c` or a shell heredoc on ARM. Isaac
Sim's preload check starts a child process, and those forms do not provide a
normal guarded script path for multiprocessing to reopen.

## 7. Install Isaac Lab v2.3.2

```bash
cd ~/research
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
./isaaclab.sh -i none
```

If the repository already exists, only confirm the tag and install:

```bash
cd ~/research/IsaacLab
git checkout v2.3.2
./isaaclab.sh -i none
```

Keep the `vaic-dgx-spark` environment active for these commands. A warning
about no learning-framework extra being selected is expected for `-i none`.

## 8. Install VAIC

From this repository:

```bash
cd ~/research/FastSAC_RLPD_HER
python -m pip install -e .
```

On DGX Spark, `setup.py` selects the available ARM package pair and prevents
the editable install from replacing CUDA PyTorch with the CPU build:

- Torch `>=2.9,<2.10`
- TorchRL `0.8.0`
- TensorDict `0.8.1`
- W&B `0.19.11`
- ONNX `1.18.0`

Verify the combined Python stack without importing Isaac Sim from stdin:

```bash
python -c 'import torch, torchrl, tensordict, wandb, onnx; print(torch.__version__, torch.cuda.is_available()); print(torchrl.__version__, tensordict.__version__, wandb.__version__, onnx.__version__)'
```

## 9. Verify VAIC Startup

First compose the requested Hydra configuration without starting a training
run:

```bash
nohup python scripts/train.py \
  algo=fastsac_vel_train \
  task=G1/vaic/skateboard_tea \
  > sac_skateboard_tea.log 2>&1 &
```

Run one bounded training iteration with W&B networking disabled. Eight
environments are the minimum for this configuration because it uses eight
minibatches:

```bash
python scripts/train.py \
  algo=fastsac_vel_train \
  task=G1/vaic/skateboard_tea \
  task.num_envs=8 \
  total_frames=256 \
  save_interval=-1 \
  wandb.mode=disabled
```

After that succeeds, start the normal training job:

```bash
python scripts/train.py \
  algo=fastsac_vel_train \
  task=G1/vaic/skateboard_tea
```

## Repair an Existing CPU-Torch Environment

If `torch.__version__` reports `2.7.0+cpu`, repair the current venv with:

```bash
source ~/.venvs/vaic-dgx-spark/bin/activate

python -m pip install --upgrade \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install \
  wandb==0.19.11 onnx==1.18.0 protobuf==6.33.6 \
  click==8.1.7 typing_extensions==4.12.2 \
  torchrl==0.8.0 tensordict==0.8.1

cd ~/research/FastSAC_RLPD_HER
python -m pip install -e .
```

## Expected Metadata Warnings

`pip check` cannot be completely clean for this official aarch64 combination:

- Isaac Sim 5.1 metadata declares Torch 2.7, while the official Isaac Lab
  Spark instructions override it with CUDA Torch 2.9.
- Isaac Lab 2.3.2 pins `starlette==0.49.1`, while Isaac Sim's FastAPI package
  declares `starlette<0.46`.
- The CUDA index may install an SBSA-tagged cuSPARSELt wheel that older pip
  compatibility checks describe as unsupported even though CUDA PyTorch loads
  it on Spark.

Use the CUDA tensor test, Isaac Sim compatibility checker, and VAIC startup
test above as the runtime checks.

## DGX Spark Limitations

Isaac Sim 5.1 on DGX Spark does not support livestreaming, Hub Workstation
Cache, OBJ import, the Application Template, cuRobo, or cuMotion. OBJ import
also affects URDF assets that reference `.obj` meshes.

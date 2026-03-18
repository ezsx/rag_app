# Bypassing V100 TCC poisoning for GPU Docker in WSL2

**The root problem is architectural, not configurational: `nvidia-container-cli` enumerates all GPUs via NVML before processing any device filter, and NVIDIA offers no config option to prevent this.** The V100 in TCC mode is officially unsupported in WSL2 — NVIDIA's documentation explicitly states GPU acceleration is only available for WDDM-mode GPUs. Because NVML's initialization queries every GPU in the system, the V100's "gpu access blocked by the operating system" error fatally crashes `nvidia-container-cli` before `NVIDIA_VISIBLE_DEVICES=0` or `--device 0` can take effect. This is tracked as an open WSL issue (microsoft/WSL#8134 requests per-GPU blacklisting; #13027 documents TCC+WDDM dual-GPU poisoning). No config.toml pre-filter exists, no NVML environment variable restricts enumeration, and `CUDA_VISIBLE_DEVICES` only affects the CUDA runtime — not NVML.

The critical constraint — V100 must stay active on Windows for `llama-server.exe` while the RTX 5060 Ti serves TEI in WSL2 Docker — eliminates the simplest fix (disabling V100 in Device Manager). The five viable paths below are ordered by probability of success and effort.

## Solution 1: handcraft a CDI spec and switch to CDI mode

CDI (Container Device Interface) mode is the **most architecturally sound fix** because it completely sidesteps `nvidia-container-cli` and NVML at container startup. Instead of runtime GPU enumeration, the container runtime reads a pre-generated YAML spec that declares exactly which devices, mounts, and hooks to inject. Since `nvidia-ctk cdi generate` also fails (it calls NVML internally), the spec must be created manually.

**Step 1 — Identify your driver store directory:**
```bash
ls /usr/lib/wsl/drivers/
# Example output: nv_dispig.inf_amd64_xxxxxxxxxxxxxxxx
```
Only the RTX 5060 Ti's WDDM driver will appear here; the V100 TCC driver won't have a directory since TCC adapters are not exposed via DXCore.

**Step 2 — Create `/etc/cdi/nvidia.yaml`:**
```yaml
cdiVersion: "0.7.0"
kind: "nvidia.com/gpu"
devices:
  - name: "all"
    containerEdits:
      deviceNodes:
        - path: /dev/dxg
  - name: "0"
    containerEdits:
      deviceNodes:
        - path: /dev/dxg
containerEdits:
  env:
    - NVIDIA_VISIBLE_DEVICES=void
  hooks:
    - hookName: createContainer
      path: /usr/bin/nvidia-cdi-hook
      args:
        - nvidia-cdi-hook
        - update-ldcache
        - "--ldconfig-path"
        - /sbin/ldconfig
        - "--folder"
        - /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>
        - "--folder"
        - /usr/lib/wsl/lib
  mounts:
    - hostPath: /usr/lib/wsl/lib/libdxcore.so
      containerPath: /usr/lib/wsl/lib/libdxcore.so
      options: [ro, nosuid, nodev, bind]
    - hostPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/libcuda.so.1.1
      containerPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/libcuda.so.1.1
      options: [ro, nosuid, nodev, bind]
    - hostPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/libnvidia-ml.so.1
      containerPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/libnvidia-ml.so.1
      options: [ro, nosuid, nodev, bind]
    - hostPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/libnvidia-ptxjitcompiler.so.1
      containerPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/libnvidia-ptxjitcompiler.so.1
      options: [ro, nosuid, nodev, bind]
    - hostPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/nvcubins.bin
      containerPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/nvcubins.bin
      options: [ro, nosuid, nodev, bind]
    - hostPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/nvidia-smi
      containerPath: /usr/lib/wsl/drivers/<YOUR_DRIVER_DIR>/nvidia-smi
      options: [ro, nosuid, nodev, bind]
```
Replace `<YOUR_DRIVER_DIR>` with the actual directory from Step 1. You may need to add additional libraries depending on your driver version — list the full contents of the driver directory and `/usr/lib/wsl/lib/` and mount any `.so` files the TEI container needs.

**Step 3 — Switch Docker to CDI mode:**
```bash
sudo nvidia-ctk config --in-place --set nvidia-container-runtime.mode=cdi
sudo systemctl restart docker
```

**Step 4 — Run containers with CDI device syntax (NOT --gpus):**
```bash
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all \
  nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```
The `--gpus` flag is **incompatible** with CDI mode and will produce an error. Use `-e NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all` or `--device nvidia.com/gpu=all` (Docker 25+).

**Why this works**: CDI mode bypasses `nvidia-container-cli` entirely. The container receives `/dev/dxg` and the WSL driver libraries. Inside the container, `libcuda.so` calls DXCore's `D3DKMTEnumAdapters3()`, which only enumerates **WDDM adapters** — meaning it sees only the RTX 5060 Ti. The V100 in TCC mode is invisible to DXCore. CUDA operations proceed normally. This approach was confirmed working on WSL2 single-GPU systems in nvidia-container-toolkit issue #452, and the principle extends directly to this dual-GPU scenario because DXCore never exposes the TCC adapter.

## Solution 2: run an NVML shim that hides the V100

If CDI mode proves difficult to configure for your specific TEI container, an alternative is to intercept NVML at the library level. `nvidia-container-cli` uses `dlopen("libnvidia-ml.so.1")` to load NVML, then resolves symbols with `dlsym`. A wrapper library placed earlier in the search path can intercept this.

**Create the shim (`nvml_shim.c`):**
```c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

static void *real_nvml = NULL;

static void ensure_real_nvml(void) {
    if (!real_nvml) {
        // Load the REAL libnvidia-ml.so.1 by absolute path
        real_nvml = dlopen("/usr/lib/wsl/lib/libnvidia-ml.so.1", RTLD_LAZY);
    }
}

nvmlReturn_t nvmlInit_v2(void) {
    ensure_real_nvml();
    nvmlReturn_t (*real_fn)(void) = dlsym(real_nvml, "nvmlInit_v2");
    return real_fn();
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count) {
    ensure_real_nvml();
    nvmlReturn_t (*real_fn)(unsigned int*) = dlsym(real_nvml, "nvmlDeviceGetCount_v2");
    nvmlReturn_t ret = real_fn(count);
    if (ret == 0 && *count > 1) *count = 1; // Hide V100
    return ret;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int idx, nvmlDevice_t *dev) {
    ensure_real_nvml();
    nvmlReturn_t (*real_fn)(unsigned int, nvmlDevice_t*) =
        dlsym(real_nvml, "nvmlDeviceGetHandleByIndex_v2");
    if (idx >= 1) return 999; // NVML_ERROR_NOT_FOUND
    return real_fn(idx, dev);
}

// Forward the unversioned macros (nvml.h #defines these to _v2)
nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }
nvmlReturn_t nvmlDeviceGetCount(unsigned int *c) { return nvmlDeviceGetCount_v2(c); }
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int i, nvmlDevice_t *d) {
    return nvmlDeviceGetHandleByIndex_v2(i, d);
}

// Shutdown must also forward
nvmlReturn_t nvmlShutdown(void) {
    ensure_real_nvml();
    nvmlReturn_t (*real_fn)(void) = dlsym(real_nvml, "nvmlShutdown");
    return real_fn();
}
```

**Compile and deploy:**
```bash
gcc -shared -fPIC -o /opt/nvml-shim/libnvidia-ml.so.1 nvml_shim.c -ldl
```

**Inject via config.toml** (the `environment` field in `[nvidia-container-cli]` passes env vars to the CLI process):
```toml
[nvidia-container-cli]
environment = ["LD_LIBRARY_PATH=/opt/nvml-shim"]
```

This is a hacky approach. The shim must export every NVML symbol that `nvidia-container-cli` resolves via `dlsym`. If `libnvidia-container` calls functions not forwarded by the shim, it will crash. The shim shown above covers the critical initialization path, but you may need to add more forwarding functions. Check the debug log (`debug = "/var/log/nvidia-container-toolkit.log"`) to see exactly which NVML calls fail.

**Important caveat**: Because `libnvidia-container` loads NVML via `dlopen` rather than standard linking, the `LD_LIBRARY_PATH` trick works only if the code uses `dlopen("libnvidia-ml.so.1", ...)` without an absolute path. In WSL2 mode, libnvidia-container may resolve the library path through DXCore's driver store, potentially bypassing `LD_LIBRARY_PATH`. Test thoroughly.

## Solution 3: skip Docker entirely with infinity-emb

The fastest path to a working embedding server that avoids all Docker/NVML issues is **infinity-emb**, a Python-based embedding server with an OpenAI-compatible API. It runs natively in WSL2 without containers.

```bash
# Install PyTorch with CUDA 12.8 support (required for RTX 5060 Ti / Blackwell sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install infinity-emb
pip install "infinity-emb[torch,optimum]"

# Launch server
infinity_emb v2 \
  --model-id BAAI/bge-small-en-v1.5 \
  --engine torch \
  --device cuda \
  --port 7997
```

The API endpoint at `http://localhost:7997/v1/embeddings` is compatible with OpenAI's embedding format, making it a drop-in replacement for TEI in most pipelines. infinity-emb supports **dynamic batching**, multiple simultaneous models, and optimized backends (torch, ONNX, CTranslate2). If the torch engine has issues with sm_120, fall back to the ONNX engine (`--engine optimum`) which uses ONNX Runtime and typically has broader architecture support.

**The RTX 5060 Ti (Blackwell, sm_120) requires PyTorch built with CUDA 12.8.** Standard PyTorch pip packages only support up to sm_90. The `cu128` index URL provides Blackwell-compatible builds. Verify with `python -c "import torch; print(torch.cuda.is_available())"` before launching the server.

## Solution 4: build TEI from source on Ubuntu 22.04

TEI is a Rust application. Compiling from source links against your system's glibc **2.35**, eliminating the glibc 2.38/2.39 requirement of pre-built Docker binaries.

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install build deps
sudo apt-get install -y libssl-dev gcc pkg-config protobuf-compiler

# Clone and build
git clone https://github.com/huggingface/text-embeddings-inference
cd text-embeddings-inference

# For Blackwell (sm_120) — use candle-cuda-turing for cuBLAS path
# (Flash Attention kernels may not yet support sm_120)
cargo install --path router -F candle-cuda-turing -F http --no-default-features
```

Build takes **30–60 minutes** due to CUDA kernel compilation. The `candle-cuda-turing` feature uses cuBLAS/cuBLASLt operations that are architecture-agnostic and should work on sm_120 with CUDA Toolkit 12.8+. If that fails, try `candle-cuda` (includes Flash Attention, but may need sm_120 kernel updates). You'll need the CUDA Toolkit installed in WSL2 — use the WSL-specific installer from NVIDIA's download page (not the Windows one).

Alternatively, install a **fresh Ubuntu 24.04 WSL2 instance** alongside your existing 22.04:
```powershell
wsl --install Ubuntu-24.04
```
Ubuntu 24.04 ships glibc 2.39, letting you run the pre-built TEI binary extracted from the Docker image directly. Both WSL2 instances coexist — keep 22.04 as fallback and gradually migrate.

## Automate V100 management for sleep/wake cycles

While the solutions above address the container startup problem, V100 NVML state corruption after sleep/hibernate remains a nuisance. A Task Scheduler task triggered on wake can automatically reset the V100.

**Create `C:\Scripts\reset-v100.ps1`:**
```powershell
$v100 = Get-PnpDevice | Where-Object {
    $_.FriendlyName -like "*V100*" -and $_.Class -eq "Display"
}
if ($v100) {
    Disable-PnpDevice -InstanceId $v100.InstanceId -Confirm:$false
    Start-Sleep -Seconds 3
    Enable-PnpDevice -InstanceId $v100.InstanceId -Confirm:$false
    wsl --shutdown  # Force WSL2 to reinitialize GPU state
}
```

**Bind to wake event via Task Scheduler:**
Create a task triggered by Event ID **1** from source **Power-Troubleshooter** in the **System** log (this fires reliably on every wake from sleep/hibernate). Set it to run with highest privileges. The XML event filter:
```xml
<QueryList>
  <Query Id="0" Path="System">
    <Select Path="System">
      *[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]
    </Select>
  </Query>
</QueryList>
```

Running before sleep is unreliable — Event ID 42 (entering sleep) fires during the transition, and scripts triggered by it actually execute on the *next* wake. The disable-then-re-enable cycle on wake forces Windows to re-initialize the V100's driver state cleanly, and `wsl --shutdown` ensures WSL2 picks up the fresh GPU state on next launch.

## Conclusion

The fundamental issue is an unhandled edge case in `libnvidia-container`: NVML initialization fails fatally on any inaccessible GPU and blocks all GPUs, with no pre-enumeration filter available. **The CDI mode approach (Solution 1) is the cleanest Docker-based fix** because it bypasses the entire NVML enumeration path, relying instead on a static spec that only mounts DXCore resources for the working WDDM adapter. The RTX 5060 Ti remains fully functional through DXCore inside the container because TCC GPUs are invisible to the DXCore/WDDM stack — the poisoning occurs exclusively in NVML.

For fastest time-to-working-system, **infinity-emb (Solution 3) eliminates Docker from the equation entirely** and can be running in under 10 minutes. For long-term Docker usage, invest the time in crafting the CDI spec. The NVML shim (Solution 2) is a creative fallback but fragile.

Three actions worth taking in parallel: file a feature request on NVIDIA/libnvidia-container for graceful per-GPU error handling during NVML enumeration, upvote microsoft/WSL#8134 (GPU blacklist in .wslconfig), and upvote microsoft/WSL#13027 (TCC+WDDM dual-GPU poisoning). These are the proper long-term fixes that would make the workarounds unnecessary.
# Why Flash Attention hangs in TEI on WSL2 with RTX 5060 Ti

The hang stems from **three compounding failures**, not one. Flash Attention flat-out lacks sm_120 kernels, the CDI spec is almost certainly missing the `update-ldcache` hook needed for CUDA library resolution, and the WSL2 Blackwell driver stack itself has a documented CUDA initialization defect. Even fixing two of these leaves the third blocking you. The fastest path to working embeddings is abandoning TEI's Flash Attention entirely — either via `USE_FLASH_ATTENTION=False` or by switching to infinity-emb with standard PyTorch SDPA.

---

## "CUDA Version: N/A" reveals a broken driver stack, not a cosmetic glitch

The `nvidia-smi` output showing **"CUDA Version: N/A"** is not normal for WSL2. Many WSL2 users see the CUDA version correctly (e.g., "CUDA Version: 12.8"). This field reports the maximum CUDA version supported by the driver, queried via `nvmlSystemGetCudaDriverVersion()`. When it reads N/A, **NVML cannot communicate with the CUDA driver layer** — meaning `libcuda.so` is either missing, unmounted, or non-functional inside the container.

Critically, nvidia-smi can enumerate GPU properties (name, memory, temperature) through DXCore/WDDM *without* a working `libcuda.so`. DXCore alone handles basic GPU discovery via `/dev/dxg` and `libdxcore.so`. So **seeing the GPU and even observing VRAM allocation does not confirm CUDA compute works**. Memory allocation through DXCore and CUDA kernel dispatch are separate code paths. The fact that TEI reports "2347 MB allocated" means DXCore initialization succeeded, but CUDA context creation for compute likely did not.

On WSL2, all CUDA operations flow through a specific chain: **libcuda.so → libdxcore.so → /dev/dxg → VMBUS → Windows host driver**. DXCore fully supports CUDA kernel launch, PTX JIT compilation, and all standard CUDA runtime operations — NVIDIA's performance benchmarks show WSL2 achieves ~90%+ of native Linux CUDA performance. The bottleneck here is not DXCore's capabilities but whether the container has a properly linked `libcuda.so`.

---

## The CDI spec is almost certainly incomplete

Your hand-crafted CDI spec mounts `/dev/dxg`, `/usr/lib/wsl/lib/`, and the driver store directory. This covers the raw files but likely misses the **`createContainer` hook** that the NVIDIA Container Toolkit normally injects. This hook runs `nvidia-cdi-hook update-ldcache --ldconfig-path /sbin/ldconfig --folder /usr/lib/wsl/lib --folder /usr/lib/wsl/drivers/<path>`, which does two critical things: registers the mounted library paths in the container's dynamic linker cache, and creates proper symlinks for `libcuda.so → libcuda.so.1 → libcuda.so.1.1`. Without it, mounted libraries exist at the correct paths but **the dynamic linker cannot find them**.

WSL2's `/usr/lib/wsl/lib/libcuda.so.1` is famously not a proper symbolic link — it's a regular file copy (documented in Microsoft/WSL issues #5663 and #8587). The `update-ldcache` hook works around this by creating proper symlinks inside the container. A complete CDI spec for WSL2 requires:

- **Device node:** `/dev/dxg` (the only device node — there is no `/dev/dxcore`)
- **Library mounts:** Everything from `/usr/lib/wsl/lib/` and `/usr/lib/wsl/drivers/<inf_folder>/`, including `libcuda.so.1.1`, `libdxcore.so`, `libnvidia-ml.so.1`, `libnvdxgdmal.so.1` (required since driver ≥550.x), and `nvcubins.bin` (critical for CUDA compute, only tracked by toolkit since v1.12.0)
- **Hook:** `nvidia-cdi-hook update-ldcache` as a `createContainer` hook
- **Environment:** `NVIDIA_VISIBLE_DEVICES=void` (prevents the legacy container runtime hook from conflicting)

The recommended fix is to install `nvidia-container-toolkit` on WSL2 and run `nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` rather than hand-crafting the spec. Also ensure `no-cgroups = true` is set in `/etc/nvidia-container-runtime/config.toml`, as WSL2 does not support Linux cgroups for GPU management.

---

## Flash Attention fundamentally cannot run on sm_120 today

Even with a perfect CUDA stack, **Flash Attention will not work on the RTX 5060 Ti**. The situation is more nuanced than "Blackwell unsupported" because sm_120 (desktop Blackwell: RTX 50-series) and sm_100 (data center Blackwell: B200/B300) are architecturally different at the silicon level. SM100 has dedicated Tensor Memory (TMEM) hardware with `tcgen05` instructions; **SM120 physically lacks TMEM** and instead uses the older HMMA register-to-register MMA instruction family from Ampere.

The current Flash Attention landscape for sm_120:

- **Flash Attention 2** compiles for sm_120 (the build system includes `-gencode arch=compute_120,code=sm_120`) but the **runtime gating logic** in `flash_api.cpp` blocks execution with "FlashAttention only supports Ampere GPUs or newer." Multiple open GitHub issues (#1665, #1929, #1987) request sm_120 support with no fix timeline from maintainers.
- **Flash Attention 3** explicitly excludes all Blackwell architectures (≥sm_100). It uses Hopper-specific GMMA and TMA instructions that sm_120 does not support in the same way.
- **Flash Attention 4** targets SM100 exclusively using `tcgen05.mma` instructions that **do not exist on SM120 hardware** — `ptxas` rejects them outright for `sm_120a`.
- **TEI's router-120 binary** uses `candle-cuda-turing` (Flash Attention v1), but candle itself has a compute capability validation bug where even matching compile-time and runtime sm_120 produces an error: "Runtime compute cap 120 is not compatible with compile time compute cap 120" (TEI issue #652).

The irony is that SM120's tensor cores can execute the SM80-compatible `mma.sync.aligned.m16n8k16` instructions that FA2 kernels actually use. A community developer confirmed this by writing a [custom FA2 implementation in CUDA C++](https://gau-nernst.github.io/fa-5090/) that runs correctly on RTX 5090. The blocker is purely software-side runtime checks and untested code paths, not hardware capability.

PTX forward compatibility (compiling for `compute_80` and JIT-compiling to sm_120 at runtime) could theoretically work, but TEI's binaries embed only SASS cubins for their target architectures and candle's runtime check would reject the mismatch anyway.

---

## USE_FLASH_ATTENTION=False is the TEI-specific fix

TEI supports disabling Flash Attention via the **`USE_FLASH_ATTENTION=False` environment variable** (formalized in PR #692, released in v1.8.1). This causes TEI to use non-flash model implementations with **cuBLASLt** for attention computation instead of custom CUDA kernels.

```bash
docker run --gpus all \
  -e USE_FLASH_ATTENTION=False \
  -p 8080:80 -v $PWD/data:/data \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id Qwen/Qwen3-Embedding-0.6B
```

There is no `--disable-flash-attention` CLI flag — only the environment variable. There are two important caveats. First, the `cuda-1.9` image's router-120 binary must have been compiled with non-flash code paths for sm_120, and the candle compute capability validation bug (issue #652) may still trigger. Second, some Qwen3 model architectures initially required Flash Attention and lacked non-flash implementations (TEI issue #630), though this was later fixed. If `USE_FLASH_ATTENTION=False` still fails, building TEI from source with `CUDA_COMPUTE_CAP=120` and the `candle-cuda-turing` feature flag (which skips FA2 compilation entirely) is the next step.

---

## The WSL2 Blackwell driver defect adds a third failure layer

Independent of Flash Attention, there is a **documented WSL2 driver-level issue with Blackwell GPUs**. An NVIDIA Developer Forum post from September 2025 describes an RTX 5060 Ti where nvidia-smi works perfectly on WSL2 but `torch.cuda.is_available()` returns `False`. A corresponding PyTorch issue (#162403) was filed and tagged "Blackwell" and "module: wsl" but closed as "needs reproduction." The reporter described it as a failure "in the driver's API layer as it is exposed to the WSL environment."

Driver version 581.80 (your version, corresponding to Linux-side 580.105.07) was among the early Blackwell WSL2 drivers. Later driver releases may have resolved this. Before investing time in TEI configuration, run a minimal CUDA test inside your container:

```bash
python3 -c "import ctypes; cuda = ctypes.CDLL('libcuda.so.1'); print('cuInit result:', cuda.cuInit(0))"
```

If `libcuda.so.1` fails to load, the CDI spec is the problem. If `cuInit` returns a non-zero error code, the WSL2 Blackwell driver is the problem. Only if both succeed should you proceed to debug TEI/Flash Attention.

---

## Alternative embedding servers that bypass all three issues

The most pragmatic path forward avoids TEI's custom CUDA kernels entirely. Three alternatives stand out, all supporting both Qwen3-Embedding-0.6B and BGE-M3:

**infinity-emb** (recommended) is a high-throughput FastAPI-based embedding server with explicit Blackwell support added in July 2025. Using `--engine torch`, it runs standard PyTorch SDPA attention — no Flash Attention dependency. Install PyTorch 2.7+ with cu128, install `infinity_emb[torch]`, and *do not* install the `flash-attn` package. It provides an OpenAI-compatible API, dynamic batching, and multi-model serving:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "infinity_emb[torch]"
infinity_emb v2 --model-id Qwen/Qwen3-Embedding-0.6B --model-id BAAI/bge-m3 --engine torch --device cuda
```

**Ollama** is the zero-configuration option. It uses llama.cpp (not PyTorch or Flash Attention) and already supports Blackwell natively. Both models are available as pre-quantized GGUF (`ollama pull qwen3-embedding:0.6b`). The trade-off is quantized-only inference and no OpenAI-compatible API natively.

**sentence-transformers + FastAPI** is the maximum-control fallback. Pure PyTorch inference path, works on any GPU that PyTorch supports. PyTorch 2.7.0+ stable with cu128 wheels fully supports sm_120 — confirmed by PyTorch staff. Requires manual API implementation and lacks built-in dynamic batching.

## Conclusion

The hang has three independent causes acting simultaneously. The CDI spec's missing `update-ldcache` hook prevents `libcuda.so` from being resolvable, which explains the "CUDA Version: N/A" signal and would cause any CUDA kernel launch to fail silently. The WSL2 Blackwell driver stack (v581.80) has a documented defect where CUDA context initialization fails despite nvidia-smi working. And Flash Attention's runtime gating explicitly blocks sm_120, with no fix timeline from maintainers — the desktop Blackwell architecture lacks the TMEM hardware that FA3/FA4 target, and FA2's runtime checks reject it despite hardware compatibility.

The diagnostic sequence should be: fix the CDI spec (or use `nvidia-ctk cdi generate`), test raw CUDA initialization with `cuInit()`, and then either set `USE_FLASH_ATTENTION=False` or — more reliably — switch to infinity-emb with `--engine torch` on PyTorch cu128. The SM120 Flash Attention gap will likely close as the ecosystem matures, but for production embedding serving today, standard PyTorch SDPA on Blackwell delivers correct results without the kernel compatibility minefield.
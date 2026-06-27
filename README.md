# GGUF Chatbot — RTX 4050 6 GB Optimised

Single-file Python chatbot for GGUF models.  
Streams output · GPU offloading · auto context trimming · zero idle RAM.

---

## Quick start

```bash
bash install.sh                         # one-time setup
source venv/bin/activate
python chat.py --model /path/to/model.gguf
```

---

## Recommended models for 6 GB VRAM + 8 GB RAM

All figures are approximate. Always download Q4_K_M or Q3_K_M quants.

| Model              | Quant     | Size   | GPU layers | RAM use | Notes                      |
|--------------------|-----------|--------|------------|---------|----------------------------|
| Llama 3.1 8B       | Q4_K_M    | ~5 GB  | -1 (all)   | ~1 GB   | Fully on GPU, fastest      |
| Mistral 7B v0.3    | Q4_K_M    | ~4.5 GB| -1 (all)   | ~1 GB   | Great for coding/chat      |
| Qwen2.5 14B        | Q4_K_M    | ~9 GB  | 20         | ~5 GB   | Mixed GPU+CPU              |
| Llama 3.1 70B      | Q2_K      | ~26 GB | 10–14      | ~8 GB   | Slow but huge model        |
| Llama 3.3 70B      | IQ3_M     | ~29 GB | 10         | ~8 GB   | Best 70B quality at Q3     |
| DeepSeek-R1 32B    | Q3_K_M    | ~13 GB | 18–22      | ~8 GB   | Strong reasoning           |
| Qwen2.5 32B        | Q3_K_M    | ~14 GB | 16–20      | ~8 GB   | Best 32B overall           |
| Phi-4 14B          | Q4_K_M    | ~9 GB  | 20         | ~5 GB   | Strong for its size        |

**Rule of thumb for GPU layers:**
- Every 7B layer ≈ 40–60 MB VRAM (Q4_K_M)  
- Every 30B layer ≈ 120–150 MB VRAM (Q4_K_M)  
- Start with `--gpu-layers 20`, then increase until you hit OOM, then back off 2.

**Where to get models:**  
https://huggingface.co/bartowski  (pre-quantised GGUF, well-maintained)  
https://huggingface.co/TheBloke   (older but huge selection)

---

## Key flags

```
--model      PATH   path to .gguf file             (required)
--gpu-layers N      layers on GPU (999=auto-max)   default: 999
--ctx        N      context window tokens           default: 4096
--batch      N      prompt batch size               default: 512
--threads    N      CPU threads                     default: 6
--kv-bits    N      KV cache precision 1/8/2        default: 1 (f16)
--max-tokens N      max reply tokens                default: 1024
--temp       F      temperature                     default: 0.7
--system     TEXT   system prompt                   default: helpful assistant
--seed       N      RNG seed                        default: -1 (random)
```

### Save VRAM with KV cache quantisation

```bash
# Use q8_0 KV cache — ~40 % less VRAM on cache, minimal quality loss
python chat.py --model model.gguf --kv-bits 8

# Use q4_0 KV cache — ~60 % less VRAM, slight quality drop at long context
python chat.py --model model.gguf --kv-bits 2
```

### Example: 32B model, mixed GPU/CPU

```bash
python chat.py \
  --model ~/models/qwen2.5-32b-instruct-q3_k_m.gguf \
  --gpu-layers 20 \
  --ctx 4096 \
  --kv-bits 8 \
  --threads 6 \
  --max-tokens 512
```

---

## In-chat commands

| Command          | Description                                 |
|------------------|---------------------------------------------|
| `/clear`         | Reset conversation history                  |
| `/info`          | Show model metadata + memory stats          |
| `/ctx`           | Show context token usage bar                |
| `/sys <text>`    | Change system prompt (clears history)       |
| `/save <file>`   | Save conversation to JSON                   |
| `/load <file>`   | Restore conversation from JSON              |
| `/exit`          | Quit                                        |

---

## Memory layout (how it works)

```
┌─────────────────────────────────────────────────────────┐
│  .gguf file on disk  (memory-mapped, not fully loaded)   │
└────────────────────┬────────────────────────────────────┘
                     │  mmap — OS pages in on demand
        ┌────────────▼────────────┐
        │  RAM  (CPU layers)      │  ← remaining layers
        │  up to ~8 GB            │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  VRAM (GPU layers)      │  ← --gpu-layers N
        │  RTX 4050  6 GB         │  ← + KV cache (flash attn)
        └─────────────────────────┘
```

- `use_mmap=True`  — the full model is **not** loaded into RAM; the OS maps it.
- `use_mlock=False` — pages can be swapped; critical for laptop with 16 GB.
- `flash_attn=True` — reduces KV cache VRAM by ~30 %.
- History is auto-trimmed when it exceeds 75 % of the context window.

---

## Troubleshooting

**CUDA out of memory**  
Reduce `--gpu-layers` by 2–4 or switch to `--kv-bits 8`.

**Slow generation (< 3 tok/s)**  
Most of the model is on CPU. Either use a smaller/more-quantised model, or accept the speed — CPU inference of big models is inherently slow.

**`CUDA error: no kernel image`**  
Your CUDA toolkit and driver versions are mismatched. Run `nvidia-smi` and check driver-supported CUDA, then reinstall toolkit.

**`libcuda.so not found` during build**  
```bash
sudo ldconfig /usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Then reinstall llama-cpp-python.

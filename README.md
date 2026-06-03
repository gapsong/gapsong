## ML Engineer · LLM Efficiency & Quantization

I work at the intersection of extreme model compression and 
production-grade fine-tuning.

**What I actually do:** I make large language models run on hardware 
that wasn't supposed to support them — without destroying what makes 
them useful.

**Shipped:**
- 🔧 Official QA-LoRA implementation in Hugging Face PEFT
  → [PR #2571](https://github.com/huggingface/peft/pull/2571) · 
     [PR #2664](https://github.com/huggingface/peft/pull/2664)
- 📄 Master Thesis @ TU Berlin (supervised by Prof. Samek & 
  Prof. Müller, Fraunhofer HHI):
  "Accelerating Quantization-Aware Training of 2-bit Compact LLMs"
  → Proposed SA-SVD: -63% training VRAM vs. standard LoRA, 
    +150 perplexity points recovery on broken 2-bit models
  → Proposed DRA: error-based adapter initialization for 
    parameter-efficient fine-tuning on resilient architectures
- 🧪 SA-SVD reference implementation (open source, reproducible)
  → [gapsong/sa-svd-qa-lora](https://github.com/gapsong/sa-svd-qa-lora)
  → Measured at 2-bit across three LLMs: better WikiText perplexity 
    on every model tested (-15% to -48%) at identical training budget — 
    and it makes Qwen2-1.5B trainable where standard random-init 
    QA-LoRA diverges with inf gradients

**Stack:** PyTorch · Hugging Face (PEFT, Transformers, TRL) · 
GPTQ · bitsandbytes · Slurm · CUDA · AWS

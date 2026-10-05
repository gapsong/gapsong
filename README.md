## Fullstack and ML Engineer · LLM Efficiency & Quantization

I work at the intersection of extreme model compression and 
production-grade fine-tuning.

**What I actually do:** I make large language models run on hardware 
that wasn't supposed to support them without destroying what makes 
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
- 🧰 qpeft: quantization-aware training and PEFT for LLMs at 2/3/4 bits
  → [gapsong/qpeft](https://github.com/gapsong/qpeft) ·
    EfficientQAT, QA-LoRA and PEQA as configs over one quantized substrate;
    the merge stays an integer model, tested to equal the trained one

**Tools I build for myself** (agent-assisted with Claude Code, used daily, open source):
- 🎙️ [mac-voice-dictation](https://github.com/gapsong/mac-voice-dictation) ·
  push-to-talk dictation for macOS: hold a key, speak, release.
  Native Swift, Whisper on the Mac's own GPU, nothing leaves the machine.
- ⚡ [whisper-service](https://github.com/gapsong/whisper-service) ·
  the on-device Whisper server behind it: MLX on Apple Silicon,
  Silero VAD, about 0.2 s per sentence.
- 📊 [claude-code-statusline](https://github.com/gapsong/claude-code-statusline) ·
  model, context, usage limits and git state in one portable bash statusline.

**Stack:** PyTorch · Hugging Face (PEFT, Transformers, TRL) · 
GPTQ · bitsandbytes · Slurm · CUDA · AWS

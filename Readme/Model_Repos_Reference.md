# Model Repository Reference

Quick reference of model repositories for testing with Smart Language Model Loader templates.

## Qwen Vision-Language Models

### Qwen3-VL Series (Latest)

**2B Models:**
- `Qwen/Qwen3-VL-2B-Instruct` - Standard 2B model (4GB VRAM)
- `Qwen/Qwen3-VL-2B-Instruct-FP8` - FP8 quantized (2.5GB VRAM)
- `Qwen/Qwen3-VL-2B-Thinking` - Reasoning variant (4GB VRAM)
- `Qwen/Qwen3-VL-2B-Thinking-FP8` - FP8 quantized reasoning (2.5GB VRAM)

**4B Models:**
- `Qwen/Qwen3-VL-4B-Instruct` - Standard 4B model (6GB VRAM)
- `Qwen/Qwen3-VL-4B-Instruct-FP8` - FP8 quantized (3.5GB VRAM)
- `Qwen/Qwen3-VL-4B-Thinking` - Reasoning variant (6GB VRAM)
- `Qwen/Qwen3-VL-4B-Thinking-FP8` - FP8 quantized reasoning (3.5GB VRAM)

**8B Models:**
- `Qwen/Qwen3-VL-8B-Instruct` - Standard 8B model (12GB VRAM)
- `Qwen/Qwen3-VL-8B-Instruct-FP8` - FP8 quantized (7.5GB VRAM)
- `Qwen/Qwen3-VL-8B-Thinking` - Reasoning variant (12GB VRAM)
- `Qwen/Qwen3-VL-8B-Thinking-FP8` - FP8 quantized reasoning (7.5GB VRAM)

**32B Models:**
- `Qwen/Qwen3-VL-32B-Instruct` - Standard 32B model (28GB VRAM)
- `Qwen/Qwen3-VL-32B-Instruct-FP8` - FP8 quantized (24GB VRAM)
- `Qwen/Qwen3-VL-32B-Thinking` - Reasoning variant (28GB VRAM)
- `Qwen/Qwen3-VL-32B-Thinking-FP8` - FP8 quantized reasoning (24GB VRAM)

### Qwen2.5-VL Series (Previous Generation)

- `Qwen/Qwen2.5-VL-3B-Instruct` - 3B model (6GB VRAM)
- `Qwen/Qwen2.5-VL-7B-Instruct` - 7B model (15GB VRAM)

## Florence-2 Models

### Official Microsoft Models

**Base Models (230M params, ~0.5GB):**
- `microsoft/Florence-2-base` - Base pretrained
- `microsoft/Florence-2-base-ft` - Fine-tuned variant

**Large Models (770M params, ~1.5GB):**
- `microsoft/Florence-2-large` - Large pretrained
- `microsoft/Florence-2-large-ft` - Fine-tuned variant

### Specialized Fine-tunes

**Document Understanding:**
- `HuggingFaceM4/Florence-2-DocVQA` - Document Q&A

**Community Fine-tunes:**
- `thwri/CogFlorence-2.1-Large` - CogFlorence variant
- `thwri/CogFlorence-2.2-Large` - Updated CogFlorence

**Caption/Prompt Generation:**
- `gokaygokay/Florence-2-SD3-Captioner` - SD3 optimized captions
- `gokaygokay/Florence-2-Flux-Large` - Flux optimized captions
- `MiaoshouAI/Florence-2-base-PromptGen-v1.5` - Base prompt gen v1.5
- `MiaoshouAI/Florence-2-large-PromptGen-v1.5` - Large prompt gen v1.5
- `MiaoshouAI/Florence-2-base-PromptGen-v2.0` - Base prompt gen v2.0
- `MiaoshouAI/Florence-2-large-PromptGen-v2.0` - Large prompt gen v2.0
- `PJMixers-Images/Florence-2-base-Castollux-v0.5` - Castollux variant

### LoRA Adapters

- `NikshepShetty/Florence-2-pixelprose` - Pixel prose style

## Usage Examples

### Creating QwenVL Template (Transformers)
```
Template Name: Qwen3-VL-2B-Instruct
Model Type: QwenVL (Transformers)
Model Source: HuggingFace
Repo ID: Qwen/Qwen3-VL-2B-Instruct
Quantized: No
VRAM Full: 4.0 GB
```

### Creating QwenVL Template (Pre-quantized FP8)
```
Template Name: Qwen3-VL-2B-FP8
Model Type: QwenVL (Transformers)
Model Source: HuggingFace
Repo ID: Qwen/Qwen3-VL-2B-Instruct-FP8
Quantized: Yes
VRAM Full: 2.5 GB
```

### Creating Florence-2 Template
```
Template Name: Florence-2-Base
Model Type: Florence2 (Transformers)
Model Source: HuggingFace
Repo ID: microsoft/Florence-2-base
Quantized: No
VRAM Full: 0.5 GB
```

### Creating Florence-2 Prompt Gen Template
```
Template Name: Florence-PromptGen-Large
Model Type: Florence2 (Transformers)
Model Source: HuggingFace
Repo ID: MiaoshouAI/Florence-2-large-PromptGen-v2.0
Quantized: No
VRAM Full: 1.5 GB
```

## Notes

- **FP8 Models**: Pre-quantized, set "Quantized" to Yes
- **Transformers Models**: Can apply runtime quantization (4bit/8bit) if not pre-quantized
- **GGUF Models**: Must specify context_size, no runtime quantization available
- **VRAM Values**: Approximate full precision requirements, actual usage varies by context size and batch
- **Model Type**: 
  - QwenVL supports image + video inputs
  - Florence-2 supports image only (no video)
  - LLM is text-only (no vision)

## Testing Checklist

- [ ] QwenVL Transformers (standard)
- [ ] QwenVL Transformers (pre-quantized FP8)
- [ ] QwenVL GGUF (with mmproj)
- [ ] Florence-2 Base
- [ ] Florence-2 Large
- [ ] Florence-2 specialized (DocVQA, PromptGen)
- [ ] Text-only LLM (GGUF)
- [ ] Local model selection (from models/LLM folder)
- [ ] Template auto-save functionality
- [ ] Runtime quantization (4bit/8bit on non-quantized models)

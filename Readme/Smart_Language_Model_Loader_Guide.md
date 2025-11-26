# Smart Language Model Loader Guide

**Unified Vision-Language Models for Image & Video Understanding**

The Smart Language Model Loader (Smart LML) provides unified access to multiple model types for analyzing images, videos, and text within your ComfyUI workflows. This single node replaces multiple specialized loaders by supporting QwenVL vision-language models, Florence-2 fast captioning models, and text-only LLM models - all with pre-configured templates for easy model selection.

## What Can This Node Do?

Smart Language Model Loader enables you to:

- **Analyze Images:** Generate detailed descriptions, tags, captions, or custom analyses of any image
- **Understand Videos:** Summarize video content, analyze scenes across multiple frames (QwenVL only)
- **Auto-Tag Content:** Fast, accurate tag generation optimized for Stable Diffusion and Flux prompts
- **Extract Text:** OCR for documents, screenshots, and images with text
- **Detect Objects:** Find and locate specific objects with bounding boxes (Florence-2)
- **Create Training Data:** Generate detailed captions for LoRA/DreamBooth training
- **Enhance Prompts:** Expand and refine text prompts with LLM models
- **Ask Questions:** Query images with natural language ("What is the person wearing?")

The node automatically downloads models on first use, manages VRAM efficiently, and supports both transformers (full quality) and GGUF (low VRAM) formats.

## Table of Contents

- [What is Smart LML?](#what-is-smart-lml)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Node Overview](#node-overview)
- [Template System](#template-system)
- [Model Types](#model-types)
- [Common Use Cases](#common-use-cases)
- [Parameters Guide](#parameters-guide)
- [Performance & VRAM](#performance--vram)
- [Troubleshooting](#troubleshooting)
- [Tips & Best Practices](#tips--best-practices)
- [Model Repository Reference](Model_Repos_Reference.md) - HuggingFace URLs for all supported models

---

## How Does It Work?

The node provides a unified interface for three types of AI models. Simply:
1. Select a template (pre-configured model setup)
2. Connect your images or provide text
3. Choose a task or prompt style
4. Run the workflow

Models download automatically and are cached in `ComfyUI/models/LLM/` for offline use. Pre-configured templates are included for all supported models.

## Model Types Explained

Smart Language Model Loader is a unified framework that supports three types of AI models:

### 1. **QwenVL Models** (Vision-Language)
- Analyze images and videos with natural language understanding
- Generate detailed descriptions, tags, captions
- Support video analysis with multiple frames
- Available in sizes from 2B to 32B parameters
- Support both transformers and GGUF formats

### 2. **Florence-2 Models** (Vision-Language)
- Fast, efficient image captioning and analysis
- Specialized tasks (OCR, object detection, prompt generation)
- Optimized for real-time workflows
- Small model sizes (230M-770M parameters)
- Images only (no video support)

### 3. **LLM Models** (Text-Only)
- Text-to-text processing without vision
- Prompt refinement and expansion
- Tags to natural language conversion
- Supports both transformers and GGUF formats

---

## Quick Start

### Your First Image Analysis (5 Minutes)

**Goal:** Get a detailed description of an image.

**Steps:**
1. In ComfyUI, add **Smart Language Model Loader [Eclipse]** node to your workflow
2. Connect an **IMAGE** output (from Load Image, generation node, etc.) to the `images` input
3. In the node settings:
   - `template_name`: Select `Qwen2.5-VL-3B-Instruct` (default, downloads ~6GB on first use)
   - `qwen_preset_prompt`: Select `Detailed Description`
4. Click **Queue Prompt** to run

The model downloads automatically on first use (stored in `ComfyUI/models/LLM/`). After loading completes (~30-60 seconds first time), you'll see a detailed paragraph describing your image in the `text` output.

**Expected Result:** 
```
A young woman with long brown hair stands in a sunlit garden, 
wearing a flowing white dress. Cherry blossoms bloom on nearby 
trees, their pink petals contrasting with the bright blue sky. 
She gazes toward the camera with a gentle smile.
```

### Fast Auto-Tagging (1-2 Seconds Per Image)

**Goal:** Generate Stable Diffusion / Flux tags quickly.

**Steps:**
1. Add **Smart Language Model Loader [Eclipse]** node
2. Connect image to `images` input
3. In node settings:
   - `template_name`: Select `Florence-2-base-PromptGen-v2.0` (downloads ~500MB first use)
   - `florence_task`: Select `prompt_gen_tags`
4. Run workflow

**Expected Result:**
```
1girl, long hair, brown hair, white dress, outdoors, garden, cherry blossoms, blue sky, smile, detailed
```

Florence-2 is optimized for speed (~1 second) and low VRAM (~2GB). Perfect for batch processing generated images.

---

## Supported Models

All models listed below are pre-configured as templates in `ComfyUI_Eclipse/templates/smartlm_templates/` (bundled) and copied to `ComfyUI/models/Eclipse/smartlm_templates/` (user-editable) on first run. Select any template from the dropdown to automatically download and configure the model. VRAM requirements are based on actual template specifications.

### QwenVL Models (Transformers)

| Template Name | Size | VRAM (fp16) | VRAM (8bit) | VRAM (4bit) | Video | Description |
|--------------|------|-------------|-------------|-------------|-------|-------------|
| **Qwen3-VL-2B-Instruct** | 2B | 4.0 GB | 2.5 GB | 1.5 GB | ✅ | Fastest, great quality |
| **Qwen3-VL-2B-Instruct-FP8** | 2B | ~2 GB | N/A | N/A | ✅ | Pre-quantized FP8 |
| **Qwen3-VL-2B-Thinking** | 2B | 4.0 GB | 2.5 GB | 1.5 GB | ✅ | Reasoning-focused variant |
| **Qwen3-VL-4B-Instruct** | 4B | ~8 GB | ~5 GB | ~3 GB | ✅ | Balanced quality/speed |
| **Qwen3-VL-4B-Instruct-FP8** | 4B | ~4 GB | N/A | N/A | ✅ | Pre-quantized FP8 |
| **Qwen3-VL-4B-Thinking** | 4B | ~8 GB | ~5 GB | ~3 GB | ✅ | Reasoning-focused 4B |
| **Qwen3-VL-8B-Instruct** | 8B | 12.0 GB | 7.0 GB | 4.5 GB | ✅ | High quality |
| **Qwen3-VL-8B-Instruct-FP8** | 8B | ~6 GB | N/A | N/A | ✅ | Pre-quantized FP8 |
| **Qwen3-VL-8B-Thinking** | 8B | 12.0 GB | 7.0 GB | 4.5 GB | ✅ | Reasoning-focused 8B |
| **Qwen3-VL-32B-Instruct** | 32B | 28.0 GB | 14.0 GB | 8.5 GB | ✅ | Best quality (high-end GPU) |
| **Qwen3-VL-32B-Instruct-FP8** | 32B | ~14 GB | N/A | N/A | ✅ | Pre-quantized FP8 |
| **Qwen3-VL-32B-Thinking** | 32B | 28.0 GB | 14.0 GB | 8.5 GB | ✅ | Reasoning-focused 32B |
| **Qwen2.5-VL-3B-Instruct** | 3B | 6.0 GB | 3.5 GB | 2.0 GB | ✅ | Previous gen (default) |
| **Qwen2.5-VL-7B-Instruct** | 7B | ~14 GB | ~8 GB | ~5 GB | ✅ | Previous generation |

**Recommended starter:** `Qwen3-VL-2B-Instruct` or `Qwen2.5-VL-3B-Instruct` (default)

### QwenVL Models (GGUF)

| Template Name | Size | Quant | VRAM | Context | Video | Description |
|--------------|------|-------|------|---------|-------|-------------|
| **Qwen2.5-VL-3B-Instruct-GGUF** | 3B | Q4_K_M | 3.7 GB | Default | ✅ | Q4 quantized, includes MMProj |
| **Qwen2.5-VL-7B-Ablit-Q8** | 7B | Q8_0 | 10.0 GB | 32768 | ✅ | Abliterated (uncensored), Q8 quantized |

**Note:** GGUF QwenVL models require matching MMProj files for vision support. Both templates include MMProj downloads automatically. These models are always unloaded after each run to prevent VRAM accumulation.

### Florence-2 Models

| Template Name | Size | VRAM (fp16) | Default Task | Best For |
|--------------|------|-------------|--------------|----------|
| **Florence-2-base** | 230M | 0.5 GB | more_detailed_caption | General captions, fast inference |
| **Florence-2-base-PromptGen-v1.5** | 230M | 1.2 GB | more_detailed_caption | Tag generation (v1.5) |
| **Florence-2-base-PromptGen-v2.0** | 230M | 1.2 GB | prompt_gen_tags | **Tag generation for SD/Flux** |
| **Florence-2-base-PromptGen** | 230M | 1.2 GB | prompt_gen_tags | Original PromptGen |
| **Florence-2-large-PromptGen-v1.5** | 770M | 3.7 GB | more_detailed_caption | High-quality v1.5 |
| **Florence-2-large-PromptGen-v2.0** | 770M | 3.7 GB | prompt_gen_tags | High-quality v2.0 |
| **Florence-2-Flux-Large** | 770M | 3.7 GB | more_detailed_caption | High-quality Flux captions |
| **Florence-2-DocVQA** | 230M | 3.7 GB | ocr | Document analysis, OCR |
| **Florence2_FaceDetection** | 770M | 3.7 GB (4bit) | caption_to_phrase_grounding | Face detection (text: "face") |
| **Florence2_EyeDetection** | 770M | 3.7 GB (4bit) | caption_to_phrase_grounding | Eye detection |
| **Florence2_MouthDetection** | 770M | 3.7 GB (4bit) | caption_to_phrase_grounding | Mouth detection |

**Recommended starter:** `Florence-2-base-PromptGen-v2.0` (best for prompt tags)
**Note:** Detection templates are pre-configured with specific text inputs and convert_to_bboxes=False

### Text-Only LLM Models (GGUF)

| Template Name | Size | Quant | VRAM | Context | Description |
|--------------|------|-------|------|---------|-------------|
| **Mistral-7B-Instruct-v0.3_Q5_K_M** | 7B | Q5_K_M | 5.7 GB | 4096 | General text processing |
| **Lexi-Llama-3-8B-Uncensored_Q4_K_M** | 8B | Q4_K_M | 5.5 GB | Default | Uncensored text generation |
| **llava-v1.6-mistral-7b_Q4_K_M** | 7B | Q4_K_M | ~5 GB | Default | LLaVA vision model (GGUF) |

**Note:** Text-only LLM models don't process images. Use for prompt refinement, text expansion, tags-to-description conversion, etc. All LLM templates default to "Tags to Natural Language" instruction mode.

---

## Node Overview

### Smart Language Model Loader [Eclipse]

**The main unified node** supporting all model types with template management.

#### What This Node Does

This single node replaces multiple specialized loaders by providing:

- **Unified Interface:** One node for QwenVL, Florence-2, and LLM models
- **Template System:** Pre-configured templates with optional auto-save for persistent preferences
- **Auto-Download:** Models download automatically from HuggingFace on first use
- **Format Flexibility:** Supports transformers (full quality) and GGUF (low VRAM) formats
- **Task Specialization:** Optimized presets for common tasks (tagging, OCR, description, etc.)
- **Visual Feedback:** Displays bounding boxes for object detection tasks (Florence-2)
- **Type Detection:** Automatically identifies model type from template
- **Memory Management:** Efficient VRAM usage with configurable cleanup and model persistence
- **Context Control:** Adjustable context window for GGUF models (2K-131K tokens)
- **Advanced Options:** Optional pipe input for advanced generation parameters

#### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| `images` | IMAGE | Optional | Single image or batch of video frames to analyze |
| `text` | STRING | Optional | Text input (overrides widget fields for prompts/instructions) |
| `pipe_opt` | LMOPTIONSPIPE | Optional | Advanced parameters from LM Advanced Options Pipe node |

**Notes:**
- If no `images` connected: Text-only mode (LLM models only)
- If no `text` connected: Uses widget fields (`qwen_custom_prompt`, `llm_prompt`, etc.)
- Video support: Connect batch of IMAGE frames (QwenVL models only)

#### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Original input image or visualization with bounding boxes (detection tasks) |
| `text` | STRING | Generated description, tags, caption, or processed text |
| `data` | JSON | Structured detection data (bounding boxes, labels, polygons) for Florence-2 tasks |

**Output Behavior:**
- **QwenVL:** `text` contains natural language description/answer
- **Florence-2 Captions:** `text` contains caption text
- **Florence-2 Detection:** `text` contains labels, `data` contains coordinates, `image` shows boxes
- **LLM:** `text` contains processed/generated text (no image output)

### Pipe Out LM Advanced Options [Eclipse]

**Optional companion node** for advanced generation parameters.

#### What This Node Does

This node provides fine-grained control over generation parameters through a pipe output. It features:

- **Automatic Defaults Saving:** Your parameter changes are automatically saved per model type to `models/Eclipse/config/smartlm_advanced_defaults.json`
- **Persistent Preferences:** Settings persist across ComfyUI sessions
- **Model-Specific Settings:** Separate configurations for QwenVL, Florence2, and LLM models
- **UI Filtering:** Select model type to show only relevant parameters (all parameters passed through pipe)
- **No Manual Config:** Settings are saved automatically when node executes

#### Usage

1. Add **Pipe Out LM Advanced Options [Eclipse]** to your workflow
2. Select `model_type` (QwenVL, Florence2, or LLM) to filter visible parameters
3. Adjust parameters as needed
4. Connect `pipe` output to Smart Language Model Loader's `pipe_opt` input
5. Run workflow - settings are automatically saved for next time

#### Parameters

**Common Parameters:**
- `device` - cuda (GPU) or cpu
- `use_torch_compile` - Enable PyTorch compilation (QwenVL, Florence2)
- `num_beams` - Beam search width (1-10, all models)
- `do_sample` - Enable sampling vs greedy decoding (all models)

**QwenVL Specific:**
- `temperature` - Sampling randomness (0.1-2.0)
- `top_p` - Nucleus sampling (0.1-1.0)
- `top_k` - Top-K sampling (0-100)
- `repetition_penalty` - Penalize repeated tokens (1.0-2.0)
- `frame_count` - Video frames to sample (1-32)

**Florence2 Specific:**
- Same as common parameters

**LLM Specific:**
- `temperature` - Sampling randomness (0.1-2.0)
- `top_p` - Nucleus sampling (0.1-1.0)
- `top_k` - Top-K sampling (0-100)
- `repetition_penalty` - Penalize repeated tokens (1.0-2.0)

**Note:** The `model_type` selector is for UI convenience only - it doesn't change which model runs, just which parameters are visible. All parameters are passed through the pipe regardless of selection.

---

## Template System

Templates are pre-configured JSON files that define model settings, paths, and capabilities. All templates are bundled with ComfyUI_Eclipse and ready to use.

### Template Locations

- **User templates:** `ComfyUI/models/Eclipse/smartlm_templates/` (editable)
- **Bundled templates:** `ComfyUI_Eclipse/templates/smartlm_templates/` (defaults)

### Using Templates

Simply select a template from the `template_name` dropdown in the node. Each template includes:
- Model path or HuggingFace repository ID
- Recommended quantization settings
- VRAM requirements
- Supported features (video, tasks, etc.)
- Optimal parameters (max_tokens, etc.)

Models download automatically when first selected. Templates are automatically updated with `local_path` and actual `vram_requirement` after models are downloaded.

**Need a model repository URL?** See [MODEL_REPOS_REFERENCE.md](MODEL_REPOS_REFERENCE.md) for a complete list of HuggingFace repository IDs you can copy-paste when creating templates manually.

### Template System Modes

The Smart Language Model Loader provides four template operation modes. **None** is the default mode - the node does NOT auto-load templates on startup. This gives you full flexibility to work with templates exactly when and how you need them.

**Template Actions:**
- **None** (default) - Manual configuration mode. Configure model settings directly without using a template. Node always starts in None mode - you decide if and when to load a template.
- **Load** - Load and apply a saved template. Model settings come from the template file.
- **Save** - Save current configuration as a new template.
- **Delete** - Remove selected template from disk.

**Key Behavior:**
- **No Auto-Loading:** The node never automatically loads a template. You start with a blank slate in None mode every time.
- **Multi-Node Workflows:** You can use the same template across multiple nodes in one workflow, then customize each node independently. Workflow: Load template → switch to None → adjust settings for specific node.
- **Template Name Auto-Fill:** When switching to Save mode, the `new_template_name` field intelligently auto-populates using: (1) currently loaded template name if in Load mode, (2) `local_model` path for Local source, or (3) extracted repo name from `repo_id` for HuggingFace source.

**Using Templates (Load Mode):**

1. Set `template_action` to **Load**
2. Select a template from the `template_name` dropdown
3. Run workflow - model loads with template settings
4. **Auto-Detection:** If template has URLs but empty local paths, the node automatically searches for matching models in `ComfyUI/models/LLM/` and uses them if found
5. Generation parameters (max_tokens, quantization, etc.) and task defaults are applied from template
6. Configuration widgets (model_type, model_source, etc.) are hidden in Load mode - switch to **None** to see detected paths
7. **Optional:** Switch to **None** mode to customize settings while keeping template as starting point

**Creating New Templates (Save Mode):**

1. Set `template_action` to **None**
2. Configure all model settings:
   - `model_type`: Select model type - QwenVL, QwenVL (GGUF), Florence2, Florence2 (GGUF), LLM, or LLM (GGUF)
   - `model_source`: Choose **HuggingFace** (auto-download) or **Local** (already downloaded)
   - For HuggingFace:
     - `repo_id`: HuggingFace repo ID or direct download URL
     - `local_path`: Filename after download (folder for transformers, file for GGUF)
   - For Local:
     - `local_model`: Select from dropdown of models in `ComfyUI/models/LLM/`
   - For GGUF QwenVL models:
     - `mmproj_source`: HuggingFace or Local
     - `mmproj_url` / `mmproj_local`: MMProj file location
     - `mmproj_path`: MMProj filename
   - Generation parameters: `max_tokens`, `quantization`, `attention_mode`
   - Context window: `context_size` (GGUF only, default 32768)
   - Task defaults: Select default task/preset and text input for your model type
3. Set `template_action` to **Save**
4. Enter a unique name in `new_template_name` (auto-filled intelligently based on context)
5. Click the **⚡Execute Template Action** button
6. Template is saved to `ComfyUI/models/Eclipse/smartlm_templates/`
7. Mode automatically switches back to **None** after save

**Deleting Templates:**

1. Set `template_action` to **Delete**
2. Select template to delete from `template_name` dropdown
3. Click the **⚡Execute Template Action** button
4. Template file is removed from disk

**Switching Between Modes:**

When switching from **Load** to **None** or **Save** mode:
- Configuration widgets are automatically populated with values from the loaded template
- Prefers "Local" source if model has been downloaded (`local_path` exists)
- Falls back to "HuggingFace" source if model not yet downloaded (only `repo_id` exists)
- Generation parameters and task defaults remain unchanged (already set by template)

When switching to **Save** mode:
- `new_template_name` field auto-fills based on context (loaded template name → `local_model` path → extracted repo name)
- Edit as needed before saving

**Automatic Template Updates:**

Templates are automatically updated after model download:
- `local_path` is set after successful download
- `vram_requirement` is calculated from actual file sizes
- No user interaction required - happens in background

**Auto-Detection of Local Models:**

When loading templates that have HuggingFace URLs (`repo_id`, `mmproj_url`) but empty local paths, the node automatically searches for matching models in your `ComfyUI/models/LLM/` directory:

- **Model Detection:** Searches by repo name/filename from URL
- **MMProj Detection:** For GGUF QwenVL models, searches for mmproj files in the model's folder (even with different filenames)
- **Seamless Usage:** If found locally, uses local files instead of downloading
- **URL Preservation:** When saving templates after auto-detection, both URLs and local paths are preserved for sharing

**Example Flow:**
1. Download Qwen2.5-VL-7B model from HuggingFace to `models/LLM/Qwen-VL/Qwen2.5-VL-7B-Abliterated/`
2. Load template with only `repo_id` URL (empty `local_path`)
3. Node automatically detects and uses your local files
4. Save template → both URL (for sharing) and local path (for your use) are saved

This means you can share templates with HuggingFace URLs, and users who already have the model downloaded will use their local copy automatically.

**Multi-Node Workflow Pattern:**

The flexible None-default design supports advanced workflows:

1. **Load Template:** Switch to Load mode, select template, run to load model
2. **Switch to None:** Change `template_action` to None - configuration widgets appear with template values
3. **Customize Per Node:** Adjust settings (quantization, max_tokens, task) for this specific node
4. **Repeat:** Add another Smart LML node, load same template, customize differently

This allows you to use one template (e.g., "Qwen3-VL-2B-Instruct") across multiple nodes in a workflow, with each node having different generation parameters or task settings.

### Template Structure

All templates include an `_available_tasks` field that documents the available tasks, presets, or instruction modes for that model type. This field is automatically added when creating new templates and serves as a user reference when manually editing templates. It has no functional impact on model execution.

**Example QwenVL Template (Transformers):**
```json
{
  "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
  "local_path": "",
  "mmproj_path": "",
  "model_type": "qwenvl",
  "default": false,
  "default_task": "Detailed Description",
  "default_text_input": "",
  "max_tokens": 1024,
  "quantization": "fp16",
  "attention_mode": "auto",
  "quantized": false,
  "vram_requirement": {
    "full": 4.0,
    "8bit": 2.5,
    "4bit": 1.5
  },
  "_available_tasks": [
    "Custom",
    "Tags",
    "Simple Description",
    "Detailed Description",
    "Ultra Detailed Description",
    "Cinematic Description",
    "Detailed Analysis",
    "Video Summary",
    "Short Story",
    "Prompt Refine & Expand"
  ]
}
```

**Example QwenVL GGUF Template:**
```json
{
  "repo_id": "https://huggingface.co/.../Qwen2.5-VL-7B-Abliterated-Caption-it.Q8_0.gguf",
  "local_path": "Qwen-VL/Qwen2.5-VL-7B-Abliterated-Caption-it.Q8_0/Qwen2.5-VL-7B-Abliterated-Caption-it.Q8_0.gguf",
  "mmproj_path": "Qwen-VL/Qwen2.5-VL-7B-Abliterated-Caption-it.Q8_0/Qwen2.5-VL-7B-Abliterated-Caption-it.mmproj-Q8_0.gguf",
  "mmproj_url": "https://huggingface.co/.../Qwen2.5-VL-7B-Abliterated-Caption-it.mmproj-Q8_0.gguf",
  "model_type": "qwenvl",
  "default": false,
  "default_task": "Video Summary",
  "default_text_input": "",
  "max_tokens": 1024,
  "quantization": "fp16",
  "attention_mode": "auto",
  "quantized": true,
  "vram_requirement": {
    "full": 7.5
  },
  "context_size": 32768,
  "_available_tasks": [
    "Custom",
    "Tags",
    "Simple Description",
    "Detailed Description",
    "Ultra Detailed Description",
    "Cinematic Description",
    "Detailed Analysis",
    "Video Summary",
    "Short Story",
    "Prompt Refine & Expand"
  ]
}
```

**Note:** Templates now preserve both URLs and local paths for hybrid use - local files are used when available, URLs enable auto-download when shared with others who don't have the files yet.

**Example Florence-2 Template:**
```json
{
  "repo_id": "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
  "local_path": "Florence-2-base-PromptGen-v2.0",
  "mmproj_path": "",
  "model_type": "florence2",
  "default": false,
  "default_task": "prompt_gen_tags",
  "default_text_input": "",
  "max_tokens": 512,
  "quantization": "fp16",
  "attention_mode": "auto",
  "quantized": false,
  "vram_requirement": {
    "full": 1.2,
    "8bit": 0.7,
    "4bit": 0.5
  },
  "_available_tasks": "caption, detailed_caption, more_detailed_caption, prompt_gen_tags, ocr, caption_to_phrase_grounding, ocr_with_region, region_proposal, dense_region_caption, region_caption, referring_expression_segmentation"
}
```

**Example LLM Template (GGUF):**
```json
{
  "repo_id": "https://huggingface.co/.../Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
  "local_path": "Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
  "mmproj_path": "",
  "model_type": "llm",
  "default": false,
  "default_task": "Tags to Natural Language",
  "default_text_input": "",
  "max_tokens": 2048,
  "quantization": "fp16",
  "attention_mode": "auto",
  "quantized": true,
  "vram_requirement": {
    "full": 5.7
  },
  "context_size": 4096,
  "_available_tasks": [
    "Tags to Natural Language",
    "Expand Description",
    "Refine Prompt",
    "Custom Instruction",
    "Direct Chat"
  ]
}
```

**Example Detection Template (Florence-2):**
```json
{
  "repo_id": "gokaygokay/Florence-2-Flux-Large",
  "local_path": "Florence-2-Flux-Large",
  "model_type": "florence2",
  "default": false,
  "default_task": "caption_to_phrase_grounding",
  "default_text_input": "face",
  "max_tokens": 256,
  "quantization": "4bit",
  "attention_mode": "auto",
  "quantized": false,
  "vram_requirement": {
    "full": 3.7,
    "8bit": 2.2,
    "4bit": 1.4
  },
  "convert_to_bboxes": false
}
```

**Template Fields:**

**Core Fields (All Models):**
- `model_type`: `"qwenvl"`, `"florence2"`, or `"llm"` - Determines model type and available features
- `repo_id`: HuggingFace repository ID or direct download URL (auto-downloads if not local)
- `local_path`: Path relative to `models/LLM/` (folder for transformers, file for GGUF) - automatically set after download
- `quantized`: Whether model is pre-quantized (FP8, GGUF, etc.)
- `vram_requirement`: VRAM estimates in GB - `{"full": X, "8bit": Y, "4bit": Z}` - automatically calculated from actual file size after download

**Default Widget Values:**
- `max_tokens`: Default maximum tokens to generate (64-2048)
- `quantization`: Default quantization setting ("auto", "fp16", "bf16", "fp32", "8bit", "4bit")
  - **"auto"** (recommended): VRAM-aware automatic selection - analyzes available memory and model VRAM requirements, then selects fp16 (if sufficient VRAM), 8bit (if fp16 too large), or 4bit (if 8bit too large) with 20% safety margin
  - Transformers only - GGUF models have quantization baked into the model file
- `attention_mode`: Default attention implementation ("auto", "flash_attention_2", "sdpa", "eager")
- `default_task`: Default task/preset/mode selection (unified across all model types)
- `default_text_input`: Default text input for custom prompts/instructions (unified across all model types)

**Documentation Fields (All Model Types):**
- `_available_tasks`: Documentation field listing available tasks/presets/modes for this model type. Automatically added when creating new templates. Format varies by model type:
  - **QwenVL**: Array of preset prompt names (e.g., `["Custom", "Tags", "Detailed Description", ...]`)
  - **Florence-2**: Comma-separated string of task IDs (e.g., `"caption, ocr, prompt_gen_tags, ..."`)
  - **LLM**: Array of instruction mode names (e.g., `["Tags to Natural Language", "Expand Description", ...]`)
  - This field serves as a reference when manually editing templates and has no functional impact on execution

**QwenVL Specific:**
- `context_size`: GGUF only - Context window size (2048-131072, default 32768)

**Florence-2 Specific:**
- `convert_to_bboxes`: Default bbox conversion setting (true/false)

**LLM Specific:**
- `context_size`: Context window size for GGUF models (2048-131072)

**Template Field Naming:**
All model types now use unified field names for consistency:
- `default_task` maps to: `qwen_preset_prompt` (QwenVL), `florence_task` (Florence-2), `llm_instruction_mode` (LLM)
- `default_text_input` maps to: `qwen_custom_prompt` (QwenVL), `florence_text_input` (Florence-2), `llm_custom_instruction` (LLM)

**GGUF Vision Models (QwenVL):**
- `mmproj_path`: MMProj filename for vision support (required, relative to `models/LLM/`)
- `mmproj_url`: Direct download URL for MMProj file (auto-downloads if missing)

---

## Model Types

### QwenVL (Vision-Language)

**Supports:** Images, Videos (transformers and GGUF)

**Strengths:**
- ✅ Natural language understanding (ask questions about images)
- ✅ Video analysis with multiple frames
- ✅ Long, detailed, coherent descriptions
- ✅ Custom prompts for specific queries
- ✅ Multiple model sizes (2B-32B)
- ✅ GGUF support with llama-cpp-python

**Use Cases:**
- Video summarization and analysis
- Ultra-detailed image descriptions
- Custom queries ("What is the person wearing?")
- Training data caption generation
- Creative writing from visuals

**Preset Prompts:**
- **Custom** - Use only your custom prompt
- **Tags** - Comma-separated keywords (max 50)
- **Simple Description** - Single sentence overview
- **Detailed Description** - 2-3 sentence paragraph
- **Ultra Detailed Description** - Rich, immersive description
- **Cinematic Description** - Film shot style description
- **Detailed Analysis** - Structured breakdown (subject, attire, background, composition)
- **Video Summary** - Narrative summary of video events
- **Short Story** - Creative story inspired by visual
- **Prompt Refine & Expand** - Enhance user text into richer prompt

**Custom Prompts:**
Type your own question in `qwen_custom_prompt`:
```
What is the person in the image wearing? Describe the clothing style and colors.
```
```
Analyze the lighting in this scene. What time of day is it?
```

### Florence-2 (Vision-Language)

**Supports:** Images only (no video)

**Strengths:**
- ✅ Fast inference (<1 second on most GPUs)
- ✅ Low VRAM usage (~1-3 GB)
- ✅ Specialized tasks (OCR, object detection, tags)
- ✅ Optimized for prompt generation (PromptGen models)
- ✅ Small model sizes (230M-770M)

**Use Cases:**
- Real-time image tagging in workflows
- Generating SD/Flux prompts
- Quick captions for large batches
- OCR and document analysis
- Object detection with bounding boxes

**Available Tasks:**

| Task ID | Description | Output Format |
|---------|-------------|---------------|
| **caption** | Short single sentence | Plain text |
| **detailed_caption** | Detailed paragraph | Plain text |
| **more_detailed_caption** | Very detailed rich description | Plain text |
| **prompt_gen_tags** | Comma-separated tags | `keyword, keyword, keyword` |
| **prompt_gen_mixed_caption** | Mixed-style caption | Plain text |
| **prompt_gen_analyze** | Analytical description | Plain text |
| **prompt_gen_mixed_caption_plus** | Enhanced mixed caption | Plain text |
| **ocr** | Extract text | Text string |
| **ocr_with_region** | Text with locations | Text + bounding boxes |
| **caption_to_phrase_grounding** | Detect specific object | Bounding boxes + labels |
| **region_proposal** | Object region proposals | Bounding boxes |
| **dense_region_caption** | Multiple region captions | Bounding boxes + captions |
| **region_caption** | Object detection | Bounding boxes + labels |
| **referring_expression_segmentation** | Text-based segmentation | Polygons + labels |

**Detection Tasks:**
- Use `florence_text_input` to specify what to detect (e.g., "face", "eye", "person")
- Returns bounding boxes visualized on output image
- Structured data available in `data` output (JSON)
- `convert_to_bboxes` option converts quad_boxes/polygons to standard bboxes

### LLM (Text-Only)

**Supports:** Text input only (no images)

**Strengths:**
- ✅ Fast text-to-text processing
- ✅ Prompt refinement and expansion
- ✅ Tags to natural language conversion
- ✅ Flexible instruction modes

**Use Cases:**
- Convert tags to natural language descriptions
- Expand short descriptions into detailed prompts
- Refine prompts for better quality
- General text processing and chat

**Instruction Modes:**
- **Tags to Natural Language** - Convert comma-separated tags to prose
- **Expand Description** - Expand short text into detailed description
- **Refine Prompt** - Enhance prompt quality and expressiveness
- **Custom Instruction** - Use your own instruction template with `{prompt}` placeholder
- **Direct Chat** - Direct conversation without instruction wrapper

**Custom Instructions:**
Use `llm_custom_instruction` with `{prompt}` placeholder:
```
Generate a detailed, cinematic prompt from: "{prompt}"
```
```
Rewrite the following as natural language: "{prompt}"
```

---

## Common Use Cases

### 1. Auto-Tagging Generated Images

**Goal:** Generate tags for workflow organization.

**Setup:**
- Template: `Florence-2-base-PromptGen-v2.0`
- Task: `prompt_gen_tags`
- Connect: Generated image → `images`

**Result:** `1girl, long hair, blue eyes, school uniform, outdoors, cherry blossoms`

### 2. Creating Training Captions

**Goal:** Detailed captions for LoRA/DreamBooth training.

**Setup:**
- Template: `Qwen3-VL-4B-Instruct`
- Preset: "Detailed Description" or "Ultra Detailed Description"
- Connect: Training image → `images`

**Result:** Rich descriptive paragraph.

### 3. Video Content Summarization

**Goal:** Summarize video clip.

**Setup:**
- Template: `Qwen3-VL-2B-Instruct` (or larger)
- Preset: "Video Summary"
- Connect: Video frames (IMAGE batch) → `images`

**Result:** Narrative summary of events.

### 4. Object Detection

**Goal:** Find specific objects in image.

**Setup:**
- Template: `Florence-2-base`
- Task: `caption_to_phrase_grounding`
- Enter text: "face" (or "eye", "person", etc.)
- Connect: Image → `images`

**Result:** Bounding boxes drawn on output image, structured data in `data` output.

### 5. OCR Text Extraction

**Goal:** Extract text from documents/screenshots.

**Setup:**
- Template: `Florence-2-DocVQA` or `Florence-2-base`
- Task: `ocr` or `ocr_with_region`
- Connect: Document image → `images`

**Result:** Extracted text (and locations if using `ocr_with_region`).

### 6. Prompt Enhancement

**Goal:** Improve basic prompts.

**Setup:**
- Template: `Qwen3-VL-4B-Instruct`
- Preset: "Prompt Refine & Expand"
- Custom prompt: "a girl in a garden"
- Connect: (optional reference image) → `images`

**Result:** Enhanced, detailed prompt.

### 7. Tags to Description (Text-Only)

**Goal:** Convert tags to natural language.

**Setup:**
- Template: `Mistral-7B-Instruct-v0.3_Q5_K_M` (or other LLM)
- Mode: "Tags to Natural Language"
- Input: "1girl, long hair, blue dress, forest, sunlight"

**Result:** "A girl with long hair wearing a blue dress stands in a sunlit forest."

---

## Parameters Guide

### Template Selection

**template_name**: Select pre-configured template to use
- Dropdown lists all available templates
- Templates include model path, recommended settings, and VRAM requirements
- Models download automatically on first use
- See [Supported Models](#supported-models) section for complete list

### Generation Parameters

**quantization**: Runtime quantization (transformers models)
- `auto` (default) - **VRAM-aware automatic selection**: Analyzes available GPU/system memory and template VRAM requirements, then intelligently selects the best quantization level (fp16 → 8bit → 4bit) that fits in available memory with 20% safety margin
- `fp16` - Half precision, best balance quality/VRAM (force fp16 regardless of available memory)
- `bf16` - Brain float 16 (similar to fp16, better numerical stability)
- `fp32` - Full precision (2x VRAM, highest quality)
- `8bit` - 8-bit quantization (~40% VRAM reduction, requires bitsandbytes)
- `4bit` - 4-bit quantization (~75% VRAM reduction, requires bitsandbytes)

**Note:** Auto mode reads template `vram_requirement` field (or estimates from model size) to determine memory needs, then selects fp16/8bit/4bit based on what fits. Example: 12GB available, model needs 14GB for fp16 → selects 8bit (7GB needed). GGUF models ignore this setting as quantization is baked into the model file.

**attention_mode**: Attention implementation
- `auto` (default) - Best available
- `flash_attention_2` - Fastest (requires flash-attn package)
- `sdpa` - PyTorch built-in (good balance)
- `eager` - Standard (slowest, fallback)

**max_tokens**: Maximum output length
- Range: 64-2048 (512 default)
- Tags: 256-512
- Captions: 512-1024
- Detailed: 1024-2048

**context_size**: Context window size (GGUF models only)
- Range: 2048-131072 (32768 default)
- Controls total input+output token capacity
- Text-only LLM: 2048-4096
- Vision models: 16384-32768 for images, 32768-65536 for videos
- Note: Each video frame uses ~1000 tokens

**memory_cleanup**: Clear unused memory before loading
- `True` (default) - Clear cache before loading (safe, won't unload generation models)
- `False` - Skip cleanup (faster if model already loaded)

**keep_model_loaded**: Keep model in VRAM between runs
- `False` (default) - Free VRAM after each run
- `True` - Keep loaded (faster for multiple runs)
- Note: GGUF models are always unloaded to prevent VRAM accumulation

**auto_save_template**: Auto-save widget changes to template
- `False` (default) - Don't modify templates
- `True` - Save changes (max_tokens, tasks, quantization) for next load
- Useful for persistent preferences per model

**seed**: Random seed for reproducibility (0 to 2^64-1, -3 to -1 for random)

### QwenVL Parameters

**qwen_preset_prompt**: Select system prompt template
- See [QwenVL Model Type](#qwenvl-vision-language) for full list

**qwen_custom_prompt**: Your custom question/instruction
- For "Custom" preset: main instruction
- For other presets: additional hints appended to system prompt
- Takes priority if `text` input is connected

### Florence-2 Parameters

**florence_task**: Select Florence-2 task
- See [Florence-2 Model Type](#florence-2-vision-language) for full list

**florence_text_input**: Text input for detection tasks
- Example: "face", "eye", "person" for `caption_to_phrase_grounding`
- Leave empty for caption tasks

**convert_to_bboxes**: Convert detection formats
- `False` (default) - Preserve original quad_boxes/polygons format
- `True` - Convert quad_boxes/polygons to standard normalized bboxes

### LLM Parameters

**llm_instruction_mode**: Instruction template mode
- `Tags to Natural Language` - Convert tags to prose
- `Expand Description` - Expand short text
- `Refine Prompt` - Enhance prompt quality
- `Custom Instruction` - Use custom template
- `Direct Chat` - No instruction wrapper

**llm_custom_instruction**: Custom instruction template
- Use `{prompt}` as placeholder for user input
- Only used when mode is "Custom Instruction"

**llm_prompt**: Your input text (used when `text` input not connected)

### Advanced Parameters (via pipe_opt)

Use **Pipe Out LM Advanced Options [Eclipse]** node to provide advanced generation parameters. The node automatically saves your parameter changes per model type (QwenVL, Florence2, LLM) to `models/Eclipse/config/smartlm_advanced_defaults.json` for persistent preferences across sessions.

**Model Type Selector:**
- `model_type` - Select QwenVL, Florence2, or LLM to show only relevant parameters
- This is for UI filtering only - all parameters are passed through the pipe
- Your settings are automatically saved per model type

**Available Parameters:**

**device**: Computation device
- `cuda` (default) - GPU
- `cpu` - CPU (very slow)

**use_torch_compile**: Enable PyTorch compilation
- `False` (default)
- `True` - Compile model (first run slow, subsequent faster)

**temperature**: Sampling randomness (0.1-2.0, default 0.7)
- Lower: More focused, deterministic
- Higher: More creative, varied

**top_p**: Nucleus sampling (0.1-1.0, default 0.9)

**top_k**: Top-K sampling (1-100, default 50)

**num_beams**: Beam search width (1-10, default 3)
- 1: Greedy (fastest)
- 3-5: Good balance
- 8-10: Best quality (slow)

**do_sample**: Enable sampling
- `True` (default) - Use temperature/top_p
- `False` - Greedy/beam search (deterministic)

**repetition_penalty**: Penalize repeated tokens (1.0-2.0, default 1.0)
- 1.0: No penalty
- 1.2-1.5: Moderate
- 1.8-2.0: Strong (may affect quality)

**frame_count**: Video frames to sample (1-32, default 8, QwenVL only)

---

## Performance & VRAM

### VRAM Requirements

**QwenVL (Transformers):**

| Model | fp16 | 8bit | 4bit |
|-------|------|------|------|
| 2B | ~4 GB | ~2.5 GB | ~1.5 GB |
| 3B | ~6 GB | ~3.5 GB | ~2 GB |
| 4B | ~8 GB | ~5 GB | ~2.5 GB |
| 7-8B | ~14-16 GB | ~10 GB | ~4-5 GB |
| 32B | ~64 GB | ~40 GB | ~18 GB |

**QwenVL (GGUF):**
- Depends on quantization (Q4, Q8, etc.)
- Generally 2-6 GB for 3-7B models
- Requires matching MMProj file (+0.5-1 GB)

**Florence-2:**

| Model | fp16 |
|-------|------|
| Base (230M) | ~1-2 GB |
| Large (770M) | ~3 GB |

**LLM (GGUF):**
- Q4: ~4 GB for 7-8B models
- Q5: ~5 GB for 7-8B models

### Speed Estimates (RTX 4090)

**QwenVL:**
- Single image: 2-5 seconds
- Video (8 frames): 10-20 seconds

**Florence-2:**
- Single image: <1 second
- Batch (10 images): ~5 seconds

**LLM:**
- Text generation: 1-3 seconds (depends on length)

### Optimization Tips

1. **Use Florence-2 for speed-critical tasks** (tagging, quick captions)
2. **Enable Flash Attention 2** if available: `pip install flash-attn --no-build-isolation`
3. **Leave quantization on `auto`** (recommended) - intelligently downgrades quantization (fp16 → 8bit → 4bit) if VRAM is insufficient, preventing OOM errors
4. **Manual quantization override**: Use `4bit` to force 4-bit on low VRAM GPUs (RTX 3060, 4060 Ti), or `fp16` to force full precision on high VRAM GPUs (RTX 4090) - requires `pip install bitsandbytes` for 8bit/4bit
5. **Keep model loaded** for batch processing
6. **Reduce max_tokens** to minimum needed
7. **Use smaller models** when quality difference negligible
8. **GGUF models** have quantization baked in - use pre-quantized models (Q4, Q5, Q8) for VRAM control

---

## Troubleshooting

### Model Not Downloading

**Problem:** Download stalls or fails.

**Solutions:**
- Check internet connection
- Verify HuggingFace access (not blocked)
- Check disk space in `ComfyUI/models/LLM/`
- Manual download:
  ```powershell
  cd ComfyUI\models\LLM
  git lfs install
  git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
  ```

### Out of Memory (OOM)

**Problem:** CUDA out of memory.

**Solutions:**
1. Use smaller model (2B instead of 8B)
2. Enable quantization (8bit or 4bit)
3. Close other GPU applications
4. Disable `keep_model_loaded`
5. Use `sdpa` attention instead of `flash_attention_2`
6. Reduce `max_tokens`

### GGUF Models Not Working

**Problem:** GGUF model fails to load or generate.

**Solutions:**
1. Install llama-cpp-python: `pip install llama-cpp-python`
2. For QwenVL GGUF: Ensure matching MMProj is selected in template
3. Increase `context_size` if processing long videos or text (default 32768)
4. Check console for specific errors
5. Verify GGUF file is not corrupted (re-download if needed)

**Note:** GGUF models are automatically unloaded after each run to prevent VRAM accumulation, even with `keep_model_loaded=True`. This is required because llama-cpp-python's vision encoder holds VRAM that can only be freed by unloading.

### QwenVL GGUF Vision Not Working

**Problem:** GGUF generates text unrelated to image.

**Solutions:**
1. Verify MMProj file is selected in template (required for vision)
2. Ensure MMProj quantization matches model quantization (Q4 model → Q4 MMProj)
3. Check llama-cpp-python version: `pip install --upgrade llama-cpp-python>=0.3.16`
4. Verify Qwen2.5VLChatHandler is available (v0.3.10+)

### Florence-2 AttributeError

**Problem:** Error about missing `_supports_sdpa` attribute.

**Solution:** Delete cached model code:
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\modules\transformers_modules\*florence*"
```

### Poor Quality Output

**Problem:** Generated text is low quality or nonsensical.

**Solutions:**
1. Try larger model (4B or 8B instead of 2B)
2. Use fp16 instead of 4bit quantization
3. Adjust generation parameters:
   - Increase `num_beams` (3-5)
   - Set `repetition_penalty=1.2`
4. Verify correct task/preset for your goal
5. Check template configuration

### Slow Inference

**Problem:** Generation takes too long.

**Solutions:**
1. Use Florence-2 for fast tasks
2. Use smaller QwenVL model (2B)
3. Enable Flash Attention 2
4. Use fp16 (4bit has overhead)
5. Reduce `max_tokens`
6. Reduce `num_beams` to 1-2

---

## Tips & Best Practices

### General Tips

1. **Start small:** Begin with 2B models to learn the system
2. **Match model to task:**
   - Tags/prompts → Florence-2 PromptGen
   - Rich descriptions → QwenVL 4B+
   - Video → QwenVL (any size)
   - OCR → Florence-2-DocVQA
   - Speed → Florence-2
   - Text-only → LLM GGUF
3. **Enable auto-save:** Turn on `auto_save_template` to persist your preferred settings per model
4. **Use advanced options:** Add Pipe Out LM Advanced Options node for fine control - settings auto-save per model type
5. **Adjust context for GGUF:** Increase `context_size` for long videos or complex text (GGUF models only)
6. **Monitor VRAM:** Use Task Manager/nvidia-smi
7. **Test before batch:** Run single image first
8. **Memory cleanup:** Leave `memory_cleanup=True` (default) unless model is pre-loaded

### Prompt Engineering (QwenVL)

1. **Be specific:** "Describe the lighting and color mood" not "Describe this"
2. **Structure requests:** Use numbered lists for multiple aspects
3. **Use examples:** Show desired output format in prompt
4. **Iterate:** Refine prompts based on outputs

### Template Sharing

1. **Create shareable templates:** Templates with both URLs and local paths work for everyone
2. **After download:** Save your template to add local paths while keeping URLs
3. **Share with confidence:** Recipients without local files will auto-download; those with files will use local copy
4. **Filename flexibility:** MMProj files are detected even if filenames differ from template
5. **URL preservation:** When saving templates after auto-detection, both repo_id and mmproj_url are preserved automatically

### Workflow Integration

**Auto-tagging pipeline:**
```
[Image Generator] → [Smart LML (Florence-2)] → [Save with metadata]
```

**Prompt refinement loop:**
```
[Text] → [Image Generator] → [Smart LML (QwenVL)] → [Merge] → [Regenerate]
```

**Batch analysis:**
```
[Load Images] → [Smart LML (keep_model_loaded=True)] → [Save text]
```

**Video summarization:**
```
[Load Video] → [Smart LML (QwenVL)] → [Text output]
```

### Quality vs. Speed Trade-offs

| Priority | Model | Quantization | Params |
|----------|-------|--------------|--------|
| **Maximum Speed** | Florence-2-base-PromptGen-v2.0 | fp16 | max_tokens=256 |
| **Balanced** | Qwen3-VL-2B-Instruct | fp16 | max_tokens=512, num_beams=3 |
| **Maximum Quality** | Qwen3-VL-8B-Instruct | fp16 | max_tokens=2048, num_beams=5 |
| **Low VRAM** | Qwen3-VL-2B-Instruct | 4bit | max_tokens=512, num_beams=3 |

### Creating Training Data

**For LoRA/DreamBooth:**
1. Use Qwen3-VL-4B-Instruct or larger
2. Preset: "Ultra Detailed Description"
3. Set `max_tokens=1024+`
4. Generate caption per training image
5. Save as text files matching image filenames

**For tag-based training (Danbooru style):**
1. Use Florence-2-base-PromptGen-v2.0
2. Task: `prompt_gen_tags`
3. Set `max_tokens=512`
4. Fast, consistent tag generation
5. Review/edit tags manually

---

## Advanced Topics

### Using GGUF Models

**Requirements:**
- Install llama-cpp-python: `pip install llama-cpp-python`
- For GPU acceleration: Install CUDA-enabled version
- For QwenVL vision: Matching MMProj file required

**Creating GGUF Templates:**
1. Download GGUF model to `ComfyUI/models/LLM/`
2. For QwenVL: Download matching MMProj file
3. Create template with `model_source="Local File"`
4. Select GGUF file from `local_path` dropdown
5. Select MMProj from `mmproj_path` dropdown
6. Save template

### Offline Usage

Models load from `ComfyUI/models/LLM/` first, then HuggingFace Hub.

**To use offline:**
1. Download model to `ComfyUI/models/LLM/<model_name>/`
2. Create template with `local_path` pointing to folder
3. Disconnect internet
4. Model loads from local cache

### Integration with ComfyUI-Florence2

If [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) extension is installed:
- Smart LML automatically uses its optimized implementation
- Falls back to standard transformers if import fails
- No configuration needed

---

## Support & Resources

- **GitHub:** [ComfyUI_Eclipse Repository](https://github.com/r-vage/ComfyUI_Eclipse)
- **Issues:** [Report bugs or request features](https://github.com/r-vage/ComfyUI_Eclipse/issues)
- **Model Repository Reference:** [MODEL_REPOS_REFERENCE.md](MODEL_REPOS_REFERENCE.md) - Complete list of HuggingFace repository URLs for all supported Qwen and Florence-2 models
- **QwenVL:** [Qwen-VL GitHub](https://github.com/QwenLM/Qwen-VL)
- **Florence-2:** [Florence-2 HuggingFace](https://huggingface.co/microsoft/Florence-2-large)
- **llama-cpp-python:** [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)

---

**Happy analyzing! 📸🎥✨**

If this guide helped you, consider starring the [ComfyUI_Eclipse repository](https://github.com/r-vage/ComfyUI_Eclipse) ⭐

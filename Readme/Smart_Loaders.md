# Smart Loader User Guide

This guide explains how to use the Smart Loader series - next-generation model loaders with advanced features, multi-format support, and intelligent workflow integration.

## Table of Contents
- [Overview](#overview)
- [Smart Loader Plus](#smart-loader-plus)
- [Smart Loader](#smart-loader)
- [Model Types & Formats](#model-types--formats)
- [Template System](#template-system)
- [Configuration Guide](#configuration-guide)
- [Tips & Best Practices](#tips--best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Smart Loader series represents the most advanced model loading system in ComfyUI_Eclipse, offering:

- **Multi-Format Support:** Standard Checkpoints, UNet, Nunchaku Flux/Qwen (quantized), GGUF (quantized)
- **Template System:** Save and load complete configurations instantly
- **CLIP Ensemble:** Load up to 4 CLIP modules simultaneously
- **Automatic Configuration:** Optional latent and sampler setup
- **Quantization Support:** Load compressed models for reduced VRAM usage
- **Intelligent Defaults:** Smart configuration based on model type

### Two Variants

**Smart Loader Plus:**
- Full-featured loader with everything built-in
- Includes latent and sampler configuration
- Best for complete, self-contained workflows
- Outputs comprehensive pipe with all settings

**Smart Loader:**
- Streamlined version without latent/sampler
- Focus on model/CLIP/VAE loading only
- Best for workflows with separate sampler nodes
- Lighter and more flexible

Both loaders share template compatibility and multi-format support.

---

## Smart Loader Plus

**Node Name:** `Smart Loader Plus [Eclipse]`

### What It Does

The ultimate all-in-one loader that handles model loading, CLIP configuration, VAE setup, latent creation, and sampler settings in a single node. Perfect for power users and complex workflows.

### Core Features

#### 1. Multi-Format Model Support

Load any of these model types from a single node:

**Standard Checkpoint**
- Traditional `.safetensors` or `.ckpt` files
- Contains MODEL, CLIP, and VAE in one file
- Most common format
- **Location:** `ComfyUI/models/checkpoints/`

**UNet Model**
- Standalone diffusion model without CLIP/VAE
- Smaller file size
- Requires external CLIP and VAE
- **Location:** `ComfyUI/models/diffusion_models/`

**Nunchaku Flux** (Quantized)
- Compressed Flux models using SVDQuant
- INT4/FP4/FP8 formats for reduced VRAM
- Requires ComfyUI-Nunchaku extension
- **Location:** `ComfyUI/models/diffusion_models/`

**Nunchaku Qwen** (Quantized)
- Compressed Qwen image understanding models
- Reduced memory footprint
- Requires ComfyUI-Nunchaku extension
- **Location:** `ComfyUI/models/diffusion_models/`

**GGUF Model** (Quantized)
- GGUF format quantized models
- Various quantization levels
- Requires ComfyUI-GGUF extension
- **Location:** `ComfyUI/models/diffusion_models/`

#### 2. Template System

Save and load complete loader configurations:

**Save Templates:**
- Captures all settings (model, CLIP, VAE, latent, sampler)
- Store configurations for different workflows
- Quick switching between setups
- **Storage:** `ComfyUI/models/smart_loader_templates/` (primary location, auto-created on first run)

**Load Templates:**
- Instantly restore saved configurations
- Works across Smart Loader and Smart Loader Plus
- Graceful degradation (Plus templates work in regular Loader)

**Delete Templates:**
- Remove unused configurations
- Clean template library

#### 3. CLIP Ensemble System

Configure up to 4 CLIP modules:

**Source Options:**
- **Baked:** Use CLIP from checkpoint (Standard Checkpoint only)
- **External:** Load separate CLIP files from `ComfyUI/models/clip/` and `ComfyUI/models/text_encoders/` folders

**CLIP Count:**
- 1 to 4 modules
- Each module can use different CLIP types

**Supported CLIP File Formats:**
- `.safetensors` - Standard format (recommended)
- `.gguf` - Quantized CLIP models (requires ComfyUI-GGUF extension)

**CLIP Types Supported:**
- Flux (flux_text_encoders)
- SD3 (sd3_clip)
- SDXL (sdxl_clip)
- Qwen Image (qwen_clip)
- HiDream (hidream_clip)
- Hunyuan Image (hunyuan_image_clip)
- Wan (wan_clip)
- Standard (clip)

**CLIP Layer Trimming:**
- Enable/disable per configuration
- `-24` to `-1` range
- Saves memory and affects prompt adherence

#### 4. Latent Configuration

Built-in empty latent creation:

**Resolution Presets:**
- Flux ratios (1:1, 16:9, 9:16, 21:9, 3:2, 2:3, etc.)
- SDXL sizes (1024x1024, landscape, portrait, widescreen)
- HiDream, Qwen, and more
- **Custom:** Manual width/height input

**Batch Size:**
- 1 to 64 images per generation
- Matches latent batch dimension

**Toggle:**
- Enable to use built-in latent
- Disable to use separate Empty Latent node

#### 5. Sampler Settings

Integrated sampler configuration:

**Sampler Selection:**
- euler, euler_a, dpmpp_2m, dpmpp_sde, and more
- All ComfyUI samplers supported

**Scheduler:**
- normal, karras, exponential, sgm_uniform, etc.

**Parameters:**
- Steps: 1-150
- CFG Scale: 0.0-100.0
- Flux Guidance: 0.0-100.0 (for Flux models)

**Toggle:**
- Enable to use built-in sampler settings
- Disable to use separate KSampler node

#### 6. Model Sampling Configuration

Advanced sampling method selection for different model architectures:

**Sampling Method:**
- **None (Default):** Standard sampling without modifications
- **SD3:** Stable Diffusion 3 models (flow-based)
- **AuraFlow:** AuraFlow models (flow-based)
- **Flux:** Flux models (flow-based with dimension-aware shift)
- **Stable Cascade:** Stable Cascade models
- **LCM:** Latent Consistency Models (distilled)
- **ContinuousEDM:** Continuous EDM sampling with subtypes
- **ContinuousV:** Continuous V-prediction sampling
- **LTXV:** Video models (token-based)

**Method-Specific Parameters:**

**Flow-Based (SD3, AuraFlow, Stable Cascade):**
- `shift`: Controls sampling schedule (default varies by method)
- `multiplier`: Timestep scaling (SD3/AuraFlow only, default 1000.0 for SD3, 1.0 for AuraFlow)

**Flux:**
- `max_shift`: Maximum shift value (default 1.15)
- `base_shift`: Base shift value (default 0.5)
- `sampling_width/height`: Image dimensions for shift calculation (auto-filled in Smart Loader Plus)
- Shift calculated from: `((width * height / 1024) * slope) + intercept`

**Stable Cascade:**
- `shift`: Sampling schedule control (default 2.0)

**LCM (Latent Consistency Models):**
- `original_timesteps`: Number of timesteps in original model (1-1000, default 50)
- `zsnr`: Zero-terminal SNR mode for better quality

**ContinuousEDM:**
- `sampling_subtype`: EDM variant (eps, v_prediction, edm, edm_playground_v2.5, cosmos_rflow)
- `sigma_max`: Maximum noise level (default 120.0)
- `sigma_min`: Minimum noise level (default 0.002)

**ContinuousV:**
- `sigma_max`: Maximum noise level (default 500.0)
- `sigma_min`: Minimum noise level (default 0.03)

**LTXV (Video Models):**
- `max_shift`: Maximum shift value (default 2.05)
- `base_shift`: Base shift value (default 0.95)
- Uses token count for shift calculation instead of dimensions

**When to Use Each Method:**
- **SD3:** Stable Diffusion 3 and SD3.5 models
- **AuraFlow:** AuraFlow architecture models
- **Flux:** Flux Dev, Flux Schnell, and Flux variants
- **Stable Cascade:** Stable Cascade models specifically
- **LCM:** Distilled models designed for few-step generation
- **ContinuousEDM:** Models with continuous noise schedules (EDM-based)
- **ContinuousV:** Models using V-prediction parameterization
- **LTXV:** Lightricks video generation models

**Toggle:**
- Enable `configure_model_sampling` to show sampling options
- Disable to use standard sampling (no modifications)

**Template Integration:**
- Sampling method and parameters saved in templates
- Only active method's parameters are saved (snapshot approach)
- Templates preserve sampling configuration across sessions

#### 7. LoRA Configuration (Model-Only)

Integrated LoRA support with model-only weights:

**LoRA Slots:**
- Up to 3 LoRA slots available
- Each slot has: switch (on/off), LoRA selection, model weight

**Model-Only Mode:**
- Single weight slider per LoRA (applies to model only)
- No separate CLIP weight configuration
- Simpler interface for model-focused LoRA usage

**LoRA Count:**
- Controls how many slots are visible (1-3)
- Reduces UI clutter when fewer LoRAs needed

**Template Integration:**
- LoRA configuration saved in templates
- All 3 slots saved (even if disabled) for easy toggling
- Switch states, names, and weights preserved

**Toggle:**
- Enable `configure_model_only_lora` to show LoRA options
- Disable to hide LoRA section entirely

#### 8. Quantization Settings

Model-specific options for reduced VRAM:

**Nunchaku Flux:**
- Data type: fp8, fp4, int4
- Cache threshold: Memory management
- Attention mode: flash_attn or sdpa
- I2F mode: Image-to-feature conversion
- CPU offload: Move components to RAM

**Nunchaku Qwen:**
- GPU block allocation
- Pinned memory optimization
- CPU offload

**GGUF:**
- Dequantization dtype: auto, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32
- Patch dtype: Layer-specific precision
- Device placement: auto, CPU, GPU

### Step-by-Step Usage

#### Basic Setup (Standard Checkpoint)

1. **Add Node:** `Eclipse > Loader > Smart Loader Plus`

2. **Select Model Type:** 
   - Set `model_type` to "Standard Checkpoint"

3. **Choose Checkpoint:**
   - Select from `ckpt_name` dropdown
   - Example: `juggernaut_aftermath.safetensors`

4. **Configure CLIP:**
   - Keep `configure_clip` enabled
   - Set `clip_source` to "Baked" (uses embedded CLIP)
   - Or choose "External" for custom CLIP files

5. **Configure VAE:**
   - Keep `configure_vae` enabled
   - Set `vae_source` to "Baked" (uses embedded VAE)
   - Or choose "External" for custom VAE

6. **Set Resolution:**
   - Keep `configure_latent` enabled
   - Choose preset from `resolution` (e.g., "Flux 1:1 (1024x1024)")
   - Set `batch_size` to 1

7. **Configure Sampler:**
   - Keep `configure_sampler` enabled
   - Set `sampler_name` (e.g., "euler")
   - Set `scheduler` (e.g., "simple")
   - Set `steps` (e.g., 20)
   - Set `cfg` (e.g., 7.0)
   - Set `flux_guidance` if using Flux models (e.g., 3.5)

8. **Configure Model Sampling (Optional):**
   - Enable `configure_model_sampling` if needed
   - Select appropriate `sampling_method` for your model:
     - SD3 for Stable Diffusion 3 models
     - Flux for Flux models (auto-fills dimensions from latent)
     - None for standard models
   - Adjust method-specific parameters if needed
   - Most users can leave this disabled (uses standard sampling)

9. **Connect Output:**
   - Connect `pipe` to downstream nodes
   - Use Pipe Out nodes to extract components

#### Quantized Model Setup (Nunchaku Flux)

1. **Prerequisites:**
   - Install ComfyUI-Nunchaku extension
   - Download Nunchaku quantized model
   - Place in `ComfyUI/models/diffusion_models/`

2. **Configure Node:**
   - Set `model_type` to "Nunchaku Flux"
   - Select model from `nunchaku_name` dropdown
   - Set `data_type` (fp8, fp4, or int4)
   - Configure `cache_threshold` (default: 0.1)
   - Set `attention` mode (flash_attn recommended)
   - Enable `cpu_offload` if VRAM-limited

3. **CLIP Setup:**
   - Set `clip_source` to "External" (required for UNet-style models)
   - Set `clip_count` to 2 (typical for Flux)
   - Select CLIP files for each slot
   - Set `clip_type` to "flux_text_encoders"

4. **Model Sampling:**
   - Enable `configure_model_sampling`
   - Set `sampling_method` to "Flux"
   - `max_shift` and `base_shift` default to 1.15 and 0.5 (recommended)
   - Width/height auto-fill from latent configuration
   - Shift calculated automatically from dimensions

5. **Complete Configuration:**
   - Configure VAE (external recommended)
   - Set resolution for Flux (1024x1024 or similar)
   - Configure sampler with Flux-appropriate settings

#### Template Workflow

**Saving a Template:**

1. Configure node completely (model, CLIP, VAE, latent, sampler, LoRAs if needed)
2. Set `template_action` to "Save"
3. Enter name in `new_template_name` (auto-fills if updating existing template)
4. Click "Execute Template Action" button
5. Template saved to `ComfyUI/models/Eclipse/loader_templates/<name>.json`
6. Node automatically switches back to "Load" mode with saved template selected

**Updating an Existing Template:**

1. Load template you want to update via `template_name`
2. Make desired changes to settings
3. Switch `template_action` to "Save"
4. `new_template_name` auto-fills with loaded template name
5. Click "Execute Template Action" to overwrite
6. Or edit the name to save as a new template instead

**Loading a Template:**

1. Set `template_action` to "Load"
2. Select template from `template_name` dropdown
3. Template loads automatically - all settings restored instantly
4. All values reset to defaults first, then template values applied

**Deleting a Template:**

1. Set `template_action` to "Delete"
2. Select template from `template_name` dropdown
3. Click "Execute Template Action" button
4. Template removed from library
5. Node automatically switches back to "Load" mode

### Output

The node outputs a comprehensive **pipe** containing:

- `model` - Loaded diffusion model (with LoRAs applied if enabled)
- `clip` - CLIP ensemble (single or multi-module)
- `vae` - VAE encoder/decoder
- `latent` - Pre-configured empty latent tensor (if enabled)
- `width` - Latent width
- `height` - Latent height
- `batch_size` - Batch count
- `sampler_name` - Sampler selection
- `scheduler` - Scheduler type
- `steps` - Generation steps
- `cfg` - CFG scale
- `flux_guidance` - Flux guidance value
- `model_name` - Model filename
- `vae_name` - VAE filename
- `clip_skip` - CLIP layer setting
- `is_nunchaku` - Quantization flag
- `lora_names` - List of active LoRA names (if configure_model_only_lora enabled)
- `sampling_method` - Selected sampling method (if configure_model_sampling enabled)

---

## Smart Loader

**Node Name:** `Smart Loader [Eclipse]`

### What It Does

Streamlined loader focused on model/CLIP/VAE loading without latent or sampler configuration. Perfect for workflows where you want separate control over sampling parameters.

### Key Differences from Smart Loader Plus

**What's Included:**
- Multi-format model support (same as Plus)
- Template system (same as Plus)
- CLIP ensemble configuration (same as Plus)
- VAE configuration (same as Plus)
- Quantization support (same as Plus)

**What's Removed:**
- No latent configuration (use separate Empty Latent node)
- No sampler settings (use separate KSampler node)
- Simpler, more focused interface
- Lighter pipe output

**What's Included (Same as Plus):**
- Model sampling configuration (all 8 sampling methods)
- LoRA configuration (model-only weights)
- All quantization options
- Template system with full feature support

### When to Use Smart Loader

Choose Smart Loader when:
- You prefer separate Empty Latent and KSampler nodes
- Building modular workflows
- Need flexibility in sampler configuration
- Want minimal, focused loader
- Working with custom sampling workflows

### Usage

Configuration is identical to Smart Loader Plus except:

1. **No Latent Options:**
   - `configure_latent` toggle removed
   - No resolution, width, height, batch_size settings
   - Use Empty Latent node separately

2. **No Sampler Options:**
   - `configure_sampler` toggle removed
   - No sampler_name, scheduler, steps, cfg settings
   - Use KSampler node separately

3. **Model Sampling Available:**
   - `configure_model_sampling` toggle present (same as Plus)
   - All 8 sampling methods supported
   - Method-specific parameters available
   - Template system includes sampling configuration

4. **Simpler Output:**
   - Pipe contains: model, clip, vae, model_name, vae_name, clip_skip, is_nunchaku, sampling_method
   - No latent, dimensions, or sampler data

### Example Workflow

```
Smart Loader
├─ pipe ──────────> Pipe Out Checkpoint Loader
│                   ├─ model ──────> KSampler
│                   ├─ clip ───────> CLIP Text Encode
│                   └─ vae ────────> VAE Decode
│
Empty Latent Image ─> KSampler (latent input)
```

---

## Model Types & Formats

### Standard Checkpoint

**What it is:** Traditional all-in-one model file

**File formats:** `.safetensors`, `.ckpt`, `.pt`

**Contains:** MODEL + CLIP + VAE (usually)

**Best for:** 
- Most common use case
- Single-file convenience
- Standard Stable Diffusion models

**Location:** `ComfyUI/models/checkpoints/`

**Examples:**
- `sd_xl_base_1.0.safetensors`
- `epicrealism_v5.safetensors`

### UNet Model

**What it is:** Standalone diffusion model without CLIP/VAE

**File format:** `.safetensors`

**Contains:** MODEL only

**Requires:** External CLIP and VAE files

**Best for:**
- Smaller file sizes
- Sharing just the diffusion model
- Combining different CLIPs with one model

**Location:** `ComfyUI/models/diffusion_models/`

**Configuration:**
- CLIP source must be "External"
- VAE source must be "External"
- Select appropriate CLIP type for your model

### Nunchaku Flux (Quantized)

**What it is:** Compressed Flux models using SVDQuant technology

**Quantization levels:** INT4, FP4, FP8

**Memory savings:** 
- INT4: ~75% less VRAM
- FP4: ~50% less VRAM
- FP8: ~25-40% less VRAM

**Quality:** Minimal loss with proper settings

**Requirements:**
- ComfyUI-Nunchaku extension installed
- Compatible Nunchaku Flux model

**Best for:**
- VRAM-limited GPUs (8GB, 12GB)
- Running larger models on smaller GPUs
- Faster loading times

**Location:** `ComfyUI/models/diffusion_models/`

**Recommended Settings:**
- Data type: fp8 (best quality) or int4 (most compression)
- Attention: flash_attn (faster)
- CPU offload: Enable if VRAM-limited

### Nunchaku Qwen (Quantized)

**What it is:** Compressed Qwen image understanding models

**Specialization:** Image understanding and description

**Requirements:**
- ComfyUI-Nunchaku extension
- Qwen model file

**Best for:**
- Image analysis workflows
- Caption generation
- Visual question answering

**Location:** `ComfyUI/models/diffusion_models/`

**Recommended Settings:**
- Adjust GPU blocks based on available VRAM
- Enable pinned memory for speed

### GGUF Model (Quantized)

**What it is:** GGUF format quantized models (diffusion models and CLIP encoders)

**Quantization levels:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32

**Compatibility:** Wide range of quantized models

**Requirements:**
- ComfyUI-GGUF extension installed
- GGUF model file

**Best for:**
- Alternative quantization format
- Community-shared quantized models
- Experimenting with different compression levels
- Quantized CLIP encoders for reduced VRAM

**File Locations:**
- Diffusion models: `ComfyUI/models/diffusion_models/`
- CLIP encoders: `ComfyUI/models/clip/` or `ComfyUI/models/text_encoders/`

**Recommended Settings:**
- Dequant dtype: auto (let system decide)
- Device: auto (automatic placement)

**Note:** Both diffusion models and CLIP encoders in GGUF format are supported. The Smart Loaders automatically detect GGUF files in both `clip` and `text_encoders` folders when the ComfyUI-GGUF extension is installed.

---

## Template System

### What Are Templates?

Templates save your complete loader configuration to a JSON file, allowing instant restoration of complex setups.

### What Gets Saved

The template system intelligently saves only relevant configuration for your setup, keeping template files clean and minimal.

**Always Saved:**
- Model type selection
- Configuration toggles (configure_clip, configure_vae, configure_latent, configure_sampler, configure_model_only_lora)

**Model Settings (only the active model type):**
- Standard Checkpoint: `ckpt_name` only
- UNet Model: `unet_name` + weight_dtype, data_type, cache_threshold
- Nunchaku Flux: `nunchaku_name` + attention, i2f_mode, cpu_offload, num_blocks_on_gpu, use_pin_memory
- Nunchaku Qwen: `qwen_name` + attention, i2f_mode, cpu_offload, num_blocks_on_gpu, use_pin_memory
- GGUF Model: `gguf_name` + gguf_dequant_dtype, gguf_patch_dtype, gguf_patch_on_device

**CLIP Configuration (only if configure_clip is enabled):**
- CLIP source (Baked/External)
- CLIP count (1-4 modules)
- CLIP type selection
- CLIP layer trimming settings (enable_clip_layer, stop_at_clip_layer)
- Individual CLIP file selections (only if clip_source is "External" and not "None")

**VAE Configuration (only if configure_vae is enabled):**
- VAE source (Baked/External)
- Selected VAE file (only if vae_source is "External" and not "None")

**Latent Settings (only if configure_latent is enabled, Smart Loader Plus only):**
- Resolution preset
- Batch size
- Width and height (only if resolution is "Custom")

**Sampler Settings (only if configure_sampler is enabled, Smart Loader Plus only):**
- Sampler name
- Scheduler
- Steps
- CFG scale
- Flux guidance

**Model Sampling Settings (only if configure_model_sampling is enabled):**
- Sampling method selection
- Method-specific parameters (only for active method):
  - SD3/AuraFlow: shift, multiplier
  - Flux: max_shift, base_shift, sampling_width, sampling_height
  - Stable Cascade: shift
  - LCM: original_timesteps, zsnr
  - ContinuousEDM: sampling_subtype, sigma_max, sigma_min
  - ContinuousV: sigma_max, sigma_min
  - LTXV: max_shift, base_shift

**LoRA Settings (only if configure_model_only_lora is enabled, Smart Loader Plus only):**
- LoRA count (1-3)
- All 3 LoRA slots: switch state, lora name, model weight (saved even if switch is off for easy toggling)

### Creating Templates

**When to create:**
- You have a working configuration you'll reuse
- Setting up different models for different tasks
- Want to quickly switch between workflows

**How to create:**

1. Configure the loader completely
2. Test that settings work correctly
3. Set `template_action` to "Save"
4. Enter descriptive name in `new_template_name`
5. Click "Execute Template Action"
6. Template saved to `ComfyUI/models/Eclipse/loader_templates/<name>.json`

**Naming Tips:**
- Use descriptive names: `Flux_Dev_1024`, `SDXL_Portrait`, `Qwen_Analysis`
- Include model type: `Standard_Realistic`, `Nunchaku_Flux_Compressed`
- Indicate purpose: `Quick_Test`, `High_Quality`, `Memory_Saver`

### Using Templates

**Loading:**

1. Set `template_action` to "Load"
2. Select from `template_name` dropdown
3. Template loads automatically (no button needed)
4. All settings restored instantly
5. Verify settings are correct
6. Adjust if needed (resolution, steps, etc.)

**Saving:**

1. Configure all desired settings
2. Set `template_action` to "Save"
3. Enter name in `new_template_name` (auto-fills with loaded template name if updating)
4. Click "Execute Template Action" button
5. Template saves to `ComfyUI/models/Eclipse/loader_templates/<name>.json`
6. Node automatically switches back to "Load" mode

**Auto-Fill Feature:**
- When switching from "Load" to "Save", the `new_template_name` field automatically fills with the currently loaded template name
- Makes updating existing templates effortless - just switch to Save and execute
- Edit the name if you want to save as a new template instead

**Default Reset System:**
- When loading a template, ALL settings reset to defaults first, then template values are applied
- Ensures clean state with no leftover values from previous configurations
- Template files are minimal and only contain non-default values

**Cross-Loader Compatibility:**
- Templates from Smart Loader Plus work in Smart Loader
- Latent/sampler settings ignored gracefully in Smart Loader
- Model, CLIP, VAE, and quantization settings preserved

### Managing Templates

**Template File System:**

Smart Loader uses a dual-location template system similar to SmartPrompt's wildcard system:

**Primary Location:** `ComfyUI/models/smart_loader_templates/`
- Active template storage (created automatically on first run)
- Templates you create are saved here
- Safe to customize without affecting the extension
- Delete this folder to reset to bundled defaults

**Bundled Templates:** `ComfyUI/custom_nodes/ComfyUI_Eclipse/templates/loader_templates/`
- Default templates included with the extension
- Auto-copied to models folder on first run
- Updated when you update the extension
- Don't edit these directly - they'll be overwritten

**How It Works:**
1. First time you run ComfyUI with Eclipse extension, templates auto-copy from `json/loader_templates/` to `models/smart_loader_templates/`
2. All template operations (save/load/delete) use the models folder
3. Your custom templates stay safe during extension updates
4. To reset templates: delete `models/smart_loader_templates/` folder and restart ComfyUI

**Organization:**
- Files named: `<template_name>.json`
- Edit JSON directly for advanced changes
- Share templates by copying JSON files from models folder

**Deleting:**

1. Set `template_action` to "Delete"
2. Select unwanted template
3. Click "Execute Template Action"
4. Template removed immediately

**Backup:**
- Copy entire `ComfyUI/models/Eclipse/loader_templates/` folder
- Keep backups of important configurations
- Version control with Git if desired

---

## Configuration Guide

### CLIP Configuration

**Baked CLIP (Standard Checkpoints):**
- Simplest option
- Uses CLIP embedded in checkpoint
- Single CLIP module
- Fastest loading

**External CLIP:**
- Load separate CLIP files from `ComfyUI/models/clip/` or `ComfyUI/models/text_encoders/` folders
- Supports `.safetensors` and `.gguf` formats (GGUF requires ComfyUI-GGUF extension)
- Mix different CLIP types
- Build CLIP ensembles (2-4 modules)
- More control over text encoding

**CLIP File Locations:**
- Primary: `ComfyUI/models/clip/`
- Alternative: `ComfyUI/models/text_encoders/`
- Both folders are scanned automatically
- GGUF CLIP models supported in both locations

**CLIP Types Guide:**

| Type | Use For |
|------|---------|
| Flux | Flux models (typically 2 modules) |
| SD3 | Stable Diffusion 3 models |
| SDXL | SDXL models |
| Qwen Image | Qwen image understanding |
| HiDream | HiDream models |
| Standard | SD 1.5, SD 2.x models |

**CLIP Ensemble Tips:**
- Flux typically uses 2 CLIP modules
- SDXL can use 1-2 modules
- More modules = better text understanding (but slower)
- Match CLIP type to your model architecture

**CLIP Layer Trimming:**
- Enable for Standard Checkpoints
- `-2` is balanced default
- `-1` for maximum prompt adherence
- `-3` or `-4` for more creative freedom
- Saves minimal VRAM
- Affects how literally prompts are followed

### VAE Configuration

**Baked VAE:**
- Use when checkpoint has good embedded VAE
- Modern models usually have excellent VAEs
- Single file convenience
- Faster loading

**External VAE:**
- Use for checkpoints with poor/missing VAE
- Experiment with different VAE models
- Required for UNet models
- Common external VAEs:
  - `vae-ft-mse-840000-ema-pruned.safetensors` (SD 1.5)
  - `sdxl_vae.safetensors` (SDXL)

**When to Use External:**
- UNet models (no embedded VAE)
- Checkpoint has poor VAE (blurry, artifacts)
- Experimenting with VAE effects
- Specific VAE requirements

### Latent Configuration (Smart Loader Plus)

**Using Built-in Latent:**

**Pros:**
- One-node convenience
- Preset resolutions
- Integrated with model type
- Cleaner workflow

**Cons:**
- Less flexible than separate node
- Can't modify mid-workflow
- Single batch size throughout

**Disable When:**
- Using dynamic latent sizing
- Need multiple different sizes
- Complex multi-stage workflows
- Prefer modular approach

**Resolution Presets:**
- Choose based on model type
- Flux: Use Flux presets (1024x1024, 832x1216, etc.)
- SDXL: Use SDXL presets (1024x1024, 896x1152, etc.)
- Custom: Enter exact dimensions

**Batch Size:**
- Start with 1 for testing
- Increase for batch generation
- VRAM usage scales linearly
- 4 images = 4x VRAM

### Sampler Configuration (Smart Loader Plus)

**Using Built-in Sampler:**

**Pros:**
- All settings in one place
- Template includes sampler config
- Quick iterations
- Fewer nodes

**Cons:**
- Can't change mid-workflow
- Less flexibility
- Single sampler setup

**Disable When:**
- Using multiple samplers
- Need dynamic sampler changes
- Complex sampling workflows
- Prefer traditional KSampler

**Sampler Selection Guide:**

| Sampler | Speed | Quality | Best For |
|---------|-------|---------|----------|
| euler | Fast | Good | Quick tests |
| euler_a | Fast | Good | General use |
| dpmpp_2m | Medium | Excellent | Quality work |
| dpmpp_sde | Slower | Excellent | High quality |
| dpmpp_2m_sde | Slower | Excellent | Best quality |

**Scheduler Guide:**

| Scheduler | Characteristics |
|-----------|-----------------|
| normal | Standard, predictable |
| karras | Smoother, often better |
| exponential | Aggressive early, gentle late |
| sgm_uniform | Uniform steps |
| simple | Simplified, fast |

**Steps Recommendations:**
- Testing: 15-20 steps
- General: 20-30 steps
- High quality: 30-50 steps
- Flux models: Often good at 20-25 steps

**CFG Scale:**
- Low (1-5): More creative, less prompt adherence
- Medium (6-8): Balanced (recommended start)
- High (9-15): Strong prompt adherence
- Very high (15+): Can cause over-saturation

**Flux Guidance:**
- Flux-specific parameter
- Typical range: 3.0-4.0
- Lower = more creative
- Higher = more controlled
- Start at 3.5

### Quantization Configuration

**Nunchaku Flux Settings:**

**Data Type:**
- `fp8`: Best quality, ~40% VRAM savings
- `fp4`: Good quality, ~50% VRAM savings
- `int4`: Maximum compression, ~75% VRAM savings

**Cache Threshold:**
- Default: 0.1
- Higher = more caching (faster, more VRAM)
- Lower = less caching (slower, less VRAM)

**Attention Mode:**
- `flash_attn`: Faster, recommended (requires Flash Attention)
- `sdpa`: Slower, compatible fallback

**I2F Mode:**
- Image-to-feature conversion method
- Default usually optimal

**CPU Offload:**
- Enable: Moves some processing to CPU (slower, saves VRAM)
- Disable: All on GPU (faster, more VRAM)

**Nunchaku Qwen Settings:**

**GPU Blocks:**
- Number of model blocks on GPU
- Higher = faster, more VRAM
- Lower = slower, less VRAM
- Auto-detect usually best

**Pinned Memory:**
- Speeds up CPU-GPU transfers
- Enable unless system RAM limited

**GGUF Settings:**

**Dequant Dtype:**
- `auto`: Recommended (system decides)
- Q4_0, Q4_1, Q5_0, Q5_1: Various quantization levels
- Q8_0: High quality quantization
- F16, F32: Higher precision

**Patch Dtype:**
- Layer-specific precision
- Usually leave at default

**Patch on Device:**
- `auto`: Recommended
- `CPU`: Force CPU (slower, saves VRAM)
- `GPU`: Force GPU (faster, more VRAM)

---

## Troubleshooting

### Extension Not Found

**Problem:** "Nunchaku support not available" or "GGUF support not available"

**Solution:**
1. Install required extension:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku
   git clone https://github.com/city96/ComfyUI-GGUF
   ```
2. Restart ComfyUI
3. Quantized options now enabled

### Model Loading Fails

**Problem:** Error loading selected model file

**Check:**
1. File in correct folder:
   - Standard: `models/checkpoints/`
   - UNet: `models/unet/`
   - Nunchaku Flux: `models/nunchaku/`
   - Nunchaku Qwen: `models/qwen/`
   - GGUF: `models/gguf/`

2. File extension correct (.safetensors preferred)
3. File not corrupted (redownload if needed)
4. Sufficient storage space
5. File permissions readable

### CLIP Configuration Error

**Problem:** CLIP fails to load or gives error

**Solutions:**
1. **For Standard Checkpoint:**
   - Use "Baked" CLIP source
   - Enable CLIP configuration
   - Check CLIP layer range (-24 to -1)

2. **For UNet/Quantized:**
   - Must use "External" CLIP
   - Select appropriate CLIP type
   - Ensure CLIP files exist
   - Match CLIP count to model needs

3. **For Flux models:**
   - Set CLIP count to 2
   - Select both CLIP files
   - Use "flux_text_encoders" type

### VAE Issues

**Problem:** Images blurry, artifacts, or VAE error

**Solutions:**
1. Try external VAE:
   - Download appropriate VAE
   - Place in `models/vae/`
   - Set VAE source to "External"
   - Select VAE file

2. Verify VAE compatibility:
   - SD 1.5 needs SD 1.5 VAE
   - SDXL needs SDXL VAE
   - Match VAE to model type

3. Check VAE file:
   - Not corrupted
   - Correct format (.safetensors)
   - Sufficient permissions

### Out of VRAM

**Problem:** CUDA out of memory error during loading

**Solutions:**

1. **Use Quantization:**
   - Try fp8 first (good quality)
   - Use int4 for maximum savings
   - Enable Nunchaku or GGUF models

2. **Enable CPU Offload:**
   - Nunchaku: Set cpu_offload = true
   - Trades speed for VRAM

3. **Reduce Configuration:**
   - Lower CLIP count (4→2→1)
   - Disable CLIP layer trimming
   - Use smaller batch size
   - Lower resolution

4. **System Level:**
   - Close other applications
   - Check VRAM usage in Task Manager
   - Restart ComfyUI
   - Update GPU drivers

### Template Won't Load

**Problem:** Template fails to load or settings incorrect

**Solutions:**
1. Verify template file exists:
   - Check `ComfyUI/models/smart_loader_templates/`
   - File named `<template_name>.json`

2. Check template format:
   - Valid JSON structure
   - Not corrupted
   - Compatible version

3. Reload template list:
   - Set template_action to "None"
   - Wait a moment
   - Set back to "Load"
   - Refresh dropdown

4. Re-create if needed:
   - Delete corrupted template
   - Configure manually
   - Save new template

### Sampler Settings Not Applied

**Problem:** Using Smart Loader Plus but sampler settings ignored

**Check:**
1. `configure_sampler` is enabled
2. Values are in valid ranges:
   - Steps: 1-150
   - CFG: 0-100
   - Flux guidance: 0-100

3. Connecting pipe to correct node
4. Not overriding with separate KSampler

### Model Sampling Not Working

**Problem:** Sampling method selected but not taking effect

**Solutions:**
1. **Verify Configuration:**
   - `configure_model_sampling` is enabled
   - Selected method matches model architecture
   - Method-specific parameters in valid ranges

2. **Check Model Compatibility:**
   - SD3: Use with Stable Diffusion 3 models only
   - Flux: Use with Flux models (Dev, Schnell, variants)
   - LCM: Use with distilled/LCM models only
   - Method must match model architecture

3. **Flux Specific:**
   - Dimensions auto-fill from latent (Smart Loader Plus)
   - Verify width/height set if auto-fill fails
   - Check console for shift calculation messages

4. **LCM Specific:**
   - Reduce steps significantly (4-8 typical)
   - original_timesteps should match training (usually 50)
   - Try enabling zsnr for better quality

5. **Continuous Methods:**
   - Verify sigma_max > sigma_min
   - Use recommended defaults initially
   - ContinuousEDM: Select appropriate subtype for model

### Quantization Quality Issues

**Problem:** Generated images lower quality with quantized models

**Improvements:**
1. **Nunchaku Settings:**
   - Use fp8 instead of int4
   - Set cache_threshold higher (0.2-0.5)
   - Use flash_attn attention
   - Disable CPU offload if possible

2. **Generation Settings:**
   - Increase steps (30-40)
   - Adjust CFG scale
   - Try different samplers
   - Use better prompts

3. **Model Choice:**
   - Some models quantize better than others
   - Try different quantization levels
   - Test unquantized version for comparison

### Dynamic UI Not Updating

**Problem:** Widgets not showing/hiding when changing settings

**Solutions:**
1. Refresh page (Ctrl+F5)
2. Restart ComfyUI
3. Check browser console for errors
4. Verify JavaScript not blocked
5. Try different browser
6. Update ComfyUI to latest version

### Template Button Missing

**Problem:** Can't find "Execute Template Action" button

**Solution:**
- Set `template_action` to anything except "None"
- Button appears automatically
- Set to "None" to hide button

---

## Advanced Topics

### Model Sampling Configuration

**Understanding Sampling Methods:**

Different model architectures require specific sampling approaches for optimal quality. The Smart Loaders support 8 major sampling methods:

**Flow-Based Methods** (SD3, AuraFlow, Flux):
- Use flow-matching sampling for smooth generation
- Require `shift` parameter to control sampling schedule
- Higher shift = more refinement early in generation

**Discrete Methods** (Stable Cascade, LCM):
- Use discrete timestep schedules
- LCM designed for few-step generation (4-8 steps)
- Stable Cascade has unique architecture requirements

**Continuous Methods** (ContinuousEDM, ContinuousV):
- Use continuous noise schedules
- Control via sigma_max and sigma_min
- More flexible than discrete schedules

**Video Methods** (LTXV):
- Specialized for video generation
- Uses token-based calculations
- Handles temporal consistency

**Choosing the Right Method:**

1. **Check Model Type:**
   - Read model card/documentation
   - Look for architecture mentions (SD3, Flux, etc.)
   - Test with "None" first to see if special sampling needed

2. **SD3 Models:**
   - Use "SD3" method
   - Default shift (3.0) works for most
   - Increase for more detail, decrease for faster generation

3. **Flux Models:**
   - Use "Flux" method
   - max_shift=1.15, base_shift=0.5 (defaults)
   - Dimensions auto-filled in Smart Loader Plus
   - Lower max_shift for more creative results

4. **AuraFlow Models:**
   - Use "AuraFlow" method
   - shift=1.73 (default) is optimal
   - multiplier=1.0 (lower than SD3)

5. **LCM/Distilled Models:**
   - Use "LCM" method
   - Reduce steps to 4-8 (critical for LCM)
   - original_timesteps=50 (match training)
   - Enable zsnr for better quality

6. **Stable Cascade:**
   - Use "Stable Cascade" method
   - shift=2.0 (default)
   - Requires Stable Cascade architecture

7. **EDM-Based Models:**
   - Use "ContinuousEDM" method
   - Select appropriate subtype:
     - `eps`: Epsilon prediction (most common)
     - `v_prediction`: V-prediction parameterization
     - `edm`: Standard EDM
     - `edm_playground_v2.5`: Playground 2.5 models
     - `cosmos_rflow`: Cosmos models
   - Adjust sigma_max/min if needed

8. **V-Prediction Models:**
   - Use "ContinuousV" method
   - Higher sigma values than EDM
   - sigma_max=500.0, sigma_min=0.03

**Flux Dimension Calculation:**

Flux uses image dimensions to calculate shift dynamically:
```
x1 = 256, x2 = 4096
slope = (max_shift - base_shift) / (x2 - x1)
intercept = base_shift - slope * x1
shift = ((width * height / 1024) * slope) + intercept
```

**Smart Loader Plus** auto-fills dimensions from latent configuration.
**Smart Loader** requires manual width/height input if using Flux.

**Parameter Tuning:**

**shift (Flow methods):**
- Lower (1.0-2.0): Faster, more creative, less refined
- Medium (2.0-3.0): Balanced quality
- Higher (3.0-5.0): More refinement, slower, more detail

**sigma_max/sigma_min (Continuous methods):**
- Wider range (high max, low min): More noise schedule steps
- Narrower range: Fewer steps, faster
- Match to model training for best results

**original_timesteps (LCM):**
- Must match distillation training
- Usually 50 for standard LCM
- Check model documentation

**zsnr (LCM):**
- Enables zero-terminal SNR
- Often improves quality
- May affect color/brightness slightly

### CLIP Ensemble Configuration

**Single CLIP (Simple):**
```
clip_source: Baked (or External)
clip_count: 1
clip_type: (appropriate for model)
clip_name1: (if external)
```

**Dual CLIP (Flux):**
```
clip_source: External
clip_count: 2
clip_type: flux_text_encoders
clip_name1: t5xxl_fp8_e4m3fn.safetensors
clip_name2: clip_l.safetensors
```

**Quad CLIP (Advanced):**
```
clip_source: External
clip_count: 4
clip_type: (depends on architecture)
clip_name1-4: (four different CLIP files)
```

### Custom Resolution Configuration

**Portrait Photography:**
```
resolution: Custom
width: 768
height: 1024
batch_size: 1
```

**Landscape Wallpaper:**
```
resolution: Custom
width: 2048
height: 1152
batch_size: 1
```

**Square Social Media:**
```
resolution: Flux 1:1 (1024x1024)
batch_size: 4
```

### Memory-Optimized Configuration

**For 8GB VRAM:**
```
model_type: Nunchaku Flux
data_type: int4
cpu_offload: true
clip_count: 2
resolution: Flux 1:1 (1024x1024)
batch_size: 1
steps: 20
```

**For 12GB VRAM:**
```
model_type: Nunchaku Flux
data_type: fp8
cpu_offload: false
clip_count: 2
resolution: Flux 16:9 (1344x768)
batch_size: 1
steps: 25
```

**For 24GB+ VRAM:**
```
model_type: Standard Checkpoint
clip_source: Baked
vae_source: Baked
resolution: Custom (1536x1536)
batch_size: 2
steps: 30
```

---

## Related Documentation

- [Checkpoint Loaders Guide](Checkpoint_Loaders.md) - Traditional checkpoint loaders
- [Pipe System Guide](Pipe_System.md) - Understanding Eclipse pipes
- [Template Management](Templates.md) - Advanced template techniques

---

## Quick Reference

### Smart Loader Plus

**Minimum Configuration:**
- model_type
- Model file selection
- configure_clip: true
- configure_vae: true

**Recommended Start:**
- Standard Checkpoint
- Baked CLIP and VAE
- CLIP layer: -2
- Resolution preset
- Batch: 1
- Sampler: euler
- Steps: 20
- CFG: 7

### Smart Loader

**Minimum Configuration:**
- model_type
- Model file selection
- configure_clip: true
- configure_vae: true

**Recommended Start:**
- Standard Checkpoint
- Baked CLIP and VAE
- CLIP layer: -2
- Use separate Empty Latent
- Use separate KSampler

---

**Need help?** Check the main [README](../README.md) or open an issue on the [GitHub repository](https://github.com/r-vage/ComfyUI_Eclipse).

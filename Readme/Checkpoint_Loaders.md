# Checkpoint Loader User Guide

This guide explains how to use the traditional checkpoint loaders in ComfyUI-RvTools. These loaders are simple, reliable options for loading AI models without advanced features.

## Table of Contents
- [Overview](#overview)
- [Checkpoint Loader Small](#checkpoint-loader-small)
- [Checkpoint Loader Small (Pipe)](#checkpoint-loader-small-pipe)
- [Common Settings](#common-settings)
- [Tips & Best Practices](#tips--best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

ComfyUI-RvTools provides two traditional checkpoint loaders:

1. **Checkpoint Loader Small** - Outputs individual components (model, CLIP, VAE)
2. **Checkpoint Loader Small (Pipe)** - Outputs a single pipe containing all components

Both loaders provide the same core functionality but differ in their output format. Use the regular version for standard workflows and the Pipe version when working with RvTools pipe-based workflows.

### When to Use These Loaders

Use these loaders when you need:
- Simple checkpoint loading without advanced features
- Basic model/CLIP/VAE configuration
- Compatibility with traditional ComfyUI workflows
- Lightweight, straightforward model loading

For advanced features like multi-format support, quantization, templates, or automatic latent/sampler configuration, consider using the [Smart Loader series](Smart_Loaders.md) instead.

---

## Checkpoint Loader Small

**Node Name:** `Checkpoint Loader Small [RvTools]`

### What It Does

Loads a checkpoint file (AI model) and provides separate outputs for the model, CLIP, and VAE components. This is the traditional way to load models in ComfyUI.

### Inputs

#### ckpt_name (Required)
- **What it is:** The checkpoint file to load
- **Where to find it:** Dropdown shows all `.safetensors`, `.ckpt`, `.pt`, and `.pth` files in your `ComfyUI/models/checkpoints` folder
- **What to choose:** Select the AI model you want to use for generation
- **Example:** `sd_xl_base_1.0.safetensors`, `flux1-dev.safetensors`

**Security Note:** The loader prefers modern `.safetensors` files and will warn you if you use older `.ckpt` or `.pt` formats. Safetensors are safer and faster.

#### vae_name (Required)
- **What it is:** The VAE (Variational AutoEncoder) to use for encoding/decoding images
- **Options:**
  - `Baked VAE` - Use the VAE embedded in the checkpoint file (recommended for most models)
  - Any VAE file from `ComfyUI/models/vae` folder
- **When to change:** Only use external VAE if your model specifically requires it or if you want to experiment with different VAE models
- **Example:** `vae-ft-mse-840000-ema-pruned.safetensors`

**Tip:** Most modern checkpoints have excellent built-in VAEs. Start with "Baked VAE" unless you have a specific reason to change it.

#### stop_at_clip_layer (Required)
- **What it is:** Controls how much of the CLIP text encoder to use
- **Default:** `-2` (recommended for most use cases)
- **Range:** `-24` to `-1`
- **What the numbers mean:**
  - `-1` = Use the full CLIP model (all layers)
  - `-2` = Skip the last layer (commonly used, good balance)
  - `-3` to `-24` = Skip more layers (progressively reduces CLIP influence)

**What it does:** Trimming CLIP layers can sometimes improve image quality by reducing over-fitting to prompts, and it also saves a small amount of memory.

**Recommendation:** Leave at `-2` unless you're experimenting. Lower numbers (like `-3` or `-4`) sometimes help with very detailed prompts.

### Outputs

The node provides 4 separate outputs:

1. **model** (MODEL) - The diffusion model for image generation
2. **clip** (CLIP) - The text encoder for understanding prompts
3. **vae** (VAE) - The encoder/decoder for image conversion
4. **model_name** (STRING) - The filename of the loaded checkpoint

### How to Use

1. **Add the node** to your workflow from the node menu: `RvTools > Loader > Checkpoint Loader Small`
2. **Select your checkpoint** from the `ckpt_name` dropdown
3. **Choose VAE setting:**
   - Leave as `Baked VAE` for most models
   - Select a custom VAE if needed
4. **Adjust CLIP layers** (optional):
   - Keep default `-2` for normal use
   - Try `-3` or `-4` if experimenting with prompt adherence
5. **Connect the outputs:**
   - Connect **model** to KSampler's model input
   - Connect **clip** to CLIP Text Encode nodes
   - Connect **vae** to VAE Decode node

### Example Workflow Connection

```
Checkpoint Loader Small
├─ model ────────> KSampler (model input)
├─ clip ─────────> CLIP Text Encode (Positive)
├─ clip ─────────> CLIP Text Encode (Negative)
└─ vae ──────────> VAE Decode
```

---

## Checkpoint Loader Small (Pipe)

**Node Name:** `Checkpoint Loader Small (Pipe) [RvTools]`

### What It Does

Same as the regular Checkpoint Loader Small, but outputs all components in a single pipe object instead of separate connections. This reduces visual clutter in complex workflows.

### Inputs

Identical to Checkpoint Loader Small:
- **ckpt_name** - Select your checkpoint file
- **vae_name** - Choose VAE (Baked or external)
- **stop_at_clip_layer** - CLIP layer trimming (-24 to -1)

See the [Checkpoint Loader Small](#checkpoint-loader-small) section above for detailed explanations of each input.

### Outputs

The node provides 1 pipe output containing all components:

**pipe** (PIPE) - A dictionary-style pipe containing:
- `model` - The diffusion model
- `clip` - The text encoder
- `vae` - The image encoder/decoder
- `latent` - Placeholder (empty, not configured)
- `width` - Not configured (None)
- `height` - Not configured (None)
- `batch_size` - Set to 1
- `model_name` - Checkpoint filename
- `vae_name` - VAE filename (empty if using Baked VAE)
- `clip_skip` - The CLIP layer setting

### How to Use

1. **Add the node** from: `RvTools > Loader > Checkpoint Loader Small (Pipe)`
2. **Configure settings** (same as regular loader)
3. **Connect the pipe** to downstream nodes:
   - Use **Pipe Out** nodes to extract individual components
   - Or connect directly to other pipe-compatible nodes

### Example Workflow with Pipe

```
Checkpoint Loader Small (Pipe)
└─ pipe ──────────> Pipe Out Checkpoint Loader
                    ├─ model ──────> KSampler
                    ├─ clip ───────> CLIP Text Encode
                    └─ vae ────────> VAE Decode
```

### When to Use Pipe Version

**Use Pipe version when:**
- Working with RvTools pipe ecosystem
- Building complex workflows with many connections
- You want cleaner, more organized node graphs
- Connecting to Context (Image) or other pipe nodes

**Use Regular version when:**
- Working with standard ComfyUI nodes
- You prefer traditional separate outputs
- Building simple workflows

---

## Common Settings

### Checkpoint File Selection

**Supported Formats:**
- `.safetensors` - **Recommended** (modern, safe, fast)
- `.sft` - Safetensors alternative extension
- `.ckpt` - Legacy format (shows warning)
- `.pt` - Legacy format (shows warning)
- `.pth` - Legacy format (shows warning)

**File Location:** Place checkpoint files in `ComfyUI/models/checkpoints/`

**Security:** Both loaders verify that checkpoint files are:
- Located in the checkpoints folder (prevents unauthorized file access)
- Readable and accessible
- Using safe formats (warns on legacy formats)

### VAE Configuration

**Baked VAE (Default):**
- Uses the VAE embedded in the checkpoint
- No additional file needed
- Recommended for most users
- Fastest loading

**External VAE:**
- Uses a separate VAE file
- Useful for:
  - Checkpoints without embedded VAE
  - Experimenting with different VAE models
  - Fixing VAE-related image issues
- Place VAE files in `ComfyUI/models/vae/`

**Popular External VAEs:**
- `vae-ft-mse-840000-ema-pruned.safetensors` (Stable Diffusion 1.5)
- `sdxl_vae.safetensors` (SDXL models)

### CLIP Layer Trimming

**What It Affects:**
- How closely images follow your text prompts
- Memory usage (minimal difference)
- Generation style

**Common Values:**
- `-1` - Full CLIP, maximum prompt adherence
- `-2` - **Default**, balanced performance (recommended)
- `-3` - Slightly reduced prompt adherence
- `-4` - More creative, less literal interpretation
- `-5` to `-24` - Progressively less prompt influence

**When to Adjust:**
- Start at `-2` (default)
- If images are too literal/rigid, try `-3` or `-4`
- If prompts aren't followed well enough, try `-1`
- Extreme values (below -10) rarely useful

---

## Tips & Best Practices

### General Usage

1. **Start Simple:** Use "Baked VAE" and `-2` CLIP layer for most models
2. **Use Safetensors:** Prefer `.safetensors` files for safety and speed
3. **Organize Models:** Keep checkpoints organized in subfolders within `ComfyUI/models/checkpoints/`
4. **Check Console:** Watch the console for warnings about file formats or paths

### Performance Optimization

1. **File Format:** `.safetensors` loads faster than `.ckpt` or `.pt`
2. **CLIP Trimming:** Higher values (like `-2`) use slightly less memory than `-1`
3. **VAE Choice:** "Baked VAE" loads faster than external VAE (one file vs two)

### Workflow Organization

**Use Regular Loader when:**
- Building standard ComfyUI workflows
- You need direct access to individual components
- Working with non-pipe nodes

**Use Pipe Loader when:**
- Working with RvTools pipe ecosystem
- Building complex multi-stage workflows
- You want to reduce connection clutter
- Passing model data through multiple processing stages

### Model Compatibility

Both loaders work with:
- Stable Diffusion 1.5 models
- SDXL models
- Stable Diffusion 2.x models
- Most community fine-tunes and merges

They do NOT work with:
- UNet-only models (no CLIP/VAE)
- Nunchaku quantized models
- GGUF quantized models

For these formats, use the [Smart Loader series](Smart_Loaders.md).

---

## Troubleshooting

### "Checkpoint not found" Error

**Problem:** The selected checkpoint file cannot be found.

**Solutions:**
1. Verify file is in `ComfyUI/models/checkpoints/` folder
2. Check file extension is supported (.safetensors, .ckpt, .pt, .pth)
3. Restart ComfyUI to refresh file list
4. Check file permissions (must be readable)

### "VAE not found" Error

**Problem:** Selected external VAE file is missing.

**Solutions:**
1. Switch to "Baked VAE" if the checkpoint has embedded VAE
2. Verify VAE file is in `ComfyUI/models/vae/` folder
3. Check VAE filename in dropdown matches actual file
4. Restart ComfyUI to refresh VAE list

### Legacy Format Warning

**Message:** "Warning: legacy checkpoint extension detected"

**What it means:** You're loading an older `.ckpt`, `.pt`, or `.pth` file.

**Solutions:**
1. **Safe option:** Convert checkpoint to `.safetensors` format using conversion tools
2. **Quick option:** Ignore warning if you trust the file source
3. Download modern `.safetensors` version of the model if available

**Why it matters:** Legacy formats can contain malicious code. Safetensors cannot.

### Path Outside Checkpoints Folder Warning

**Message:** "Warning: resolved checkpoint path is outside the checkpoints folder"

**What it means:** Security check detected potential unauthorized file access.

**Solutions:**
1. Move the checkpoint file into `ComfyUI/models/checkpoints/`
2. If using symlinks, verify they point to safe locations
3. Check that you're not using absolute paths or `..` in filenames

### Out of Memory / VRAM Issues

**Problem:** Loading checkpoint fails with memory error.

**Solutions:**
1. Close other GPU-intensive applications
2. Use CLIP trimming (try `-3` or `-4`) to save small amount of VRAM
3. Consider using quantized models with Smart Loader Plus
4. Upgrade GPU or reduce batch size in your sampler

### CLIP Layer Setting Not Working

**Problem:** Changing `stop_at_clip_layer` doesn't seem to affect output.

**Solutions:**
1. Ensure value is between `-24` and `-1`
2. Try more extreme values (like `-1` vs `-4`) to see difference
3. Clear ComfyUI cache and regenerate
4. Check that CLIP is actually connected to your text encoding nodes

### Pipe Output Not Connecting

**Problem:** Using Pipe version, but connections fail.

**Solutions:**
1. Verify you're connecting to pipe-compatible nodes
2. Use "Pipe Out Checkpoint Loader" node to extract components
3. Check that downstream node expects pipe input
4. For standard nodes, use regular Checkpoint Loader Small instead

---

## Related Documentation

- [Smart Loader User Guide](Smart_Loaders.md) - Advanced multi-format loaders with templates and quantization
- [Pipe Ecosystem Guide](Pipe_System.md) - Understanding RvTools pipe-based workflows
- [Model Management Guide](Model_Management.md) - Organizing and managing your model collection

---

## Quick Reference

### Checkpoint Loader Small

| Setting | Default | Purpose |
|---------|---------|---------|
| ckpt_name | (required) | Which model to load |
| vae_name | Baked VAE | Which VAE to use |
| stop_at_clip_layer | -2 | CLIP layer trimming |

**Outputs:** model, clip, vae, model_name

### Checkpoint Loader Small (Pipe)

| Setting | Default | Purpose |
|---------|---------|---------|
| ckpt_name | (required) | Which model to load |
| vae_name | Baked VAE | Which VAE to use |
| stop_at_clip_layer | -2 | CLIP layer trimming |

**Outputs:** pipe (containing model, clip, vae, metadata)

---

**Need help?** Check the main [README](../README.md) or open an issue on the [GitHub repository](https://github.com/r-vage/ComfyUI_RvTools).

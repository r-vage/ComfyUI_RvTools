# Smart Loader Templates

This folder contains configuration templates for both Smart Loader and Smart Loader Plus nodes.

## Usage

### Loading a Template
1. Set `template_action` to **"Load"**
2. Select a template from `template_name`
3. The configuration will be applied automatically
4. All settings reset to defaults first, then template values are applied

### Saving a Template
1. Configure your loader node with desired settings
2. Set `template_action` to **"Save"**
3. Enter a name in `new_template_name` (auto-fills with loaded template name when updating)
4. Click the "Execute Template Action" button
5. Node automatically switches back to "Load" mode with your template selected

### Updating an Existing Template
1. Load the template you want to update
2. Make your changes
3. Switch to **"Save"** mode
4. The `new_template_name` field auto-fills with the loaded template name
5. Click "Execute Template Action" to overwrite (or edit the name to save as new)

### Deleting a Template
1. Set `template_action` to **"Delete"**
2. Select the template from `template_name`
3. Click the "Execute Template Action" button
4. Template will be removed

## Intelligent Template Saving

Templates use **smart field filtering** - only relevant settings for your configuration are saved:

### Always Saved
- Model type selection
- Configuration toggles (configure_clip, configure_vae, configure_latent, configure_sampler, configure_model_only_lora)

### Model-Specific Settings

**Standard Checkpoint:**
```json
{
  "model_type": "Standard Checkpoint",
  "ckpt_name": "model.safetensors",
  "clip_source": "Baked",
  "enable_clip_layer": true,
  "stop_at_clip_layer": -2,
  "vae_source": "Baked"
}
```
- CLIP layer trimming only saved for Standard Checkpoints
- No clip_count, clip_type (not needed for baked CLIP)

**UNet Model:**
```json
{
  "model_type": "UNet Model",
  "unet_name": "flux-dev-unet.safetensors",
  "weight_dtype": "fp8_e4m3fn",
  "clip_source": "External",
  "clip_count": "2",
  "clip_type": "flux",
  "clip_name1": "clip_l.safetensors",
  "clip_name2": "t5xxl_fp8.safetensors",
  "vae_source": "External",
  "vae_name": "ae.safetensors"
}
```
- Only saves weight_dtype (not Nunchaku settings)
- Requires external CLIP and VAE

**Nunchaku Flux:**
```json
{
  "model_type": "Nunchaku Flux",
  "nunchaku_name": "svdq-fp4-flux.safetensors",
  "data_type": "bfloat16",
  "cache_threshold": 0.1,
  "attention": "flash-attention2",
  "i2f_mode": "enabled",
  "cpu_offload": "auto",
  "clip_source": "External",
  "clip_count": "2",
  "clip_type": "flux"
}
```
- Flux-specific quantization settings
- Requires external CLIP

**Nunchaku Qwen:**
```json
{
  "model_type": "Nunchaku Qwen",
  "qwen_name": "qwen-quant.safetensors",
  "cpu_offload": "auto",
  "num_blocks_on_gpu": 1,
  "use_pin_memory": "disable",
  "clip_source": "External",
  "clip_type": "qwen_image"
}
```
- Only offload parameters (attention/i2f_mode not used by Qwen)

**GGUF Model:**
```json
{
  "model_type": "GGUF Model",
  "gguf_name": "flux-q4.gguf",
  "gguf_dequant_dtype": "default",
  "gguf_patch_dtype": "default",
  "gguf_patch_on_device": false,
  "clip_source": "External",
  "clip_type": "flux"
}
```

### CLIP Configuration
- **Baked CLIP**: Only saves clip_source, enable_clip_layer, stop_at_clip_layer
- **External CLIP**: Saves clip_count, clip_type, clip_name1-4 (if not "None")

### VAE Configuration
- Always saves vae_source
- Only saves vae_name if External and not "None"

### Latent Configuration (Smart Loader Plus only)
- Saves resolution preset
- Does NOT save batch_size (local/workflow value)
- Only saves width/height if resolution is "Custom"

### Sampler Configuration (Smart Loader Plus only)
- Saves sampler_name, scheduler, steps, cfg
- Only saves flux_guidance for Flux models (Nunchaku Flux, UNet/GGUF with clip_type="flux")

### LoRA Configuration
- Saves lora_count
- Saves all 3 LoRA slots (switch, name, weight) even if disabled for easy toggling

## Cross-Compatibility

Templates work across both Smart Loader and Smart Loader Plus:
- **Smart Loader Plus → Smart Loader**: Latent/sampler settings gracefully ignored
- **Smart Loader → Smart Loader Plus**: Missing fields use defaults (configure_latent=false, configure_sampler=false)

## What's NOT Saved

These values are intentionally excluded:
- ❌ **batch_size** - Local/workflow value, not configuration
- ❌ **flux_guidance** - For non-Flux models (Standard Checkpoint, Nunchaku Qwen)
- ❌ **clip_count/clip_type** - For Standard Checkpoints with baked CLIP
- ❌ **enable_clip_layer/stop_at_clip_layer** - For UNet/Nunchaku/GGUF models
- ❌ **attention/i2f_mode** - For Nunchaku Qwen (not used by loader)
- ❌ **data_type/cache_threshold** - For UNet models (Nunchaku Flux only)

## Example Templates

### Complete Flux Workflow (Smart Loader Plus)
```json
{
  "model_type": "UNet Model",
  "configure_clip": true,
  "configure_vae": true,
  "configure_latent": true,
  "configure_sampler": true,
  "unet_name": "flux1-dev.safetensors",
  "weight_dtype": "fp8_e4m3fn",
  "clip_source": "External",
  "clip_count": "2",
  "clip_type": "flux",
  "clip_name1": "clip_l.safetensors",
  "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
  "vae_source": "External",
  "vae_name": "ae.safetensors",
  "resolution": "1024x1024 (1:1)",
  "sampler_name": "euler",
  "scheduler": "simple",
  "steps": 20,
  "cfg": 1.0,
  "flux_guidance": 3.5
}
```

### SDXL Portrait (Standard Checkpoint)
```json
{
  "model_type": "Standard Checkpoint",
  "configure_clip": true,
  "configure_vae": true,
  "ckpt_name": "epicrealism_xl.safetensors",
  "clip_source": "Baked",
  "enable_clip_layer": true,
  "stop_at_clip_layer": -2,
  "vae_source": "Baked",
  "resolution": "832x1216 (2:3 Portrait)",
  "sampler_name": "dpmpp_2m_sde",
  "scheduler": "karras",
  "steps": 28,
  "cfg": 7.0
}
```

### Memory-Optimized Flux (Nunchaku)
```json
{
  "model_type": "Nunchaku Flux",
  "configure_clip": true,
  "configure_vae": true,
  "nunchaku_name": "svdq-int4-flux.safetensors",
  "data_type": "bfloat16",
  "cache_threshold": 0.0,
  "attention": "flash-attention2",
  "cpu_offload": "enable",
  "clip_source": "External",
  "clip_count": "2",
  "clip_type": "flux"
}
```

## Tips

- Use descriptive names: `Flux_Dev_1024`, `SDXL_Portrait`, `Qwen_Analysis`
- Templates are cross-compatible between Smart Loader and Smart Loader Plus
- Auto-fill feature makes updating templates effortless
- Templates only store what matters - cleaner files, faster loading
- Share templates by copying JSON files between installations

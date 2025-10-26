# Smart Loader Templates

This folder contains configuration templates for the Smart Loader node.

## Usage

### Loading a Template
1. Set `template_action` to **"Load"**
2. Select a template from `template_name`
3. The configuration will be applied automatically

### Saving a Template
1. Configure your Smart Loader node with desired settings
2. Set `template_action` to **"Save"**
3. Enter a name in `new_template_name`
4. Execute the node - your configuration will be saved

### Deleting a Template
1. Set `template_action` to **"Delete"**
2. Select the template from `template_name`
3. Execute the node - the template will be removed

## Template Format

Templates are JSON files with the following structure:

```json
{
  "model_type": "Standard Checkpoint",
  "ckpt_name": "flux1-dev.safetensors",
  "unet_name": "None",
  "weight_dtype": "default",
  "configure_clip": true,
  "configure_vae": true,
  "configure_latent": true,
  "clip_source": "Baked",
  "clip_count": "1",
  "clip_name1": "None",
  "clip_name2": "None",
  "clip_name3": "None",
  "clip_name4": "None",
  "clip_type": "flux",
  "enable_clip_layer": false,
  "stop_at_clip_layer": -2,
  "vae_source": "Baked",
  "vae_name": "None",
  "resolution": "1024x1024",
  "width": 1024,
  "height": 1024,
  "batch_size": 1
}
```

## What's Saved

Templates now save **both configuration AND file selections**:
- ✅ Model type (Standard/UNet)
- ✅ Selected checkpoint/UNet file
- ✅ Component toggles (CLIP/VAE/Latent)
- ✅ Source settings (Baked/External)
- ✅ Selected CLIP files (1-4)
- ✅ Selected VAE file
- ✅ CLIP count, type, layer trimming
- ✅ Resolution presets, batch size
- ✅ Weight dtype (for UNet)

This means you can have complete preset workflows like:
- **"MyFluxWorkflow"** → Specific Flux checkpoint + settings
- **"SDXL_Refiner"** → Specific SDXL checkpoint + 2 external CLIPs + custom VAE

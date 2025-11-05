# ComfyUI_Eclipse

ComfyUI_Eclipse is a collection of custom nodes, helpers and utilities for ComfyUI designed to make workflow building easier and more reliable. It includes convenience nodes for loading checkpoints and pipelines, type conversions, folder and filename helpers, simple image utilities, logic and flow helpers, and small toolkits for working with VAE/CLIP and latents.

Note: Workflows created with RvTools_v2 are NOT compatible with this version. This release contains a substantial cleanup and many improvements.

## Documentation

Detailed documentation for specific features:

- **[Smart Loaders Guide](Readme/Smart_Loaders.md)** - Complete guide to Smart Loader and Smart Loader Plus with multi-format support
- **[Smart Prompt Guide](Readme/Smart_Prompt.md)** - How to use and customize the Smart Prompt system
- **[Wildcard Processor Guide](Readme/Wildcard_Processor.md)** - Advanced wildcard syntax and usage examples
- **[Checkpoint Loaders Guide](Readme/Checkpoint_Loaders.md)** - Legacy checkpoint loader documentation
- **[Nunchaku Installation](Readme/Nunchaku_Installation.md)** - Step-by-step guide for installing Nunchaku quantization support
- **[User Documentation Index](Readme/README.md)** - Complete index of all user guides

## Highlights

- **Smart Loader Series:** Next-generation model loaders with multi-format support (Standard Checkpoints, UNet, Nunchaku quantized Flux/Qwen, GGUF quantized models), featuring template management, automatic memory cleanup, and graceful extension fallbacks. [→ Documentation](Readme/Smart_Loaders.md)
  - **Smart Loader Plus:** Full-featured loader with latent/sampler configuration, resolution presets, CLIP ensemble (up to 4 modules), and comprehensive quantization support.
  - **Smart Loader:** Streamlined variant for minimal workflows - loads model/CLIP/VAE without latent or sampler configuration.
- **Smart Prompt System:** Quick prompt building with dropdown selectors loaded from organized text files. Pre-configured with subjects, settings, and environments. Users can create custom prompt files by adding numbered `.txt` files (e.g., `1_my_prompts.txt`) - each line becomes a selectable option. Supports folder filtering and random selection with seed control for reproducible prompt generation. Files are automatically copied to `ComfyUI/models/wildcards/smartprompt/` for wildcard integration on first run. [→ Documentation](Readme/Smart_Prompt.md)
- **Wildcard Processor:** Advanced wildcard system for dynamic prompt generation. Create custom wildcard files in the `ComfyUI/models/wildcards/` directory using `.txt` files with one option per line. Supports weighted options (`option:weight` format), nested wildcards, and dynamic seed integration for complex prompt variations. Example wildcards are automatically copied on first launch. [→ Documentation](Readme/Wildcard_Processor.md)
- **Legacy Checkpoint Loaders:** Traditional loaders including Checkpoint Loader Small and Small (Pipe) variants for basic checkpoint loading workflows.
- **Sophisticated Pipe Ecosystem:** Standardized data interchange system with context pipes, generation data pipes, concatenation, and extraction nodes to eliminate spaghetti connections in complex workflows. (More detailed documentation can be found below.)
- **Comprehensive Switching System:** Extensive switch and multi-switch nodes for all ComfyUI data types, enabling dynamic workflow branching and conditional execution.
- **Advanced Text Processing:** Prompt generation with environment/subject sliders, smart prompt systems, multiline string inputs, and regex-based string replacement.
- **Video Workflow Tools:** Video clip combination, seamless joining, WAN frame helpers, loop calculators, and video-specific pipe contexts for professional video generation.
- **Sampler Settings Management:** Specialized sampler configurations for Flux, SDXL, and standard models with preset management and pipe-based distribution.
- **Type Conversion Suite:** Comprehensive conversion nodes (Any → Float/Integer/String/Combo), list/batch transformations, mask operations, and string merging utilities.
- **Universal Passers:** Type-safe data passing nodes for all ComfyUI types (models, latents, images, conditioning, pipes, etc.) to maintain data integrity through workflows.
- **Resolution & Settings Presets:** Built-in resolution presets for popular aspect ratios (Flux, SDXL, HiDream, Qwen, etc.) and directory-based settings management.
- **Core Utilities:** VRAM purging helpers, colored console logging, path management, and comprehensive sampler/scheduler lists for all model types.

The nodes live under the `py/` directory and are grouped by function. The `core/` directory contains shared utilities and constants used by the nodes.

## Contents

- `py/` — All custom node implementations (checkpoint loaders, conversion nodes, folder utilities, image helpers, logic nodes, passers, pipes, etc.).
- `core/` — Shared code: categories, logging helpers (`cstr`), VRAM purge helper, configuration and keys.
- `json/`, `settings/`, `workflow/`, `web/` — Assets, example settings, sample workflows and a small web frontend helper.
- `requirements.txt` / `pyproject.toml` — Declared dependencies and packaging metadata.

## License

This project is licensed under the Apache License 2.0 (see `LICENSE`). Check the license before embedding parts of this project in other software.

## Beginner-friendly installation

The easiest way to install ComfyUI_Eclipse is to place it in ComfyUI's `custom_nodes` folder so ComfyUI will discover the nodes automatically.

1. Locate your ComfyUI installation folder.
2. Inside ComfyUI, find (or create) the `custom_nodes` folder.
3. Copy the entire `ComfyUI_Eclipse` folder into `custom_nodes` so the tree looks like:

```
ComfyUI/
  custom_nodes/
    ComfyUI_Eclipse/
      py/
      core/
      README.md
      ...
```

Or, clone directly into `custom_nodes`:

```powershell
# from your ComfyUI directory (PowerShell)
git clone https://github.com/r-vage/ComfyUI_Eclipse custom_nodes/ComfyUI_Eclipse
```

4. Install any optional Python dependencies required by specific nodes. From the repository root (or your ComfyUI root), run:

```powershell
# optional - only if your ComfyUI environment is missing packages from requirements.txt
pip install -r custom_nodes/ComfyUI_Eclipse/requirements.txt

# For ComfyUI portable installations:
python_embeded\python.exe -m pip install -r custom_nodes/ComfyUI_Eclipse/requirements.txt
```

Common dependencies referenced by nodes include: torch, numpy, Pillow, opencv-python, piexif and others. ComfyUI itself usually provides the main ML stack (torch, torchvision, safetensors), but if you see errors you may need to install missing packages.

5. Restart ComfyUI. The new nodes should appear in the node list under categories provided by the package.

### Opening a console / terminal in the ComfyUI folder (beginner)

If you're new to command lines, here's a very short guide to open a terminal (console) already located in your ComfyUI folder so you can run commands there.

Windows (PowerShell / Windows Terminal):

- Option A — From File Explorer:
  1. Open File Explorer and navigate to the ComfyUI installation folder (the folder that contains `run_nvidia_gpu.bat`, `webui.bat`, `main.py` or similar files).
  2. Hold Shift, right-click on an empty area in the folder and choose "Open PowerShell window here" or "Open in Windows Terminal".

- Option B — From any PowerShell window:
  1. Open PowerShell or Windows Terminal.
  2. Change directory to the ComfyUI folder, for example:

```powershell
# replace the path below with your actual ComfyUI path
cd 'D:\path\to\ComfyUI'
# or using Set-Location
Set-Location 'D:\path\to\ComfyUI'
```

Notes for Windows:
- If your path contains spaces, wrap it in single or double quotes.
- Your default shell may be PowerShell (`pwsh.exe`) or Command Prompt (`cmd.exe`); PowerShell and Windows Terminal are recommended.

macOS / Linux (Terminal):

1. Open Terminal (Spotlight → "Terminal" on macOS, or your terminal emulator on Linux).
2. Change directory to the ComfyUI folder, for example:

```bash
# replace the path below with your actual ComfyUI path
cd /home/you/ComfyUI
```

Tips:
- Use Tab to autocomplete long folder names.
- If you use a Python virtual environment, activate it from the same console before running ComfyUI.

## Quick start — using the Smart Loaders

The Smart Loader series provides modern, flexible model loading with support for multiple formats and quantization methods.

### Smart Loader Plus [Eclipse]
The full-featured loader for complex workflows:

- **Multi-Format Support:** Standard Checkpoints, UNet models, Nunchaku quantized Flux/Qwen (SVDQuant INT4/FP4/FP8), and GGUF quantized models.
- **Template System:** Save and load complete configurations including model selections, CLIP/VAE settings, and sampler parameters.
- **CLIP Ensemble:** Support for up to 4 CLIP modules with multiple architecture types (Flux, SD3, SDXL, Qwen, HiDream, Hunyuan, Wan, etc.).
- **Advanced Configuration:** 
  - Latent configuration with resolution presets or custom dimensions
  - Sampler settings (sampler, scheduler, steps, CFG, flux_guidance)
  - CLIP layer trimming for memory optimization
  - Weight dtype control (fp8 variants)
- **Quantization Options:**
  - Nunchaku Flux: Data type, cache threshold, attention mode, CPU offload
  - Nunchaku Qwen: GPU block allocation, pinned memory
  - GGUF: Dequantization dtype, patch dtype, device placement
- **Outputs:** Single pipe containing model, CLIP, VAE, latent, dimensions, batch size, sampler settings, and metadata.

### Smart Loader [Eclipse]
Simplified loader for streamlined workflows:

- **Same Format Support:** Standard Checkpoints, UNet, Nunchaku Flux/Qwen, GGUF models.
- **Template Compatibility:** Load templates from Smart Loader Plus (latent/sampler settings ignored gracefully).
- **Minimal Configuration:** Focus on model/CLIP/VAE loading only.
- **No Latent/Sampler:** Use separate nodes for Empty Latent Image and KSampler configuration.
- **Outputs:** Pipe containing model, CLIP, VAE, model name, and metadata.

### Required Extensions for Quantized Models

To use Nunchaku or GGUF quantized models with the Smart Loaders, you need to install the following ComfyUI extensions:

**For Nunchaku Support (SVDQuant INT4/FP4/FP8):**
- Repository: [ComfyUI-Nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku)
- Installation: Clone into your `custom_nodes` folder
- Supports: Nunchaku Flux and Nunchaku Qwen quantized models

**For GGUF Support:**
- Repository: [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- Installation: Clone into your `custom_nodes` folder
- Supports: GGUF quantized model formats

```powershell
# Navigate to your ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes

# Install Nunchaku support
git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku

# Install GGUF support
git clone https://github.com/city96/ComfyUI-GGUF
```

**Note:** The Smart Loaders will work without these extensions installed, but quantized model options will be disabled. Standard Checkpoints and UNet models work without additional dependencies.

Basic usage:

1. Add Smart Loader or Smart Loader Plus to your workflow.
2. Select model type (Standard Checkpoint, UNet, Nunchaku Flux, Nunchaku Qwen, or GGUF).
3. Choose the appropriate model file from the dropdown.
4. Configure CLIP (baked or external) and VAE (baked or external).
5. For Smart Loader Plus: Set resolution, batch size, and sampler settings.
6. Use templates to save/load configurations for quick workflow iteration.
7. Connect the pipe output to downstream nodes or use Pipe Out nodes to extract components.

The Smart Loaders include comprehensive error handling, automatic VRAM cleanup, and graceful fallbacks when optional extensions (Nunchaku, GGUF) are not installed.

## Tips & troubleshooting

- If a node raises an import error for a package, install the missing package into the same Python environment that runs ComfyUI.
- If you place the folder under `custom_nodes` but the nodes don't show up, restart ComfyUI and check the server logs for import errors.

## Contributing

Contributions, bug reports, and PRs are welcome. Please fork the repository, make changes in a feature branch, and open a PR with a short description of the change.

If opening issues, include the ComfyUI version, Python version, torch/CUDA details (if relevant), and error tracebacks.

## Node categories overview

This project groups nodes into categories to make them easier to find in ComfyUI. Below is a short summary of the categories provided by ComfyUI_Eclipse:

- **Eclipse (Main)** — Top-level group for general Eclipse nodes and primary entry points. Contains high-level helpers and commonly used nodes.
- **Loader** — Smart loaders and checkpoint loaders (model / VAE / CLIP / latent). Advanced loaders with multi-format support including Standard Checkpoints, UNet, Nunchaku quantized models, and GGUF formats.
- **Conversion** — Type conversion helpers (Any → Float/Integer/String/Combo, lists ↔ batches, image/mask conversions, string merging, pipe concatenation, etc.).
- **Folder** — Nodes for creating and managing project folders, filename prefixing, and smart folder utilities with placeholder support to organize outputs.
- **Image** — Image utilities for loading from various sources, previewing, saving with advanced metadata, and manipulating images in workflows.
- **Router** — Routing and control nodes for conditional execution, switches, multi-switches, and any-type data passing through workflows.
- **Pipe** — Pipeline and composition helpers (12-channel pipes, context managers for image/video workflows, generation data, sampler settings, and pipe extraction nodes).
- **Primitives** — Small building-block nodes for basic values (Boolean, Integer, Float, String) used in control flow and logic operations.
- **Settings** — Nodes for sampler configurations, resolution presets, directory settings, ControlNet union types, and video name generators used to tune pipelines.
- **Text** — String and text-processing helpers (multiline input, smart prompts, wildcard processing, regex replacement, dual text inputs).
- **Video** — Video workflow utilities (loop/keep calculators, video clip combination, seamless joining, frame helpers for professional video generation).
- **Utilities** — General utility nodes (LoRA stack management, Show Any for debugging, workflow control with Stop, RAM/VRAM cleanup).

If you open ComfyUI after installing the package you'll find these categories in the node chooser; categories are intended to be concise and practical so you can quickly locate the right node for your workflow.

## Files by category

### Conversion
Convenience nodes for type conversion, list/batch transforms, string merging, and context/pipe manipulation.
- Concat Multi - Concatenate multiple pipes
- Convert Primitive - Convert Any to String/Integer/Float/Combo
- Convert To Batch - Convert image/mask lists to batches
- Convert to List - Convert image/mask batches to lists
- Image Convert - Convert images to RGB format
- Join - Join strings, lists, and pipes
- Lora Stack to String - Convert LoRA stack to formatted string
- Merge Strings - Merge multiple strings
- String from List - Extract string from list by index
- Widget to String - Convert widget values to strings

### Folder
Nodes for creating and managing project folders, filename prefixing, and smart folder utilities to organize outputs.
- Add Folder - Add folder prefix to paths
- Filename Prefix - Add customizable filename prefix
- Smart Folder - Advanced folder management with placeholder support

### Image
Image utilities for loading, previewing, saving, and manipulating images in workflows and output nodes.
- Load Image - Load single image with metadata
- Load Image Path - Load image from custom path
- Load Image Path (Pipe) - Load image from path with pipe output
- Preview Image - Preview images in workflow
- Preview Mask - Preview masks in workflow
- Save Images - Advanced image saving with metadata and placeholders

### Loader
Nodes for loading model checkpoints with support for Standard, UNet, Nunchaku quantized, and GGUF formats.
- Smart Loader - Streamlined loader (model/CLIP/VAE only)
- Smart Loader Plus - Full-featured loader with latent/sampler configuration
- Checkpoint Loader Small - Basic checkpoint loader
- Checkpoint Loader Small (Pipe) - Basic checkpoint loader with pipe output

### Primitives (Logic / Basic values)
Small building-block nodes for booleans, numbers, and strings, used in control flow and logic operations.
- Boolean - Boolean value input
- Float - Float value input
- Integer - Integer value input
- Integer (Gen) - Integer generator with range
- String - String value input

### Router
Routing and control nodes for conditional execution, switches, and data passing.
- Any Passer - Pass any data type through workflow
- Any Dual Switch - Switch between two any-type inputs
- Any Multi-Switch - Switch between multiple any-type inputs
- If Execute - Conditional execution control

### Pipe
Pipeline and composition helpers: context managers, multi-channel pipes, generation data, and out nodes for assembling or emitting pipeline data.
- Pipe 12CH Any - 12-channel any-type pipe
- Context (Image) - Image generation context pipe
- Context (Video) - Video generation context pipe
- Context (WanVideo) - WanVideo wrapper context pipe
- Generation Data - Generation metadata pipe
- Sampler Settings - Sampler configuration pipe
- Pipe Out Checkpoint Loader - Extract checkpoint loader data from pipe
- Pipe Out Load Directory Settings - Extract directory settings from pipe
- Pipe Out Load Image - Extract image data from pipe
- Pipe Out Sampler Settings - Extract sampler settings from pipe
- Pipe Out Smart Folder - Extract smart folder data from pipe
- Pipe Out VCNameGen - Extract video name generator data from pipe
- Pipe Out WanVideo Setup - Extract WanVideo setup from pipe

### Settings
Nodes that expose or compose small settings objects (sampler presets, resolution helpers, directory settings) used to tune pipelines.
- ControlNet Union Type - ControlNet union type selector for Flux
- Custom Size - Custom resolution input
- Image Resolutions - Resolution presets
- Load Directory Settings - Directory configuration for outputs
- Sampler Selection - Sampler and scheduler selector
- Sampler Settings - Basic sampler configuration
- Sampler Settings (NI) - Sampler settings with noise injection configuration
- Sampler Settings (NI+Seed) - Sampler settings with noise injection configuration + seed
- Sampler Settings (Seed) - Sampler settings with seed control
- Sampler Settings (Small) - Minimal sampler configuration
- Sampler Settings (Small+Seed) - Minimal sampler configuration with seed
- VCNameGen v1 - Video/checkpoint name generator v1
- VCNameGen v2 - Video/checkpoint name generator v2
- WanVideo Setup - WanVideo configuration

### Text
Nodes for prompt construction, text processing, and string manipulation with advanced placeholder and wildcard support.
- Dual Text - Two independent text inputs
- Multiline Text - Multiline string input that also outputs the string as list
- Replace String - Simple string replacement
- Replace String v2 - Advanced regex string replacement
- Smart Prompt - Dynamic prompt generation, dynamic seed widget when seed_input is connected
- Wildcard Processor - Process wildcards in prompts, allows wildcard files, weights, and seed control, dynamic seed widget when seed_input is connected

### Video
Nodes for video clip composition, frame utilities, and loop/frame calculations for video-friendly pipelines.
- Loop Calculator - Calculate loop frame counts
- Keep Calculator - Calculate frame keep/trim values
- Combine Video Clips - Concatenate video clips
- Seamless Join Video Clips - Join video clips with seamless transitions

### Utilities
General utility nodes for LoRA management, debugging, resource management, and workflow control.
- LoRA Stack - Build LoRA stack configuration
- LoRA Stack Apply - Apply LoRA stack to model/CLIP (supports nunchaku quantized models)
- Show Any - Display any data type for debugging, also allows tensor to image conversion to display images and masks.
- Stop - Stop workflow execution
- RAM Cleanup - Manual RAM cleanup
- VRAM Cleanup - Manual VRAM cleanup

## Node Spotlight: Save Images [Eclipse]

The **Save Images** node is a highly advanced and flexible output node designed for robust image saving in ComfyUI workflows, offering extensive customization and metadata support.

### Outstanding Abilities
- **Flexible Input Handling:** Accepts images directly via the `images` input or through a connected metadata pipe (`pipe_opt`), allowing saving from any workflow stage or metadata-only operations.
- **Dynamic Output Organization:** Supports customizable output paths and filenames using a rich set of placeholders for automatic organization by date, model, seed, and generation parameters.
- **Comprehensive Metadata Embedding:** Can embed full workflow data, prompts, and generation parameters directly into PNG or WEBP metadata for traceability and reproducibility.
- **Generation Data Preservation:** When enabled, saves all generation parameters (prompts, model info, sampler, seed, CFG, steps, etc.) as embedded metadata and/or as a separate JSON file alongside images.
- **Lora and Embedding Hashing:** Automatically extracts and hashes used Loras and embeddings from prompts, storing short hashes for Civitai compatibility.
- **UI Integration and Previews:** Optionally returns preview data for ComfyUI's UI, including filenames and subfolder paths for easy navigation.
- **Advanced Filename Management:** Features robust sanitization, collision avoidance with numeric counters, and support for custom delimiters and padding.
- **Multi-Format Support:** Saves images in PNG, JPEG, TIFF, GIF, BMP, and WEBP formats, with options for DPI, quality, lossless compression, and optimization.
- **Automatic Directory Creation:** Creates output folders on-the-fly if they don't exist, ensuring seamless saving.
- **Civitai Compatibility:** Extracts model, Lora, and embedding hashes in Civitai's expected format for metadata sharing.

### Placeholder System
The node supports placeholders in both output paths and filename prefixes for dynamic value insertion. Placeholders are replaced with actual values at save time; unknown or empty placeholders default to readable alternatives.

Supported placeholders:
- `%today`, `%date` — Current date (YYYY-MM-DD)
- `%time` — Current time (HHMMSS)
- `%Y`, `%m`, `%d`, `%H`, `%S` — Individual date/time components
- `%basemodel`, `%model` — Model names (base model and full model)
- `%seed`, `%sampler_name`, `%scheduler`, `%steps`, `%cfg`, `%denoise`, `%clip_skip` — Generation parameters

Example: `%today\%basemodel\%seed_%sampler_name_%steps` creates organized folder structures like `2025-09-27\ModelName\12345_euler_20`.

### Connection Possibilities
- **Direct Image Input:** Connect any image output to the `images` input for standard saving scenarios.
- **Pipe Input:** Connect a metadata pipe (from context or logic nodes) to `pipe_opt` to save images and extract metadata from complex workflows automatically.
- **Hybrid Usage:** Combine both inputs for maximum flexibility, allowing images from one source and metadata from another.

### Generation Data Saving
When the `save_generation_data` option is enabled:
- Embeds all generation parameters (prompts, model names, sampler settings, seed, CFG, steps, etc.) into image metadata.
- Optionally saves the full workflow as a separate JSON file alongside each image.
- Extracts and includes short SHA-256 hashes for models, Loras, and embeddings in Civitai-compatible format.
- Supports prompt removal for privacy and Lora token appending for full traceability.

## Node Spotlight: Smart Loader Plus [Eclipse]

The **Smart Loader Plus** is the flagship model loader for ComfyUI_Eclipse, representing the next generation of checkpoint loading with comprehensive multi-format support, advanced configuration options, and intelligent workflow integration. It's designed for power users who need maximum flexibility and control over their model loading pipeline.

### Outstanding Abilities

#### Multi-Format Model Support
- **Standard Checkpoints:** Traditional safetensors/ckpt checkpoints with full MODEL/CLIP/VAE support
- **UNet Models:** Standalone diffusion models without embedded CLIP/VAE
- **Nunchaku Flux Quantized:** SVDQuant INT4/FP4/FP8 compressed Flux models (requires ComfyUI-Nunchaku)
- **Nunchaku Qwen Quantized:** SVDQuant compressed Qwen models with specialized image understanding (requires ComfyUI-Nunchaku)
- **GGUF Quantized:** GGUF format models with flexible quantization options (requires ComfyUI-GGUF)
- **Automatic Format Detection:** Intelligently detects model type and configures appropriate loading pipeline

#### Advanced CLIP Ensemble System
- **Multi-CLIP Support:** Load up to 4 CLIP modules simultaneously for ensemble configurations
- **Architecture Flexibility:** Supports Flux, SD3, SDXL, Qwen Image, HiDream, Hunyuan Image, Wan, and standard CLIP architectures
- **Layer Trimming:** Optional CLIP layer trimming (clip_skip) for memory optimization and generation control
- **Baked vs External:** Choose between checkpoint-embedded CLIP or external CLIP files per slot
- **Smart Defaults:** Automatically configures sensible CLIP settings based on model type

#### Template Management System
- **Save Configurations:** Store complete loader states including model paths, CLIP/VAE choices, and all settings
- **Quick Loading:** Instantly restore complex configurations with a single template selection
- **Template Organization:** Manage templates via dropdown with create/load/delete operations
- **Cross-Version Compatibility:** Templates saved in Smart Loader Plus gracefully degrade when loaded in Smart Loader
- **Persistent Storage:** Templates stored in `json/loader_templates/` directory

#### Latent Configuration
- **Resolution Presets:** Extensive preset library for common aspect ratios:
  - Flux ratios (1:1, 16:9, 9:16, 21:9, 3:2, 2:3, etc.)
  - SDXL resolutions (1024x1024, landscape, portrait, widescreen)
  - HiDream, Qwen, and custom ratios
- **Custom Dimensions:** Manual width/height input with validation
- **Batch Size Control:** Configure batch generation count (1-64)
- **Automatic Latent Creation:** Generates empty latent tensor based on configuration
- **VAE Channel Detection:** Automatically adapts latent channels to VAE architecture

#### Sampler Settings Integration
- **Embedded Sampler Config:** Built-in sampler/scheduler selection with immediate workflow integration
- **Complete Parameter Set:**
  - Sampler selection (euler, euler_a, dpmpp_2m, etc.)
  - Scheduler (normal, karras, exponential, sgm_uniform, etc.)
  - Steps (1-150)
  - CFG scale (0.0-100.0)
  - Flux guidance (0.0-100.0 for Flux models)
- **Pipe Integration:** Sampler settings embedded in output pipe for downstream nodes
- **Optional Configuration:** Can be disabled to use external sampler nodes instead

#### Quantization Configuration
Each quantized format has specialized settings:

**Nunchaku Flux:**
- Data type selection (fp8, fp4, int4)
- Cache threshold for memory management
- Attention mode (flash_attn, sdpa)
- CPU offload toggle for VRAM conservation

**Nunchaku Qwen:**
- GPU block allocation control
- Pinned memory optimization

**GGUF:**
- Dequantization dtype (auto, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32)
- Patch dtype for layer-specific precision
- Device placement (auto, CPU, GPU)

#### VAE Management
- **Baked VAE Support:** Use VAE embedded in checkpoint
- **External VAE Loading:** Load standalone VAE files
- **Automatic Fallback:** Falls back to baked VAE if external loading fails
- **Format Validation:** Warns about legacy extensions and prefers .safetensors

#### Weight Dtype Control
- **FP8 Variants:** Support for E4M3FN and E5M2 fp8 formats for memory-efficient loading
- **Default Precision:** Auto-selects appropriate precision based on model type
- **Manual Override:** Fine-tune weight precision for performance optimization

#### Error Handling & Validation
- **Fail-Fast Validation:** Early detection of missing files, invalid configurations
- **Graceful Degradation:** Continues operation when optional extensions unavailable
- **Informative Warnings:** Detailed console messages for legacy formats, missing files
- **Extension Detection:** Automatically disables quantized options if extensions not installed
- **Memory Management:** Automatic VRAM cleanup after loading operations

#### Pipe Output System
The loader outputs a comprehensive dict-style pipe containing:
- **model:** Loaded MODEL object (or None for external handling)
- **clip:** CLIP ensemble (single or multi-module)
- **vae:** VAE decoder/encoder
- **latent:** Pre-configured empty latent tensor
- **width, height:** Latent dimensions
- **batch_size:** Batch count for generation
- **sampler_name, scheduler:** Sampler configuration
- **steps, cfg, flux_guidance:** Generation parameters
- **model_name, vae_name:** Asset names for metadata
- **clip_skip:** CLIP layer trimming setting
- **is_nunchaku:** Quantization detection flag

### Advanced Features

#### Dynamic UI Visibility
The JavaScript frontend dynamically shows/hides options based on selections:
- Model type determines available options (UNet disables baked CLIP/VAE)
- Configure latent/sampler toggles control visibility of respective sections
- Quantization options appear only for appropriate model types
- Template operations show/hide based on template selection

#### Memory Optimization
- **Lazy Loading:** Components loaded only when needed
- **VRAM Purging:** Automatic cleanup after loading operations
- **Quantization Support:** Reduced memory footprint with INT4/FP4/FP8 models
- **CPU Offloading:** Option to offload model components to system RAM

#### Workflow Integration
- **Single-Pipe Architecture:** One connection carries all generation data
- **Pipe Out Compatibility:** Works with all Eclipse Pipe Out nodes for component extraction
- **Context Pipe Support:** Directly feeds into Context (Image) and Context (Video) nodes
- **Metadata Preservation:** Model/VAE names preserved for Save Images and generation tracking

### Use Cases

**Complex Multi-Model Workflows:**
Perfect for workflows requiring CLIP ensembles, mixed precision models, or advanced quantization.

**Rapid Prototyping:**
Template system enables instant switching between model configurations for A/B testing.

**Memory-Constrained Environments:**
Quantization support (INT4/FP4/FP8) and CPU offloading enable running larger models on limited VRAM.

**Production Pipelines:**
Comprehensive error handling and validation ensure reliable operation in automated workflows.

**Cross-Platform Compatibility:**
Graceful fallbacks ensure workflows remain portable even without optional extensions.

### Best Practices

- **Use Templates:** Save frequently-used configurations for instant recall
- **Enable Configure Latent:** When using separate Empty Latent nodes, disable to avoid redundancy
- **Match Sampler Settings:** Disable built-in sampler config if using external KSampler nodes
- **Monitor VRAM:** Use quantized models (Nunchaku/GGUF) for VRAM-limited systems
- **Validate Extensions:** Install ComfyUI-Nunchaku and ComfyUI-GGUF for full feature access
- **Check Console:** Review loading messages for warnings about legacy formats or missing files

The Smart Loader Plus represents the pinnacle of flexible, powerful model loading in ComfyUI, designed to handle everything from simple checkpoint loading to complex multi-model quantized workflows with ease.

## Node Spotlight: Smart Prompt [Eclipse]

The **Smart Prompt** node is an intuitive prompt building system that transforms text file collections into organized dropdown menus, enabling rapid prompt composition and experimentation with reproducible results.

### Outstanding Abilities

#### File-Based Dropdown System
- **Automatic Widget Generation:** Each text file in the `prompt/` directory automatically becomes a labeled dropdown widget in the node
- **Organized Categories:** Files are grouped by subfolder (subjects, settings, environments, etc.) for logical organization
- **Clean Naming:** File numbering (e.g., `1_styles.txt`, `2_lighting.txt`) controls display order while showing clean names in the UI
- **Line-Based Options:** Each line in a text file becomes a selectable option in the corresponding dropdown
- **Dynamic Loading:** Adding new files or modifying existing ones updates the node without restart

#### Folder Structure & Organization
The node scans the `prompt/` directory structure:
- **Subfolders as Categories:** Each subfolder represents a prompt category (e.g., `subjects/`, `settings/`, `environments/`)
- **Numbered Files:** Files prefixed with numbers (e.g., `1_character.txt`, `2_pose.txt`) control widget order
- **Multiple Folders Supported:** Organize prompts across multiple category folders for clean separation

#### Selection Modes
Each dropdown offers three selection modes:
- **None:** Skip this option in the final prompt
- **Random:** Randomly select one line from the file (seed-controlled for reproducibility)
- **Specific Selection:** Choose any specific line from the file's contents

#### Seed Control & Reproducibility
- **Deterministic Randomization:** When using "Random" selections, the seed ensures identical results
- **Seed Input Support:** Optional seed input port allows dynamic seed control from workflow
- **Special Seeds:** Supports standard ComfyUI seed conventions (-1 for random, with server-generated seeds)
- **Workflow Integration:** Resolved seeds are saved to workflow metadata for perfect reproducibility

#### Folder Filtering
- **All Folders View:** Display all prompt options from all subfolders simultaneously
- **Single Folder Filter:** Show only widgets from a specific category folder for focused prompt building
- **Dynamic UI:** Folder selection instantly updates visible widgets

#### Prompt Assembly
- **Automatic Concatenation:** Selected options are intelligently joined with commas
- **Smart Cleanup:** Removes trailing punctuation, extra spaces, and empty selections
- **Natural Language Output:** Produces clean, readable prompts ready for generation

### Creating Custom Prompt Files

Users can easily create custom prompt libraries:

1. **Navigate to the `prompt/` directory** in the ComfyUI_Eclipse installation
2. **Choose or create a subfolder** (e.g., `subjects/`, `settings/`, or create your own like `styles/`)
3. **Create a numbered text file** (e.g., `1_my_prompts.txt`, `2_lighting_styles.txt`)
4. **Add one option per line:**
   ```
   dramatic lighting
   soft natural light
   golden hour glow
   studio lighting
   moody shadows
   ```
5. **Save the file** and the node will automatically detect it on next load

**Naming Convention:**
- Start with a number for ordering: `1_`, `2_`, `3_`, etc.
- Follow with a descriptive name: `character`, `pose`, `lighting`, etc.
- Use `.txt` extension
- Example: `1_character.txt`, `2_pose.txt`, `3_clothing.txt`

### Use Cases

**Rapid Prompt Iteration:**
Build complex prompts quickly by selecting from curated options rather than typing repeatedly.

**A/B Testing:**
Easily swap between different prompt components to test variations systematically.

**Team Collaboration:**
Share prompt libraries across teams by distributing text files with standardized options.

**Workflow Templates:**
Create reusable prompt structures for specific art styles, subjects, or generation types.

**Learning Tool:**
Explore effective prompting by browsing pre-configured options and understanding what works.

### Best Practices

- **Organize by Category:** Keep related prompts in the same subfolder for easier navigation
- **Use Clear Names:** Name files descriptively so widgets are self-explanatory
- **One Concept Per Line:** Each line should represent a complete, coherent prompt element
- **Number for Order:** Use file numbering to control the logical flow of prompt building
- **Document Libraries:** Consider adding a `_desc` version of files with explanations
- **Seed Everything:** Always use seed control when using Random selections for reproducible results

The Smart Prompt node transforms prompt engineering from manual typing to curated selection, dramatically speeding up workflow iteration and experimentation.

## Node Spotlight: Wildcard Processor [Eclipse]

The **Wildcard Processor** node is a powerful prompt expansion system that enables dynamic, randomized prompt generation through template-based wildcards, supporting nested expansions, weighted selections, and advanced pattern matching for infinite prompt variations.

### Outstanding Abilities

#### Wildcard Syntax & Expansion
- **Basic Wildcards:** Use `{wildcard_name}` syntax to insert random selections from wildcard files
- **Nested Wildcards:** Wildcards can contain other wildcards for complex, layered expansions
- **Multiple Wildcards:** Use multiple wildcards in a single prompt for combinatorial variations
- **Inline Options:** Define options directly in prompts using `{option1|option2|option3}` syntax
- **File-Based Wildcards:** Load wildcard options from `.txt` files in the `ComfyUI/models/wildcards/` directory

#### Weighted Selection System
- **Weight Syntax:** Assign probability weights using `option:weight` format
- **Flexible Weighting:** Higher weights increase selection probability (e.g., `red:5` is 5x more likely than `blue:1`)
- **Mixed Weighting:** Combine weighted and unweighted options (unweighted defaults to weight of 1)
- **Normalized Distribution:** Weights are automatically normalized to create proper probability distributions

**Example weighted wildcard file (`colors.txt`):**
```
red:5
blue:3
green:2
yellow:1
```

#### Wildcard File Management
- **Custom Wildcards:** Create `.txt` files in the `ComfyUI/models/wildcards/` directory
- **One Option Per Line:** Each line represents one possible expansion
- **Automatic Discovery:** New wildcard files are automatically detected
- **Hierarchical Organization:** Use subfolders to organize wildcards by category
- **File Naming:** Wildcard filename (without .txt) becomes the wildcard name
- **Example Files:** Example wildcards are automatically copied to the directory on first launch

**Creating a wildcard file:**
1. Navigate to `ComfyUI/models/wildcards/` directory
2. Create a text file (e.g., `emotions.txt`)
3. Add one option per line:
   ```
   happy:3
   sad:2
   excited
   calm
   mysterious:4
   ```
4. Use in prompts as `{emotions}`

#### Seed Control & Reproducibility
- **Deterministic Randomization:** Same seed always produces same expansion
- **Seed Input Port:** Optional seed input for dynamic seed control from workflow
- **Dynamic Seed Widget:** Seed widget appears/updates when seed_input is connected
- **Workflow Integration:** Ensures reproducible prompt variations across generations

#### Advanced Pattern Matching
- **Recursive Expansion:** Wildcards are expanded recursively until no wildcards remain
- **Escape Sequences:** Support for escaping special characters when needed
- **Multiple Passes:** Continues expansion until all nested wildcards are resolved
- **Error Handling:** Graceful fallback for missing or invalid wildcard files

#### Prompt Output
- **Expanded Prompt:** Returns fully expanded prompt with all wildcards replaced
- **Clean Output:** Removes extra spaces, normalizes formatting
- **Reproducible:** Identical seed + template = identical output

### Wildcard Syntax Examples

**Basic Wildcard:**
```
A {color} {animal} in a {location}
```
With `color.txt`, `animal.txt`, `location.txt` files in wildcards directory.

**Inline Options:**
```
A {red|blue|green} car driving through {city|forest|desert}
```

**Weighted Inline Options:**
```
{portrait:5|landscape:2|abstract:1} painting of {cats:3|dogs:2|birds}
```

**Nested Wildcards:**
```
{art_style} artwork featuring {subject}
```
Where `art_style.txt` might contain `{modern|classical} {painting|sketch}`.

**Complex Combination:**
```
{quality:10|} {art_style}, {subject} {pose:3|}, {lighting}, {background:2|}
```
Combines weighted selections, optional elements (empty string), and multiple wildcards.

### Creating Custom Wildcard Libraries

**Simple Wildcard (`styles.txt`):**
```
photorealistic
anime style
oil painting
watercolor
digital art
sketch
```

**Weighted Wildcard (`quality.txt`):**
```
masterpiece:10
best quality:8
high quality:5
normal quality:2
low quality:1
```

**Nested Wildcard (`character.txt`):**
```
{male|female} {human|elf|dwarf}
{young|old} {warrior|mage|rogue}
mysterious {hero|villain}
```

### Use Cases

**Infinite Variation:**
Generate endless unique prompts from template structures for large batch generation.

**Dataset Creation:**
Create diverse training datasets by generating thousands of unique prompts automatically.

**Exploration & Discovery:**
Discover unexpected prompt combinations by letting wildcards randomly combine elements.

**Style Experimentation:**
Test different art styles, subjects, and compositions systematically with controlled randomization.

**Batch Processing:**
Generate varied images in batch workflows while maintaining structural consistency.

### Best Practices

- **Start Simple:** Begin with basic wildcards before moving to nested/weighted versions
- **Use Weights Wisely:** Weight your most desired outcomes higher for better results
- **Organize Categories:** Keep related wildcards in separate files for maintainability
- **Test Templates:** Test wildcard templates with multiple seeds to ensure good variation
- **Document Wildcards:** Consider naming conventions that make wildcard purpose clear
- **Combine with Smart Prompt:** Use both nodes together - Smart Prompt for structure, Wildcards for variation
- **Seed Control is Critical:** Always use seed management for reproducible or iteratable results

The Wildcard Processor transforms static prompts into dynamic templates, enabling systematic exploration of prompt space and effortless generation of diverse, reproducible variations.

## The Pipe Ecosystem of [Eclipse]

The pipe ecosystem in ComfyUI_Eclipse is a sophisticated data interchange system designed to standardize and simplify the flow of complex data structures through ComfyUI workflows. Pipes act as containers that bundle related parameters, models, and settings into single, manageable objects, eliminating the need for dozens of individual node connections.

### Core Concept

A pipe is fundamentally a Python dictionary that encapsulates multiple related pieces of data. Instead of connecting separate wires for model, CLIP, VAE, latent tensor, dimensions, sampler settings, and metadata, all of this information can be passed through a single pipe connection. This approach dramatically reduces workflow complexity and improves maintainability.

### Pipe Types and Variants

#### Context Pipes
Context pipes are the foundation of the ecosystem, holding the core components of a generation pipeline:

- **Context (Image) (`Context (Image) [Eclipse]`):** Standard image generation context containing model, CLIP, VAE, conditioning (positive/negative), latent, sampler/scheduler, generation parameters (steps, cfg, seed, dimensions), and text prompts. Ideal for standard image generation workflows.
- **Context (Video) (`Context (Video) [Eclipse]`):** Extended context for video workflows, adding video-specific parameters like frame rate, frame load cap, skip frames, select every nth frame, and audio/image inputs/outputs. Designed for video generation pipelines.
- **Context (WanVideo) (`Context (WanVideo) [Eclipse]`):** Specialized wrapper for WAN Video Workflows, supporting WANVIDEOMODEL and WANTEXTENCODER types with additional video processing parameters for WAN-based video generation.

#### Generation Data Pipes
These pipes focus on sampler and generation settings:

- **Generation Data (`Generation Data [Eclipse]`):** Contains sampler/scheduler names, steps, cfg, seed, dimensions, text prompts, model/VAE names, LoRA names, denoise strength, and CLIP skip settings. Perfect for metadata tracking and parameter preservation.

#### Sampler Settings Pipes
Specialized pipes for different sampling configurations:

- **Sampler Settings (`Sampler Settings [Eclipse]`):** Comprehensive sampler configuration with sampler/scheduler, steps, CFG, seed, and denoise parameters.
- **Sampler Settings (Small) (`Sampler Settings (Small) [Eclipse]`):** Minimal sampler configuration with basic sampler/scheduler, steps, and CFG.
- **Sampler Settings (Small+Seed) (`Sampler Settings (Small+Seed) [Eclipse]`):** Minimal configuration with added seed control.
- **Sampler Settings (Seed) (`Sampler Settings (Seed) [Eclipse]`):** Full sampler settings with integrated seed management.
- **Sampler Settings (NI) (`Sampler Settings (NI) [Eclipse]`):** Noise Injection Parameters with generation settings (no seed).
- **Sampler Settings (NI+Seed) (`Sampler Settings (NI+Seed) [Eclipse]`):** Noise Injection Parameters with seed and generation settings.

#### Multi-Channel Pipes
Flexible any-type data pipes for custom workflows:

- **Pipe 12CH Any (`Pipe 12CH Any [Eclipse]`):** 12-channel any-type pipe for complex custom workflows requiring multiple arbitrary data streams.

### Key Abilities

#### 1. Standardized Data Interchange
- **Dict-Style Format:** All pipes use consistent dictionary structures with canonical key names.
- **Type Safety:** Each pipe component has defined types (MODEL, CLIP, VAE, LATENT, INT, FLOAT, STRING, etc.).
- **Extensibility:** New fields can be added without breaking existing workflows.

#### 2. Workflow Simplification
- **Reduced Connections:** Bundle 10+ parameters into single connections.
- **Cleaner Layouts:** Workflows become more readable and easier to debug.
- **Modular Design:** Components can be mixed and matched across different pipeline types.

#### 3. Data Manipulation Capabilities
- **Pipe Concatenation:** Merge multiple pipes using the Concat Multi node with strategies (overwrite, preserve, merge).
- **Component Extraction:** Extract individual elements (model, CLIP, VAE, latent) from pipes using Pipe Out nodes.
- **Context Building:** Construct pipes from scratch or modify existing ones.

#### 4. Advanced Features
- **Latent Generation:** Automatic latent tensor creation based on dimensions and batch size.
- **Metadata Preservation:** Maintain model names, VAE names, LoRA lists for reference.
- **Error Handling:** Graceful fallbacks and validation for missing or invalid data.
- **Memory Optimization:** Support for different weight dtypes and CLIP trimming.

### Pipe Output Nodes

Specialized nodes extract specific data from pipes:

- **Pipe Out Checkpoint Loader (`Pipe Out Checkpoint Loader [Eclipse]`):** Extracts model, CLIP, VAE, latent, dimensions, batch size, and model/VAE names from checkpoint loader pipes.
- **Pipe Out Smart Folder (`Pipe Out Smart Folder [Eclipse]`):** Extracts smart folder configuration including paths, dimensions, and placeholder data.
- **Pipe Out Sampler Settings (`Pipe Out Sampler Settings [Eclipse]`):** Extracts all sampler and generation parameters (sampler, scheduler, steps, CFG, seed, denoise, etc.).
- **Pipe Out Load Directory Settings (`Pipe Out Load Directory Settings [Eclipse]`):** Extracts directory settings for output path management.
- **Pipe Out Load Image (`Pipe Out Load Image [Eclipse]`):** Extracts image data and associated metadata from image loading pipes.
- **Pipe Out VCNameGen (`Pipe Out VCNameGen [Eclipse]`):** Extracts video/checkpoint name generator configuration.
- **Pipe Out WanVideo Setup (`Pipe Out WanVideo Setup [Eclipse]`):** Extracts WanVideo workflow setup parameters.

### Practical Applications

#### Complex Workflows
Pipes excel in workflows requiring multiple model components, ensemble CLIP setups, or video processing pipelines where managing dozens of individual connections becomes impractical.

#### Batch Processing
When processing multiple images or videos with consistent settings, pipes allow settings to be defined once and reused across batch operations.

#### Modular Pipeline Construction
Build reusable pipeline segments that can be connected together, with pipes handling the data flow between modules.

#### Memory Management
Pipes support efficient memory usage through dtype control and component lazy loading.

### Best Practices

- **Use Dict Pipes:** Prefer dict-style pipes over legacy tuple formats for maximum compatibility.
- **Validate Components:** Use pipe output nodes to ensure all required components are present.
- **Merge Strategically:** When concatenating pipes, choose appropriate merge strategies (merge for combining, overwrite for replacement).
- **Type Consistency:** Ensure pipe components match expected types for downstream nodes.
- **Documentation:** Include pipe metadata (model names, settings) for workflow reproducibility.

The pipe ecosystem transforms ComfyUI workflow construction from a web of individual connections into a streamlined, professional data flow system capable of handling the most complex AI generation pipelines.

to be continued...

---

Enjoy — and happy workflow-building!

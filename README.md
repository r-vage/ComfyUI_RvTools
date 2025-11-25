# ComfyUI_Eclipse

ComfyUI_Eclipse is a collection of custom nodes, helpers and utilities for ComfyUI designed to make workflow building easier and more reliable. It includes convenience nodes for loading checkpoints and pipelines, type conversions, folder and filename helpers, simple image utilities, logic and flow helpers, and small toolkits for working with VAE/CLIP and latents.

Note: Workflows created with RvTools_v2 are NOT compatible with this version. This release contains a substantial cleanup and many improvements.

## Documentation

Detailed documentation for specific features:

- **[Smart Loaders Guide](Readme/Smart_Loaders.md)** - Complete guide to Smart Loader and Smart Loader Plus with multi-format support
- **[Smart Prompt Guide](Readme/Smart_Prompt.md)** - How to use and customize the Smart Prompt system
- **[Wildcard Processor Guide](Readme/Wildcard_Processor.md)** - Advanced wildcard syntax and usage examples
- **[Smart Language Model Loader Guide](Readme/Smart_Language_Model_Loader_Guide.md)** ⭐ NEW - Complete guide for the unified vision-language and LLM loader. Analyze images/videos with QwenVL, generate fast tags with Florence-2, or process text with LLM models. Includes template management, model comparison, and practical use cases.
- **[Model Repository Reference](Readme/Model_Repos_Reference.md)** - Quick reference with HuggingFace repository URLs for all supported models (Qwen, Florence-2) for easy copy-paste when creating templates
- **[Save Images Guide](Readme/Save_Images.md)** - Advanced image saving with metadata, placeholders, and generation data
- **[Checkpoint Loaders Guide](Readme/Checkpoint_Loaders.md)** - Legacy checkpoint loader documentation
- **[Nunchaku Installation](Readme/Nunchaku_Installation.md)** - Step-by-step guide for installing Nunchaku quantization support
- **[User Documentation Index](Readme/README.md)** - Complete index of all user guides

## Highlights

- **Smart Loader Series:** Next-generation model loaders with multi-format support (Standard Checkpoints, UNet, Nunchaku quantized Flux/Qwen, GGUF quantized models), featuring template management, automatic memory cleanup, and graceful extension fallbacks. [→ Documentation](Readme/Smart_Loaders.md)
  - **Smart Loader Plus:** Full-featured loader with latent/sampler configuration, resolution presets, CLIP ensemble (up to 4 modules), and comprehensive quantization support.
  - **Smart Loader:** Streamlined variant for minimal workflows - loads model/CLIP/VAE without latent or sampler configuration.
- **Smart Prompt System:** Quick prompt building with dropdown selectors loaded from organized text files. Pre-configured with subjects, settings, and environments. Users can create custom prompt files by adding numbered `.txt` files (e.g., `1_my_prompts.txt`) - each line becomes a selectable option. Supports folder filtering and random selection with seed control for reproducible prompt generation. Files are automatically copied to `ComfyUI/models/Eclipse/smart_prompt/` on first run, with wildcard integration via junction to `wildcards/smart_prompt/`. [→ Documentation](Readme/Smart_Prompt.md)
- **Wildcard Processor:** Advanced wildcard system for dynamic prompt generation. Create custom wildcard files in the `ComfyUI/models/wildcards/` directory using `.txt` files with one option per line. Supports weighted options (`option:weight` format), nested wildcards, and dynamic seed integration for complex prompt variations. Example wildcards are automatically copied on first launch. [→ Documentation](Readme/Wildcard_Processor.md)
- **Smart Language Model Loader:** Single node for vision-language and text-only AI models. Analyze images with QwenVL (2B-32B parameters, video support, detailed descriptions), generate fast tags with Florence-2 (<1s, SD/Flux optimized), or process text with LLM models (prompt refinement, tags-to-description). Pre-configured templates for all supported models with auto-documented available tasks, auto-downloading from HuggingFace, supports both transformers (quality) and GGUF (low VRAM) formats. Flexible template system: **None mode** (default) for manual configuration, **Load mode** to apply saved templates, **Save mode** to create new templates, **Delete mode** to remove templates. No automatic template loading - start from scratch or use templates as needed. Supports multiple nodes in one workflow using the same template with different customizations (load template → switch to None → adjust per node). Templates auto-update with local paths and actual VRAM requirements after download. Perfect for auto-tagging generations, creating training captions, video analysis, OCR, and object detection. [→ Documentation](Readme/Smart_Language_Model_Loader_Guide.md)
- **Legacy Checkpoint Loaders:** Traditional loaders including Checkpoint Loader Small and Small (Pipe) variants for basic checkpoint loading workflows.
- **Sophisticated Pipe Ecosystem:** Standardized data interchange system with context pipes, generation data pipes, concatenation, and extraction nodes to eliminate spaghetti connections in complex workflows. (More detailed documentation can be found below.)
- **Comprehensive Switching System:** Extensive switch and multi-switch nodes for all ComfyUI data types, enabling dynamic workflow branching and conditional execution.
- **Advanced Text Processing:** Prompt generation with smart prompt systems, multiline string inputs, and regex-based string replacement.
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

### Eclipse Folder Structure (First Launch)

On first launch, ComfyUI_Eclipse automatically creates a folder structure in your ComfyUI models directory for user-editable templates and prompts:

```
ComfyUI/
  models/
    Eclipse/                          # Eclipse user files (edit freely!)
      smart_prompt/                   # Smart Prompt template files
        environment/                  # Environment descriptions
        settings/                     # Style and quality settings
        subjects/                     # Subject categories
        ...
      loader_templates/               # Smart Loader templates (checkpoint configurations)
        Qwen2.5-VL-3B-Instruct.json
        Florence-2-large-ft.json
        ...
      smartlm_templates/              # Smart Language Model Loader templates
        Qwen2.5-VL-3B-Instruct.json
        Florence-2-large-ft.json
        ...
      config/                         # Configuration files (user-editable)
        smartlm_prompt_defaults.json  # QwenVL/Florence-2 task definitions
        llm_few_shot_training.json    # LLM instruction mode examples
        smartlm_advanced_defaults.json # Advanced model parameters
    wildcards/
      smart_prompt/                   # Junction/symlink → Eclipse/smart_prompt/
```

**Important Notes:**
- **Edit files in `models/Eclipse/`** - This is your personal workspace. Changes persist across updates.
- **Git updates won't overwrite** - Files in `models/Eclipse/` are independent from the repository.
- **Wildcard integration** - The `wildcards/smart_prompt/` is a junction (Windows) or symlink (Unix) pointing to `Eclipse/smart_prompt/` for seamless wildcard processor integration.
- **One-time copy** - Templates are copied from the repository to `Eclipse/` only on first run. To get new templates from updates, manually copy from `custom_nodes/ComfyUI_Eclipse/templates/` or delete the Eclipse folder and restart.
- **Automatic migration** - If you're upgrading from a previous version, your existing files from `models/smart_loader_templates/` and `models/wildcards/smartprompt/` will be automatically moved to the new `Eclipse/` structure on first launch. Old folders are removed after successful migration.

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
- **Template System:** Save and load complete configurations including model selections, CLIP/VAE settings, sampler parameters, and sampling methods.
- **CLIP Ensemble:** Support for up to 4 CLIP modules with multiple architecture types (Flux, SD3, SDXL, Qwen, HiDream, Hunyuan, Wan, etc.).
- **Model Sampling:** Advanced sampling method support for different architectures (SD3, AuraFlow, Flux, Stable Cascade, LCM, ContinuousEDM, ContinuousV, LTXV) with automatic parameter management.
- **Advanced Configuration:** 
  - Latent configuration with resolution presets or custom dimensions
  - Sampler settings (sampler, scheduler, steps, CFG, flux_guidance)
  - Model sampling configuration (method-specific parameters, auto-defaults)
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
5. Optionally enable model sampling and select appropriate method (SD3, Flux, etc.) for your model architecture.
6. For Smart Loader Plus: Set resolution, batch size, and sampler settings.
7. Use templates to save/load configurations for quick workflow iteration.
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
- Add Watermark Image - Add watermark to images with positioning and scaling options
- Load Image - Load single image with metadata
- Load Image Path - Load image from custom path
- Load Image Path (Pipe) - Load image from path with pipe output
- Preview Image - Preview images in workflow
- Preview Mask - Preview masks in workflow
- Save Images - Advanced image saving with metadata and placeholders

### Loader
Nodes for loading model checkpoints with support for Standard, UNet, Nunchaku quantized, and GGUF formats, plus vision-language and text-only LLM models.
- Smart Loader - Streamlined loader (model/CLIP/VAE only)
- Smart Loader Plus - Full-featured loader with latent/sampler configuration
- Smart Language Model Loader - Unified vision-language and LLM loader supporting QwenVL (image/video analysis), Florence-2 (fast tagging/OCR), and text-only LLM models with pre-configured templates and auto-download
- Checkpoint Loader Small - Basic checkpoint loader
- Checkpoint Loader Small (Pipe) - Basic checkpoint loader with pipe output

### Primitives (Logic / Basic values)
Small building-block nodes for booleans, numbers, and strings, used in control flow and logic operations.
- Boolean - Boolean value input
- Float - Float value input
- Integer - Integer value input
- Integer (Gen) - Integer with generate_after widget to increment number after each generation
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
- Pipe IO Sampler Settings - Input/output node for sampler settings with pipe passthrough
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
- Sampler Settings (NI+Seed) - Sampler settings with noise injection and seed
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
- Smart Prompt - Dynamic prompt generation with dropdown selectors, seed control, and folder filtering
- Wildcard Processor - Process wildcards in prompts with weighted options, nested wildcards, and seed control

### Video
Nodes for video clip composition, frame utilities, and loop/frame calculations for video-friendly pipelines.
- Loop Calculator - Calculate loop frame counts for video generation
- Keep Calculator - Calculate frame keep/trim values for video processing
- Combine Video Clips - Concatenate video clips
- Seamless Join Video Clips - Join video clips with seamless transitions

### Utilities
General utility nodes for LoRA management, debugging, resource management, and workflow control.
- LoRA Stack - Build LoRA stack configuration
- LoRA Stack Apply - Apply LoRA stack to model/CLIP (supports nunchaku quantized models)
- Show Any - Display any data type for debugging, tensor to image conversion for images and masks
- Stop - Stop workflow execution
- RAM Cleanup - Manual RAM cleanup
- VRAM Cleanup - Manual VRAM cleanup

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

# ComfyUI_Eclipse User Documentation

Welcome to the user documentation for ComfyUI_Eclipse! This guide is designed for artists, creators, and users who want to understand how to use the nodes effectively - not for developers.

## Documentation Index

### Model Loaders

**[Checkpoint Loaders Guide](Checkpoint_Loaders.md)**
- Traditional checkpoint loading
- Simple, reliable model loading
- Understanding CLIP and VAE settings
- Basic troubleshooting

**[Smart Loaders Guide](Smart_Loaders.md)**
- Advanced multi-format loaders
- Template system for quick configuration
- Quantized model support (Nunchaku, GGUF)
- CLIP ensemble configuration
- Memory optimization techniques

### Text Processing

**[Smart Prompt Guide](Smart_Prompt.md)**
- Dropdown-based prompt building
- File-based prompt organization
- Folder filtering and categories
- Seed-controlled random selection
- Creating custom prompt libraries

**[Wildcard Processor Guide](Wildcard_Processor.md)**
- Template-based prompt expansion
- Wildcard syntax and patterns
- Weighted random selection
- Nested wildcards
- Creating wildcard files

### Installation & Setup

**[Nunchaku Installation Guide](Nunchaku_Installation.md)**
- Installing Nunchaku for quantized Flux models
- Step-by-step installation for ComfyUI Portable
- GPU compatibility information
- Troubleshooting dependency issues
- Understanding performance on different GPU architectures

### Getting Started

If you're new to ComfyUI_Eclipse loaders:

1. **Start Here:** [Checkpoint Loaders Guide](Checkpoint_Loaders.md)
   - Learn the basics with traditional loaders
   - Understand core concepts (CLIP, VAE, model files)
   - Get comfortable with basic settings

2. **Level Up:** [Smart Loaders Guide](Smart_Loaders.md)
   - Move to advanced features
   - Learn template management
   - Explore quantized models
   - Optimize for your system

3. **Text Processing:** [Smart Prompt](Smart_Prompt.md) & [Wildcard Processor](Wildcard_Processor.md)
   - Build prompts efficiently
   - Create prompt templates
   - Generate infinite variations
   - Control randomization

4. **Advanced Setup:** [Nunchaku Installation](Nunchaku_Installation.md)
   - Install quantized model support (optional)
   - Reduce VRAM usage significantly
   - Understand GPU compatibility
   - Optimize for your hardware

### Quick Help

**I want to...**

- **Load a basic model** → [Checkpoint Loader Small](Checkpoint_Loaders.md#checkpoint-loader-small)
- **Save/load configurations** → [Template System](Smart_Loaders.md#template-system)
- **Use quantized models** → [Model Types & Formats](Smart_Loaders.md#model-types--formats)
- **Reduce VRAM usage** → [Quantization Configuration](Smart_Loaders.md#quantization-configuration)
- **Build CLIP ensembles** → [CLIP Configuration](Smart_Loaders.md#clip-configuration)
- **Work with pipes** → [Checkpoint Loader Small (Pipe)](Checkpoint_Loaders.md#checkpoint-loader-small-pipe)
- **Build prompts from files** → [Smart Prompt Guide](Smart_Prompt.md)
- **Create prompt templates** → [Wildcard Processor Guide](Wildcard_Processor.md)
- **Install Nunchaku support** → [Nunchaku Installation](Nunchaku_Installation.md)
- **Reduce VRAM with quantization** → [Nunchaku Installation](Nunchaku_Installation.md)

### Common Questions

**Q: Which loader should I use?**

A: Start with **Checkpoint Loader Small** for simplicity. Move to **Smart Loader Plus** when you need:
- Multiple model formats (UNet, Nunchaku, GGUF)
- Template management for quick switching
- Quantized models for VRAM savings
- All-in-one latent and sampler configuration

**Q: What's the difference between Smart Loader and Smart Loader Plus?**

A: **Smart Loader Plus** includes latent and sampler configuration built-in. **Smart Loader** is the streamlined version without those - you use separate Empty Latent and KSampler nodes instead.

**Q: How do I reduce VRAM usage?**

A: Use quantized models (Nunchaku or GGUF) with Smart Loaders. See [Quantization Configuration](Smart_Loaders.md#quantization-configuration) for details.

**Q: What are templates?**

A: Templates save your complete loader configuration (model, CLIP, VAE, sampler, etc.) so you can restore it instantly later. See [Template System](Smart_Loaders.md#template-system).

**Q: My checkpoint won't load, what do I do?**

A: Check the [Troubleshooting](Checkpoint_Loaders.md#troubleshooting) sections in both guides for solutions to common problems.

**Q: How do I build prompts quickly?**

A: Use [Smart Prompt](Smart_Prompt.md) for dropdown-based selection from organized prompt files, or [Wildcard Processor](Wildcard_Processor.md) for template-based generation with infinite variations.

**Q: What's the difference between Smart Prompt and Wildcard Processor?**

A: **Smart Prompt** uses numbered text files to create dropdown menus (select from curated options). **Wildcard Processor** uses template syntax like `{option1|option2}` for dynamic expansion (infinite variations from templates).

**Q: How do I install Nunchaku for quantized models?**

A: Follow the detailed [Nunchaku Installation Guide](Nunchaku_Installation.md). It includes step-by-step commands for ComfyUI Portable, dependency management, and GPU compatibility information.

**Q: What GPU do I need for Nunchaku/quantized models?**

A: RTX 30 and 40 series GPUs work well with the primary benefit being lower VRAM usage. RTX 50 series (Blackwell) will add native FP4 acceleration for additional speed. See [GPU Compatibility](Nunchaku_Installation.md#gpu-compatibility) for details.

### File Locations Reference

| Item | Location |
|------|----------|
| Standard Checkpoints | `ComfyUI/models/checkpoints/` |
| UNet Models | `ComfyUI/models/unet/` |
| Nunchaku Models | `ComfyUI/models/nunchaku/` |
| Qwen Models | `ComfyUI/models/qwen/` |
| GGUF Models | `ComfyUI/models/gguf/` |
| CLIP Files | `ComfyUI/models/clip/` |
| VAE Files | `ComfyUI/models/vae/` |
| Templates | `ComfyUI/custom_nodes/ComfyUI_Eclipse/json/loader_templates/` |
| Smart Prompt Files | `ComfyUI/models/wildcards/smartprompt/` (primary)<br>`ComfyUI_Eclipse/prompt/` (fallback) |
| Wildcard Files | `ComfyUI/models/wildcards/` |

### Required Extensions

Some features require additional extensions:

**For Nunchaku Models (Quantized Flux/Qwen):**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nunchaku-tech/ComfyUI-nunchaku
```

**For GGUF Models:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
```

### Recommended File Formats

| Format | Status | Notes |
|--------|--------|-------|
| `.safetensors` | ✅ Recommended | Safe, fast, modern |
| `.sft` | ✅ Recommended | Safetensors alternative |
| `.ckpt` | ⚠️ Legacy | Works but shows warning |
| `.pt` | ⚠️ Legacy | Works but shows warning |
| `.pth` | ⚠️ Legacy | Works but shows warning |

Always prefer `.safetensors` when available for safety and speed.

### Support & Help

- **Main README:** [../README.md](../README.md) - Overview and feature highlights
- **GitHub Issues:** [Report bugs or request features](https://github.com/r-vage/ComfyUI_Eclipse/issues)
- **License:** GPL-3.0 - See [LICENSE](../LICENSE)

### What's Not Covered Here

This user documentation focuses on model loaders. For other Eclipse features:

- **Pipe System** - See main [README](../README.md#the-pipe-ecosystem-of-eclipse)
- **Text Processing** - See [Smart Prompt](../README.md#node-spotlight-smart-prompt-eclipse) and [Wildcard Processor](../README.md#node-spotlight-wildcard-processor-eclipse)
- **Image Saving** - See [Save Images](../README.md#node-spotlight-save-images-eclipse)
- **Other Nodes** - See [Files by Category](../README.md#files-by-category)

---

## Contributing to Documentation

Found an error or want to improve these guides?

1. Documentation lives in `Readme/` folder
2. Written in Markdown for easy editing
3. Focus on user-friendly language (not developer jargon)
4. Include examples and step-by-step instructions
5. Submit PRs with improvements

---

**Happy creating!** If these guides helped you, consider starring the [repository](https://github.com/r-vage/ComfyUI_Eclipse) ⭐

# Save Images [Eclipse]

The **Save Images** node is a highly advanced and flexible output node designed for robust image saving in ComfyUI workflows, offering extensive customization and metadata support.

## Outstanding Abilities

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

## Placeholder System

The node supports placeholders in both output paths and filename prefixes for dynamic value insertion. Placeholders are replaced with actual values at save time; unknown or empty placeholders default to readable alternatives.

Supported placeholders:
- `%today`, `%date` — Current date (YYYY-MM-DD)
- `%time` — Current time (HHMMSS)
- `%Y`, `%m`, `%d`, `%H`, `%S` — Individual date/time components
- `%basemodel`, `%model` — Model names (base model and full model)
- `%seed`, `%sampler_name`, `%scheduler`, `%steps`, `%cfg`, `%denoise`, `%clip_skip` — Generation parameters

Example: `%today\%basemodel\%seed_%sampler_name_%steps` creates organized folder structures like `2025-09-27\ModelName\12345_euler_20`.

## Connection Possibilities

- **Direct Image Input:** Connect any image output to the `images` input for standard saving scenarios.
- **Pipe Input:** Connect a metadata pipe (from context or logic nodes) to `pipe_opt` to save images and extract metadata from complex workflows automatically.
- **Hybrid Usage:** Combine both inputs for maximum flexibility, allowing images from one source and metadata from another.

## Generation Data Saving

When the `save_generation_data` option is enabled:
- Embeds all generation parameters (prompts, model names, sampler settings, seed, CFG, steps, etc.) into image metadata.
- Optionally saves the full workflow as a separate JSON file alongside each image.
- Extracts and includes short SHA-256 hashes for models, Loras, and embeddings in Civitai-compatible format.
- Supports prompt removal for privacy and Lora token appending for full traceability.

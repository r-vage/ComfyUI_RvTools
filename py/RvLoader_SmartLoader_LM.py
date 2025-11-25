# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RvLoader_SmartLoader_LM - Smart Language Model Loader
#
# Smart node that automatically handles QwenVL, Florence-2, and text-only LLM models
# with template-based configuration and auto-detection.

import nodes
import torch
import gc
from ..core import CATEGORY, cstr
from ..core.smartlm_base import (
    SmartLMLBase,
    MODEL_CONFIGS,
    FLORENCE_TASKS,
    get_template_list,
    load_template,
    detect_model_type,
    ModelType,
    get_llm_model_list,
    get_mmproj_list,
)


class RvLoader_SmartLoader_LM(SmartLMLBase):
    # Smart Language Model Loader - Auto-detects model type (QwenVL, Florence-2, LLM) from template
    
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        templates = get_template_list()
        qwen_prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail."])
        preferred_qwen = "Detailed Description"
        default_qwen_prompt = preferred_qwen if preferred_qwen in qwen_prompts else qwen_prompts[0]
        florence_tasks = list(FLORENCE_TASKS.keys())
        default_florence_task = "detailed_caption"
        llm_models = get_llm_model_list()
        mmproj_files = get_mmproj_list()
        
        return {
            "required": {
                "template_action": (["None", "Load", "Save", "Delete"], {"default": "None", "tooltip": "None: Manual configuration | Load: Use template | Save: Save as template | Delete: Remove template"}),
                "template_name": (templates, {"default": templates[0] if templates else "None", "tooltip": "Select template to load or delete"}),
                "new_template_name": ("STRING", {"default": "", "tooltip": "Save mode: Name for new template"}),
                "model_type": (["QwenVL", "QwenVL (GGUF)", "Florence2", "Florence2 (GGUF)", "LLM", "LLM (GGUF)"], {"default": "QwenVL", "tooltip": "Model type and format: QwenVL/Florence2/LLM (Transformers) or with (GGUF) suffix for llama.cpp models"}),
                "model_source": (["HuggingFace", "Local"], {"default": "HuggingFace", "tooltip": "Model source - HuggingFace (download) or Local (models/LLM folder)"}),
                "repo_id": ("STRING", {"default": "", "multiline": False, "tooltip": "HuggingFace: Repo ID or direct download URL"}),
                "local_model": (llm_models, {"default": llm_models[0] if llm_models else "None", "tooltip": "Local: Select model from models/LLM folder"}),
                "local_path": ("STRING", {"default": "", "tooltip": "HuggingFace: Local filename after download (leave empty to use repo filename)"}),
                "mmproj_source": (["HuggingFace", "Local"], {"default": "Local", "tooltip": "GGUF vision only: mmproj source"}),
                "mmproj_url": ("STRING", {"default": "", "multiline": True, "tooltip": "HuggingFace: mmproj download URL"}),
                "mmproj_local": (mmproj_files, {"default": mmproj_files[0] if mmproj_files else "None", "tooltip": "Local: Select mmproj from models/LLM folder"}),
                "mmproj_path": ("STRING", {"default": "", "tooltip": "HuggingFace: mmproj filename after download"}),
                "quantization": (["auto", "4bit", "8bit", "fp16", "bf16", "fp32"], {"default": "auto", "tooltip": "Model precision/quantization: auto (best for available memory), FP32 (full), BF16 (balanced), FP16 (standard), 8-bit/4-bit (lower VRAM, reduced quality)"}),
                "attention_mode": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto", "tooltip": "Attention implementation: auto (best available), flash_attention_2 (fastest), sdpa (balanced), eager (fallback)"}),
                "context_size": ("INT", {"default": 32768, "min": 2048, "max": 131072, "step": 1024, "tooltip": "Context window size. Transformers: auto | GGUF: Controls total input+output token capacity. Text-only LLM: 2048-4096, Vision models: 16384-32768 for images, 32768-65536 for videos."}),
                "qwen_preset_prompt": (qwen_prompts, {"default": default_qwen_prompt, "tooltip": "QwenVL: Pre-configured prompt templates for common tasks. Select 'Custom' to use only your custom prompt"}),
                "qwen_custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "QwenVL: For 'Custom' preset: main instruction. For other presets: additional hints/context appended to system prompt"}),
                "florence_task": (florence_tasks, {"default": default_florence_task, "tooltip": "Florence-2: Select task type (caption, detailed_caption, prompt generation, grounding, detection, OCR, etc.)"}),
                "convert_to_bboxes": ("BOOLEAN", {"default": False, "tooltip": "Florence-2: Convert quad_boxes (OCR) and polygons (segmentation) to normalized bboxes for standard workflows. Disable to preserve original formats."}),
                "florence_text_input": ("STRING", {"default": "", "tooltip": "Florence-2: Text input for detection tasks (e.g., 'face' for caption_to_phrase_grounding). Leave empty for caption tasks."}),
                "llm_instruction_mode": (["Tags to Natural Language", "Expand Description", "Refine Prompt", "Custom Instruction", "Direct Chat"], {"default": "Tags to Natural Language", "tooltip": "LLM: Select instruction template mode"}),
                "llm_custom_instruction": ("STRING", {"default": 'Generate a detailed prompt from "{prompt}"', "multiline": True, "tooltip": 'LLM: Custom instruction template (only used when mode is Custom Instruction). Use {prompt} as placeholder.'}),
                "llm_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "LLM: Your input text/tags/description to process (used when text input is not connected)"}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": "Maximum number of tokens to generate in the output. Larger values allow longer responses but take more time. QwenVL: 512-1024, Florence-2: 512-1024, LLM: 512-2048"}),
                "memory_cleanup": ("BOOLEAN", {"default": True, "tooltip": "Clear unused memory cache before loading LM model (safe, won't unload generation models). Disable if LM model is already loaded for faster reload."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep model in VRAM between generations (faster, uses more VRAM)"}),
                "seed": ("INT", {"default": 0, "min": -3, "max": 2**64 - 1, "control_after_generate": True, "tooltip": "Random seed for reproducible results."}),
            },
            "optional": {
                "images": ("IMAGE",),
                "text": ("STRING", {"forceInput": True, "tooltip": "Text input for all models: QwenVL custom prompt, Florence-2 phrase grounding text, or LLM prompt. Takes priority over widget fields."}),
                "pipe_opt": ("SMARTLM_ADVANCED_PIPE", {"tooltip": "Optional advanced parameters pipe from LM Advanced Options Pipe node"}),
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.LOADER.value
    RETURN_TYPES = ("IMAGE", "STRING", "JSON")
    RETURN_NAMES = ("image", "text", "data")
    FUNCTION = "execute"

    def execute(
        self,
        template_action,
        template_name,
        new_template_name,
        model_type,
        model_source,
        repo_id,
        local_model,
        local_path,
        mmproj_source,
        mmproj_url,
        mmproj_local,
        mmproj_path,
        quantization,
        attention_mode,
        context_size,
        qwen_preset_prompt,
        qwen_custom_prompt,
        florence_task,
        convert_to_bboxes,
        florence_text_input,
        llm_instruction_mode,
        llm_custom_instruction,
        llm_prompt,
        max_tokens,
        memory_cleanup,
        keep_model_loaded,
        seed,
        images=None,
        text=None,
        pipe_opt=None,
    ):
        import time
        import os
        import json
        from pathlib import Path
        start_time = time.time()
        
        # Extract base model type and format from combined model_type
        is_gguf = "(GGUF)" in model_type
        base_model_type = model_type.replace(" (GGUF)", "").strip()
        internal_model_type = base_model_type.lower()
        
        # Handle template actions (Save/Delete)
        if template_action == "Save" and new_template_name and new_template_name.strip():
            # Save current configuration as template
            from ..core.smartlm_base import TEMPLATE_DIR
            template_path = TEMPLATE_DIR / f"{new_template_name.strip()}.json"
            
            # Determine repo_id and local_path based on source
            if model_source == "Local":
                # Local model: use selected model as local_path, repo_id empty
                final_repo_id = ""
                # Ensure trailing slash for directory consistency
                final_local_path = local_model.rstrip('/') + '/' if local_model and not local_model.endswith('.gguf') else local_model
            else:
                # HuggingFace: use repo_id and optional local_path
                final_repo_id = repo_id.strip()
                # Normalize local_path - add trailing slash for directories
                final_local_path = local_path.strip()
                if final_local_path and not final_local_path.endswith('.gguf'):
                    final_local_path = final_local_path.rstrip('/') + '/'
            
            # Determine mmproj_path and mmproj_url based on source (for GGUF vision models)
            final_mmproj_path = ""
            final_mmproj_url = ""
            if is_gguf and internal_model_type == "qwenvl":
                if mmproj_source == "Local":
                    final_mmproj_path = mmproj_local
                else:
                    final_mmproj_url = mmproj_url.strip()
                    final_mmproj_path = mmproj_path.strip()
            
            # Auto-detect quantized status from model path/name
            model_name_lower = (final_local_path or final_repo_id or "").lower()
            has_quant_markers = any(marker in model_name_lower for marker in [
                "fp8", "int8", "int4", "q4", "q5", "q6", "q8", "_q4_", "_q5_", "_q8_",
                "gptq", "awq", "gguf"
            ])
            auto_quantized = has_quant_markers or is_gguf
            
            # Auto-detect model size - try file size first for local models, then fallback to name parsing
            vram_full = 3.0  # Default estimate
            if model_source == "Local" and final_local_path:
                try:
                    import folder_paths
                    llm_dir = Path(folder_paths.models_dir) / "LLM"
                    model_path = llm_dir / final_local_path
                    
                    total_size_gb = 0.0
                    if model_path.is_file():
                        # Single file (GGUF, safetensors, etc.)
                        total_size_gb = model_path.stat().st_size / (1024**3)
                    elif model_path.is_dir():
                        # Model folder - sum all model files (safetensors, bin, pt, etc.)
                        model_extensions = ['.safetensors', '.bin', '.pt', '.gguf']
                        for file in model_path.rglob('*'):
                            if file.is_file() and file.suffix in model_extensions:
                                total_size_gb += file.stat().st_size / (1024**3)
                    
                    if total_size_gb > 0.1:  # Only use if we found significant files (>100MB)
                        vram_full = round(total_size_gb, 1)
                except Exception as e:
                    cstr(f"[SmartLM] Could not calculate model size from files, using name-based estimate: {e}").warning.print()
            
            # Fallback: estimate from model name if file size calculation failed or not local
            if vram_full == 3.0:  # Still default, use name-based detection
                if "2b" in model_name_lower or "2.5b" in model_name_lower:
                    vram_full = 4.0
                elif "3b" in model_name_lower:
                    vram_full = 6.0
                elif "4b" in model_name_lower:
                    vram_full = 8.0
                elif "7b" in model_name_lower or "8b" in model_name_lower:
                    vram_full = 14.0
                elif "32b" in model_name_lower:
                    vram_full = 64.0
                elif "florence-2-base" in model_name_lower or "florence2-base" in model_name_lower:
                    vram_full = 0.5  # ~230M params
                elif "florence-2-large" in model_name_lower or "florence2-large" in model_name_lower:
                    vram_full = 1.5  # ~770M params
            
            # Build template configuration from current widgets
            new_config = {
                "repo_id": final_repo_id,
                "local_path": final_local_path,
                "mmproj_path": final_mmproj_path,
                "model_type": internal_model_type,
                "default_task": "",  # Will be set based on model type
                "default_text_input": "",
                "max_tokens": max_tokens,
                "quantization": quantization,
                "attention_mode": attention_mode,
                "quantized": auto_quantized,
                "vram_requirement": {
                    "full": vram_full
                }
            }
            
            # Add mmproj_url if provided (GGUF vision models)
            if final_mmproj_url:
                new_config["mmproj_url"] = final_mmproj_url
            
            # Auto-calculate 8bit/4bit VRAM from model size (for non-quantized models)
            # Skip for pre-quantized models (GGUF, GPTQ, AWQ, etc.)
            if not auto_quantized:
                # 8-bit: ~50% of full precision size
                new_config["vram_requirement"]["8bit"] = round(vram_full * 0.5, 1)
                # 4-bit: ~25% of full precision size
                new_config["vram_requirement"]["4bit"] = round(vram_full * 0.25, 1)
            
            # Add context_size for GGUF models
            if is_gguf:
                new_config["context_size"] = context_size
            
            # Set default_task based on model type
            if internal_model_type == "qwenvl":
                new_config["default_task"] = qwen_preset_prompt
                new_config["default_text_input"] = qwen_custom_prompt
            elif internal_model_type == "florence2":
                new_config["default_task"] = florence_task
                new_config["default_text_input"] = florence_text_input
            elif internal_model_type == "llm":
                new_config["default_task"] = llm_instruction_mode
                new_config["default_text_input"] = llm_custom_instruction
            
            # Add _available_tasks at the end as a user hint (documentation field)
            if internal_model_type == "qwenvl":
                new_config["_available_tasks"] = MODEL_CONFIGS.get("_preset_prompts", [])
            elif internal_model_type == "florence2":
                new_config["_available_tasks"] = ", ".join(FLORENCE_TASKS.keys())
            elif internal_model_type == "llm":
                new_config["_available_tasks"] = [
                    "Tags to Natural Language",
                    "Expand Description",
                    "Refine Prompt",
                    "Custom Instruction",
                    "Direct Chat"
                ]
            
            # Save template
            try:
                os.makedirs(TEMPLATE_DIR, exist_ok=True)
                with open(template_path, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, indent=2)
                cstr(f"[SmartLM] ✓ Created new template: {new_template_name.strip()}").msg.print()
            except Exception as e:
                cstr(f"[SmartLM] Failed to create template: {e}").error.print()
            
            # Stop execution - template created, no model loading needed
            empty_image = torch.zeros((1, 64, 64, 3))
            nodes.interrupt_processing()
            return (empty_image, "", {})
        
        elif template_action == "Delete" and template_name and template_name != "None":
            # Delete selected template
            from ..core.smartlm_base import TEMPLATE_DIR
            template_path = TEMPLATE_DIR / f"{template_name}.json"
            
            try:
                if template_path.exists():
                    os.remove(template_path)
                    cstr(f"[SmartLM] ✓ Deleted template: {template_name}").msg.print()
                else:
                    cstr(f"[SmartLM] Template not found: {template_name}").warning.print()
            except Exception as e:
                cstr(f"[SmartLM] Failed to delete template: {e}").error.print()
            
            # Stop execution - template deleted, no model loading needed
            empty_image = torch.zeros((1, 64, 64, 3))
            nodes.interrupt_processing()
            return (empty_image, "", {})
        
        # Note: Template auto-save is now handled in JavaScript after execution
        # This allows JS to only save visible widgets based on detected model type
        
        # Extract advanced parameters from pipe if provided
        device = "cuda"
        use_torch_compile = False
        temperature = 0.7
        top_p = 0.9
        num_beams = 3
        do_sample = True
        repetition_penalty = 1.0
        frame_count = 8
        
        if pipe_opt is not None:
            device = pipe_opt.get("device", device)
            use_torch_compile = pipe_opt.get("use_torch_compile", use_torch_compile)
            temperature = pipe_opt.get("temperature", temperature)
            top_p = pipe_opt.get("top_p", top_p)
            top_k = pipe_opt.get("top_k", 50)
            num_beams = pipe_opt.get("num_beams", num_beams)
            do_sample = pipe_opt.get("do_sample", do_sample)
            repetition_penalty = pipe_opt.get("repetition_penalty", repetition_penalty)
            frame_count = pipe_opt.get("frame_count", frame_count)
        else:
            top_k = 50
        
        # Determine configuration source: Load template or use direct configuration (None mode)
        if template_action == "Load" and template_name and template_name != "None":
            # Load mode: Use template configuration
            template_info = load_template(template_name)
            detected_model_type = detect_model_type(template_info)
            
            # Check if model is GGUF
            is_gguf = False
            if template_info:
                final_local_path = template_info.get("local_path", "")
                final_repo_id = template_info.get("repo_id", "")
                is_gguf = final_local_path.lower().endswith(".gguf") or final_repo_id.lower().endswith(".gguf") or "gguf" in final_repo_id.lower()
            
            # Load model from template
            if self.current_template != template_name or self.model is None:
                if is_gguf:
                    cstr(f"[SmartLM] Loading template '{template_name}' (type: {detected_model_type.value}, max_tokens: {max_tokens}, context: {context_size})").msg.print()
                else:
                    cstr(f"[SmartLM] Loading template '{template_name}' (type: {detected_model_type.value}, max_tokens: {max_tokens})").msg.print()
                self.load_model(template_name, quantization, attention_mode, device=device, context_size=context_size, 
                              memory_cleanup=memory_cleanup, use_torch_compile=use_torch_compile)
        else:
            # None mode: Use direct configuration from widgets
            # Determine repo_id and local_path
            if model_source == "Local":
                final_repo_id = ""
                final_local_path = local_model
            else:
                final_repo_id = repo_id.strip()
                final_local_path = local_path.strip()
            
            # Determine mmproj
            final_mmproj_path = ""
            final_mmproj_url = ""
            if is_gguf and internal_model_type == "qwenvl":
                if mmproj_source == "Local":
                    final_mmproj_path = mmproj_local
                else:
                    final_mmproj_url = mmproj_url.strip()
                    final_mmproj_path = mmproj_path.strip()
            
            # Auto-detect quantized status and VRAM requirements
            model_name_lower = (final_local_path or final_repo_id or "").lower()
            has_quant_markers = any(marker in model_name_lower for marker in [
                "fp8", "int8", "int4", "q4", "q5", "q6", "q8", "_q4_", "_q5_", "_q8_",
                "gptq", "awq", "gguf"
            ])
            auto_quantized = has_quant_markers or is_gguf
            
            # Auto-detect model size - try file size first for local models, then fallback to name parsing
            vram_full = 3.0  # Default
            if model_source == "Local" and final_local_path:
                try:
                    import folder_paths
                    llm_dir = Path(folder_paths.models_dir) / "LLM"
                    model_path = llm_dir / final_local_path
                    
                    total_size_gb = 0.0
                    if model_path.is_file():
                        # Single file (GGUF, safetensors, etc.)
                        total_size_gb = model_path.stat().st_size / (1024**3)
                    elif model_path.is_dir():
                        # Model folder - check for sharded models first, then single files
                        # Priority: .safetensors (preferred) > .bin > .pt
                        all_files = list(model_path.rglob('*'))
                        model_files = [f for f in all_files if f.is_file()]
                        
                        # Check for shard files (e.g., model-00001-of-00005.safetensors)
                        safetensors_files = [f for f in model_files if f.suffix == '.safetensors']
                        bin_files = [f for f in model_files if f.suffix == '.bin']
                        pt_files = [f for f in model_files if f.suffix == '.pt']
                        gguf_files = [f for f in model_files if f.suffix == '.gguf']
                        
                        # Check if we have shards (files with -of- pattern)
                        has_shards = lambda files: any('-of-' in f.name for f in files)
                        
                        # Priority: safetensors shards > single safetensors > bin shards > single bin > pt > gguf
                        if has_shards(safetensors_files):
                            # Use safetensors shards
                            for file in safetensors_files:
                                if '-of-' in file.name:
                                    total_size_gb += file.stat().st_size / (1024**3)
                        elif safetensors_files:
                            # Single safetensors file (no shards)
                            for file in safetensors_files:
                                total_size_gb += file.stat().st_size / (1024**3)
                        elif has_shards(bin_files):
                            # Use bin shards
                            for file in bin_files:
                                if '-of-' in file.name:
                                    total_size_gb += file.stat().st_size / (1024**3)
                        elif bin_files:
                            # Single bin file
                            for file in bin_files:
                                total_size_gb += file.stat().st_size / (1024**3)
                        elif pt_files:
                            # PT files
                            for file in pt_files:
                                total_size_gb += file.stat().st_size / (1024**3)
                        elif gguf_files:
                            # GGUF files
                            for file in gguf_files:
                                total_size_gb += file.stat().st_size / (1024**3)
                    
                    if total_size_gb > 0.1:  # Only use if we found significant files (>100MB)
                        vram_full = round(total_size_gb, 1)
                except Exception as e:
                    cstr(f"[SmartLM] Could not calculate model size from files, using name-based estimate: {e}").warning.print()
            
            # Fallback: estimate from model name if file size calculation failed or not local
            if vram_full == 3.0:  # Still default, use name-based detection
                if "2b" in model_name_lower or "2.5b" in model_name_lower:
                    vram_full = 4.0
                elif "3b" in model_name_lower:
                    vram_full = 6.0
                elif "4b" in model_name_lower:
                    vram_full = 8.0
                elif "7b" in model_name_lower or "8b" in model_name_lower:
                    vram_full = 14.0
                elif "32b" in model_name_lower:
                    vram_full = 64.0
                elif "florence" in model_name_lower:
                    vram_full = 0.5 if "base" in model_name_lower else 1.5
            
            # Build temporary template_info for None mode
            vram_req = {"full": vram_full}
            # Auto-calculate 8bit/4bit VRAM for non-quantized models
            if not auto_quantized:
                vram_req["8bit"] = round(vram_full * 0.5, 1)
                vram_req["4bit"] = round(vram_full * 0.25, 1)
            
            template_info = {
                "repo_id": final_repo_id,
                "local_path": final_local_path,
                "mmproj_path": final_mmproj_path,
                "model_type": internal_model_type,
                "quantization": quantization,
                "attention_mode": attention_mode,
                "quantized": auto_quantized,
                "vram_requirement": vram_req
            }
            
            if final_mmproj_url:
                template_info["mmproj_url"] = final_mmproj_url
            
            if is_gguf:
                template_info["context_size"] = context_size
            
            detected_model_type = detect_model_type(template_info)
            
            # Load model from direct configuration (use None as cache key for manual config)
            config_key = f"manual_{internal_model_type}_{final_repo_id or final_local_path}"
            if self.current_template != config_key or self.model is None:
                if is_gguf:
                    cstr(f"[SmartLM] Loading model (type: {detected_model_type.value}, format: GGUF, max_tokens: {max_tokens}, context: {context_size})").msg.print()
                else:
                    cstr(f"[SmartLM] Loading model (type: {detected_model_type.value}, format: Transformers, max_tokens: {max_tokens})").msg.print()
                
                # Use direct loading method (create temporary template-like structure)
                self.current_template = config_key
                self.template_info = template_info
                
                # Call load_model_direct or similar - for now use existing load_model by creating temp template
                from ..core.smartlm_base import TEMPLATE_DIR
                temp_template_name = "__temp_manual_config__"
                temp_template_path = TEMPLATE_DIR / f"{temp_template_name}.json"
                
                try:
                    os.makedirs(TEMPLATE_DIR, exist_ok=True)
                    with open(temp_template_path, 'w', encoding='utf-8') as f:
                        json.dump(template_info, f, indent=2)
                    
                    self.load_model(temp_template_name, quantization, attention_mode, device=device, context_size=context_size,
                                  memory_cleanup=memory_cleanup, use_torch_compile=use_torch_compile)
                    
                    # Clean up temp template
                    if temp_template_path.exists():
                        os.remove(temp_template_path)
                except Exception as e:
                    cstr(f"[SmartLM] Failed to load model from manual configuration: {e}").error.print()
                    raise
        
        # Prepare prompt based on model type
        if detected_model_type == ModelType.QWENVL:
            # Use QwenVL prompts
            from ..core.smartlm_base import SYSTEM_PROMPTS
            
            # Use text input if connected (takes priority)
            if text is not None:
                prompt_text = text
            # Handle Custom preset: use custom prompt as main instruction
            elif qwen_preset_prompt == "Custom":
                prompt_text = qwen_custom_prompt.strip() if qwen_custom_prompt.strip() else "Describe this image in detail."
            else:
                # Build prompt: system instruction + preset name + optional custom hints
                system_prompt = SYSTEM_PROMPTS.get(qwen_preset_prompt, "")
                prompt_text = f"{system_prompt}\n\n{qwen_preset_prompt}" if system_prompt else qwen_preset_prompt
                
                # Append custom prompt as additional hints/context
                if qwen_custom_prompt.strip():
                    prompt_text = f"{prompt_text}\n\nAdditional context: {qwen_custom_prompt.strip()}"
            
            result, data = self.generate(
                image=images,
                prompt=prompt_text,
                max_tokens=max_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                seed=seed,
                repetition_penalty=repetition_penalty,
                frame_count=frame_count,
            )
            
        elif detected_model_type == ModelType.FLORENCE2:
            # Florence-2 doesn't support video (multiple frames)
            input_image = images
            if images is not None and images.shape[0] > 1:
                cstr("[SmartLM] Warning - Florence-2 doesn't support video, using first frame").warning.print()
                input_image = images[0:1]
            
            # Use text input if connected, otherwise use florence_text_input widget
            florence_text = text if text is not None else florence_text_input
            
            result, data = self.generate(
                image=input_image,
                prompt="",  # Not used
                task=florence_task,
                text_input=florence_text,
                max_tokens=max_tokens,
                num_beams=num_beams,
                top_k=top_k,
                do_sample=do_sample,
                seed=seed,
                repetition_penalty=repetition_penalty,
                convert_to_bboxes=convert_to_bboxes,
            )
        
        elif detected_model_type == ModelType.LLM:
            # Text-only LLM: Use instruction mode to determine template and behavior
            # Map display names to config keys
            mode_map = {
                "Tags to Natural Language": "tags_to_natural_language",
                "Expand Description": "expand_description",
                "Refine Prompt": "refine_prompt",
                "Custom Instruction": "custom_instruction",
                "Direct Chat": "direct_chat"
            }
            
            mode_key = mode_map.get(llm_instruction_mode, "direct_chat")
            
            # For Custom Instruction, use user-provided template, otherwise pass mode key
            if llm_instruction_mode == "Custom Instruction":
                instruction_template = llm_custom_instruction
            else:
                instruction_template = ""  # Will use template from config
            
            # Use text input if connected, otherwise use llm_prompt widget
            prompt_text = text if text is not None else llm_prompt
            
            result, data = self.generate(
                image=None,  # Text-only
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                repetition_penalty=repetition_penalty,
                llm_mode=mode_key,
                instruction_template=instruction_template,
            )
        
        else:
            raise ValueError(f"Unknown model type for template '{template_name}'")
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        char_count = len(result)
        cstr(f"[SmartLM] Generated {char_count} characters in {elapsed:.2f}s").msg.print()
        
        # Generate visualization image if Florence-2 detection task
        output_image = None
        
        if detected_model_type == ModelType.FLORENCE2 and data and ("bboxes" in data or "quad_boxes" in data or "polygons" in data):
            try:
                # Draw visualization with bboxes
                output_image = self._draw_bboxes(input_image if detected_model_type == ModelType.FLORENCE2 else images, data)
            except Exception as e:
                cstr(f"[SmartLM] Could not draw bounding boxes: {e}").warning.print()
                # Return original image if drawing fails
                output_image = input_image if detected_model_type == ModelType.FLORENCE2 else images
        else:
            # No detections - return original image or blank
            if images is not None:
                output_image = images
            else:
                # Create blank image tensor
                output_image = torch.zeros((1, 64, 64, 3))
        
        # Cleanup if requested OR if GGUF model (GGUF models must be unloaded to free VRAM)
        # llama-cpp-python's vision encoder (mtmd context) holds VRAM that can only be freed by unloading
        is_gguf_model = hasattr(self, 'is_gguf') and self.is_gguf
        should_cleanup = not keep_model_loaded or is_gguf_model
        
        if should_cleanup:
            if is_gguf_model and keep_model_loaded:
                cstr("[SmartLM] Note: GGUF models must be unloaded after each run to prevent VRAM accumulation").msg.print()
            
            self.clear()
            
            # Extra CUDA cleanup for GGUF
            if is_gguf_model:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        return (output_image, result, data)

NODE_NAME = 'Smart Language Model Loader [Eclipse]'
NODE_DESC = 'Smart Language Model Loader'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvLoader_SmartLoader_LM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
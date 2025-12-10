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

# QwenVL Model Wrapper - Handles both Transformers and GGUF formats
#
# Supports:
# - Qwen3-VL (requires transformers >= 4.57.1)
# - Qwen2.5-VL
# - Video and image multimodal analysis
# - GGUF format with llama-cpp-python (v0.3.10+)

import torch
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from . import cstr


def load_qwenvl(smart_lm_instance, template_name: str, quantization: str, attention: str, 
                device: str, context_size: int = 32768, use_torch_compile: bool = False):
    """Load QwenVL model (supports both transformers and GGUF)"""
    from .smartlm_base import load_template, LLAMA_CPP_AVAILABLE
    from .smartlm_files import verify_model_integrity
    
    template_info = load_template(template_name)
    
    # Ensure model path exists first to get actual file path
    model_path, model_folder, repo_id = smart_lm_instance.ensure_model_path(template_name)
    
    # Check if this is a GGUF model (check actual model_path, not just local_path from template)
    model_path_lower = model_path.lower()
    has_gguf_ext = model_path_lower.endswith(".gguf")
    
    # Check for GGUF quantization markers (Q4_K_M, Q5_K_S, Q8_0, etc.)
    gguf_quant_markers = ["_q4_", "_q5_", "_q6_", "_q8_", "-q4-", "-q5-", "-q6-", "-q8-",
                          "_k_m", "_k_s", "_k_l", "q4_k", "q5_k", "q6_k", "q8_0", ".q4_", ".q5_", ".q6_", ".q8_"]
    has_gguf_quant = any(marker in model_path_lower for marker in gguf_quant_markers)
    
    is_gguf = has_gguf_ext or has_gguf_quant
    
    if is_gguf:
        # Load GGUF model with llama-cpp-python
        load_qwenvl_gguf(smart_lm_instance, template_name, template_info, device, context_size)
        return
    
    # For transformers models, model_path already retrieved above
    
    # Verify model integrity if loading from local cache
    if not verify_model_integrity(Path(model_path), repo_id):
        raise RuntimeError(f"Model integrity check failed for {model_path}. The model may be corrupted. Please delete and re-download.")
    
    # Load transformers model
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
    from transformers import BitsAndBytesConfig
    
    # Auto-detect if model is pre-quantized (FP8 models usually have it in name/folder)
    local_path = template_info.get("local_path", "")
    has_quant_markers = any(marker in local_path.lower() or marker in repo_id.lower() or marker in model_path.lower()
                            for marker in ["fp8", "int8", "int4", "q4", "q5", "q6", "q8", "_q4_", "_q5_", "_q8_"])
    
    # Auto-detect, but allow template override (GGUF already handled above)
    is_prequantized = template_info.get("quantized", has_quant_markers)
    
    # Determine quantization config and dtype (matching qwenvl_base logic)
    if is_prequantized:
        # Pre-quantized models (FP8) - no additional quantization
        quant_config = None
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    elif quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        dtype = None  # BitsAndBytes handles dtype
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = None  # BitsAndBytes handles dtype
    else:
        # Non-quantized mode: fp16, bf16, or fp32
        quant_config = None
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map.get(quantization, torch.float16)
        # Fallback to fp32 on CPU if not fp32 already
        if device == "cpu" and dtype != torch.float32:
            dtype = torch.float32
    
    # Resolve attention mode (QwenVL doesn't support "auto")
    if attention == "auto" or attention == "flash_attention_2":
        # Try flash_attention_2 if available
        try:
            import flash_attn
            attention = "flash_attention_2"
        except ImportError:
            attention = "sdpa"
            if attention == "flash_attention_2":
                cstr("[SmartLM] Flash-Attn requested but unavailable, falling back to SDPA").warning.print()
    elif attention == "eager":
        attention = "eager"
    else:
        attention = "sdpa"
    
    # Build load_kwargs based on model type
    load_kwargs = {
        "dtype": dtype,
        "attn_implementation": attention,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,  # Required to avoid "Cannot copy out of meta tensor" errors
    }
    
    # HYBRID APPROACH: quantized vs non-quantized
    if is_prequantized:
        # Pre-quantized models: load without device_map to avoid dispatch issues
        load_kwargs["device_map"] = None
        cstr(f"[SmartLM] Loading {template_name} ({quantization}, attn={attention}) [pre-quantized]").msg.print()
        
    elif quant_config:
        # Quantized (4bit/8bit): MUST use device_map="auto" for BitsAndBytes
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
        cstr(f"[SmartLM] Loading {template_name} ({quantization}, attn={attention}) [quantized, device_map=auto]").msg.print()
        
    else:
        # Non-quantized (fp16/bf16): Use ComfyUI offload for better integration
        load_kwargs["device_map"] = None
        cstr(f"[SmartLM] Loading {template_name} ({quantization}, attn={attention}) [non-quantized, ComfyUI offload]").msg.print()
    
    try:
        smart_lm_instance.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
    except ValueError as e:
        error_str = str(e)
        
        # Check for configuration class mismatch (wrong model type selected)
        if "Unrecognized configuration class" in error_str and "Florence2Config" in error_str:
            cstr(f"[SmartLM] ✗ Model type mismatch detected").error.print()
            cstr(f"[SmartLM] You selected 'QwenVL' model type but the model at {model_path} is a Florence-2 model").error.print()
            cstr(f"[SmartLM] Please change 'model_type' to 'Florence2' or select a different model").error.print()
            raise RuntimeError(f"Model type mismatch: Selected QwenVL but model is Florence-2. Please change model_type to 'Florence2'.") from None
        
        # Check for other configuration mismatches
        if "Unrecognized configuration class" in error_str:
            # Try to extract actual model type from error
            if "Qwen2_5_VLConfig" in error_str or "Qwen2VLConfig" in error_str:
                actual_type = "QwenVL"
            elif "Florence2Config" in error_str:
                actual_type = "Florence2"
            else:
                actual_type = "unknown"
            
            cstr(f"[SmartLM] ✗ Model type mismatch detected").error.print()
            cstr(f"[SmartLM] The selected model type doesn't match the actual model architecture (detected: {actual_type})").error.print()
            cstr(f"[SmartLM] Please select the correct 'model_type' for your model").error.print()
            raise RuntimeError(f"Model type mismatch. Detected model type: {actual_type}. Please change model_type accordingly.") from None
        
        # Check for architecture version issues
        if "does not recognize this architecture" in error_str or "model type" in error_str:
            model_type = "unknown"
            if "model type `" in error_str:
                # Extract model type from error message
                import re
                match = re.search(r"model type `([^`]+)`", str(e))
                if match:
                    model_type = match.group(1)
            
            cstr(f"[SmartLM] ✗ Model architecture '{model_type}' not supported by installed transformers version").error.print()
            cstr(f"[SmartLM] This model is too new for your transformers library").error.print()
            cstr(f"[SmartLM] Update with: pip install --upgrade transformers").error.print()
            raise RuntimeError(f"Unsupported model architecture '{model_type}'. Please update transformers library.") from None
        else:
            raise
    
    # Post-loading device placement
    if is_prequantized and device == "cuda" and torch.cuda.is_available():
        # Pre-quantized: manually move to device
        smart_lm_instance.model = smart_lm_instance.model.to("cuda")
    elif not quant_config:
        # Non-quantized: use ComfyUI offload device
        try:
            import comfy.model_management as mm
            offload_device = mm.unet_offload_device()
            smart_lm_instance.model = smart_lm_instance.model.to(offload_device)
        except:
            pass
    
    # Track quantization state for generation
    # Pre-quantized models are quantized, even though quant_config is None
    smart_lm_instance.is_quantized = is_prequantized or bool(quant_config)
    
    # Apply torch.compile if requested (requires CUDA and Torch 2.1+)
    if use_torch_compile and torch.cuda.is_available():
        try:
            smart_lm_instance.model = torch.compile(smart_lm_instance.model, mode="reduce-overhead")
            cstr("[SmartLM] ✓ torch.compile enabled for QwenVL").msg.print()
        except Exception as e:
            cstr(f"[SmartLM] torch.compile skipped: {e}").warning.print()
    
    # Enable KV cache for better performance
    smart_lm_instance.model.config.use_cache = True
    if hasattr(smart_lm_instance.model, "generation_config"):
        smart_lm_instance.model.generation_config.use_cache = True
    
    smart_lm_instance.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    smart_lm_instance.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Mark as non-GGUF model
    smart_lm_instance.is_gguf = False


def load_qwenvl_gguf(smart_lm_instance, template_name: str, template_info: dict, device: str, context_size: int = 32768):
    """Load QwenVL GGUF model with llama-cpp-python"""
    from .smartlm_base import LLAMA_CPP_AVAILABLE
    from .smartlm_files import verify_model_integrity, extract_repo_id_from_url
    
    # Check if llama-cpp-python is available (detected at startup)
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is required for GGUF models but was not found at startup. "
            "Install with: pip install llama-cpp-python\n"
            "Then restart ComfyUI."
        )
    
    # Get model path first to detect model type from actual filename
    model_path, model_folder, repo_id = smart_lm_instance.ensure_model_path(template_name)
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"GGUF model file not found: {model_file}")
    
    # Verify model integrity (will use cached hash if available)
    if repo_id:
        if not verify_model_integrity(model_file, extract_repo_id_from_url(repo_id)):
            raise RuntimeError(f"Model integrity verification failed for {model_file.name}. File may be corrupted.")
    
    # Detect model type from actual model filename to choose appropriate chat handler
    model_name_lower = model_file.name.lower()
    is_llava = "llava" in model_name_lower
    is_qwen = "qwen" in model_name_lower
    
    # Import llama.cpp components
    ChatHandler: Optional[Any] = None
    try:
        from llama_cpp import Llama
        
        # Import appropriate chat handler for vision support
        if is_llava:
            try:
                from llama_cpp.llama_chat_format import Llava16ChatHandler
                ChatHandler = Llava16ChatHandler
            except ImportError:
                cstr(f"[SmartLM] Warning: Llava16ChatHandler not available").warning.print()
                
        elif is_qwen:
            # Try to import Qwen handler (name changed between versions)
            # Note: Qwen2.5-VL support added in llama-cpp-python v0.3.10+
            try:
                # Try Qwen25VLChatHandler (v0.3.10+, name confirmed in v0.3.16)
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                ChatHandler = Qwen25VLChatHandler
                cstr(f"[SmartLM] Using Qwen25VLChatHandler").msg.print()
            except ImportError:
                try:
                    # Try Qwen2VLChatHandler (fallback for potential intermediate versions)
                    from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                    ChatHandler = Qwen2VLChatHandler
                    cstr(f"[SmartLM] Using Qwen2VLChatHandler").msg.print()
                except ImportError:
                    # Handler not available - need v0.3.10 or newer
                    import llama_cpp
                    version = getattr(llama_cpp, '__version__', 'unknown')
                    cstr(f"[SmartLM] Qwen chat handler not available in llama-cpp-python {version}").warning.print()
                    cstr(f"[SmartLM] Qwen2.5-VL requires llama-cpp-python >= 0.3.10").warning.print()
                    cstr(f"[SmartLM] Upgrade with: pip install --upgrade llama-cpp-python>=0.3.16").warning.print()
                    ChatHandler = None
                    
    except ImportError as e:
        raise ImportError(
            f"Failed to import llama-cpp-python components: {e}\n"
            "Install with: pip install llama-cpp-python\n"
            "Then restart ComfyUI."
        )
    
    # MMProj is required for vision support - use helper to download if needed
    mmproj_file_path = smart_lm_instance.ensure_mmproj_path(template_info, model_folder, template_name)
    mmproj_file = Path(mmproj_file_path) if mmproj_file_path else None
    
    # Verify mmproj integrity if it exists and we have a repo_id
    if mmproj_file and mmproj_file.exists():
        mmproj_url = template_info.get("mmproj_url")
        if mmproj_url:
            # Extract original filename from URL for hash lookup
            original_mmproj_filename = mmproj_url.split('/')[-1]
            if not verify_model_integrity(mmproj_file, extract_repo_id_from_url(mmproj_url), original_mmproj_filename):
                cstr(f"[SmartLM] Warning: MMProj integrity verification failed. File may be corrupted.").warning.print()
    
    if not mmproj_file:
        cstr(f"[SmartLM] Warning: No MMProj available. Vision features may not work.").warning.print()
    
    cstr(f"[SmartLM] Loading GGUF model: {model_file.name}").msg.print()
    if mmproj_file:
        cstr(f"[SmartLM] Using MMProj: {mmproj_file.name}").msg.print()
    
    # Configure llama.cpp parameters
    n_gpu_layers = -1 if device == "cuda" and torch.cuda.is_available() else 0
    
    # Create chat handler with mmproj if available
    chat_handler = None
    if mmproj_file and ChatHandler is not None:
        try:
            chat_handler = ChatHandler(
                clip_model_path=str(mmproj_file),
                verbose=False
            )
            handler_type = "LLaVA 1.6" if is_llava else "Qwen2-VL"
            cstr(f"[SmartLM] ✓ MMProj loaded successfully ({handler_type} handler)").msg.print()
        except Exception as e:
            cstr(f"[SmartLM] Failed to load MMProj: {e}").error.print()
            chat_handler = None
    elif mmproj_file and ChatHandler is None:
        cstr(f"[SmartLM] Warning: MMProj file found but no chat handler available for this model type").warning.print()
    
    # Load GGUF model
    # Note: llama.cpp handles its own quantization and device management
    # We don't use ComfyUI device management or BitsAndBytes for GGUF models
    # Vision models need large context for image embeddings (especially videos)
    # Each frame can use 500-1000 tokens, so 16 frames = 8k-16k tokens minimum
    from llama_cpp import Llama
    smart_lm_instance.model = Llama(
        model_path=str(model_file),
        chat_handler=chat_handler,
        n_gpu_layers=n_gpu_layers,  # llama.cpp manages GPU offloading
        n_ctx=context_size,  # User-configurable context size
        n_batch=512,  # Batch size for processing
        verbose=False,
    )
    
    # CRITICAL: Store chat_handler reference separately for cleanup
    # The chat_handler is created BEFORE Llama() and holds the CLIP model in VRAM
    # We must track it separately to properly free VRAM on cleanup
    smart_lm_instance.chat_handler_ref = chat_handler
    
    # Mark as GGUF model
    smart_lm_instance.is_gguf = True
    smart_lm_instance.is_quantized = True  # GGUF models are pre-quantized (Q8, Q4, etc.)
    
    # Store template for reference
    smart_lm_instance.gguf_template = template_info
    
    cstr(f"[SmartLM] ✓ GGUF model loaded with {n_gpu_layers} GPU layers").msg.print()


def generate_qwenvl(smart_lm_instance, image: Any, prompt: str, max_tokens: int, 
                   temperature: float, top_p: float, top_k: int, num_beams: int, 
                   do_sample: bool, seed: Optional[int], repetition_penalty: float = 1.0, 
                   frame_count: int = 8) -> str:
    """Generate with QwenVL (supports both transformers and GGUF)"""
    
    # Seed is already clamped by JavaScript for QwenVL (int32 range)
    # Florence-2 and LLM use full uint32 range
    
    # Check if this is a GGUF model
    if hasattr(smart_lm_instance, 'is_gguf') and smart_lm_instance.is_gguf:
        return generate_qwenvl_gguf(smart_lm_instance, image, prompt, max_tokens, temperature, top_p, top_k, seed, repetition_penalty, frame_count)
    
    # Transformers model generation
    import numpy as np
    
    # Parse prompt to extract instruction and optional user hints (same as GGUF)
    system_prompt = "You are a helpful assistant."
    user_hints = ""
    
    if "\n\n" in prompt:
        parts = prompt.split("\n\n")
        if len(parts) >= 2:
            system_prompt = parts[0].strip()
            if len(parts) >= 3 and parts[2].strip().startswith("Additional context:"):
                user_hints = parts[2].strip().replace("Additional context:", "").strip()
    
    # Build final prompt: instruction + optional user hints
    final_prompt = system_prompt
    if user_hints:
        final_prompt = f"{system_prompt} {user_hints}"
    
    # HYBRID APPROACH: Handle device management based on quantization
    # Set defaults first to ensure variables are always defined
    device = next(smart_lm_instance.model.parameters()).device
    offload_device = device
    
    if hasattr(smart_lm_instance, 'is_quantized') and not smart_lm_instance.is_quantized:
        # Non-quantized: Use ComfyUI device management
        try:
            import comfy.model_management as mm
            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()
            smart_lm_instance.model.to(device)
        except:
            # ComfyUI not available, use model's current device
            pass
    # Quantized: Model stays where device_map placed it (uses defaults set above)
    
    # Handle video frames if input has multiple frames
    frames = None
    if image is not None and len(image.shape) == 4 and image.shape[0] > 1:
        # This is a video (multiple frames) - limit to frame_count
        total_frames = image.shape[0]
        actual_frame_count = min(frame_count, total_frames)
        frames = [smart_lm_instance.tensor_to_pil(image[i]) for i in range(actual_frame_count)]
    
    image_pil = smart_lm_instance.tensor_to_pil(image) if image is not None else None
    
    conversation: list[dict[str, Any]] = [{"role": "user", "content": []}]
    
    # Add image if single frame
    if image_pil and frames is None:
        conversation[0]["content"].append({"type": "image", "image": image_pil})
    
    # Add video if multiple frames
    if frames and len(frames) > 1:
        conversation[0]["content"].append({"type": "video", "video": frames})
    
    conversation[0]["content"].append({"type": "text", "text": final_prompt})
    
    chat = smart_lm_instance.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    images = [image_pil] if (image_pil and frames is None) else None
    videos = [frames] if frames and len(frames) > 1 else None
    
    processed = smart_lm_instance.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in processed.items()
    }
    
    stop_tokens: list[int] = [smart_lm_instance.tokenizer.eos_token_id]
    if hasattr(smart_lm_instance.tokenizer, "eot_id") and smart_lm_instance.tokenizer.eot_id is not None:
        stop_tokens.append(smart_lm_instance.tokenizer.eot_id)
    
    kwargs = {
        "max_new_tokens": max_tokens,
        "repetition_penalty": 1.0,  # Default value matching qwenvl_base
        "num_beams": num_beams,
        "eos_token_id": stop_tokens,
        "pad_token_id": smart_lm_instance.tokenizer.pad_token_id,
    }
    
    if num_beams == 1:
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        kwargs["do_sample"] = False
    
    outputs = smart_lm_instance.model.generate(**model_inputs, **kwargs)
    
    # Synchronize CUDA to ensure generation is complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    input_len = model_inputs["input_ids"].shape[-1]
    text = smart_lm_instance.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
    
    # HYBRID APPROACH: Offload non-quantized models back
    if hasattr(smart_lm_instance, 'is_quantized') and not smart_lm_instance.is_quantized and offload_device != device:
        try:
            smart_lm_instance.model.to(offload_device)
            import comfy.model_management as mm
            mm.soft_empty_cache()
        except:
            pass
    
    return text.strip()


def generate_qwenvl_gguf(smart_lm_instance, image: Any, prompt: str, max_tokens: int,
                         temperature: float, top_p: float, top_k: int, seed: Optional[int],
                         repetition_penalty: float = 1.0, frame_count: int = 8) -> str:
    """Generate with QwenVL GGUF model using llama-cpp-python"""
    
    # Set seed if provided (already clamped by JavaScript)
    if seed is not None:
        smart_lm_instance.model.set_seed(seed)
    
    # Parse prompt format: "system instruction\n\ntask name" or just "task"
    # Extract system instruction and check for user hints
    system_prompt = "You are a helpful assistant."
    user_hints = ""  # Optional hints from user_prompt widget
    
    # Check if prompt has the format: "instruction\n\ntask_name\n\nAdditional context: hints"
    if "\n\n" in prompt:
        parts = prompt.split("\n\n")
        if len(parts) >= 2:
            # First part is the detailed instruction - use as system prompt
            system_prompt = parts[0].strip()
            # Check if there's additional user context (starts with "Additional context:")
            if len(parts) >= 3 and parts[2].strip().startswith("Additional context:"):
                user_hints = parts[2].strip().replace("Additional context:", "").strip()
    
    # Prepare messages for chat completion with extracted system prompt
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    
    # Add image if provided
    if image is not None:
        # Handle video (multiple frames) or single image
        if len(image.shape) == 4 and image.shape[0] > 1:
            # For video, llama.cpp expects images in the content array - limit to frame_count
            total_frames = image.shape[0]
            actual_frame_count = min(frame_count, total_frames)
            image_content = []
            for i in range(actual_frame_count):
                # Add instruction + optional user hints before each image
                frame_text = system_prompt
                if user_hints:
                    frame_text = f"{system_prompt} {user_hints}"
                image_content.append({"type": "text", "text": frame_text})
                
                pil_image = smart_lm_instance.tensor_to_pil(image[i])
                # Convert to base64 data URL for llama.cpp
                buffered = BytesIO()
                try:
                    pil_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    image_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_str}"}
                    })
                finally:
                    # Cleanup memory per frame
                    buffered.close()
                    del pil_image, buffered
            
            # Force garbage collection after processing all frames
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            messages.append({
                "role": "user",
                "content": image_content
            })
        else:
            # Single image
            pil_image = smart_lm_instance.tensor_to_pil(image)
            buffered = BytesIO()
            try:
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Add text prompt with image - model needs explicit instruction
                prompt_text = system_prompt
                if user_hints:
                    prompt_text = f"{system_prompt} {user_hints}"
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                })
            finally:
                # Cleanup memory
                buffered.close()
                del pil_image, buffered
    else:
        # Text-only prompt (no image) - need to include the prompt in user message
        prompt_text = system_prompt
        if user_hints:
            prompt_text = f"{system_prompt} {user_hints}"
        messages.append({
            "role": "user",
            "content": prompt_text
        })
    
    # Generate response
    try:
        response = smart_lm_instance.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stream=False
        )
        
        # Extract text from response
        text = response['choices'][0]['message']['content']
        
        # Clear messages to free base64 image data from memory
        messages.clear()
        del response
        
        # Clean up common formatting artifacts
        text = text.strip()
        # Remove leading colon and space (e.g., ": The video shows..." -> "The video shows...")
        if text.startswith(": "):
            text = text[2:]
        elif text.startswith(":"):
            text = text[1:].lstrip()
        
        return text
        
    except ValueError as e:
        if "Failed to evaluate chunk: error code 1" in str(e):
            error_msg = (
                f"[SmartLM] GGUF model ran out of context. "
                f"Try: 1) Reduce frame_count (currently processing {image.shape[0] if image is not None and len(image.shape) == 4 else 1} frames), "
                f"2) Increase context_size when loading model (32768 for videos recommended)"
            )
            cstr(error_msg).error.print()
            raise ValueError(error_msg) from e
        else:
            raise

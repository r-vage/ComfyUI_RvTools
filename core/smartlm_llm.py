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

# LLM Model Wrapper - Handles text-only language models
#
# Supports:
# - Transformers models (Mistral, Llama, Qwen, etc.)
# - GGUF format with llama-cpp-python
# - Few-shot learning and instruction templates
# - Direct chat mode

import torch
from pathlib import Path
from typing import Optional
from . import cstr


def load_llm(smart_lm_instance, template_name: str, quantization: str, attention: str, device: str, context_size: int = 32768):
    """Load text-only LLM model (supports both transformers and GGUF)"""
    from .smartlm_base import load_template, LLAMA_CPP_AVAILABLE
    from .smartlm_files import verify_model_integrity, extract_repo_id_from_url
    
    template_info = load_template(template_name)
    
    # Ensure model path exists first to get actual file path
    model_path, model_folder, repo_id = smart_lm_instance.ensure_model_path(template_name)
    
    # Detect GGUF from actual model path and template values
    model_path_lower = model_path.lower()
    local_path = template_info.get("local_path", "")
    local_lower = local_path.lower()
    repo_lower = repo_id.lower() if repo_id else ""
    
    # Check for .gguf extension
    has_gguf_ext = model_path_lower.endswith(".gguf") or local_lower.endswith(".gguf") or repo_lower.endswith(".gguf")
    
    # Check for "gguf" in repo/path name (e.g., "model-GGUF" repos)
    has_gguf_name = "gguf" in model_path_lower or "gguf" in local_lower or "gguf" in repo_lower
    
    # Check for GGUF quantization markers (Q4_K_M, Q5_K_S, Q8_0, etc.)
    gguf_quant_markers = ["_q4_", "_q5_", "_q6_", "_q8_", "-q4-", "-q5-", "-q6-", "-q8-",
                          "_k_m", "_k_s", "_k_l", "q4_k", "q5_k", "q6_k", "q8_0", ".q4_", ".q5_", ".q6_", ".q8_"]
    has_gguf_quant = any(marker in model_path_lower or marker in local_lower or marker in repo_lower for marker in gguf_quant_markers)
    
    is_gguf = has_gguf_ext or has_gguf_name or has_gguf_quant
    
    if is_gguf:
        load_llm_gguf(smart_lm_instance, template_name, template_info, device, context_size)
    else:
        load_llm_transformers(smart_lm_instance, template_name, quantization, attention, device)


def load_llm_gguf(smart_lm_instance, template_name: str, template_info: dict, device: str, context_size: int = 32768):
    """Load text-only GGUF model with llama-cpp-python (no vision support)"""
    from .smartlm_base import LLAMA_CPP_AVAILABLE
    from .smartlm_files import verify_model_integrity, extract_repo_id_from_url
    
    # Check if llama-cpp-python is available
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError(
            "llama-cpp-python is required for GGUF models but was not found at startup. "
            "Install with: pip install llama-cpp-python\n"
            "Then restart ComfyUI."
        )
    
    # Import llama.cpp
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            f"Failed to import llama-cpp-python: {e}\n"
            "Install with: pip install llama-cpp-python\n"
            "Then restart ComfyUI."
        )
    
    # Get model path - ensure_model_path handles auto-download
    model_path, model_folder, repo_id = smart_lm_instance.ensure_model_path(template_name)
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"GGUF model file not found: {model_file}")
    
    # Verify model integrity (will use cached hash if available)
    if repo_id:
        if not verify_model_integrity(model_file, extract_repo_id_from_url(repo_id)):
            raise RuntimeError(f"Model integrity verification failed for {model_file.name}. File may be corrupted.")
    
    cstr(f"[SmartLM] Loading text-only LLM GGUF model: {model_file.name}").msg.print()
    
    # Configure llama.cpp parameters
    n_gpu_layers = -1 if device == "cuda" and torch.cuda.is_available() else 0
    
    # Load GGUF model (NO chat_handler for text-only)
    smart_lm_instance.model = Llama(
        model_path=str(model_file),
        n_gpu_layers=n_gpu_layers,
        n_ctx=context_size,  # User-configurable context size
        verbose=False,
    )
    
    # Mark as GGUF model
    smart_lm_instance.is_gguf = True
    smart_lm_instance.is_quantized = True
    
    # Store template for reference
    smart_lm_instance.gguf_template = template_info
    
    cstr(f"[SmartLM] ✓ Text-only LLM GGUF loaded with {n_gpu_layers} GPU layers").msg.print()


def load_llm_transformers(smart_lm_instance, template_name: str, quantization: str, attention: str, device: str):
    """Load text-only LLM model with transformers (Mistral, Llama, etc.)"""
    from .smartlm_base import load_template
    from .smartlm_files import verify_model_integrity
    
    template_info = load_template(template_name)
    
    # Ensure model path exists
    model_path, model_folder, repo_id = smart_lm_instance.ensure_model_path(template_name)
    
    # Verify model integrity if loading from local cache
    if not repo_id:
        repo_id = template_info.get("repo_id", "")
    if not verify_model_integrity(Path(model_path), repo_id):
        raise RuntimeError(f"Model integrity check failed for {model_path}. The model may be corrupted. Please delete and re-download.")
    
    # Load transformers model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
    
    # Auto-detect if model is pre-quantized
    repo_id = template_info.get("repo_id", "")
    local_path = template_info.get("local_path", "")
    has_quant_markers = any(marker in local_path.lower() or marker in repo_id.lower() or marker in model_path.lower()
                            for marker in ["fp8", "int8", "int4", "q4", "q5", "q6", "q8", "_q4_", "_q5_", "_q8_"])
    
    is_prequantized = template_info.get("quantized", has_quant_markers)
    
    # Determine quantization config and dtype
    if is_prequantized:
        quant_config = None
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    elif quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        dtype = None
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = None
    else:
        quant_config = None
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map.get(quantization, torch.float16)
        if device == "cpu" and dtype != torch.float32:
            dtype = torch.float32
    
    cstr(f"[SmartLM] Loading text-only LLM (transformers): {model_path}").msg.print()
    cstr(f"[SmartLM] Quantization: {quantization}, dtype: {dtype}, device: {device}").msg.print()
    
    # Load model
    if quant_config:
        smart_lm_instance.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        smart_lm_instance.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cuda":
            smart_lm_instance.model.to("cuda")
    
    # Load tokenizer
    smart_lm_instance.processor = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # Set padding token if not set
    if smart_lm_instance.processor.pad_token is None:
        smart_lm_instance.processor.pad_token = smart_lm_instance.processor.eos_token
    
    # Mark as loaded
    smart_lm_instance.is_gguf = False
    smart_lm_instance.is_quantized = quantization in ["4bit", "8bit"] or is_prequantized
    
    cstr(f"[SmartLM] ✓ Text-only LLM loaded (transformers, {quantization})").msg.print()


def generate_llm(smart_lm_instance, prompt: str, max_tokens: int, temperature: float,
                 top_p: float, top_k: int, seed: Optional[int], repetition_penalty: float,
                 llm_mode: str, instruction_template: str) -> str:
    """Generate text-only completion with LLM (no images)"""
    from .smartlm_base import LLM_FEW_SHOT_EXAMPLES
    
    # Check if model is GGUF (llama-cpp-python) or transformers
    is_gguf_model = hasattr(smart_lm_instance, 'is_gguf') and smart_lm_instance.is_gguf
    
    # Set seed if provided (method differs between GGUF and transformers)
    if seed is not None:
        if is_gguf_model:
            smart_lm_instance.model.set_seed(seed)
        else:
            # Transformers - set global seed
            from transformers import set_seed
            import hashlib
            seed_bytes = str(seed).encode('utf-8')
            hash_object = hashlib.sha256(seed_bytes)
            hashed_seed = int(hash_object.hexdigest(), 16) % (2**32)
            set_seed(hashed_seed)
    
    # Load configuration for the selected mode
    config = LLM_FEW_SHOT_EXAMPLES.get(llm_mode, LLM_FEW_SHOT_EXAMPLES.get("direct_chat", {}))
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    examples = config.get("examples", [])
    
    # Get instruction template (custom or from config)
    if instruction_template:
        # Custom instruction provided
        template = instruction_template
    else:
        # Use template from config
        template = config.get("instruction_template", "")
    
    # Build messages based on mode
    if llm_mode != "direct_chat" and template:
        # Apply instruction template with few-shot examples
        req = template.replace("{prompt}", prompt) if "{prompt}" in template else f"{template} {prompt}"
        
        # Build messages: system + examples + user request
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(examples)
        messages.append({"role": "user", "content": req})
    else:
        # Direct chat mode - no instruction wrapper or examples
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    
    # Generate response - different methods for GGUF vs transformers
    try:
        if is_gguf_model:
            # GGUF model (llama-cpp-python) - use create_chat_completion
            response = smart_lm_instance.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stream=False
            )
            text = response['choices'][0]['message']['content']
        else:
            # Transformers model - use tokenizer + generate
            # Apply chat template to convert messages to text
            input_text = smart_lm_instance.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and generate
            inputs = smart_lm_instance.processor(input_text, return_tensors="pt").to(smart_lm_instance.model.device)
            
            outputs = smart_lm_instance.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=smart_lm_instance.processor.eos_token_id
            )
            
            # Decode only the generated tokens (skip input)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            text = smart_lm_instance.processor.decode(generated_tokens, skip_special_tokens=True)
        
        return text.strip()

    except Exception as e:
        cstr(f"[SmartLM] LLM generation error: {e}").error.print()
        raise

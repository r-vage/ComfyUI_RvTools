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
#
# Qwen Vision-Language Models are licensed under Apache-2.0
# Original QwenVL integration: GPL-3.0 (ComfyUI-QwenVL)
# Eclipse integration: Apache-2.0

from __future__ import annotations

"""
QwenVL Base Module - Shared utilities and base class for QwenVL nodes

This module contains:
- Configuration loading
- Model download with mirror support
- Memory management utilities
- Base class for QwenVL model loading and inference

Used by both simplified and advanced QwenVL nodes.
"""

import gc
import json
from enum import Enum
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

import folder_paths
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from . import cstr

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "json" / "qwenvl_config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

MODEL_CONFIGS = {k: v for k, v in CONFIG.items() if not k.startswith("_")}
SYSTEM_PROMPTS = CONFIG.get("_system_prompts", {})
PRESET_PROMPTS = CONFIG.get("_preset_prompts", [])
DOWNLOAD_SETTINGS = CONFIG.get("_download_settings", {
    "endpoint": "auto",
    "use_mirror": False,
    "mirror_endpoint": "https://hf-mirror.com"
})

ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa"]


class Quantization(Enum):
    """Quantization modes for model loading"""
    Q4 = "4bit"
    Q8 = "8bit"
    FP16 = "fp16"

    @classmethod
    def get_values(cls):
        return [e.value for e in cls]

    @classmethod
    def from_value(cls, value):
        for e in cls:
            if e.value == value:
                return e
        return cls.FP16


def get_device_info():
    """Detect available compute device and memory"""
    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev_id)
        total = props.total_memory / (1024**3)
        free = (props.total_memory - torch.cuda.memory_allocated(dev_id)) / (1024**3)
        return {
            "device_type": "cuda",
            "recommended_device": "cuda",
            "gpu": {
                "name": props.name,
                "total_memory": total,
                "free_memory": free,
            },
            "system_memory": {
                "total": psutil.virtual_memory().total / (1024**3),
                "available": psutil.virtual_memory().available / (1024**3),
            },
        }
    elif torch.backends.mps.is_available():
        return {
            "device_type": "mps",
            "recommended_device": "mps",
            "system_memory": {
                "total": psutil.virtual_memory().total / (1024**3),
                "available": psutil.virtual_memory().available / (1024**3),
            },
        }
    else:
        return {
            "device_type": "cpu",
            "recommended_device": "cpu",
            "system_memory": {
                "total": psutil.virtual_memory().total / (1024**3),
                "available": psutil.virtual_memory().available / (1024**3),
            },
        }


def flash_attn_available():
    """Check if Flash Attention v2 is available"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def resolve_attention_mode(mode):
    """Determine attention implementation to use"""
    if mode in {"flash_attention_2", "sdpa"}:
        return mode
    if flash_attn_available():
        return "flash_attention_2"
    cstr("[QwenVL] Flash-Attn auto mode: dependency not ready, using SDPA").msg.print()
    return "sdpa"


def ensure_model(model_name):
    """Download model if not present, return local path"""
    import os
    
    info = MODEL_CONFIGS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]
    download_size = info.get("download_size_gb", 0)
    models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]
    
    # Check if model already exists
    if target.exists() and any(target.iterdir()):
        cstr(f"[QwenVL] Using cached model: {model_name}").msg.print()
        return str(target)
    
    # Configure download endpoint
    use_mirror = DOWNLOAD_SETTINGS.get("use_mirror", False)
    endpoint_setting = DOWNLOAD_SETTINGS.get("endpoint", "auto")
    mirror_endpoint = DOWNLOAD_SETTINGS.get("mirror_endpoint", "https://hf-mirror.com")
    
    # Determine endpoint to use
    current_endpoint = os.environ.get("HF_ENDPOINT", "")
    if use_mirror and endpoint_setting != "auto":
        # Override with config mirror
        os.environ["HF_ENDPOINT"] = mirror_endpoint
        endpoint_display = mirror_endpoint
    elif current_endpoint:
        # Use environment variable
        endpoint_display = current_endpoint
    else:
        # Default HuggingFace
        endpoint_display = "huggingface.co (default)"
    
    # Download model with informative messages
    size_info = f" (~{download_size:.1f} GB)" if download_size > 0 else ""
    cstr(f"[QwenVL] Downloading {model_name}{size_info}").msg.print()
    cstr(f"[QwenVL] Repository: {repo_id}").msg.print()
    cstr(f"[QwenVL] Endpoint: {endpoint_display}").msg.print()
    cstr(f"[QwenVL] Target: {target}").msg.print()
    
    if use_mirror:
        cstr(f"[QwenVL] Using mirror mode - this should be faster in China/Asia").msg.print()
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", ".git*"],
        )
        cstr(f"[QwenVL] Download complete: {model_name}").msg.print()
    except Exception as e:
        cstr(f"[QwenVL] Download failed: {e}").error.print()
        if not use_mirror and "China" in str(e) or "timeout" in str(e).lower():
            cstr(f"[QwenVL] Tip: Set 'use_mirror: true' in qwenvl_config.json for faster downloads in China/Asia").warning.print()
        raise
    finally:
        # Restore original endpoint if we changed it
        if use_mirror and endpoint_setting != "auto" and not os.environ.get("HF_ENDPOINT_ORIGINAL"):
            if current_endpoint:
                os.environ["HF_ENDPOINT"] = current_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)
    
    return str(target)


def enforce_memory(model_name, quantization, device_info):
    """Auto-adjust quantization if insufficient memory"""
    info = MODEL_CONFIGS.get(model_name, {})
    requirements = info.get("vram_requirement", {})
    mapping = {
        Quantization.Q4: requirements.get("4bit", 0),
        Quantization.Q8: requirements.get("8bit", 0),
        Quantization.FP16: requirements.get("full", 0),
    }
    needed = mapping.get(quantization, 0)
    if not needed:
        return quantization
    if device_info["recommended_device"] in {"cpu", "mps"}:
        needed *= 1.5
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]
    if needed * 1.2 > available:
        if quantization == Quantization.FP16:
            cstr("[QwenVL] Auto-switch to 8-bit due to VRAM pressure").warning.print()
            return Quantization.Q8
        if quantization == Quantization.Q8:
            cstr("[QwenVL] Auto-switch to 4-bit due to VRAM pressure").warning.print()
            return Quantization.Q4
        raise RuntimeError("Insufficient memory for 4-bit mode")
    return quantization


def quantization_config(model_name, quantization):
    """Build quantization config for model loading"""
    info = MODEL_CONFIGS.get(model_name, {})
    # For pre-quantized models (FP8), don't apply additional quantization
    # Return None for config and torch.bfloat16 for dtype
    if info.get("quantized"):
        return None, torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if quantization == Quantization.Q4:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return cfg, None
    if quantization == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None
    return None, torch.float16 if torch.cuda.is_available() else torch.float32


class QwenVLBase:
    """Base class for Qwen Vision-Language models"""
    
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        cstr(f"[QwenVL] Node initialized on {self.device_info['device_type']}").msg.print()

    def clear(self):
        """Clear model from memory"""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        use_compile,
        device_choice,
        keep_model_loaded,
    ):
        """Load Qwen-VL model with specified configuration"""
        quant = enforce_memory(model_name, Quantization.from_value(quant_value), self.device_info)
        attn_impl = resolve_attention_mode(attention_mode)
        device = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        signature = (model_name, quant.value, attn_impl, device, use_compile)
        
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            return
        
        self.clear()
        model_path = ensure_model(model_name)
        quant_config, dtype = quantization_config(model_name, quant)
        
        # Check if this is a pre-quantized model (e.g., FP8)
        model_info = MODEL_CONFIGS.get(model_name, {})
        is_prequantized = model_info.get("quantized", False)
        
        # Build load_kwargs based on model type
        load_kwargs = {
            "dtype": dtype,
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,  # Required to avoid "Cannot copy out of meta tensor" errors
        }
        
        # For pre-quantized models, we must NOT use device_map with dispatch
        # Instead, load to CPU first, then manually move to device after loading
        if is_prequantized:
            # Pre-quantized models: load without device_map to avoid dispatch issues
            load_kwargs["device_map"] = None
            cstr(f"[QwenVL] Loading {model_name} ({quant.value}, attn={attn_impl}) [pre-quantized]").msg.print()
        else:
            # Regular models: use device_map for efficient loading
            load_kwargs["device_map"] = "auto" if device == "cuda" and torch.cuda.is_available() else device
            cstr(f"[QwenVL] Loading {model_name} ({quant.value}, attn={attn_impl})").msg.print()
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        
        # For pre-quantized models, manually move to device after loading
        if is_prequantized and device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        
        if use_compile and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                cstr("[QwenVL] torch.compile enabled").msg.print()
            except Exception as exc:
                cstr(f"[QwenVL] torch.compile skipped: {exc}").warning.print()
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_signature = signature
        cstr(f"[QwenVL] Model loaded successfully").msg.print()

    @staticmethod
    def tensor_to_pil(tensor):
        """Convert ComfyUI tensor to PIL Image"""
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
    ):
        """Generate text response from visual inputs"""
        conversation = [{"role": "user", "content": []}]
        
        if image is not None:
            conversation[0]["content"].append({"type": "image", "image": self.tensor_to_pil(image)})
        
        if video is not None:
            frames = [self.tensor_to_pil(frame) for frame in video]
            if len(frames) > frame_count:
                idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                frames = [frames[i] for i in idx]
            if frames:
                conversation[0]["content"].append({"type": "video", "video": frames})
        
        conversation[0]["content"].append({"type": "text", "text": prompt_text})
        
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
        video_frames = [frame for item in conversation[0]["content"] if item["type"] == "video" for frame in item["video"]]
        videos = [video_frames] if video_frames else None
        
        processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False
        
        outputs = self.model.generate(**model_inputs, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def run(
        self,
        model_name,
        quantization,
        preset_prompt,
        custom_prompt,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
        keep_model_loaded,
        attention_mode,
        use_torch_compile,
        device,
    ):
        """Main execution method"""
        torch.manual_seed(seed)
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
        )
        
        try:
            text = self.generate(
                prompt,
                image,
                video,
                frame_count,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()

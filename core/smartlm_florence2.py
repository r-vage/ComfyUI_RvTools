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

# Smart LML Florence-2 Module
#
# Florence-2 specific functionality extracted from smartlm_base.py
# Handles Florence-2 model loading, generation, and helper functions

import json
import re
import torch
import numpy as np
import gc
from pathlib import Path
from typing import Any, Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

# Import cstr for logging
from . import cstr
from .smartlm_files import verify_model_integrity

# Florence-2 task prompts (hardcoded to match official implementation)
FLORENCE_PROMPTS = {
    'region_caption': '<OD>',
    'dense_region_caption': '<DENSE_REGION_CAPTION>',
    'region_proposal': '<REGION_PROPOSAL>',
    'caption': '<CAPTION>',
    'detailed_caption': '<DETAILED_CAPTION>',
    'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
    'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
    'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
    'ocr': '<OCR>',
    'ocr_with_region': '<OCR_WITH_REGION>',
    'docvqa': '<DocVQA>',
    'prompt_gen_tags': '<GENERATE_TAGS>',
    'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
    'prompt_gen_analyze': '<ANALYZE>',
    'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
}

# Florence-2 tasks configuration (loaded from config)
FLORENCE_TASKS = {
    "region_caption": {"prompt": "<OD>", "description": "Object detection with captions"},
    "dense_region_caption": {"prompt": "<DENSE_REGION_CAPTION>", "description": "Dense captioning with multiple regions"},
    "region_proposal": {"prompt": "<REGION_PROPOSAL>", "description": "Generate region proposals for objects"},
    "caption": {"prompt": "<CAPTION>", "description": "Short single-sentence caption"},
    "detailed_caption": {"prompt": "<DETAILED_CAPTION>", "description": "Detailed paragraph description"},
    "more_detailed_caption": {"prompt": "<MORE_DETAILED_CAPTION>", "description": "Very detailed rich description"},
    "caption_to_phrase_grounding": {"prompt": "<CAPTION_TO_PHRASE_GROUNDING>", "description": "Detect and locate specific objects/phrases in image"},
    "referring_expression_segmentation": {"prompt": "<REFERRING_EXPRESSION_SEGMENTATION>", "description": "Segment objects based on text description"},
    "ocr": {"prompt": "<OCR>", "description": "Extract text from image"},
    "ocr_with_region": {"prompt": "<OCR_WITH_REGION>", "description": "Extract text with bounding boxes"},
    "docvqa": {"prompt": "<DocVQA>", "description": "Document visual question answering"},
    "prompt_gen_tags": {"prompt": "<GENERATE_TAGS>", "description": "Generate comma-separated tags (PromptGen models)"},
    "prompt_gen_mixed_caption": {"prompt": "<MIXED_CAPTION>", "description": "Mixed-style caption for prompt generation"},
    "prompt_gen_analyze": {"prompt": "<ANALYZE>", "description": "Analytical description"},
    "prompt_gen_mixed_caption_plus": {"prompt": "<MIXED_CAPTION_PLUS>", "description": "Enhanced mixed caption"},
}


def get_florence_tasks():
    """Get Florence-2 tasks configuration."""
    return FLORENCE_TASKS


def update_florence_tasks(tasks: dict):
    """Update Florence-2 tasks from config file."""
    global FLORENCE_TASKS
    FLORENCE_TASKS = tasks


def get_florence2_size_estimate(model_name: str) -> float:
    """
    Estimate Florence-2 model size based on model name.
    
    Args:
        model_name: Model name or path (lowercase)
        
    Returns:
        Estimated size in GB
    """
    if "florence-2-base" in model_name or "florence2-base" in model_name:
        return 0.5  # ~230M params
    elif "florence-2-large" in model_name or "florence2-large" in model_name:
        return 1.5  # ~770M params
    return 0.5  # Default to base size


def nms_filter(bboxes: List[List[float]], labels: List[str], iou_threshold: float = 0.5) -> Tuple[List[List[float]], List[str]]:
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes.
    
    Args:
        bboxes: List of bounding boxes in [x1, y1, x2, y2] format
        labels: List of labels corresponding to each bbox
        iou_threshold: IoU threshold for suppression (default: 0.5)
        
    Returns:
        Tuple of (filtered_bboxes, filtered_labels)
    """
    if not bboxes or len(bboxes) == 0:
        return [], []
    
    # Convert to numpy for easier computation
    boxes = np.array(bboxes)
    
    # Calculate areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by y2 coordinate (bottom of box) - larger boxes first
    order = np.argsort(y2)[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    # Filter bboxes and labels
    filtered_bboxes = [bboxes[i] for i in keep]
    filtered_labels = [labels[i] for i in keep] if labels else []
    
    return filtered_bboxes, filtered_labels


def load_florence2_model(base_instance, template_name: str, quantization: str, 
                         attention: str, device: str, use_torch_compile: bool = False):
    """
    Load Florence-2 model using wrapper for proper import handling.
    
    Args:
        base_instance: SmartLMLBase instance to store model/processor
        template_name: Template configuration name
        quantization: Quantization mode (fp16/bf16/fp32/8bit/4bit)
        attention: Attention implementation (auto/sdpa/eager/flash_attention_2)
        device: Target device (cuda/cpu/mps)
        use_torch_compile: Whether to apply torch.compile optimization
    """
    from transformers import BitsAndBytesConfig
    from .florence2_wrapper import load_florence2_model as load_model, load_florence2_processor
    
    # Ensure model path exists (downloads if needed)
    model_path, model_folder, repo_id = base_instance.ensure_model_path(template_name)
    
    # Load template info for configuration
    from .smartlm_base import load_template
    template_info = load_template(template_name)
    
    # Determine quantization config and dtype (same as QwenVL)
    quant_config = None
    dtype = None
    
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        # fp16/bf16/fp32
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
            quantization, torch.float16
        )
    
    # Build load_kwargs
    load_kwargs = {
        "dtype": dtype,
        "attn_implementation": attention,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "repo_id": repo_id,  # For integrity verification
    }
    
    # HYBRID APPROACH: quantized vs non-quantized
    if quant_config:
        # Quantized: Let bitsandbytes handle device placement
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
    else:
        # Non-quantized: Load to CPU first (ComfyUI will manage device)
        load_kwargs["device_map"] = None
        load_kwargs["torch_dtype"] = dtype
    
    # Load model using wrapper (handles custom implementation and fallback)
    base_instance.model = load_model(model_path, **load_kwargs).eval()
    
    # Post-loading device placement for non-quantized models
    # Use standard PyTorch .to() since Florence-2 is a HuggingFace model, not a ComfyUI model
    if not quant_config:
        if device == "cuda" and torch.cuda.is_available():
            base_instance.model = base_instance.model.to("cuda")
        elif device != "cpu":
            base_instance.model = base_instance.model.to(device)
    
    # Apply torch.compile if requested (requires CUDA and non-quantized model)
    if use_torch_compile and not quant_config and torch.cuda.is_available():
        try:
            base_instance.model = torch.compile(base_instance.model, mode="reduce-overhead")
            cstr("[SmartLM] ✓ Applied torch.compile optimization").msg.print()
        except Exception as e:
            cstr(f"[SmartLM] Warning: torch.compile failed: {e}").warning.print()
    elif use_torch_compile and quant_config:
        cstr("[SmartLM] torch.compile skipped (not compatible with quantization)").msg.print()
    
    # Load processor using wrapper
    base_instance.processor = load_florence2_processor(model_path)
    base_instance.dtype = dtype if dtype else torch.float16
    
    # Track quantization state for generation
    base_instance.is_quantized = bool(quant_config)


def generate_florence2(base_instance, image: Any, task_or_prompt: str, max_tokens: int,
                       num_beams: int, do_sample: bool, seed: Optional[int], 
                       repetition_penalty: float = 1.0, text_input: Optional[str] = None,
                       convert_to_bboxes: bool = True, detection_filter_threshold: float = 0.80,
                       nms_iou_threshold: float = 0.50) -> Tuple[str, dict]:
    """
    Generate with Florence-2 - returns (text, parsed_data).
    
    Args:
        base_instance: SmartLMLBase instance with loaded model
        image: Input image tensor
        task_or_prompt: Task name or custom prompt
        max_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        seed: Random seed
        repetition_penalty: Penalty for repetition
        text_input: Additional text input for specific tasks
        convert_to_bboxes: Convert detections to bboxes format
        detection_filter_threshold: Filter threshold for oversized detections
        nms_iou_threshold: IoU threshold for NMS filtering (0.0-1.0)
        
    Returns:
        Tuple of (generated_text, parsed_data_dict)
    """
    if seed is not None:
        # Use hash for better randomization (Florence-2 seeds can be full uint32)
        import hashlib
        seed_bytes = str(seed).encode('utf-8')
        hash_seed = int(hashlib.sha256(seed_bytes).hexdigest()[:8], 16)
        torch.manual_seed(hash_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hash_seed)
    
    # Handle image tensor - Florence-2 expects PIL image
    if image is None:
        raise ValueError("Florence-2 requires an image input")
    
    # Convert tensor to PIL - handle batch dimension
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            # Take first image from batch
            image = image[0]
        # Convert to PIL
        array = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(array)
    else:
        image_pil = image
    
    if not image_pil:
        raise ValueError("Failed to convert image to PIL format")
    
    # Get task prompt token
    task_prompt = FLORENCE_PROMPTS.get(task_or_prompt, task_or_prompt)
    
    # Tasks that require text input after the task token
    tasks_requiring_text = [
        "caption_to_phrase_grounding",
        "referring_expression_segmentation",
        "docvqa"
    ]
    
    # Build full prompt for generation (task token + text input only for tasks that need it)
    if task_or_prompt in tasks_requiring_text and text_input and text_input.strip():
        # Task token + text input (e.g., "<CAPTION_TO_PHRASE_GROUNDING>person")
        prompt = task_prompt + text_input
    else:
        # Just task token (e.g., "<CAPTION>")
        prompt = task_prompt
    
    # Get current device - Florence-2 uses standard PyTorch device management
    device = next(base_instance.model.parameters()).device
    
    # For non-quantized models, ensure model is on correct device
    if hasattr(base_instance, 'is_quantized') and not base_instance.is_quantized:
        # Move to CUDA if available and not already there
        if torch.cuda.is_available() and device.type != "cuda":
            base_instance.model = base_instance.model.to("cuda")
            device = next(base_instance.model.parameters()).device
    # Quantized: Model stays where device_map placed it
    
    dtype = base_instance.dtype if hasattr(base_instance, 'dtype') else torch.float16
    
    # Process image with do_rescale=False (ComfyUI images are already 0-1)
    inputs = base_instance.processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False)
    inputs = {k: v.to(dtype).to(device) if torch.is_tensor(v) and v.dtype.is_floating_point else v.to(device)
              for k, v in inputs.items()}
    
    generated_ids = base_instance.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty if repetition_penalty and repetition_penalty > 0 else 1.0,
        use_cache=False,
    )
    
    # Decode with skip_special_tokens=True to get clean text first
    results_clean = base_instance.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Also decode with skip_special_tokens=False for spatial tasks that need location tokens
    results = base_instance.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Define tasks that produce spatial data (bboxes, polygons, etc.)
    spatial_tasks = [
        "region_caption",  # Object detection with captions
        "dense_region_caption",
        "region_proposal",
        "caption_to_phrase_grounding",
        "ocr_with_region",
        "referring_expression_segmentation"
    ]
    
    # Parse structured output (bounding boxes, labels, etc.) ONLY for spatial tasks
    parsed_data = {}
    is_spatial_task = task_or_prompt in spatial_tasks
    
    if is_spatial_task:
        # Try to parse with processor's post_process method
        try:
            parsed_answer = base_instance.processor.post_process_generation(
                results,
                task=task_prompt,
                image_size=(image_pil.width, image_pil.height)
            )
            
            # Extract bboxes and labels
            if task_prompt in parsed_answer:
                task_result = parsed_answer[task_prompt]
                
                # Handle different result formats
                if 'bboxes' in task_result and 'labels' in task_result:
                    parsed_data['bboxes'] = task_result['bboxes']
                    parsed_data['labels'] = task_result['labels']
                elif 'quad_boxes' in task_result and 'labels' in task_result:
                    # OCR with region returns quad boxes (4 corners)
                    parsed_data['quad_boxes'] = task_result['quad_boxes']
                    parsed_data['labels'] = task_result['labels']
                elif 'polygons' in task_result and 'labels' in task_result:
                    # Segmentation returns polygons
                    parsed_data['polygons'] = task_result['polygons']
                    parsed_data['labels'] = task_result['labels']
                
        except Exception as e:
            # Fallback: Manual parsing if processor fails
            cstr(f"[SmartLM] Processor parsing failed, using manual parser: {e}").warning.print()
            parsed_data = parse_florence_location_tokens(
                results, image_pil.width, image_pil.height
            )
        
        # Convert quad boxes to regular bboxes if needed
        if 'quad_boxes' in parsed_data and convert_to_bboxes:
            quad_boxes = parsed_data['quad_boxes']
            bboxes = []
            for quad in quad_boxes:
                # Quad format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Convert to bbox: [min_x, min_y, max_x, max_y]
                if isinstance(quad[0], (list, tuple)):
                    # Nested list format
                    xs = [pt[0] for pt in quad]
                    ys = [pt[1] for pt in quad]
                else:
                    # Flat list format [x1,y1,x2,y2,...]
                    xs = [quad[i] for i in range(0, len(quad), 2)]
                    ys = [quad[i] for i in range(1, len(quad), 2)]
                
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                bboxes.append(bbox)
            
            parsed_data['bboxes'] = bboxes
            del parsed_data['quad_boxes']
        
        # Convert polygons to bboxes if needed
        if 'polygons' in parsed_data and convert_to_bboxes:
            polygons = parsed_data['polygons']
            bboxes = []
            for poly in polygons:
                # Polygon format: [[x1,y1], [x2,y2], ...] or [x1,y1,x2,y2,...]
                if isinstance(poly[0], (list, tuple)):
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                else:
                    xs = [poly[i] for i in range(0, len(poly), 2)]
                    ys = [poly[i] for i in range(1, len(poly), 2)]
                
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                bboxes.append(bbox)
            
            parsed_data['bboxes'] = bboxes
            del parsed_data['polygons']
        
        # Filter out oversized detections (likely errors)
        if 'bboxes' in parsed_data and parsed_data['bboxes']:
            img_area = image_pil.width * image_pil.height
            filtered_bboxes = []
            filtered_labels = []
            
            for i, bbox in enumerate(parsed_data['bboxes']):
                x1, y1, x2, y2 = bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_ratio = bbox_area / img_area if img_area > 0 else 0
                
                # Filter out detections that cover > threshold of image (likely errors)
                if bbox_ratio <= detection_filter_threshold:
                    filtered_bboxes.append(bbox)
                    if 'labels' in parsed_data and i < len(parsed_data['labels']):
                        filtered_labels.append(parsed_data['labels'][i])
            
            parsed_data['bboxes'] = filtered_bboxes
            if filtered_labels:
                parsed_data['labels'] = filtered_labels
            
            # Apply NMS to remove overlapping detections (skip if threshold is 1.0)
            if len(parsed_data['bboxes']) > 1 and nms_iou_threshold < 1.0:
                nms_bboxes, nms_labels = nms_filter(
                    parsed_data['bboxes'], 
                    parsed_data.get('labels', []), 
                    iou_threshold=nms_iou_threshold
                )
                if nms_bboxes:
                    parsed_data['bboxes'] = nms_bboxes
                    if nms_labels:
                        parsed_data['labels'] = nms_labels
    
    # Clean up special tokens for text output
    # Special handling for ocr_with_region to preserve line breaks
    if task_or_prompt == 'ocr_with_region':
        # For OCR with region, clean text but preserve structure
        clean_results = results_clean.strip()
    elif is_spatial_task:
        # For other spatial tasks, use clean version (skip_special_tokens=True already removed tokens)
        clean_results = results_clean.strip()
    else:
        # For non-spatial tasks, just use clean version
        clean_results = results_clean.strip()
        # Remove any remaining special tokens manually
        for token in ['<s>', '</s>', '<pad>', '<|endoftext|>']:
            clean_results = clean_results.replace(token, '')
        clean_results = clean_results.strip()
    
    # Florence-2 models stay in memory (standard HuggingFace behavior)
    # Memory is freed when SmartLMLBase.clear() is called
    
    return (clean_results, parsed_data)


def parse_florence_location_tokens(text: str, width: int, height: int) -> dict:
    """
    Parse Florence-2 location tokens manually when processor doesn't parse properly.
    
    Supports multiple formats:
    - Bboxes: "label<loc_x1><loc_y1><loc_x2><loc_y2>" (4 tokens)
    - Polygons: "label<loc_x1><loc_y1><loc_x2><loc_y2>...<loc_xn><loc_yn>" (4+ tokens, multiples of 2)
    - Quad boxes: "label<loc_x1><loc_y1><loc_x2><loc_y2><loc_x3><loc_y3><loc_x4><loc_y4>" (8 tokens for OCR)
    
    Location tokens are normalized to 0-999 range.
    
    Args:
        text: Raw model output with location tokens
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Dict with 'bboxes'/'quad_boxes'/'polygons' and 'labels'
    """
    # Pattern to match label followed by location tokens
    # Captures label and all subsequent <loc_###> tokens
    pattern = r'([^<]+?)((?:<loc_\d+>)+)'
    matches = re.findall(pattern, text)
    
    if not matches:
        return {}
    
    bboxes = []
    labels = []
    polygons = []
    quad_boxes = []
    
    for match in matches:
        label = match[0].strip()
        loc_tokens = match[1]
        
        # Extract all location values
        locs = [int(x) for x in re.findall(r'<loc_(\d+)>', loc_tokens)]
        
        if len(locs) < 4:
            continue  # Invalid
        
        # Denormalize from 0-999 to pixel coordinates
        coords = []
        for i, loc in enumerate(locs):
            if i % 2 == 0:  # x coordinate
                coords.append((loc / 999.0) * width)
            else:  # y coordinate
                coords.append((loc / 999.0) * height)
        
        # Classify by coordinate count
        if len(coords) == 4:
            # Regular bbox: [x1, y1, x2, y2]
            bboxes.append(coords)
            labels.append(label)
        elif len(coords) == 8:
            # Quad box (OCR): [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            quad = [[coords[i], coords[i+1]] for i in range(0, 8, 2)]
            quad_boxes.append(quad)
            labels.append(label)
        elif len(coords) > 4 and len(coords) % 2 == 0:
            # Polygon: [[x1,y1], [x2,y2], ...]
            poly = [[coords[i], coords[i+1]] for i in range(0, len(coords), 2)]
            polygons.append(poly)
            labels.append(label)
    
    # Return appropriate format based on what was found
    result = {}
    if bboxes:
        result['bboxes'] = bboxes
        result['labels'] = labels
    if quad_boxes:
        result['quad_boxes'] = quad_boxes
        if 'labels' not in result:
            result['labels'] = labels
    if polygons:
        result['polygons'] = polygons
        if 'labels' not in result:
            result['labels'] = labels
    
    return result


def draw_bboxes(image: Any, data: dict) -> Any:
    """
    Draw bounding boxes, quad boxes, and polygons on image and return as tensor.
    
    Args:
        image: Input image tensor
        data: Detection data dict with 'bboxes'/'quad_boxes'/'polygons' and 'labels'
        
    Returns:
        Image tensor with drawn annotations
    """
    # Convert tensor to PIL
    if image is None:
        raise ValueError("Image required for drawing")
    
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]  # Take first from batch
        array = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array)
    else:
        pil_image = image
    
    # Create a copy to draw on
    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Get detection data
    bboxes = data.get("bboxes", [])
    quad_boxes = data.get("quad_boxes", [])
    polygons = data.get("polygons", [])
    labels = data.get("labels", [])
    
    # Color palette for labels - vibrant colors with good visibility
    colormap = ['red', 'lime', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 
                'green', 'pink', 'gold', 'turquoise', 'coral', 'violet', 'springgreen', 
                'deeppink', 'dodgerblue', 'tomato', 'limegreen', 'hotpink']
    
    # Try to load font once
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw regular bounding boxes
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        # Use fixed red color for bboxes (original behavior)
        color = 'red'
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label if available
        if i < len(labels):
            label = labels[i]
            # Get text bbox for background
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw background rectangle
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
    
    # Draw quad boxes (for OCR/text detection)
    for i, quad_box in enumerate(quad_boxes):
        # quad_box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        if isinstance(quad_box[0], (list, tuple)):
            points = [(pt[0], pt[1]) for pt in quad_box]
        else:
            # Flat list format [x1,y1,x2,y2,...]
            points = [(quad_box[j], quad_box[j+1]) for j in range(0, len(quad_box), 2)]
        
        color = colormap[i % len(colormap)]
        draw.polygon(points, outline=color, width=3)
        
        # Draw index label
        if points:
            label = str(i) if i >= len(labels) else labels[i]
            text_bbox = draw.textbbox(points[0], label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([points[0][0], points[0][1] - text_height - 4, 
                           points[0][0] + text_width + 4, points[0][1]], fill=color)
            draw.text((points[0][0] + 2, points[0][1] - text_height - 2), label, fill='white', font=font)
    
    # Draw polygons
    for i, polygon in enumerate(polygons):
        # polygon format: [[x1,y1], [x2,y2], ...] or [x1,y1,x2,y2,...]
        if isinstance(polygon[0], (list, tuple)):
            points = [(pt[0], pt[1]) for pt in polygon]
        else:
            # Flat list format
            points = [(polygon[j], polygon[j+1]) for j in range(0, len(polygon), 2)]
        
        color = colormap[i % len(colormap)]
        draw.polygon(points, outline=color, width=3)
        
        # Draw label
        if points:
            label = str(i) if i >= len(labels) else labels[i]
            text_bbox = draw.textbbox(points[0], label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([points[0][0], points[0][1] - text_height - 4,
                           points[0][0] + text_width + 4, points[0][1]], fill=color)
            draw.text((points[0][0] + 2, points[0][1] - text_height - 2), label, fill='white', font=font)
    
    # Convert back to tensor format
    img_array = np.array(draw_image).astype(np.float32) / 255.0
    tensor_out = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
    
    return tensor_out

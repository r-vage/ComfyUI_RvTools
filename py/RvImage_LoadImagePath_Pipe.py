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

import os
import sys
import torch
import numpy as np
import hashlib
import requests  # type: ignore[import-untyped]
import json
import re

from io import BytesIO
from PIL import Image, ImageOps
from xml.dom import minidom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

from ..core import CATEGORY
from ..core.common import SCHEDULERS_ANY

#credits to https://github.com/Jordach/comfy-plasma for the initial code, which was modified for this project

EASYDIFFUSION_MAPPING_A = {
	"prompt": "Prompt",
	"negative_prompt": "Negative Prompt",
	"seed": "Seed",
	"use_stable_diffusion_model": "Stable Diffusion model",
	"clip_skip": "Clip Skip",
	"use_vae_model": "VAE model",
	"sampler_name": "Sampler",
	"width": "Width",
	"height": "Height",
	"num_inference_steps": "Steps",
	"guidance_scale": "Guidance Scale",
}

EASYDIFFUSION_MAPPING_B = {
	"prompt": "prompt",
	"negative_prompt": "negative_prompt",
	"seed": "seed",
	"use_stable_diffusion_model": "use_stable_diffusion_model",
	"clip_skip": "clip_skip",
	"use_vae_model": "use_vae_model",
	"sampler_name": "sampler_name",
	"width": "width",
	"height": "height",
	"num_inference_steps": "num_inference_steps",
	"guidance_scale": "guidance_scale",
}

IMV_CIVITAI_SAMPLER_MAP = {
	'Euler a':'euler_ancestral',
	'Euler':'euler',
	'LMS':'lms',
	'Heun':'heun',
	'DPM2':'dpm_2',
	'DPM2 a':'dpm_2_ancestral',
	'DPM++ 2S a':'dpmpp_2s_ancestral',
	'DPM++ 2M':'dpmpp_2m',
	'DPM++ SDE':'dpmpp_sde',
	'DPM++ 2M SDE':'dpmpp_2m_sde',
	'DPM++ 3M SDE':'dpmpp_3m_sde',
	'DPM fast':'dpm_fast',
	'DPM adaptive':'dpm_adaptive',
	'DDIM':'ddim',
	'PLMS':'plms',
	'UniPC':'uni_pc_bh2',
	'UniPC':'uni_pc',
	'LCM':'lcm',
}

INV_CIVITAI_SCHEDULER_MAP = {
	'Karras':'karras',
	'Exponential':'exponential',
	'SGM Uniform':'sgm_uniform',
	'Simple':'simple',
	'DDIM Uniform':'ddim_uniform',
	'Beta':'beta',
	'Linear Quadratic':'linear_quadratic',
	'KL Optimal':'kl_optimal',
	'Simple Test':'simple_test',
}

def handle_auto1111(params):
	if params and "\nSteps:" in params:
		# has a negative:
		if "Negative prompt:" in params:
			prompt_index = [params.index("\nNegative prompt:"), params.index("\nSteps:")]
			neg = params[prompt_index[0] + 1 + len("Negative prompt: "):prompt_index[-1]]
		else:
			index = [params.index("\nSteps:")]
			neg = ""

		pos = params[:prompt_index[0]]
		return pos, neg
	elif params:
		# has a negative:
		if "Negative prompt:" in params:
			prompt_index = [params.index("\nNegative prompt:")]
			neg = params[prompt_index[0] + 1 + len("Negative prompt: "):]
		else:
			index = [len(params)]
			neg = ""
		
		pos = params[:prompt_index[0]]
		return pos, neg
	else:
		return "", ""

def handle_ezdiff(params):
	data = json.loads(params)
	if data.get("prompt"):
		ed = EASYDIFFUSION_MAPPING_B
	else:
		ed = EASYDIFFUSION_MAPPING_A

	pos = data.get(ed["prompt"])
	data.pop(ed["prompt"])
	neg = data.get(ed["negative_prompt"])
	return pos, neg

def handle_invoke_modern(params):
	meta = json.loads(params.get("sd-metadata"))
	img = meta.get("image")
	prompt = img.get("prompt")
	index = [prompt.rfind("["), prompt.rfind("]")]

	# negative
	if -1 not in index:
		pos = prompt[:index[0]]
		neg = prompt[index[0] + 1:index[1]]
		return pos, neg
	else:
		return prompt, ""

def handle_invoke_legacy(params):
	dream = params.get("Dream")
	pi = dream.rfind('"')
	ni = [dream.rfind("["), dream.rfind("]")]

	# has neg
	if -1 not in ni:
		pos = dream[1:ni[0]]
		neg = dream[ni[0] + 1:ni[1]]
		return pos, neg
	else:
		pos = dream[1:pi]
		return pos, ""

def handle_novelai(params):
	pos = params.get("Description")
	comment = params.get("Comment", "{}")
	comment_json = json.loads(comment)  # type: ignore
	neg = comment_json.get("uc")
	return pos, neg

def handle_qdiffusion(params):
	pass

def handle_comfyui(params):
	"""Extract generation data from ComfyUI embedded metadata"""
	gen_data = {}
	
	# Extract parameters (generation settings)
	if "parameters" in params:
		gen_data["parameters"] = params["parameters"]
	
	# Extract workflow data
	if "workflow" in params:
		try:
			gen_data["workflow"] = json.loads(params["workflow"])
		except:
			gen_data["workflow"] = params["workflow"]
	
	# Extract LORA weights
	if "lora_weights" in params:
		try:
			gen_data["lora_weights"] = json.loads(params["lora_weights"])
		except:
			gen_data["lora_weights"] = params["lora_weights"]
	
	# Parse generation parameters string for structured data
	if "parameters" in gen_data:
		params_str = gen_data["parameters"]
		if "Steps:" in params_str:
			# Extract individual parameters
			try:
				# Steps
				if "Steps: " in params_str:
					steps_start = params_str.find("Steps: ") + 7
					steps_end = params_str.find(",", steps_start)
					if steps_end == -1:
						steps_end = params_str.find("\n", steps_start)
					gen_data["steps"] = int(params_str[steps_start:steps_end].strip())
				
				# Sampler
				if "Sampler: " in params_str:
					sampler_start = params_str.find("Sampler: ") + 9
					sampler_end = params_str.find(",", sampler_start)
					if sampler_end == -1:
						sampler_end = params_str.find("\n", sampler_start)
					gen_data["sampler_name"] = params_str[sampler_start:sampler_end].strip()
				
				# Separate scheduler from sampler if embedded
				if gen_data.get("sampler_name"):
					sampler_full = gen_data["sampler_name"]
					# Check if sampler ends with a known scheduler (case-insensitive)
					# Use a lower-cased, word-boundary regex so spacing/case differences
					# don't prevent detection (e.g. "DPM++ 2M SGM Uniform" vs "Uniform").
					schedulers_to_check = set(INV_CIVITAI_SCHEDULER_MAP.keys()) | set(SCHEDULERS_ANY)
					if isinstance(sampler_full, str):
						sampler_full_l = sampler_full.lower()
						for sched in schedulers_to_check:
							try:
								sched_l = str(sched).lower()
								# match scheduler at the end as a whole word (allow optional preceding space)
								if re.search(r"(?:\s|^)" + re.escape(sched_l) + r"\s*$", sampler_full_l):
									# Find the actual scheduler in the original-cased string
									sched_start = sampler_full_l.rfind(sched_l)
									if sched_start >= 0:
										actual_sched = sampler_full[sched_start:]
										gen_data["sampler_name"] = sampler_full[:sched_start].strip()
										gen_data["scheduler"] = actual_sched
										break
							except Exception:
								# Skip any problematic scheduler entries
								continue
				
				# CFG scale
				if "CFG scale: " in params_str:
					cfg_start = params_str.find("CFG scale: ") + 11
					cfg_end = params_str.find(",", cfg_start)
					if cfg_end == -1:
						cfg_end = params_str.find("\n", cfg_start)
					gen_data["cfg_scale"] = float(params_str[cfg_start:cfg_end].strip())
				
				# Seed
				if "Seed: " in params_str:
					seed_start = params_str.find("Seed: ") + 6
					seed_end = params_str.find(",", seed_start)
					if seed_end == -1:
						seed_end = params_str.find("\n", seed_start)
					gen_data["seed"] = int(params_str[seed_start:seed_end].strip())
				
				# Size
				if "Size: " in params_str:
					size_start = params_str.find("Size: ") + 6
					size_end = params_str.find(",", size_start)
					if size_end == -1:
						size_end = params_str.find("\n", size_start)
					size_str = params_str[size_start:size_end].strip()
					if "x" in size_str:
						width, height = size_str.split("x")
						gen_data["width_param"] = int(width.strip())
						gen_data["height_param"] = int(height.strip())
				
				# Model hashes
				if "Hashes: " in params_str:
					hashes_start = params_str.find("Hashes: ") + 8
					hashes_end = params_str.find("}", hashes_start) + 1
					if hashes_end > hashes_start:
						hashes_str = params_str[hashes_start:hashes_end]
						try:
							gen_data["model_hashes"] = json.loads(hashes_str)
						except:
							gen_data["model_hashes"] = hashes_str
				
				# Version
				if "Version: " in params_str:
					version_start = params_str.find("Version: ") + 9
					version_end = params_str.find("\n", version_start)
					if version_end == -1:
						version_end = len(params_str)
					gen_data["version"] = params_str[version_start:version_end].strip()
					
			except Exception as e:
				# If parsing fails, just continue
				pass
	
	# Apply Civitai mappings to convert human names to keys
	if "sampler_name" in gen_data:
		gen_data["sampler_name"] = IMV_CIVITAI_SAMPLER_MAP.get(gen_data["sampler_name"], gen_data["sampler_name"])
	if "scheduler" in gen_data:
		gen_data["scheduler"] = INV_CIVITAI_SCHEDULER_MAP.get(gen_data["scheduler"], gen_data["scheduler"])
	
	return gen_data
	
def handle_drawthings(params):
	try:
		data = minidom.parseString(params.get("XML:com.adobe.xmp"))
		data_json = json.loads(data.getElementsByTagName("exif:UserComment")[0].childNodes[1].childNodes[1].childNodes[0].data)  # type: ignore
	except:
		return "", ""
	else:
		pos = data_json.get("c")
		neg = data_json.get("uc")
		return pos, neg


class RvImage_LoadImagePath_Pipe:
	@classmethod
	def INPUT_TYPES(cls):  
		return {
				"required": 
					{
						"image": ("STRING", {"default": ""})
					}
				}

	CATEGORY = CATEGORY.MAIN.value + CATEGORY.IMAGE.value

	RETURN_TYPES = ("IMAGE", "MASK", "pipe")
	RETURN_NAMES = ("image", "mask", "pipe")
	FUNCTION = "load_image"

	def load_image(self, image):
		# Removes any quotes from Explorer
		image_path = str(image)
		image_path = image_path.replace('"', "")
		if image_path.startswith("http"):
			image_path = re.sub(r'quality=\d+', 'quality=100', image_path)
		i = None
		if image_path.startswith("http"):
			response = requests.get(image_path)
			i = Image.open(BytesIO(response.content)).convert("RGB")
		else:
			i = Image.open(image_path)
		prompt = ""
		negative = ""
		generation_data = "{}"
		width = i.width
		height = i.height
		steps = 0
		sampler = ""
		scheduler = ""
		cfg_scale = 0.0
		seed = 0
		model_hash = ""
		version = ""
		comfyui_processed = False
		
		if i.format == "PNG" or ("parameters" in i.info or "workflow" in i.info or "lora_weights" in i.info):
			if "parameters" in i.info or "workflow" in i.info or "lora_weights" in i.info:
				gen_data = handle_comfyui(i.info)
				if gen_data:
					generation_data = json.dumps(gen_data)
					comfyui_processed = True
					# Extract individual values
					steps = gen_data.get("steps", 0)
					sampler = gen_data.get("sampler_name")
					scheduler = gen_data.get("scheduler")
					cfg_scale = gen_data.get("cfg_scale", 0.0)
					seed = gen_data.get("seed", 0)
					model_hashes = gen_data.get("model_hashes", {})
					model_hash = json.dumps(model_hashes) if isinstance(model_hashes, dict) else str(model_hashes)
					version = gen_data.get("version", "")
					# Extract prompts from ComfyUI parameters if available
					if "parameters" in gen_data:
						params_str = gen_data["parameters"]
						if "Negative prompt:" in params_str:
							# Split positive and negative prompts
							parts = params_str.split("Negative prompt:")
							if len(parts) >= 2:
								prompt = parts[0].strip()
								# Extract negative prompt up to the generation parameters
								neg_part = parts[1]
								if "Steps:" in neg_part:
									negative = neg_part.split("Steps:")[0].strip()
								else:
									negative = neg_part.strip()
						else:
							# No negative prompt, just positive
							if "Steps:" in params_str:
								prompt = params_str.split("Steps:")[0].strip()
							else:
								prompt = params_str.strip()
			
			# auto1111 (only if ComfyUI data not found)
			elif "parameters" in i.info and not comfyui_processed:
				params = i.info.get("parameters")
				prompt, negative = handle_auto1111(params)

			# easy diffusion
			elif "negative_prompt" in i.info or "Negative Prompt" in i.info:
				params = str(i.info).replace("'", '"')
				prompt, negative = handle_ezdiff(params)
			# invokeai modern
			elif "sd-metadata" in i.info:
				prompt, negative = handle_invoke_modern(i.info)
			# legacy invokeai
			elif "Dream" in i.info:
				prompt, negative = handle_invoke_legacy(i.info)
			# novelai
			elif i.info.get("Software") == "NovelAI":
				prompt, negative = handle_novelai(i.info)
			# qdiffusion
			# elif ????:
			# drawthings (iPhone, iPad, macOS)
			elif "XML:com.adobe.xmp" in i.info:
				prompt, negative = handle_drawthings(i.info)
		
		# Removes EXIF rotation and other nonsense
		i = ImageOps.exif_transpose(i)
		image_rgb = i.convert("RGB")
		image_np = np.array(image_rgb).astype(np.float32) / 255.0
		image_tensor = torch.from_numpy(image_np)[None,]
		if 'A' in i.getbands():
			mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
			mask = 1. - torch.from_numpy(mask_np)
		else:
			mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
		
		# Create pipe dict with all extracted data
		# Extract model name from hashes if available
		model_name = ""
		if isinstance(model_hashes, dict):
			for key in model_hashes.keys():
				if key.startswith("Model:"):
					model_name = key.replace("Model:", "", 1)
					break
		
		pipe = {
			#"images": image_tensor,
			#"mask": mask,
			"steps": steps,
			"sampler_name": sampler,
			"scheduler": scheduler,
			"cfg": cfg_scale,
			"seed": seed,
			"width": width,
			"height": height,
			"text_pos": prompt,
			"text_neg": negative,
			"model_name": model_name,
			"path": '',
		}
		
		return (image_tensor, mask, pipe)

	@classmethod
	def IS_CHANGED(s, image):  # type: ignore
		image_path = str(image)
		image_path = image_path.replace('"', "")
		m = hashlib.sha256()
		if not image_path.startswith("http"):
			with open(image_path, 'rb') as f:
				m.update(f.read())
			return m.digest().hex()
		else:
			m.update(image.encode("utf-8"))
			return m.digest().hex()

	@classmethod
	def VALIDATE_INPUTS(s, image):  # type: ignore
		image_path = str(image)
		image_path = image_path.replace('"', "")
		if image_path.startswith("http"):
			return True
		if not os.path.isfile(image_path):
			return "No file found: {}".format(image_path)

		return True

NODE_NAME = 'Load Image from Path (Metadata Pipe) [RvTools]'
NODE_DESC = 'Load Image from Path (Metadata Pipe)'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvImage_LoadImagePath_Pipe
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
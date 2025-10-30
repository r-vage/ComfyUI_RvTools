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
import random
from datetime import datetime
import folder_paths
from ..core import CATEGORY, AnyType, cstr
from ..core.common import RESOLUTION_PRESETS, RESOLUTION_MAP

any = AnyType("*")

MAX_RESOLUTION = 32768

initial_random_state = random.getstate()
random.seed(datetime.now().timestamp())
rvtools_seed_random_state = random.getstate()
random.setstate(initial_random_state)


def new_random_seed():
    """ Gets a new random seed from the rvtools_seed_random_state and resetting the previous state."""
    global rvtools_seed_random_state
    prev_random_state = random.getstate()
    random.setstate(rvtools_seed_random_state)
    seed = random.randint(1, 1125899906842624)
    rvtools_seed_random_state = random.getstate()
    random.setstate(prev_random_state)
    return seed

def format_datetime(datetime_format):
    today = datetime.now()
    try:
        timestamp = today.strftime(datetime_format)
    except Exception:
        timestamp = today.strftime("%Y-%m-%d-%H%M%S")
    return timestamp

def format_date_time(string, position, datetime_format):
    today = datetime.now()
    if position == "prefix":
        return f"{today.strftime(datetime_format)}_{string}"
    if position == "postfix":
        return f"{string}_{today.strftime(datetime_format)}"
    return string

def format_variables(string, input_variables):
    if input_variables is not None and str(input_variables).strip():
        variables = str(input_variables).split(",")
        return string.format(*variables)
    else:
        return string

class RvFolder_SmartFolder:
    # Image resolution presets
    image_resolution = RESOLUTION_PRESETS
    image_resolution_map = RESOLUTION_MAP

    # Video resolution presets
    video_resolution = ["Custom",
                        "480x832",
                        "576x1024",
                        "--- 9:16 ---",
                        "240x426 (240p)",
                        "360x640 (360p)",
                        "480x853 (SD)",
                        "720x1280 (HD)",
                        "1080x1920 (FullHD)",
                        "1440x2560 (2K)",
                        "2160x3840 (4K)",
                        "4320x7680 (8K)",
                        "--- 16:9 ---",
                        "832x480",
                        "1024x576",
                        "426x240 (240p)",
                        "640x360 (360p)",
                        "853x480 (SD)",
                        "1280x720 (HD)",
                        "1920x1080 (FullHD)",
                        "2560x1440 (2K)",
                        "3840x2160 (4K)",
                        "7680x4320 (8K)",
                        ]

    @classmethod
    def IS_CHANGED(cls, seed, generation_mode=None, root_folder_image=None, root_folder_video=None, create_date_time_folder=None, 
                   date_time_format=None, date_time_position=None, create_batch_folder=None, 
                   batch_folder_name=None, batch_number=None, batch_number_control=None,
                   image_size=None, width=None, height=None, video_size=None, 
                   video_width=None, video_height=None, frame_rate=None, frame_load_cap=None, 
                   context_length=None, loop_count=None, overlap=None, skip_first_frames=None, 
                   skip_calculation=None, skip_calculation_control=None, 
                   select_every_nth=None, batch_size=None, prompt=None, extra_pnginfo=None, unique_id=None):
        """Forces a changed state if we happen to get a special seed, as if from the API directly."""
        if seed in (-1, -2, -3):
            # This isn't used, but a different value than previous will force it to be "changed"
            return new_random_seed()
        return seed


    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_mode": (["Image Mode", "Video Mode"], {"default": "Image Mode", "tooltip": "Select generation mode: Image or Video"}),
                "root_folder_image": ("STRING", {"multiline": False, "default": "images"}, "Root folder name for image generation."),
                "root_folder_video": ("STRING", {"multiline": False, "default": "videos"}, "Root folder name for video generation."),
                "create_date_time_folder": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}, "Create date/time subfolder."),
                "date_time_format": ("STRING", {"multiline": False, "default": "%Y-%m-%d"}, "Date/time format for folder naming (strftime syntax)."),
                "date_time_position": (["prefix", "postfix"], {"default": "postfix"}, "Where to add date/time to folder name: prefix, or postfix."),
                "create_batch_folder": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no"}, "Enable batch subfolder configuration."),
                "batch_folder_name": ("STRING", {"multiline": False, "default": "batch_{}"}, "Batch subfolder name. Supports variable formatting (e.g. batch_{})."),
                "batch_number": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "tooltip": "Batch number to use in batch folder name."}),
                "batch_number_control": (["fixed", "increment"], {"default": "fixed", "tooltip": "Control batch number behavior: fixed or increment after each queue."}),

                # Image-specific parameters
                "image_size": (cls.image_resolution, {}, "Image size preset."),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}, "Image width in pixels."),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}, "Image height in pixels."),

                # Video-specific parameters
                "video_size": (cls.video_resolution, {}, "Video size preset."),
                "video_width": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 1}, "Video width in pixels."),
                "video_height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 1}, "Video height in pixels."),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 8, "max": 240}, "Video frame rate (frames per second)."),
                "frame_load_cap": ("INT", {"default": 162, "min": 0, "max": MAX_RESOLUTION, "step": 1}, "Maximum frames to load per batch. Set 0 for no limit."),
                "context_length": ("INT", {"default": 81, "min": 1, "max": MAX_RESOLUTION, "step": 1}, "Context length for WAN models."),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}, "Calculates the frame_load_cap by using the context length * loop count. this overrides the frame_load_cap value if > 0."),
                "overlap": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}, "Overlap Frames between two clips."),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 4096}, "Number of initial frames to skip."),
                "skip_calculation": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Additional skip first frames calculation: skip (context_length * value)."}),
                "skip_calculation_control": (["fixed", "increment"], {"default": "fixed", "tooltip": "Control skip calculation behavior: fixed or increment after each queue."}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 100}, "Select every nth frame from input."),

                # Common parameters
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}, "Batch size (number of items per batch)."),
                "seed": ("INT", {"default": 0, "min": -1125899906842624, "max": 1125899906842624, "tooltip": "Random seed for folder naming. Use -1 for random, -2 to increment, -3 to decrement."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    CATEGORY = CATEGORY.MAIN.value + CATEGORY.FOLDER.value
    RETURN_TYPES = ("pipe",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "execute"

    def execute(self, generation_mode, root_folder_image, root_folder_video, create_date_time_folder, date_time_format, 
                date_time_position, create_batch_folder, batch_folder_name, batch_number, 
                batch_number_control, image_size, width, height, video_size, video_width, 
                video_height, frame_rate, frame_load_cap, context_length, loop_count, overlap, 
                skip_first_frames, skip_calculation, skip_calculation_control, 
                select_every_nth, batch_size, seed=0, prompt=None, extra_pnginfo=None, unique_id=None):

        # Type safety: ensure valid strings and numbers
        if not isinstance(root_folder_image, str) or not root_folder_image:
            root_folder_image = "images"
        if not isinstance(root_folder_video, str) or not root_folder_video:
            root_folder_video = "videos"
        if not isinstance(date_time_format, str) or not date_time_format:
            date_time_format = "%Y-%m-%d"
        if not isinstance(batch_folder_name, str) or not batch_folder_name:
            batch_folder_name = "batch_{}"

        # Validate numeric inputs
        if not isinstance(width, int) or width < 16:
            width = 512
        if not isinstance(height, int) or height < 16:
            height = 512
        if not isinstance(video_width, int) or video_width < 16:
            video_width = 576
        if not isinstance(video_height, int) or video_height < 16:
            video_height = 1024
        if not isinstance(frame_rate, (int, float)) or frame_rate < 8:
            frame_rate = 30.0
        if not isinstance(frame_load_cap, int):
            frame_load_cap = 162
        if not isinstance(context_length, int):
            context_length = 81
        if not isinstance(loop_count, int):
            loop_count = 0
        if not isinstance(overlap, int):
            overlap = 0
        if not isinstance(skip_first_frames, int):
            skip_first_frames = 0
        if not isinstance(skip_calculation, int):
            skip_calculation = 0
        if not isinstance(select_every_nth, int) or select_every_nth < 1:
            select_every_nth = 1
        if not isinstance(batch_size, int) or batch_size < 1:
            batch_size = 1
        if not isinstance(batch_number, int) or batch_number < 1:
            batch_number = 1
        
        # Ensure control values are valid
        if batch_number_control not in ["fixed", "increment"]:
            batch_number_control = "fixed"
        if skip_calculation_control not in ["fixed", "increment"]:
            skip_calculation_control = "fixed"

        # Format datetime
        mDate = format_datetime(date_time_format)
        root_folder = root_folder_image if generation_mode == "Image Mode" else root_folder_video
        new_path = root_folder

        # Add date/time prefix/postfix if configured
        if create_date_time_folder:
            if date_time_position == "prefix":
                new_path = os.path.join(mDate, root_folder)
            elif date_time_position == "postfix":
                new_path = os.path.join(root_folder, mDate)

        # Create batch folder if configured
        if create_batch_folder:
            folder_name_parsed = format_variables(batch_folder_name, batch_number)
            new_path = os.path.join(new_path, folder_name_parsed)

        # Build output path
        path_out = os.path.join(self.output_dir, new_path)

        # Create pipe based on generation mode
        if generation_mode == "Image Mode":
            # Handle image resolution preset
            if image_size in self.image_resolution_map:
                width, height = self.image_resolution_map[image_size]

            # Build Image pipe (do NOT include video keys)
            pipe = {
                "path": path_out,
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "seed": int(seed),  # Include seed in pipe for downstream use
            }

        else:  # Video
            # Handle video resolution preset
            video_resolution_map = {
                "480x832": (480, 832),
                "576x1024": (576, 1024),
                "240x426 (240p)": (240, 426),
                "360x640 (360p)": (360, 640),
                "480x853 (SD)": (480, 853),
                "720x1280 (HD)": (720, 1280),
                "1080x1920 (FullHD)": (1080, 1920),
                "1440x2560 (2K)": (1440, 2560),
                "2160x3840 (4K)": (2160, 3840),
                "4320x7680 (8K)": (4320, 7680),
                "832x480": (832, 480),
                "1024x576": (1024, 576),
                "426x240 (240p)": (426, 240),
                "640x360 (360p)": (640, 360),
                "853x480 (SD)": (853, 480),
                "1280x720 (HD)": (1280, 720),
                "1920x1080 (FullHD)": (1920, 1080),
                "2560x1440 (2K)": (2560, 1440),
                "3840x2160 (4K)": (3840, 2160),
                "7680x4320 (8K)": (7680, 4320),
            }
            if video_size in video_resolution_map:
                video_width, video_height = video_resolution_map[video_size]

            # Handle loop_count override for frame_load_cap
            if loop_count > 0:
                frame_load_cap = context_length * loop_count

            # Handle skip_calculation
            if skip_calculation > 0:
                try:
                    skip_first_frames += (context_length * skip_calculation)
                except Exception:
                    skip_first_frames = 0

            # Build Video pipe
            pipe = {
                "path": path_out,
                "width": video_width,
                "height": video_height,
                "frame_rate": float(frame_rate),
                "frame_load_cap": int(frame_load_cap),
                "context_length": int(context_length),
                "overlap": int(overlap),
                "skip_first_frames": int(skip_first_frames),
                "select_every_nth": int(select_every_nth),
                "batch_size": int(batch_size),
                "seed": int(seed),  # Include seed in pipe for downstream use
            }

        return (pipe,)


NODE_NAME = 'Smart Folder [RvTools]'
NODE_DESC = 'Smart Folder'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvFolder_SmartFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
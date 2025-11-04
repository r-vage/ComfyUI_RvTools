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
import cv2
import numpy as np
import torch

from typing import Optional
from ..core import CATEGORY, cstr

FPS = float(30.0)

class Eclipse_VideoClips_SeamlessJoin:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_videos"

    def __init__(self):
        self.video_first = ""
        self.video_second = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_load_cap": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1, "display": "number", "tooltip": "Total number of frames to load from each video."}),
                "mask_first_frames": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1, "display": "number", "tooltip": "Number of mask frames to add at the start of the transition."}),
                "mask_last_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "display": "number", "tooltip": "Number of mask frames to add at the end of the transition."}),
            },
            "optional": {
                "video_filelist": ("STRING", {"default": "", "multiline": False, "display": "text", "tooltip": "Comma-separated list of video file paths."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def load_video_frames(self, video_path: str, max_frames: Optional[int] = None) -> list[np.ndarray]:
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cstr(f"Video {video_path}: {total_frames} frames, {fps} fps").msg.print()
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        cap.release()
        if not frames:
            raise ValueError(f"No frames could be loaded from video: {video_path}")
        cstr(f"Successfully loaded {len(frames)} frames from {video_path}").msg.print()
        return frames

    def create_solid_color_image(self, reference_frame: np.ndarray, color_hex: str) -> np.ndarray:
        height, width = reference_frame.shape[:2]
        color_hex = color_hex.lstrip('#')
        r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        solid_image = np.full((height, width, 3), [r, g, b], dtype=np.uint8)
        return solid_image

    def frames_to_tensor(self, frames_list: list[np.ndarray]) -> torch.Tensor:
        if not frames_list:
            raise ValueError("Empty frames list provided")
        tensor_frames = [(frame.astype(np.float32) / 255.0) for frame in frames_list]
        tensor_output = torch.from_numpy(np.stack(tensor_frames, axis=0))
        return tensor_output

    def process_videos(
        self,
        frame_load_cap: int,
        mask_first_frames: int,
        mask_last_frames: int,
        video_filelist: Optional[str] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        videos = None
        if video_filelist not in (None, '', 'undefined', 'none'):
            videos = str(video_filelist).split(', ')
        cstr(f"Starting process with parameters:").msg.print()
        cstr(f"mask_last_frames: {mask_last_frames}").msg.print()
        cstr(f"mask_first_frames: {mask_first_frames}").msg.print()
        cstr(f"frame_load_cap: {frame_load_cap}").msg.print()
        if not videos:
            raise ValueError("No valid video files provided. Please specify video_filelist with comma-separated video paths.")
        
        # Main processing logic
        video_first = str(videos[0]).strip()
        video_second = str(videos[-1]).strip()
        cstr(f"video_first: {video_first}").msg.print()
        cstr(f"video_second: {video_second}").msg.print()
        if not os.path.exists(video_first):
            raise ValueError(f"First video file not found: {video_first}")
        if not os.path.exists(video_second):
            raise ValueError(f"Last video file not found: {video_second}")
        cstr(f"Both video files found, loading frames...").msg.print()
        try:
            first_images_list = self.load_video_frames(video_first, frame_load_cap * 2)
            second_images_list = self.load_video_frames(video_second, frame_load_cap * 2)
            cstr(f"Loaded {len(first_images_list)} frames from first video").msg.print()
            cstr(f"Loaded {len(second_images_list)} frames from second video").msg.print()
        except Exception as e:
            cstr(f"Error loading video frames: {str(e)}").error.print()
            raise ValueError(f"Error loading video frames: {str(e)}")
        if not first_images_list or not second_images_list:
            raise ValueError("Could not load frames from one or both videos")
        reference_frame = first_images_list[0]
        output_images_list: list[np.ndarray] = []
        first_images_start_index = frame_load_cap // 2
        first_images_end_index = frame_load_cap - mask_last_frames
        first_images_start_index = max(0, min(first_images_start_index, len(first_images_list)))
        first_images_end_index = max(first_images_start_index, min(first_images_end_index, len(first_images_list)))
        for idx in range(first_images_start_index, first_images_end_index):
            if idx < len(first_images_list):
                output_images_list.append(first_images_list[idx])
        total_mask_count = mask_last_frames + mask_first_frames
        grey_image = self.create_solid_color_image(reference_frame, "#7F7F7F")
        for _ in range(total_mask_count):
            output_images_list.append(grey_image.copy())
        second_images_start_index = mask_first_frames
        second_images_end_index = frame_load_cap // 2
        second_images_start_index = max(0, min(second_images_start_index, len(second_images_list)))
        second_images_end_index = max(second_images_start_index, min(second_images_end_index, len(second_images_list)))
        for idx in range(second_images_start_index, second_images_end_index):
            if idx < len(second_images_list):
                output_images_list.append(second_images_list[idx])
        output_mask_list: list[np.ndarray] = []
        first_mask_start_index = frame_load_cap // 2
        first_mask_end_index = frame_load_cap - mask_last_frames
        black_image = self.create_solid_color_image(reference_frame, "#000000")
        white_image = self.create_solid_color_image(reference_frame, "#FFFFFF")
        first_mask_count = first_mask_end_index - first_mask_start_index
        first_mask_count = max(0, first_mask_count)
        for _ in range(first_mask_count):
            output_mask_list.append(black_image.copy())
        for _ in range(total_mask_count):
            output_mask_list.append(white_image.copy())
        second_mask_start_index = mask_first_frames
        second_mask_end_index = frame_load_cap // 2
        second_mask_count = second_mask_end_index - second_mask_start_index
        second_mask_count = max(0, second_mask_count)
        for _ in range(second_mask_count):
            output_mask_list.append(black_image.copy())
        if not output_images_list:
            raise ValueError("No output images generated")
        if not output_mask_list:
            raise ValueError("No output masks generated")
        cstr(f"[WanVideo] Generated {len(output_images_list)} output images").msg.print()
        cstr(f"[WanVideo] Generated {len(output_mask_list)} output masks").msg.print()
        try:
            image_tensor = self.frames_to_tensor(output_images_list)
            mask_tensor = self.frames_to_tensor(output_mask_list)
            cstr(f"[WanVideo] Image tensor shape: {image_tensor.shape}").msg.print()
            cstr(f"[WanVideo] Mask tensor shape: {mask_tensor.shape}").msg.print()
            cstr(f"[WanVideo] Processing completed successfully").msg.print()
            return (image_tensor, mask_tensor)
        except Exception as e:
            cstr(f"[WanVideo] Error creating tensors: {str(e)}").error.print()
            raise ValueError(f"Error creating output tensors: {str(e)}")

NODE_NAME = 'Seamless Join Video Clips [Eclipse]'
NODE_DESC = 'Seamless Join Video Clips'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: Eclipse_VideoClips_SeamlessJoin
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
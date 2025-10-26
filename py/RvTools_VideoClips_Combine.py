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

class RvTools_VideoClips_Combine:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "combine_videos"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_load_cap": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 1, "display": "number", "tooltip": "Total number of frames to load from each video."}),
                "simple_combine": ("BOOLEAN", {"default": False, "tooltip": "If True, combines only the video files (ignores join files)."}),
            },
            "optional": {
                "video_filelist": ("STRING", {"default": "", "multiline": False, "display": "text", "tooltip": "Comma-separated list of video file paths."}),
                "joined_filelist": ("STRING", {"default": "", "multiline": False, "display": "text", "tooltip": "Comma-separated list of join file paths."}),
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

    def frames_to_tensor(self, frames_list: list[np.ndarray]) -> torch.Tensor:
        if not frames_list:
            raise ValueError("Empty frames list provided")
        tensor_frames = [(frame.astype(np.float32) / 255.0) for frame in frames_list]
        tensor_output = torch.from_numpy(np.stack(tensor_frames, axis=0))
        return tensor_output

    def combine_videos(
        self,
        frame_load_cap: int,
        simple_combine: bool,
        video_filelist: Optional[str] = None,
        joined_filelist: Optional[str] = None
    ) -> tuple[torch.Tensor, float]:
        videos = None
        joined = None
        if video_filelist not in (None, '', 'undefined', 'none'):
            videos = str(video_filelist).split(', ')
        if joined_filelist not in (None, '', 'undefined', 'none'):
            joined = str(joined_filelist).split(', ')

        output_images_list: list[np.ndarray] = []
        video_1_start_idx = 0
        video_1_end_idx = 0

        if videos and not simple_combine:
            last_was_join = False
            for i in range(len(videos)):
                video_1_list = []
                video_join_list = []
                video_1_exists = False
                join_exists = False
                video_1 = str(videos[i]).strip()
                video_1_exists = os.path.exists(video_1)
                if last_was_join:
                    if joined:
                        video_join = str(joined[i]).strip()
                    join_exists = os.path.exists(video_join)
                    if join_exists:
                        video_join_list.extend(self.load_video_frames(video_join))
                        if video_join_list:
                            output_images_list.extend(video_join_list)
                    else:
                        last_was_join = False
                        if video_1_exists:
                            video_1_list = self.load_video_frames(video_1, frame_load_cap)
                        if video_1_list:
                            video_1_start_idx = frame_load_cap // 2
                            video_1_start_idx = min(video_1_start_idx, len(video_1_list))
                            video_1_end_idx = frame_load_cap
                            video_1_end_idx = min(video_1_end_idx, len(video_1_list))
                            cstr(f"Adding Frames video_1 [{video_1_start_idx}:{video_1_end_idx}]").msg.print()
                            for idx in range(video_1_start_idx, video_1_end_idx):
                                if idx < len(video_1_list):
                                    output_images_list.append(video_1_list[idx])
                else:
                    if video_1_exists:
                        video_1_list = self.load_video_frames(video_1, frame_load_cap)
                    if joined:
                        video_join = str(joined[i]).strip()
                    join_exists = os.path.exists(video_join)
                    if join_exists:
                        video_join_list.extend(self.load_video_frames(video_join))
                        if video_1_list:
                            video_1_start_idx = 0
                            video_1_end_idx = frame_load_cap // 2
                            video_1_end_idx = min(video_1_end_idx, len(video_1_list))
                            cstr(f"Adding Frames video_1 [{video_1_start_idx}:{video_1_end_idx}]").msg.print()
                            for idx in range(video_1_start_idx, video_1_end_idx):
                                if idx < len(video_1_list):
                                    output_images_list.append(video_1_list[idx])
                        if video_join_list:
                            cstr(f"Adding Frames video_join: {len(video_join_list)}").msg.print()
                            output_images_list.extend(video_join_list)
                            last_was_join = True
                    else:
                        if video_1_list:
                            video_1_start_idx = 0
                            video_1_end_idx = frame_load_cap
                            video_1_end_idx = min(video_1_end_idx, len(video_1_list))
                            cstr(f"Adding Frames video_1 [{video_1_start_idx}:{video_1_end_idx}]").msg.print()
                            for idx in range(video_1_start_idx, video_1_end_idx):
                                if idx < len(video_1_list):
                                    output_images_list.append(video_1_list[idx])
        elif videos and simple_combine:
            for i in range(len(videos)):
                video = str(videos[i]).strip()
                if os.path.exists(video):
                    try:
                        output_images_list.extend(self.load_video_frames(video))
                    except Exception as e:
                        cstr(f"Error loading video frames: {str(e)}").error.print()
                        raise ValueError(f"Error loading video frames: {str(e)}")
        if not output_images_list:
            raise ValueError("No output images generated")
        cstr(f"Generated {len(output_images_list)} total output images").msg.print()
        try:
            image_tensor = self.frames_to_tensor(output_images_list)
            cstr(f"Image tensor shape: {image_tensor.shape}").msg.print()
            cstr(f"Video combination completed successfully").msg.print()
            return (image_tensor, FPS)
        except Exception as e:
            cstr(f"Error creating tensor: {str(e)}").msg.print()
            raise ValueError(f"Error creating output tensor: {str(e)}")

NODE_NAME = 'Combine Video Clips [RvTools]'
NODE_DESC = 'Combine Video Clips'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvTools_VideoClips_Combine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}

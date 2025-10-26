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
# Seed functionality adapted from rgthree

import random
from datetime import datetime
from ..core import CATEGORY, SAMPLERS_COMFY, SCHEDULERS_ANY, cstr
from typing import Any, Dict, Tuple

# Some extension must be setting a seed as server-generated seeds were not random. We'll set a new
# seed and use that state going forward.
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

class RvSettings_Sampler_Settings_Seed:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.SETTINGS.value
    RETURN_TYPES = ("pipe",)
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls, allow_overwrite=None, sampler_name=None, scheduler=None, steps=None, cfg=None, guidance=None, denoise=None, seed=None, prompt=None, extra_pnginfo=None, unique_id=None):
        """Forces a changed state if we happen to get a special seed, as if from the API directly."""
        if seed in (-1, -2, -3):
            # This isn't used, but a different value than previous will force it to be "changed"
            return new_random_seed()
        return seed

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "allow_overwrite": ("BOOLEAN", {"default": True, "tooltip": "When enabled, allows direct inputs to IO nodes to overwrite this node's values."}),
                "sampler_name": (SAMPLERS_COMFY, {"tooltip": "Select the sampler algorithm."}),
                "scheduler": (SCHEDULERS_ANY, {"tooltip": "Select the scheduler algorithm."}),
                "steps": ("INT", {"default": 20, "min": 1, "step": 1, "tooltip": "Number of sampling steps."}),
                "cfg": ("FLOAT", {"default": 3.50, "min": 0, "step": 0.1, "tooltip": "Classifier-Free Guidance scale."}),
                "guidance": ("FLOAT", {"default": 3.50, "min": 0, "step": 0.1, "tooltip": "Flux guidance scale."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1, "tooltip": "Denoise strength (0-1)."}),
                "seed": ("INT", {"default": 0, "min": -1125899906842624, "max": 1125899906842624, "tooltip": "Random seed for generation."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    def execute(self, allow_overwrite: bool, sampler_name: Any, scheduler: Any, steps: int, cfg: float, guidance: float, denoise: float, seed: int = 0, prompt=None, extra_pnginfo=None, unique_id=None) -> Tuple:
        
        # Note: allow_overwrite is stored in the pipe dict, not used for enable/disable
        # The pipe will always be returned with the _allow_overwrite flag
        
        # We generate random seeds on the frontend in the seed node before sending the workflow in for
        # many reasons. However, if we want to use this in an API call without changing the seed before
        # sending, then users _could_ pass in "-1" and get a random seed used and added to the metadata.
        # Though, this should likely be discouraged for several reasons (thus, a lot of logging).
        if seed in (-1, -2, -3):
            cstr(f'Got "{seed}" as passed seed. ' +
                  'This shouldn\'t happen when queueing from the ComfyUI frontend.').warning.print()
            if seed in (-2, -3):
                cstr(f'Cannot {"increment" if seed == -2 else "decrement"} seed from ' +
                     'server, but will generate a new random seed.').warning.print()

            original_seed = seed
            seed = new_random_seed()
            cstr(f'Server-generated random seed {seed} and saving to workflow.').msg.print()
            cstr(f'NOTE: Re-queues passing in "{seed}" and server-generated random seed won\'t be cached.').warning.print()

            if unique_id is None:
                cstr('RvSettings_Sampler_Settings', 'Cannot save server-generated seed to image metadata because ' +
                     'the node\'s id was not provided.').warning.print()
            else:
                if extra_pnginfo is None:
                    cstr('RvSettings_Sampler_Settings', 'Cannot save server-generated seed to image workflow ' +
                         'metadata because workflow was not provided.').warning.print()
                else:
                    workflow_node = next(
                        (x for x in extra_pnginfo['workflow']['nodes'] if str(x['id']) == str(unique_id)), None)
                    if workflow_node is None or 'widgets_values' not in workflow_node:
                        cstr('RvSettings_Sampler_Settings', 'Cannot save server-generated seed to image workflow ' +
                             'metadata because node was not found in the provided workflow.').warning.print()
                    else:
                        for index, widget_value in enumerate(workflow_node['widgets_values']):
                            if widget_value == original_seed:
                                workflow_node['widgets_values'][index] = seed

                if prompt is None:
                    cstr('RvSettings_Sampler_Settings', 'Cannot save server-generated seed to image API prompt ' +
                         'metadata because prompt was not provided.').warning.print()
                else:
                    prompt_node = prompt[str(unique_id)]
                    if prompt_node is None or 'inputs' not in prompt_node or 'seed' not in prompt_node['inputs']:
                        cstr('RvSettings_Sampler_Settings', 'Cannot save server-generated seed to image API prompt ' +
                             'metadata because node was not found in the provided prompt.').warning.print()
                    else:
                        prompt_node['inputs']['seed'] = seed

        pipe = {
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "steps": int(steps),
            "cfg": float(cfg),
            "guidance": float(guidance),
            "denoise": float(denoise),
            "seed": int(seed),
            "_allow_overwrite": allow_overwrite,  # Flag for IO nodes
        }
        return (pipe,)

NODE_NAME = 'Sampler Settings+Seed [RvTools]'
NODE_DESC = 'Sampler Settings+Seed'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvSettings_Sampler_Settings_Seed
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}

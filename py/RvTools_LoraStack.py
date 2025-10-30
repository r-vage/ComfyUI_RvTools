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

import folder_paths
from ..core import CATEGORY, AnyType

any = AnyType("*")

class RvTools_LoraStack:
    # A node to stack multiple LoRAs with weights and options.
    @classmethod
    def INPUT_TYPES(cls):
    
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        return {"required": {
                    "simple": ("BOOLEAN", {"default": False, "label_on": "yes", "label_off": "no", "tooltip": "When enabled, hides clip_weight widgets and uses model_weight for both"}),
                    "lora_count": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "Number of visible LoRA slots"}),
                    
                    "switch_1": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_1": (loras,),
                    "model_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_2": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_2": (loras,),
                    "model_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_3": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_3": (loras,),
                    "model_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_4": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_4": (loras,),
                    "model_weight_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_5": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_5": (loras,),
                    "model_weight_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_6": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_6": (loras,),
                    "model_weight_6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_7": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_7": (loras,),
                    "model_weight_7": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_7": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_8": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_8": (loras,),
                    "model_weight_8": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_8": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_9": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_9": (loras,),
                    "model_weight_9": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_9": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "switch_10": ("BOOLEAN", {"default": False, "label_on": "on", "label_off": "off"}),
                    "lora_name_10": (loras,),
                    "model_weight_10": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                    "clip_weight_10": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                },
                "optional": {"lora_stack": ("LORA_STACK",)
                },
        }

    RETURN_TYPES = ("LORA_STACK", )
    RETURN_NAMES = ("LORA_STACK", )
    FUNCTION = "execute"
    
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TOOLS.value

    def execute(self, simple, lora_count,
                      lora_name_1, model_weight_1, clip_weight_1, switch_1, 
                      lora_name_2, model_weight_2, clip_weight_2, switch_2, 
                      lora_name_3, model_weight_3, clip_weight_3, switch_3, 
                      lora_name_4, model_weight_4, clip_weight_4, switch_4, 
                      lora_name_5, model_weight_5, clip_weight_5, switch_5,
                      lora_name_6, model_weight_6, clip_weight_6, switch_6,
                      lora_name_7, model_weight_7, clip_weight_7, switch_7,
                      lora_name_8, model_weight_8, clip_weight_8, switch_8,
                      lora_name_9, model_weight_9, clip_weight_9, switch_9,
                      lora_name_10, model_weight_10, clip_weight_10, switch_10,
                      lora_stack=None):

        # Initialise the list
        lora_list=list()
        
        if lora_stack is not None:
            lora_list.extend([l for l in lora_stack if l[0] != "None"])
        
        # When simple mode is enabled, use model_weight for both model and clip
        # Otherwise use the provided clip_weight values
        def get_clip_weight(model_weight, clip_weight):
            return model_weight if simple else clip_weight
        
        if lora_name_1 != "None" and switch_1:
            lora_list.extend([(lora_name_1, model_weight_1, get_clip_weight(model_weight_1, clip_weight_1))])

        if lora_name_2 != "None" and switch_2:
            lora_list.extend([(lora_name_2, model_weight_2, get_clip_weight(model_weight_2, clip_weight_2))])

        if lora_name_3 != "None" and switch_3:
            lora_list.extend([(lora_name_3, model_weight_3, get_clip_weight(model_weight_3, clip_weight_3))])

        if lora_name_4 != "None" and switch_4:
            lora_list.extend([(lora_name_4, model_weight_4, get_clip_weight(model_weight_4, clip_weight_4))])

        if lora_name_5 != "None" and switch_5:
            lora_list.extend([(lora_name_5, model_weight_5, get_clip_weight(model_weight_5, clip_weight_5))])

        if lora_name_6 != "None" and switch_6:
            lora_list.extend([(lora_name_6, model_weight_6, get_clip_weight(model_weight_6, clip_weight_6))])

        if lora_name_7 != "None" and switch_7:
            lora_list.extend([(lora_name_7, model_weight_7, get_clip_weight(model_weight_7, clip_weight_7))])

        if lora_name_8 != "None" and switch_8:
            lora_list.extend([(lora_name_8, model_weight_8, get_clip_weight(model_weight_8, clip_weight_8))])

        if lora_name_9 != "None" and switch_9:
            lora_list.extend([(lora_name_9, model_weight_9, get_clip_weight(model_weight_9, clip_weight_9))])

        if lora_name_10 != "None" and switch_10:
            lora_list.extend([(lora_name_10, model_weight_10, get_clip_weight(model_weight_10, clip_weight_10))])

        return (lora_list,)


NODE_NAME = 'Lora Stack [RvTools]'
NODE_DESC = 'Lora Stack'

NODE_CLASS_MAPPINGS = {
   NODE_NAME: RvTools_LoraStack
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}
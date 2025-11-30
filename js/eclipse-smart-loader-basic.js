/** Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Dynamic widget visibility for Smart Loader
* Adds LoRA configuration management with dynamic slot visibility
*/

import { app, api } from './comfy/index.js';

const NODE_NAME = "Smart Loader Basic [Eclipse]";

app.registerExtension({
    name: "Eclipse.SmartLoaderBasic",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            const node = this;
            
            // Helper function to get widget value
            const getWidgetValue = (widgetName) => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                return widget ? widget.value : undefined;
            };
            
            const setWidgetValue = (widgetName, value) => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (!widget) return;
                
                if (widget.value !== value) {
                    widget.value = value;
                    if (widget.callback) {
                        widget.callback(value);
                    }
                }
            };
            
            const setWidgetVisible = (widgetName, visible) => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (!widget) return;
                
                if (visible) {
                    if (widget.origType) {
                        widget.type = widget.origType;
                    } else if (widget.type === "converted-widget") {
                        widget.type = "combo";
                        widget.origType = "combo";
                    }
                    delete widget.computeSize;
                    widget.hidden = false;
                } else {
                    if (widget.type !== "converted-widget" && !widget.origType) {
                        widget.origType = widget.type;
                    }
                    widget.type = "converted-widget";
                    widget.computeSize = () => [0, -4];
                    widget.hidden = true;
                }
            };
            
            // Store original unfiltered CLIP options
            const originalClipOptions = {};
            
            // Store original unfiltered model options
            const originalModelOptions = {};
            
            // Filter model widget options based on model type
            const filterModelOptions = () => {
                const modelType = getWidgetValue("model_type");
                
                // Define which models to filter and their file extensions
                const modelWidgets = {
                    "ckpt_name": {
                        show: modelType === "Standard Checkpoint",
                        extensions: ['.safetensors', '.ckpt', '.pt', '.bin', '.sft']
                    },
                    "unet_name": {
                        show: modelType === "UNet Model",
                        extensions: ['.safetensors', '.pt', '.bin', '.sft']
                    },
                    "gguf_name": {
                        show: modelType === "GGUF Model",
                        extensions: ['.gguf']
                    }
                };
                
                Object.entries(modelWidgets).forEach(([widgetName, config]) => {
                    const widget = node.widgets?.find(w => w.name === widgetName);
                    if (!widget || !widget.options) return;
                    
                    // Store original options on first run
                    if (!originalModelOptions[widgetName]) {
                        originalModelOptions[widgetName] = [...widget.options.values];
                    }
                    
                    // Get the full unfiltered list
                    const allOptions = originalModelOptions[widgetName];
                    
                    // Filter based on allowed extensions for this widget
                    const filteredOptions = allOptions.filter(name => {
                        if (name === "None") return true;
                        const nameLower = name.toLowerCase();
                        return config.extensions.some(ext => nameLower.endsWith(ext));
                    });
                    
                    // Update widget options
                    widget.options.values = filteredOptions;
                    
                    // If current value is filtered out, reset to "None"
                    if (!filteredOptions.includes(widget.value)) {
                        widget.value = "None";
                    }
                });
            };
            
            // CLIP widget options - no filtering applied
            // All CLIP files (.gguf, .safetensors, etc.) can be used with any model type
            const filterClipOptions = () => {
                const clipWidgets = ["clip_name1", "clip_name2", "clip_name3", "clip_name4"];
                
                clipWidgets.forEach(widgetName => {
                    const widget = node.widgets?.find(w => w.name === widgetName);
                    if (!widget || !widget.options) return;
                    
                    // Store original options on first run
                    if (!originalClipOptions[widgetName]) {
                        originalClipOptions[widgetName] = [...widget.options.values];
                    }
                    
                    // Always show all CLIP files - no filtering by model type
                    widget.options.values = originalClipOptions[widgetName];
                });
            };
            
            const updateVisibility = () => {
                const modelType = getWidgetValue("model_type");
                const configureClip = getWidgetValue("configure_clip");
                const configureVae = getWidgetValue("configure_vae");
                const configureLora = getWidgetValue("configure_model_only_lora");
                const clipSource = getWidgetValue("clip_source");
                const clipCount = parseInt(getWidgetValue("clip_count")) || 1;
                const vaeSource = getWidgetValue("vae_source");
                const loraCount = parseInt(getWidgetValue("lora_count")) || 3;
                
                const isStandard = (modelType === "Standard Checkpoint");
                const isUNet = (modelType === "UNet Model");
                const isGGUF = (modelType === "GGUF Model");
                const useExternalClip = (clipSource === "External");
                const useExternalVae = (vaeSource === "External");
                
                // Filter model and CLIP options based on model type
                filterModelOptions();
                filterClipOptions();
                
                // Model Selection
                setWidgetVisible("ckpt_name", isStandard);
                setWidgetVisible("unet_name", isUNet);
                setWidgetVisible("gguf_name", isGGUF);
                setWidgetVisible("weight_dtype", isUNet);
                
                // GGUF Options
                setWidgetVisible("gguf_dequant_dtype", isGGUF);
                setWidgetVisible("gguf_patch_dtype", isGGUF);
                setWidgetVisible("gguf_patch_on_device", isGGUF);
                
                // Device Selection
                setWidgetVisible("model_device", true); // Always visible
                setWidgetVisible("clip_device", configureClip);
                setWidgetVisible("vae_device", configureVae);
                
                // CLIP Configuration
                setWidgetVisible("clip_source", configureClip);
                setWidgetVisible("clip_count", configureClip && useExternalClip);
                setWidgetVisible("clip_name1", configureClip && useExternalClip && clipCount >= 1);
                setWidgetVisible("clip_name2", configureClip && useExternalClip && clipCount >= 2);
                setWidgetVisible("clip_name3", configureClip && useExternalClip && clipCount >= 3);
                setWidgetVisible("clip_name4", configureClip && useExternalClip && clipCount >= 4);
                setWidgetVisible("clip_type", configureClip && useExternalClip);
                setWidgetVisible("enable_clip_layer", configureClip && isStandard);
                setWidgetVisible("stop_at_clip_layer", configureClip && isStandard);
                
                // VAE Configuration
                setWidgetVisible("vae_source", configureVae);
                setWidgetVisible("vae_name", configureVae && useExternalVae);
                
                // LoRA Configuration
                setWidgetVisible("lora_count", configureLora);
                for (let i = 1; i <= 3; i++) {
                    const showSlot = configureLora && i <= loraCount;
                    setWidgetVisible(`lora_switch_${i}`, showSlot);
                    setWidgetVisible(`lora_name_${i}`, showSlot);
                    setWidgetVisible(`lora_weight_${i}`, showSlot);
                }
                
                // Smart resize
                setTimeout(() => {
                    node.setDirtyCanvas(true, false);
                    
                    const computedSize = node.computeSize();
                    const currentSize = node.size;
                    
                    const minWidth = 259;
                    const minHeight = 100;
                    
                    let newWidth = Math.max(currentSize[0], minWidth);
                    let newHeight = Math.max(computedSize[1], minHeight);
                    
                    newHeight += 5;
                    
                    const heightDiff = Math.abs(currentSize[1] - newHeight);
                    const isGrowing = newHeight > currentSize[1];
                    
                    if (isGrowing || heightDiff > 10) {
                        node.setSize([newWidth, newHeight]);
                    }
                    
                    node.setDirtyCanvas(true, true);
                }, 50);
            };
            
            // Hook into relevant widgets
            const relevantWidgets = [
                "model_type",
                "configure_clip",
                "configure_vae",
                "configure_model_only_lora",
                "clip_source",
                "clip_count",
                "vae_source",
                "lora_count",
            ];
            
            relevantWidgets.forEach(widgetName => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (widget) {
                    const originalCallback = widget.callback;
                    widget.callback = function() {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        updateVisibility();
                    };
                }
            });
            
            // Initial setup
            setTimeout(() => {
                updateVisibility();
            }, 10);
            
            return r;
        };
    },
});

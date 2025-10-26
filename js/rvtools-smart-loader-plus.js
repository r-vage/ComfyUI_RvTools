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
* Dynamic widget visibility for Smart Loader Plus
* Manages widget visibility for different model types (Standard, UNet, Nunchaku Flux, Nunchaku Qwen, GGUF)
*/

import { app } from "../../scripts/app.js";

const NODE_NAME = "Smart Loader Plus [RvTools]";

app.registerExtension({
    name: "RvTools.SmartLoaderPlus",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            const node = this;
            
            // Track previous template_action value to detect changes
            let lastTemplateAction = "None";
            let lastTemplateName = "None";
            
            // Refresh template list from server
            const refreshTemplateList = async () => {
                try {
                    const response = await fetch('/rvtools/loader_templates_list');
                    if (response.ok) {
                        const templates = await response.json();
                        const templateWidget = node.widgets?.find(w => w.name === "template_name");
                        if (templateWidget && templateWidget.options && templateWidget.options.values) {
                            templateWidget.options.values = templates;
                            if (!templates.includes(templateWidget.value)) {
                                templateWidget.value = "None";
                            }
                            node.setDirtyCanvas(true, true);
                        }
                    }
                } catch (e) {
                    console.error('Failed to refresh template list:', e);
                }
            };
            
            // Template action handler
            const handleTemplateAction = async () => {
                const templateAction = getWidgetValue("template_action");
                const templateName = getWidgetValue("template_name");
                const newTemplateName = getWidgetValue("new_template_name");
                
                await refreshTemplateList();
                
                if (templateAction === "Load" && templateName && templateName !== "None") {
                    await applyTemplate(templateName);
                    console.log(`✓ Template loaded: ${templateName}`);
                } else if (templateAction === "Save" && newTemplateName && newTemplateName.trim()) {
                    console.log(`✓ Queueing workflow to save template: ${newTemplateName}`);
                    const savedTemplateName = newTemplateName.trim();
                    // Queue the prompt to execute Python save logic
                    app.queuePrompt(0, 1);
                    // After save, switch to Load and select the saved template
                    setTimeout(async () => {
                        await refreshTemplateList();
                        setWidgetValue("template_action", "Load");
                        setWidgetValue("template_name", savedTemplateName);
                        setWidgetValue("new_template_name", "");
                        updateVisibility();
                        console.log(`✓ Switched to Load mode with template: ${savedTemplateName}`);
                    }, 200);
                } else if (templateAction === "Delete" && templateName && templateName !== "None") {
                    console.log(`✓ Queueing workflow to delete template: ${templateName}`);
                    // Queue the prompt to execute Python delete logic
                    app.queuePrompt(0, 1);
                    // Reset to Load with "None" after delete
                    setTimeout(async () => {
                        await refreshTemplateList();
                        setWidgetValue("template_action", "Load");
                        setWidgetValue("template_name", "None");
                        updateVisibility();
                        console.log(`✓ Template deleted, switched to Load mode`);
                    }, 500);
                }
            };
            
            let templateButton = null;
            
            const updateTemplateButton = () => {
                const templateAction = getWidgetValue("template_action");
                const hasAction = (templateAction !== "None");
                
                if (hasAction && !templateButton) {
                    templateButton = node.addWidget("button", "Execute Template Action", null, handleTemplateAction);
                    templateButton.serialize = false;
                } else if (!hasAction && templateButton) {
                    const buttonIndex = node.widgets.indexOf(templateButton);
                    if (buttonIndex >= 0) {
                        node.widgets.splice(buttonIndex, 1);
                    }
                    templateButton = null;
                }
            };
            
            const setWidgetValue = (widgetName, value) => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (widget && widget.value !== value) {
                    widget.value = value;
                    if (widget.callback) {
                        widget.callback(value);
                    }
                }
            };
            
            const loadTemplateConfig = async (templateName) => {
                if (!templateName || templateName === "None") return null;
                
                try {
                    const response = await fetch(`/rvtools/loader_templates/${templateName}.json`);
                    if (response.ok) {
                        return await response.json();
                    }
                } catch (e) {
                    console.error(`Failed to load template ${templateName}:`, e);
                }
                return null;
            };
            
            const applyTemplate = async (templateName) => {
                const config = await loadTemplateConfig(templateName);
                if (!config) return;
                
                // Apply configuration values
                if (config.model_type !== undefined) setWidgetValue("model_type", config.model_type);
                if (config.weight_dtype !== undefined) setWidgetValue("weight_dtype", config.weight_dtype);
                
                if (config.configure_clip !== undefined) setWidgetValue("configure_clip", config.configure_clip);
                if (config.configure_vae !== undefined) setWidgetValue("configure_vae", config.configure_vae);
                if (config.configure_latent !== undefined) setWidgetValue("configure_latent", config.configure_latent);
                if (config.configure_sampler !== undefined) setWidgetValue("configure_sampler", config.configure_sampler);
                
                // Apply Nunchaku Flux settings
                if (config.data_type !== undefined) setWidgetValue("data_type", config.data_type);
                if (config.cache_threshold !== undefined) setWidgetValue("cache_threshold", config.cache_threshold);
                if (config.attention !== undefined) setWidgetValue("attention", config.attention);
                if (config.i2f_mode !== undefined) setWidgetValue("i2f_mode", config.i2f_mode);
                
                // Apply shared Nunchaku settings
                if (config.cpu_offload !== undefined) setWidgetValue("cpu_offload", config.cpu_offload);
                
                // Apply Nunchaku Qwen settings
                if (config.num_blocks_on_gpu !== undefined) setWidgetValue("num_blocks_on_gpu", config.num_blocks_on_gpu);
                if (config.use_pin_memory !== undefined) setWidgetValue("use_pin_memory", config.use_pin_memory);
                
                // Apply GGUF settings
                if (config.gguf_dequant_dtype !== undefined) setWidgetValue("gguf_dequant_dtype", config.gguf_dequant_dtype);
                if (config.gguf_patch_dtype !== undefined) setWidgetValue("gguf_patch_dtype", config.gguf_patch_dtype);
                if (config.gguf_patch_on_device !== undefined) setWidgetValue("gguf_patch_on_device", config.gguf_patch_on_device);
                
                // Apply Flux guidance
                if (config.flux_guidance !== undefined) setWidgetValue("flux_guidance", config.flux_guidance);
                
                if (config.clip_source !== undefined) setWidgetValue("clip_source", config.clip_source);
                if (config.clip_count !== undefined) setWidgetValue("clip_count", config.clip_count);
                if (config.clip_type !== undefined) setWidgetValue("clip_type", config.clip_type);
                if (config.enable_clip_layer !== undefined) setWidgetValue("enable_clip_layer", config.enable_clip_layer);
                if (config.stop_at_clip_layer !== undefined) setWidgetValue("stop_at_clip_layer", config.stop_at_clip_layer);
                if (config.vae_source !== undefined) setWidgetValue("vae_source", config.vae_source);
                if (config.resolution !== undefined) setWidgetValue("resolution", config.resolution);
                if (config.width !== undefined) setWidgetValue("width", config.width);
                if (config.height !== undefined) setWidgetValue("height", config.height);
                if (config.batch_size !== undefined) setWidgetValue("batch_size", config.batch_size);
                
                // Backward compatibility: old templates had "sampler" instead of "sampler_name"
                if (config.sampler_name !== undefined) {
                    setWidgetValue("sampler_name", config.sampler_name);
                } else if (config.sampler !== undefined) {
                    setWidgetValue("sampler_name", config.sampler);
                }
                
                if (config.scheduler !== undefined) setWidgetValue("scheduler", config.scheduler);
                if (config.steps !== undefined) setWidgetValue("steps", config.steps);
                if (config.cfg !== undefined) setWidgetValue("cfg", config.cfg);
                
                // Apply file selections
                if (config.ckpt_name !== undefined) setWidgetValue("ckpt_name", config.ckpt_name);
                if (config.unet_name !== undefined) setWidgetValue("unet_name", config.unet_name);
                if (config.nunchaku_name !== undefined) setWidgetValue("nunchaku_name", config.nunchaku_name);
                if (config.qwen_name !== undefined) setWidgetValue("qwen_name", config.qwen_name);
                if (config.gguf_name !== undefined) setWidgetValue("gguf_name", config.gguf_name);
                if (config.clip_name1 !== undefined) setWidgetValue("clip_name1", config.clip_name1);
                if (config.clip_name2 !== undefined) setWidgetValue("clip_name2", config.clip_name2);
                if (config.clip_name3 !== undefined) setWidgetValue("clip_name3", config.clip_name3);
                if (config.clip_name4 !== undefined) setWidgetValue("clip_name4", config.clip_name4);
                if (config.vae_name !== undefined) setWidgetValue("vae_name", config.vae_name);
                
                console.log(`✓ Template '${templateName}' applied (with quantized model settings)`);
                updateVisibility();
            };
            
            const setWidgetVisible = (widgetName, visible) => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (!widget) return;
                
                if (visible) {
                    if (widget.origType) {
                        widget.type = widget.origType;
                    } else if (widget.type === "converted-widget") {
                        if (widgetName.includes("width") || widgetName.includes("height") || 
                            widgetName.includes("seed") || widgetName.includes("batch") ||
                            widgetName.includes("threshold") || widgetName.includes("cache")) {
                            widget.type = "number";
                            widget.origType = "number";
                        } else {
                            widget.type = "combo";
                            widget.origType = "combo";
                        }
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
            
            const getWidgetValue = (name) => {
                const widget = node.widgets?.find(w => w.name === name);
                return widget ? widget.value : null;
            };
            
            const updateVisibility = () => {
                const templateAction = getWidgetValue("template_action");
                const modelType = getWidgetValue("model_type");
                const configureClip = getWidgetValue("configure_clip");
                const configureVae = getWidgetValue("configure_vae");
                const configureLatent = getWidgetValue("configure_latent");
                const configureSampler = getWidgetValue("configure_sampler");
                const clipSource = getWidgetValue("clip_source");
                const clipCount = parseInt(getWidgetValue("clip_count")) || 1;
                const vaeSource = getWidgetValue("vae_source");
                const resolution = getWidgetValue("resolution");
                
                const isStandard = (modelType === "Standard Checkpoint");
                const isUNet = (modelType === "UNet Model");
                const isNunchaku = (modelType === "Nunchaku Flux");
                const isQwen = (modelType === "Nunchaku Qwen");
                const isGGUF = (modelType === "GGUF Model");
                const useExternalClip = (clipSource === "External");
                const useExternalVae = (vaeSource === "External");
                const isCustomResolution = (resolution === "Custom");
                
                // Template Management
                const isLoadOrDelete = (templateAction === "Load" || templateAction === "Delete");
                const isSave = (templateAction === "Save");
                setWidgetVisible("template_name", isLoadOrDelete);
                setWidgetVisible("new_template_name", isSave);
                updateTemplateButton();
                
                // Model Selection
                setWidgetVisible("ckpt_name", isStandard);
                setWidgetVisible("unet_name", isUNet);
                setWidgetVisible("nunchaku_name", isNunchaku);
                setWidgetVisible("qwen_name", isQwen);
                setWidgetVisible("gguf_name", isGGUF);
                setWidgetVisible("weight_dtype", isUNet);  // Only for UNet models
                
                // Nunchaku Flux Options
                setWidgetVisible("data_type", isNunchaku);
                setWidgetVisible("cache_threshold", isNunchaku);
                setWidgetVisible("attention", isNunchaku);
                setWidgetVisible("i2f_mode", isNunchaku);
                
                // Shared Nunchaku Options
                setWidgetVisible("cpu_offload", isNunchaku || isQwen);
                
                // Nunchaku Qwen Options
                setWidgetVisible("num_blocks_on_gpu", isQwen);
                setWidgetVisible("use_pin_memory", isQwen);
                
                // GGUF Options
                setWidgetVisible("gguf_dequant_dtype", isGGUF);
                setWidgetVisible("gguf_patch_dtype", isGGUF);
                setWidgetVisible("gguf_patch_on_device", isGGUF);
                
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
                
                // Latent Configuration
                setWidgetVisible("resolution", configureLatent);
                setWidgetVisible("width", configureLatent && isCustomResolution);
                setWidgetVisible("height", configureLatent && isCustomResolution);
                setWidgetVisible("batch_size", configureLatent);
                
                // Sampler Configuration
                setWidgetVisible("sampler_name", configureSampler);
                setWidgetVisible("scheduler", configureSampler);
                setWidgetVisible("steps", configureSampler);
                setWidgetVisible("cfg", configureSampler);
                setWidgetVisible("flux_guidance", configureSampler);
                
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
                "template_action",
                "template_name",
                "model_type",
                "configure_clip",
                "configure_vae",
                "configure_latent",
                "configure_sampler",
                "clip_source",
                "clip_count",
                "vae_source",
                "resolution",
            ];
            
            relevantWidgets.forEach(widgetName => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (widget) {
                    const originalCallback = widget.callback;
                    widget.callback = function() {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        
                        if (widgetName === "template_action" || widgetName === "template_name") {
                            const templateAction = getWidgetValue("template_action");
                            const templateName = getWidgetValue("template_name");
                            
                            if (templateAction === "Load" && templateName && templateName !== "None") {
                                if (templateName !== lastTemplateName || templateAction !== lastTemplateAction) {
                                    applyTemplate(templateName);
                                    lastTemplateName = templateName;
                                    lastTemplateAction = templateAction;
                                }
                            }
                        }
                        
                        updateVisibility();
                    };
                }
            });
            
            // Listen for execution events to refresh template list after save/delete
            const onExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                const templateAction = getWidgetValue("template_action");
                if (templateAction === "Save" || templateAction === "Delete") {
                    // Refresh template list after save/delete
                    setTimeout(() => {
                        refreshTemplateList();
                    }, 100);
                }
            };
            
            // Initial setup
            setTimeout(() => {
                updateVisibility();
                refreshTemplateList();
            }, 10);
            
            return r;
        };
    },
});

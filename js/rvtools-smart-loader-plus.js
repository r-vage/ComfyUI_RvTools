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
* Adds LoRA configuration management with dynamic slot visibility
*/

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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
            
            let lastTemplateAction = "None";
            let lastTemplateName = "None";
            let pendingTemplateSave = null;
            let pendingTemplateDelete = false;
            let isApplyingTemplate = false; // Flag to prevent callbacks during template load
            
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
                    pendingTemplateSave = newTemplateName.trim();
                    app.queuePrompt(0, 1);
                } else if (templateAction === "Delete" && templateName && templateName !== "None") {
                    console.log(`✓ Queueing workflow to delete template: ${templateName}`);
                    pendingTemplateDelete = true;
                    app.queuePrompt(0, 1);
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
                if (!widget) return;
                
                // For BOOLEAN widgets, ensure proper value setting
                if (widget.type === "toggle" || widgetName.includes("_switch_") || widgetName.startsWith("configure_") || widgetName.includes("enable_")) {
                    // Force boolean conversion and update
                    const boolValue = Boolean(value);
                    
                    // When applying template, always set the value to force visual update
                    if (isApplyingTemplate || widget.value !== boolValue) {
                        widget.value = boolValue;
                        // Only trigger callback if not applying template
                        if (widget.callback && !isApplyingTemplate) {
                            widget.callback(boolValue);
                        }
                    }
                } else {
                    // For other widgets, normal assignment
                    if (widget.value !== value) {
                        widget.value = value;
                        // Only trigger callback if not applying template
                        if (widget.callback && !isApplyingTemplate) {
                            widget.callback(value);
                        }
                    }
                }
            };
            
            const loadTemplateConfig = async (templateName) => {
                if (!templateName || templateName === "None") return null;
                
                try {
                    // Add cache-busting parameter to force fresh fetch
                    const cacheBuster = new Date().getTime();
                    const response = await fetch(`/rvtools/loader_templates/${templateName}.json?t=${cacheBuster}`, {
                        cache: 'no-store'
                    });
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
                
                // Set flag to prevent callbacks during template application
                isApplyingTemplate = true;
                
                try {
                    // Reset ALL values to their defaults first to avoid leftover values
                
                // Model selection - reset to defaults
                setWidgetValue("model_type", "Standard Checkpoint");
                setWidgetValue("ckpt_name", "None");
                setWidgetValue("unet_name", "None");
                setWidgetValue("nunchaku_name", "None");
                setWidgetValue("qwen_name", "None");
                setWidgetValue("gguf_name", "None");
                setWidgetValue("weight_dtype", "default");
                
                // Nunchaku settings - reset to defaults
                setWidgetValue("data_type", "bfloat16");
                setWidgetValue("cache_threshold", 0.0);
                setWidgetValue("attention", "flash-attention2");
                setWidgetValue("i2f_mode", "enabled");
                setWidgetValue("cpu_offload", "auto");
                setWidgetValue("num_blocks_on_gpu", 30);
                setWidgetValue("use_pin_memory", "enable");
                
                // GGUF settings - reset to defaults
                setWidgetValue("gguf_dequant_dtype", "default");
                setWidgetValue("gguf_patch_dtype", "default");
                setWidgetValue("gguf_patch_on_device", false);
                
                // Configuration toggles - reset to defaults
                setWidgetValue("configure_clip", true);
                setWidgetValue("configure_vae", true);
                setWidgetValue("configure_latent", false);
                setWidgetValue("configure_sampler", false);
                setWidgetValue("configure_model_only_lora", false);
                
                // CLIP settings - reset to defaults
                setWidgetValue("clip_source", "Baked");
                setWidgetValue("clip_count", "1");
                setWidgetValue("clip_name1", "None");
                setWidgetValue("clip_name2", "None");
                setWidgetValue("clip_name3", "None");
                setWidgetValue("clip_name4", "None");
                setWidgetValue("clip_type", "flux");
                setWidgetValue("enable_clip_layer", true);
                setWidgetValue("stop_at_clip_layer", -2);
                
                // VAE settings - reset to defaults
                setWidgetValue("vae_source", "Baked");
                setWidgetValue("vae_name", "None");
                
                // Latent settings - reset to defaults
                setWidgetValue("resolution", "1024x1024 (1:1)");
                setWidgetValue("width", 1024);
                setWidgetValue("height", 1024);
                setWidgetValue("batch_size", 1);
                
                // LoRA settings - reset to defaults
                setWidgetValue("lora_count", "1");
                for (let i = 1; i <= 3; i++) {
                    setWidgetValue(`lora_switch_${i}`, false);
                    setWidgetValue(`lora_name_${i}`, "None");
                    setWidgetValue(`lora_weight_${i}`, 1.0);
                }
                
                // Sampler settings - reset to defaults
                setWidgetValue("sampler_name", "euler");
                setWidgetValue("scheduler", "normal");
                setWidgetValue("steps", 20);
                setWidgetValue("cfg", 8.0);
                setWidgetValue("flux_guidance", 3.5);
                
                // Now apply template values (overriding defaults where specified)
                if (config.model_type !== undefined) setWidgetValue("model_type", config.model_type);
                if (config.weight_dtype !== undefined) setWidgetValue("weight_dtype", config.weight_dtype);
                
                // Now apply template values (overriding defaults where specified)
                if (config.model_type !== undefined) setWidgetValue("model_type", config.model_type);
                if (config.weight_dtype !== undefined) setWidgetValue("weight_dtype", config.weight_dtype);
                
                if (config.configure_clip !== undefined) setWidgetValue("configure_clip", config.configure_clip);
                if (config.configure_vae !== undefined) setWidgetValue("configure_vae", config.configure_vae);
                if (config.configure_latent !== undefined) setWidgetValue("configure_latent", config.configure_latent);
                if (config.configure_sampler !== undefined) setWidgetValue("configure_sampler", config.configure_sampler);
                if (config.configure_model_only_lora !== undefined) setWidgetValue("configure_model_only_lora", config.configure_model_only_lora);
                
                // Nunchaku settings
                if (config.data_type !== undefined) setWidgetValue("data_type", config.data_type);
                if (config.cache_threshold !== undefined) setWidgetValue("cache_threshold", config.cache_threshold);
                if (config.attention !== undefined) setWidgetValue("attention", config.attention);
                if (config.i2f_mode !== undefined) setWidgetValue("i2f_mode", config.i2f_mode);
                if (config.cpu_offload !== undefined) setWidgetValue("cpu_offload", config.cpu_offload);
                if (config.num_blocks_on_gpu !== undefined) setWidgetValue("num_blocks_on_gpu", config.num_blocks_on_gpu);
                if (config.use_pin_memory !== undefined) setWidgetValue("use_pin_memory", config.use_pin_memory);
                
                // GGUF settings
                if (config.gguf_dequant_dtype !== undefined) setWidgetValue("gguf_dequant_dtype", config.gguf_dequant_dtype);
                if (config.gguf_patch_dtype !== undefined) setWidgetValue("gguf_patch_dtype", config.gguf_patch_dtype);
                if (config.gguf_patch_on_device !== undefined) setWidgetValue("gguf_patch_on_device", config.gguf_patch_on_device);
                
                // CLIP settings
                if (config.clip_source !== undefined) setWidgetValue("clip_source", config.clip_source);
                if (config.clip_count !== undefined) setWidgetValue("clip_count", config.clip_count);
                if (config.clip_name1 !== undefined) setWidgetValue("clip_name1", config.clip_name1);
                if (config.clip_name2 !== undefined) setWidgetValue("clip_name2", config.clip_name2);
                if (config.clip_name3 !== undefined) setWidgetValue("clip_name3", config.clip_name3);
                if (config.clip_name4 !== undefined) setWidgetValue("clip_name4", config.clip_name4);
                if (config.clip_type !== undefined) setWidgetValue("clip_type", config.clip_type);
                if (config.enable_clip_layer !== undefined) setWidgetValue("enable_clip_layer", config.enable_clip_layer);
                if (config.stop_at_clip_layer !== undefined) setWidgetValue("stop_at_clip_layer", config.stop_at_clip_layer);
                
                // VAE settings
                if (config.vae_source !== undefined) setWidgetValue("vae_source", config.vae_source);
                if (config.vae_name !== undefined) setWidgetValue("vae_name", config.vae_name);
                
                // Latent settings
                if (config.resolution !== undefined) setWidgetValue("resolution", config.resolution);
                if (config.width !== undefined) setWidgetValue("width", config.width);
                if (config.height !== undefined) setWidgetValue("height", config.height);
                if (config.batch_size !== undefined) setWidgetValue("batch_size", config.batch_size);
                
                // LoRA settings
                if (config.lora_count !== undefined) setWidgetValue("lora_count", config.lora_count);
                for (let i = 1; i <= 3; i++) {
                    if (config[`lora_switch_${i}`] !== undefined) {
                        setWidgetValue(`lora_switch_${i}`, config[`lora_switch_${i}`]);
                    }
                    if (config[`lora_name_${i}`] !== undefined) setWidgetValue(`lora_name_${i}`, config[`lora_name_${i}`]);
                    if (config[`lora_weight_${i}`] !== undefined) setWidgetValue(`lora_weight_${i}`, config[`lora_weight_${i}`]);
                }
                
                // Sampler settings
                if (config.sampler_name !== undefined) {
                    setWidgetValue("sampler_name", config.sampler_name);
                } else if (config.sampler !== undefined) {
                    setWidgetValue("sampler_name", config.sampler);
                }
                if (config.scheduler !== undefined) setWidgetValue("scheduler", config.scheduler);
                if (config.steps !== undefined) setWidgetValue("steps", config.steps);
                if (config.cfg !== undefined) setWidgetValue("cfg", config.cfg);
                if (config.flux_guidance !== undefined) setWidgetValue("flux_guidance", config.flux_guidance);
                
                // Model file selections
                if (config.ckpt_name !== undefined) setWidgetValue("ckpt_name", config.ckpt_name);
                if (config.unet_name !== undefined) setWidgetValue("unet_name", config.unet_name);
                if (config.nunchaku_name !== undefined) setWidgetValue("nunchaku_name", config.nunchaku_name);
                if (config.qwen_name !== undefined) setWidgetValue("qwen_name", config.qwen_name);
                if (config.gguf_name !== undefined) setWidgetValue("gguf_name", config.gguf_name);
                
                console.log(`✓ Template '${templateName}' applied`);
                
                } finally {
                    // Always reset flag and update visibility, even if there's an error
                    isApplyingTemplate = false;
                    updateVisibility();
                    
                    // Force canvas redraw to ensure widget visuals are updated
                    node.setDirtyCanvas(true, true);
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
                const configureLora = getWidgetValue("configure_model_only_lora");
                const clipSource = getWidgetValue("clip_source");
                const clipCount = parseInt(getWidgetValue("clip_count")) || 1;
                const clipType = getWidgetValue("clip_type");
                const vaeSource = getWidgetValue("vae_source");
                const resolution = getWidgetValue("resolution");
                const loraCount = parseInt(getWidgetValue("lora_count")) || 3;
                
                const isStandard = (modelType === "Standard Checkpoint");
                const isUNet = (modelType === "UNet Model");
                const isNunchaku = (modelType === "Nunchaku Flux");
                const isQwen = (modelType === "Nunchaku Qwen");
                const isGGUF = (modelType === "GGUF Model");
                const useExternalClip = (clipSource === "External");
                const useExternalVae = (vaeSource === "External");
                const isCustomResolution = (resolution === "Custom");
                
                // Determine if this is a Flux model (Nunchaku Flux or UNet/GGUF with flux clip type)
                const isFluxModel = isNunchaku || (clipType === "flux" && (isUNet || isGGUF));
                
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
                setWidgetVisible("weight_dtype", isUNet);
                
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
                
                // LoRA Configuration
                setWidgetVisible("lora_count", configureLora);
                for (let i = 1; i <= 3; i++) {
                    const showSlot = configureLora && i <= loraCount;
                    setWidgetVisible(`lora_switch_${i}`, showSlot);
                    setWidgetVisible(`lora_name_${i}`, showSlot);
                    setWidgetVisible(`lora_weight_${i}`, showSlot);
                }
                
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
                // flux_guidance only relevant for Flux models
                setWidgetVisible("flux_guidance", configureSampler && isFluxModel);
                
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
                "configure_model_only_lora",
                "clip_source",
                "clip_count",
                "clip_type",
                "vae_source",
                "resolution",
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
                        
                        if (widgetName === "template_action" || widgetName === "template_name") {
                            const templateAction = getWidgetValue("template_action");
                            const templateName = getWidgetValue("template_name");
                            
                            // Auto-fill new_template_name when switching to Save mode
                            if (widgetName === "template_action" && templateAction === "Save") {
                                // If there's a loaded template, copy its name to new_template_name
                                if (templateName && templateName !== "None") {
                                    setWidgetValue("new_template_name", templateName);
                                }
                            }
                            
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
            
            // Listen for execution events
            const onExecuted = node.onExecuted;
            node.onExecuted = async function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                if (pendingTemplateSave) {
                    const savedTemplateName = pendingTemplateSave;
                    pendingTemplateSave = null;
                    
                    console.log(`✓ Save completed, refreshing template list...`);
                    await new Promise(resolve => setTimeout(resolve, 100));
                    await refreshTemplateList();
                    
                    setWidgetValue("template_action", "Load");
                    setWidgetValue("template_name", savedTemplateName);
                    setWidgetValue("new_template_name", "");
                    updateVisibility();
                    console.log(`✓ Switched to Load mode with template: ${savedTemplateName}`);
                }
                
                if (pendingTemplateDelete) {
                    pendingTemplateDelete = false;
                    
                    console.log(`✓ Delete completed, refreshing template list...`);
                    await new Promise(resolve => setTimeout(resolve, 100));
                    await refreshTemplateList();
                    
                    setWidgetValue("template_action", "Load");
                    setWidgetValue("template_name", "None");
                    updateVisibility();
                    console.log(`✓ Template deleted, switched to Load mode`);
                }
            };
            
            // Listen for execution interrupts
            api.addEventListener("execution_interrupted", async (event) => {
                console.log('[SmartLoader+] execution_interrupted event:', event.detail);
                
                if (pendingTemplateSave || pendingTemplateDelete) {
                    console.log('[SmartLoader+] Processing pending template operation...');
                    
                    if (pendingTemplateSave) {
                        const savedTemplateName = pendingTemplateSave;
                        pendingTemplateSave = null;
                        
                        console.log(`✓ Save interrupted (as expected), refreshing template list...`);
                        await new Promise(resolve => setTimeout(resolve, 300));
                        await refreshTemplateList();
                        
                        setWidgetValue("template_action", "Load");
                        setWidgetValue("template_name", savedTemplateName);
                        setWidgetValue("new_template_name", "");
                        updateVisibility();
                        console.log(`✓ Switched to Load mode with template: ${savedTemplateName}`);
                    }
                    
                    if (pendingTemplateDelete) {
                        pendingTemplateDelete = false;
                        
                        console.log(`✓ Delete interrupted (as expected), refreshing template list...`);
                        await new Promise(resolve => setTimeout(resolve, 300));
                        await refreshTemplateList();
                        
                        setWidgetValue("template_action", "Load");
                        setWidgetValue("template_name", "None");
                        updateVisibility();
                        console.log(`✓ Template deleted, switched to Load mode`);
                    }
                }
            });
            
            // Initial setup
            setTimeout(() => {
                updateVisibility();
                refreshTemplateList();
            }, 10);
            
            // Hook into onConfigure to reload template when workflow is loaded
            const onConfigure = node.onConfigure;
            node.onConfigure = function(info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                // After workflow is configured, check if a template is selected and reload it
                // Use longer delay to ensure ComfyUI has finished restoring all widget values
                setTimeout(() => {
                    const templateAction = getWidgetValue("template_action");
                    const templateName = getWidgetValue("template_name");
                    
                    if (templateAction === "Load" && templateName && templateName !== "None") {
                        console.log(`[SmartLoader+] Workflow loaded, reapplying template: ${templateName}`);
                        applyTemplate(templateName);
                    } else {
                        updateVisibility();
                    }
                }, 250); // Increased delay to ensure workflow restoration is complete
            };
            
            return r;
        };
    },
});

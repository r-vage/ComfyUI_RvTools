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
* Dynamic widget visibility for Language Model nodes
* Shows QwenVL widgets for QwenVL models, Florence-2 widgets for Florence-2 models
*/

import { app, api } from './comfy/index.js';

const NODE_NAMES = [
    "Smart Language Model Loader [Eclipse]"
];

const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";

const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

// Store last seeds per node ID
const nodeLastSeeds = {};

app.registerExtension({
    name: "Eclipse.SmartLoaderLM",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!NODE_NAMES.includes(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            const node = this;
            let currentModelType = "unknown"; // Track current model type: "qwenvl", "florence2", "unknown"
            let currentIsGGUF = false; // Track if current model is GGUF format
            let pendingTemplateSave = null;
            let pendingTemplateDelete = false;
            
            // Track if we're in workflow restoration to avoid resetting values
            node._Eclipse_isRestoring = false;
            
            // Initialize seed tracking
            this._Eclipse_lastSeed = undefined;
            this._Eclipse_randomMin = 1;
            this._Eclipse_randomMax = 2**32 - 1;
            this._Eclipse_cachedInputSeed = null;
            this._Eclipse_cachedResolvedSeed = null;
            
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
            
            // Refresh template list from server
            const refreshTemplateList = async () => {
                try {
                    const response = await fetch('/eclipse/smartlm_templates_list');
                    if (response.ok) {
                        const templates = await response.json();
                        const templateWidget = node.widgets?.find(w => w.name === "template_name");
                        if (templateWidget && templateWidget.options && templateWidget.options.values) {
                            templateWidget.options.values = templates;
                            if (!templates.includes(templateWidget.value)) {
                                templateWidget.value = templates[0] || "None";
                            }
                            node.setDirtyCanvas(true, true);
                        }
                    }
                } catch (e) {
                    console.error('[SmartLM] Failed to refresh template list:', e);
                }
            };
            
            // Template action handler
            const handleTemplateAction = async () => {
                const templateAction = getWidgetValue("template_action");
                const templateName = getWidgetValue("template_name");
                const newTemplateName = getWidgetValue("new_template_name");
                
                await refreshTemplateList();
                
                if (templateAction === "New" && newTemplateName && newTemplateName.trim()) {
                    console.log(`✓ Queueing workflow to create new template: ${newTemplateName}`);
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
                    templateButton = node.addWidget("button", "⚡️Execute Template Action", null, handleTemplateAction);
                    templateButton.serialize = false;
                } else if (!hasAction && templateButton) {
                    const buttonIndex = node.widgets.indexOf(templateButton);
                    if (buttonIndex >= 0) {
                        node.widgets.splice(buttonIndex, 1);
                    }
                    templateButton = null;
                }
            };
            
            const loadTemplateConfig = async (templateName) => {
                if (!templateName || templateName === "None") return null;
                
                try {
                    // Add cache-busting parameter to force fresh fetch
                    const cacheBuster = new Date().getTime();
                    const response = await fetch(`/eclipse/smartlm_templates/${templateName}.json?t=${cacheBuster}`, {
                        cache: 'no-store'
                    });
                    if (response.ok) {
                        return await response.json();
                    }
                } catch (e) {
                    console.error(`[SmartLM] Failed to load template ${templateName}:`, e);
                }
                return null;
            };
            
            const detectModelType = async (templateName) => {
                const config = await loadTemplateConfig(templateName);
                
                if (!config || !config.model_type) {
                    return "unknown";
                }
                
                const modelType = config.model_type.toLowerCase();
                
                if (modelType === "qwenvl") {
                    return "qwenvl";
                } else if (modelType === "florence2") {
                    return "florence2";
                } else if (modelType === "llm") {
                    return "llm";
                } else {
                    return "unknown";
                }
            };
            
            const detectIsGGUF = async (templateName) => {
                const config = await loadTemplateConfig(templateName);
                
                if (!config) {
                    return false;
                }
                
                // Check if local_path or repo_id ends with .gguf, or if repo_id contains "gguf" in the name
                const localPath = config.local_path || "";
                const repoId = config.repo_id || "";
                const hasGGUFExtension = localPath.toLowerCase().endsWith(".gguf") || 
                                        repoId.toLowerCase().endsWith(".gguf");
                const hasGGUFInName = repoId.toLowerCase().includes("gguf");
                
                return hasGGUFExtension || hasGGUFInName;
            };
            
            // Method to generate random seed
            node.generateRandomSeed = function() {
                const step = this._Eclipse_seedWidget?.options?.step || 1;
                const randomMin = this._Eclipse_randomMin || 1;
                
                // Limit seed range based on model type
                // QwenVL uses int32 range, Florence-2 and LLM can use full uint32 range
                let randomMax = this._Eclipse_randomMax || (2**32 - 1);
                if (currentModelType === "qwenvl") {
                    randomMax = Math.min(randomMax, 2147483647); // Clamp to int32 max (2^31-1)
                }
                
                const randomRange = (randomMax - randomMin) / (step / 10);
                let seed = Math.floor(Math.random() * randomRange) * (step / 10) + randomMin;
                
                // Avoid special seeds
                if (SPECIAL_SEEDS.includes(seed)) {
                    seed = 1;
                }
                return seed;
            };
            
            // Method to determine seed to use
            node.getSeedToUse = function() {
                const inputSeed = Number(this._Eclipse_seedWidget.value);
                
                // Check if we have a cached resolved seed for this input seed
                if (this._Eclipse_cachedInputSeed === inputSeed && this._Eclipse_cachedResolvedSeed != null) {
                    return this._Eclipse_cachedResolvedSeed;
                }
                
                let seedToUse = null;
                
                // If our input seed was a special seed, then handle it
                if (SPECIAL_SEEDS.includes(inputSeed)) {
                    // If the last seed was not a special seed and we have increment/decrement, then do that
                    if (typeof this._Eclipse_lastSeed === "number" && !SPECIAL_SEEDS.includes(this._Eclipse_lastSeed)) {
                        if (inputSeed === SPECIAL_SEED_INCREMENT) {
                            seedToUse = this._Eclipse_lastSeed + 1;
                        } else if (inputSeed === SPECIAL_SEED_DECREMENT) {
                            seedToUse = this._Eclipse_lastSeed - 1;
                        }
                    }
                    
                    // If we don't have a seed to use, or it's a special seed, randomize
                    if (seedToUse == null || SPECIAL_SEEDS.includes(seedToUse)) {
                        seedToUse = this.generateRandomSeed();
                    }
                }
                
                let finalSeed = seedToUse != null ? seedToUse : inputSeed;
                
                // Clamp seed for QwenVL models (int32 range)
                // Florence-2 and LLM can use full uint32 range
                if (currentModelType === "qwenvl") {
                    finalSeed = Math.max(0, Math.min(finalSeed, 2147483647)); // Clamp to int32 (0 to 2^31-1)
                }
                
                // Cache the resolved seed for this input seed
                this._Eclipse_cachedInputSeed = inputSeed;
                this._Eclipse_cachedResolvedSeed = finalSeed;
                
                return finalSeed;
            };
            
            const updateVisibility = () => {
                const isQwenVL = (currentModelType === "qwenvl");
                const isFlorence2 = (currentModelType === "florence2");
                const isLLM = (currentModelType === "llm");
                const templateAction = getWidgetValue("template_action");
                const isNewMode = (templateAction === "New");
                
                // Check if text input is connected
                const textInput = node.inputs?.find(input => input.name === "text");
                const isTextConnected = textInput && textInput.link != null;
                
                // Template management widgets visibility
                setWidgetVisible("template_action", true);
                setWidgetVisible("template_name", !isNewMode); // Hide template selector in New mode
                
                // New template widgets - only visible in New mode
                setWidgetVisible("new_template_name", isNewMode);
                setWidgetVisible("new_model_type", isNewMode);
                setWidgetVisible("new_vram_full", isNewMode);
                setWidgetVisible("new_set_default", isNewMode);
                
                // Get source selection and model type for conditional visibility
                const modelSource = isNewMode ? getWidgetValue("new_model_source") : null;
                const mmprojSource = isNewMode ? getWidgetValue("new_mmproj_source") : null;
                const newModelType = isNewMode ? getWidgetValue("new_model_type") : "";
                const isNewGGUF = newModelType && newModelType.includes("(GGUF)");
                const isVisionModel = newModelType && (newModelType.includes("QwenVL") || newModelType.includes("Florence2"));
                const showMmproj = isNewMode && isNewGGUF && newModelType.includes("QwenVL"); // Only QwenVL GGUF needs mmproj
                
                // new_quantized only visible for non-GGUF models (GGUF is always quantized)
                setWidgetVisible("new_quantized", isNewMode && !isNewGGUF);
                
                // context_size only for GGUF models
                setWidgetVisible("new_context_size", isNewMode && isNewGGUF);
                
                // Model source selection (always visible in New mode)
                setWidgetVisible("new_model_source", isNewMode);
                
                // Show/hide model input widgets based on source
                setWidgetVisible("new_local_model", isNewMode && modelSource === "Local");
                setWidgetVisible("new_repo_id", isNewMode && modelSource === "HuggingFace");
                setWidgetVisible("new_local_path", isNewMode && modelSource === "HuggingFace");
                
                // mmproj source selection and inputs (only for GGUF vision models)
                setWidgetVisible("new_mmproj_source", showMmproj);
                setWidgetVisible("new_mmproj_local", showMmproj && mmprojSource === "Local");
                setWidgetVisible("new_mmproj_url", showMmproj && mmprojSource === "HuggingFace");
                setWidgetVisible("new_mmproj_path", showMmproj && mmprojSource === "HuggingFace");
                
                // Update button visibility
                updateTemplateButton();
                
                // Hide all generation widgets in New mode (they're not used for template creation)
                if (isNewMode) {
                    setWidgetVisible("quantization", false);
                    setWidgetVisible("attention_mode", false);
                    setWidgetVisible("qwen_preset_prompt", false);
                    setWidgetVisible("qwen_custom_prompt", false);
                    setWidgetVisible("florence_task", false);
                    setWidgetVisible("florence_text_input", false);
                    setWidgetVisible("convert_to_bboxes", false);
                    setWidgetVisible("llm_instruction_mode", false);
                    setWidgetVisible("llm_custom_instruction", false);
                    setWidgetVisible("llm_prompt", false);
                    setWidgetVisible("max_tokens", false);
                    setWidgetVisible("context_size", false);
                    setWidgetVisible("memory_cleanup", false);
                    setWidgetVisible("keep_model_loaded", false);
                    setWidgetVisible("auto_save_template", false);
                    setWidgetVisible("seed", false);
                    
                    // Smart resize for New mode
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
                    
                    return; // Skip rest of visibility logic
                }
                
                // Normal mode - show generation widgets based on model type
                // Hide quantization and attention_mode for GGUF models (they don't use these parameters)
                setWidgetVisible("quantization", !currentIsGGUF);
                setWidgetVisible("attention_mode", !currentIsGGUF);
                setWidgetVisible("max_tokens", true);
                setWidgetVisible("context_size", currentIsGGUF); // Only show for GGUF models
                // Hide keep_model_loaded for GGUF models (they must be unloaded to prevent VRAM accumulation)
                setWidgetVisible("keep_model_loaded", !currentIsGGUF);
                setWidgetVisible("seed", true);
                
                // Show model-specific widgets based on detected model type
                // Show QwenVL widgets only for QwenVL models
                setWidgetVisible("qwen_preset_prompt", isQwenVL);
                setWidgetVisible("qwen_custom_prompt", isQwenVL && !isTextConnected);
                
                // Show Florence-2 widgets only for Florence-2 models
                setWidgetVisible("florence_task", isFlorence2);
                
                // Show florence_text_input based on task type (grounding/detection/ocr/region need text input)
                if (isFlorence2 && !isTextConnected) {
                    const florenceTask = getWidgetValue("florence_task");
                    const taskNeedsTextInput = florenceTask && (
                        florenceTask.includes("grounding") || 
                        florenceTask.includes("detection") || 
                        florenceTask.includes("ocr") || 
                        florenceTask.includes("region")
                    );
                    setWidgetVisible("florence_text_input", taskNeedsTextInput);
                } else {
                    setWidgetVisible("florence_text_input", false);
                }
                
                // Show convert_to_bboxes only for Florence-2 detection/grounding/OCR tasks
                if (isFlorence2) {
                    const florenceTask = getWidgetValue("florence_task");
                    const showConversionOption = florenceTask && (
                        florenceTask.includes("grounding") || 
                        florenceTask.includes("detection") || 
                        florenceTask.includes("ocr") || 
                        florenceTask.includes("region")
                    );
                    setWidgetVisible("convert_to_bboxes", showConversionOption);
                } else {
                    setWidgetVisible("convert_to_bboxes", false);
                }
                
                // Show LLM instruction widgets only for text-only LLM models
                setWidgetVisible("llm_instruction_mode", isLLM);
                setWidgetVisible("llm_prompt", isLLM && !isTextConnected);
                
                // Show custom instruction only when mode is "Custom Instruction"
                const llmMode = getWidgetValue("llm_instruction_mode");
                setWidgetVisible("llm_custom_instruction", isLLM && llmMode === "Custom Instruction");
                
                // Show advanced parameters (for Advanced node)
                const advancedWidgets = ["device", "temperature", "top_p", "num_beams", "do_sample", "repetition_penalty", "frame_count", "use_torch_compile"];
                advancedWidgets.forEach(widgetName => {
                    const widget = node.widgets?.find(w => w.name === widgetName);
                    if (widget) {
                        setWidgetVisible(widgetName, true);
                    }
                });
                
                // Show video input hint only for QwenVL (Florence-2 doesn't support video)
                // Note: video is an optional input, not a widget, so we can't hide it directly
                // But we can add a note in the tooltip or documentation
                
                // For Advanced node, temperature/top_p might need different defaults
                // QwenVL: temperature=0.7, top_p=0.9
                // Florence-2: temperature=0.4, top_p defaults fine
                // (These are handled by the Python backend based on model type)
                
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
            
            // Hook into template_action widget
            const templateActionWidget = node.widgets?.find(w => w.name === "template_action");
            if (templateActionWidget) {
                const originalActionCallback = templateActionWidget.callback;
                templateActionWidget.callback = function() {
                    if (originalActionCallback) {
                        originalActionCallback.apply(this, arguments);
                    }
                    
                    // Reset new_* widgets when switching to "New" mode
                    const action = getWidgetValue("template_action");
                    if (action === "New") {
                        setWidgetValue("new_template_name", "");
                        setWidgetValue("new_model_type", "QwenVL (Transformers)");
                        setWidgetValue("new_model_source", "HuggingFace");
                        setWidgetValue("new_repo_id", "");
                        setWidgetValue("new_local_path", "");
                        setWidgetValue("new_mmproj_source", "Local");
                        setWidgetValue("new_mmproj_url", "");
                        setWidgetValue("new_mmproj_path", "");
                        setWidgetValue("new_quantized", false);
                        setWidgetValue("new_vram_full", 3.0);
                        setWidgetValue("new_context_size", 32768);
                        setWidgetValue("new_set_default", false);
                    }
                    
                    updateVisibility();
                };
            }
            
            // Hook into new_model_source widget to show/hide model input fields
            const modelSourceWidget = node.widgets?.find(w => w.name === "new_model_source");
            if (modelSourceWidget) {
                const originalSourceCallback = modelSourceWidget.callback;
                modelSourceWidget.callback = function() {
                    if (originalSourceCallback) {
                        originalSourceCallback.apply(this, arguments);
                    }
                    
                    updateVisibility();
                };
            }
            
            // Hook into new_mmproj_source widget to show/hide mmproj input fields
            const mmprojSourceWidget = node.widgets?.find(w => w.name === "new_mmproj_source");
            if (mmprojSourceWidget) {
                const originalMmprojCallback = mmprojSourceWidget.callback;
                mmprojSourceWidget.callback = function() {
                    if (originalMmprojCallback) {
                        originalMmprojCallback.apply(this, arguments);
                    }
                    
                    updateVisibility();
                };
            }
            
            // Hook into new_model_type widget to show/hide mmproj widgets
            const newModelTypeWidget = node.widgets?.find(w => w.name === "new_model_type");
            if (newModelTypeWidget) {
                const originalTypeCallback = newModelTypeWidget.callback;
                newModelTypeWidget.callback = function() {
                    if (originalTypeCallback) {
                        originalTypeCallback.apply(this, arguments);
                    }
                    
                    updateVisibility();
                };
            }
            
            // Hook into llm_instruction_mode to show/hide custom instruction
            const llmInstructionModeWidget = node.widgets?.find(w => w.name === "llm_instruction_mode");
            if (llmInstructionModeWidget) {
                const originalInstructionCallback = llmInstructionModeWidget.callback;
                llmInstructionModeWidget.callback = function() {
                    if (originalInstructionCallback) {
                        originalInstructionCallback.apply(this, arguments);
                    }
                    // Show custom instruction field only when "Custom Instruction" is selected
                    const mode = getWidgetValue("llm_instruction_mode");
                    const showCustom = (mode === "Custom Instruction");
                    setWidgetVisible("llm_custom_instruction", showCustom);
                };
            }
            
            // Hook into florence_task to show/hide convert_to_bboxes
            const florenceTaskWidget = node.widgets?.find(w => w.name === "florence_task");
            if (florenceTaskWidget) {
                const originalTaskCallback = florenceTaskWidget.callback;
                florenceTaskWidget.callback = function() {
                    if (originalTaskCallback) {
                        originalTaskCallback.apply(this, arguments);
                    }
                    updateVisibility();
                };
            }
            
            // Hook into template_name widget
            const templateWidget = node.widgets?.find(w => w.name === "template_name");
            if (templateWidget) {
                const originalCallback = templateWidget.callback;
                templateWidget.callback = async function() {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    
                    const templateName = getWidgetValue("template_name");
                    
                    // Load template defaults whenever a valid template is selected
                    if (templateName && templateName !== "None") {
                        const config = await loadTemplateConfig(templateName);
                        const detectedType = await detectModelType(templateName);
                        const isGGUF = await detectIsGGUF(templateName);
                        currentModelType = detectedType;
                        currentIsGGUF = isGGUF;
                        
                        // Load generation parameters from template if available
                        if (config && config.max_tokens !== undefined) {
                            const maxTokensWidget = node.widgets?.find(w => w.name === "max_tokens");
                            if (maxTokensWidget) {
                                maxTokensWidget.value = config.max_tokens;
                            }
                        }
                        if (config && config.quantization !== undefined) {
                            const quantizationWidget = node.widgets?.find(w => w.name === "quantization");
                            if (quantizationWidget) {
                                quantizationWidget.value = config.quantization;
                            }
                        }
                        if (config && config.attention_mode !== undefined) {
                            const attentionWidget = node.widgets?.find(w => w.name === "attention_mode");
                            if (attentionWidget) {
                                attentionWidget.value = config.attention_mode;
                            }
                        }
                        
                        // Load task/preset defaults from template
                        if (config) {
                            // Florence-2: Load default_task and text_input
                            if (config.default_task !== undefined && detectedType === "florence2") {
                                const florenceTaskWidget = node.widgets?.find(w => w.name === "florence_task");
                                if (florenceTaskWidget) {
                                    florenceTaskWidget.value = config.default_task;
                                }
                            }
                            if (config.default_text_input !== undefined && detectedType === "florence2") {
                                const florenceTextWidget = node.widgets?.find(w => w.name === "florence_text_input");
                                if (florenceTextWidget) {
                                    florenceTextWidget.value = config.default_text_input;
                                }
                            }
                            
                            // QwenVL: Load default_task and default_text_input (same as Florence-2)
                            if (config.default_task !== undefined && detectedType === "qwenvl") {
                                const qwenPresetWidget = node.widgets?.find(w => w.name === "qwen_preset_prompt");
                                if (qwenPresetWidget) {
                                    qwenPresetWidget.value = config.default_task;
                                    console.log(`[SmartLM] Loaded default_task=${config.default_task} from template`);
                                }
                            }
                            if (config.default_text_input !== undefined && detectedType === "qwenvl") {
                                const qwenCustomWidget = node.widgets?.find(w => w.name === "qwen_custom_prompt");
                                if (qwenCustomWidget) {
                                    qwenCustomWidget.value = config.default_text_input;
                                    console.log(`[SmartLM] Loaded default_text_input from template`);
                                }
                        }
                        
                        // LLM: Load default_task and default_text_input (same as Florence-2 and QwenVL)
                        if (config.default_task !== undefined && detectedType === "llm") {
                            const llmModeWidget = node.widgets?.find(w => w.name === "llm_instruction_mode");
                            if (llmModeWidget) {
                                llmModeWidget.value = config.default_task;
                                console.log(`[SmartLM] Loaded default_task=${config.default_task} from template`);
                            }
                        }
                        if (config.default_text_input !== undefined && detectedType === "llm") {
                            const llmInstructionWidget = node.widgets?.find(w => w.name === "llm_custom_instruction");
                            if (llmInstructionWidget) {
                                llmInstructionWidget.value = config.default_text_input;
                                console.log(`[SmartLM] Loaded default_text_input from template`);
                            }
                        }
                    }                        updateVisibility();
                    }
                };
            }
            
            // Find the seed widget and remove control_after_generate (we have our own seed controls)
            let seedWidget = null;
            for (const [i, widget] of this.widgets.entries()) {
                const wname = (widget.name || '').toString().toLowerCase();
                const wlabel = (widget.label || widget.options?.label || widget.options?.name || '').toString().toLowerCase();
                const wlocalized = (widget.localized_name || '').toString().toLowerCase();
                if (wname === 'seed' || wlabel === 'seed' || wlocalized === 'seed') {
                    seedWidget = widget;
                } else if (wname === 'control_after_generate') {
                    this.widgets.splice(i, 1);
                }
            }
            
            if (seedWidget) {
                node._Eclipse_seedWidget = seedWidget;
                
                // Hook into the seed widget's value setter to clear cache when it changes
                const originalCallback = seedWidget.callback;
                seedWidget.callback = (value) => {
                    // Clear the seed cache when the seed value changes
                    node._Eclipse_cachedInputSeed = null;
                    node._Eclipse_cachedResolvedSeed = null;
                    // Call the original callback if it exists
                    if (originalCallback) {
                        return originalCallback.call(seedWidget, value);
                    }
                };
                
                const seedWidgetIndex = node.widgets.indexOf(seedWidget);
                
                // Button: Randomize Each Time
                const randomizeButton = node.addWidget(
                    "button",
                    "🎲 Randomize Each Time",
                    "",
                    () => {
                        seedWidget.value = SPECIAL_SEED_RANDOM;
                        if (seedWidget.callback) {
                            seedWidget.callback(SPECIAL_SEED_RANDOM);
                        }
                    },
                    { serialize: false }
                );
                
                // Button: New Fixed Random
                const newRandomButton = node.addWidget(
                    "button",
                    "🎲 New Fixed Random",
                    "",
                    () => {
                        const newSeed = node.generateRandomSeed();
                        seedWidget.value = newSeed;
                        if (seedWidget.callback) {
                            seedWidget.callback(newSeed);
                        }
                    },
                    { serialize: false }
                );
                
                // Button: Use Last Queued Seed
                const lastSeedButton = node.addWidget(
                    "button",
                    LAST_SEED_BUTTON_LABEL,
                    "",
                    () => {
                        if (node._Eclipse_lastSeed != null) {
                            seedWidget.value = node._Eclipse_lastSeed;
                            lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                            lastSeedButton.disabled = true;
                        }
                    },
                    { serialize: false }
                );
                lastSeedButton.disabled = true;
                node._Eclipse_lastSeedButton = lastSeedButton;
                
                // Move buttons to be right after the seed widget
                const buttonsToMove = [randomizeButton, newRandomButton, lastSeedButton];
                for (let i = buttonsToMove.length - 1; i >= 0; i--) {
                    const button = buttonsToMove[i];
                    const currentIndex = node.widgets.indexOf(button);
                    if (currentIndex !== seedWidgetIndex + 1) {
                        node.widgets.splice(currentIndex, 1);
                        node.widgets.splice(seedWidgetIndex + 1, 0, button);
                    }
                }
            }
            
            // Function to auto-save template changes (only visible widgets)
            node.autoSaveTemplate = async () => {
                const autoSave = getWidgetValue("auto_save_template");
                if (!autoSave) return;
                
                const templateName = getWidgetValue("template_name");
                if (!templateName || templateName === "None") return;
                
                // Load current template to compare
                const currentTemplate = await loadTemplateConfig(templateName);
                if (!currentTemplate) return;
                
                // Collect changes from visible widgets only
                const updates = {};
                
                // Helper to check if widget is visible and changed
                const checkWidget = (widgetName, templateKey) => {
                    const widget = node.widgets?.find(w => w.name === widgetName);
                    if (!widget || widget.hidden || widget.type === "converted-widget") return;
                    
                    const currentValue = widget.value;
                    const templateValue = currentTemplate[templateKey || widgetName];
                    
                    if (currentValue !== templateValue) {
                        updates[templateKey || widgetName] = currentValue;
                    }
                };
                
                // Check common visible widgets
                checkWidget("max_tokens");
                checkWidget("quantization");
                checkWidget("attention_mode");
                checkWidget("context_size");
                
                // Check model-specific visible widgets
                if (currentModelType === "qwenvl") {
                    checkWidget("qwen_preset_prompt", "default_task");
                    checkWidget("qwen_custom_prompt", "default_text_input");
                } else if (currentModelType === "florence2") {
                    checkWidget("florence_task", "default_task");
                    checkWidget("florence_text_input", "default_text_input");
                    checkWidget("convert_to_bboxes");
                } else if (currentModelType === "llm") {
                    checkWidget("llm_instruction_mode", "default_task");
                    checkWidget("llm_custom_instruction", "default_text_input");
                }
                
                // Send updates to server if any changes detected
                if (Object.keys(updates).length > 0) {
                    try {
                        const response = await fetch(`/eclipse/smartlm_templates/${templateName}.json`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(updates)
                        });
                        
                        if (!response.ok) {
                            console.error(`[SmartLM] Auto-save failed: ${response.status}`);
                        }
                    } catch (e) {
                        console.error(`[SmartLM] Auto-save error:`, e);
                    }
                }
            };
            
            // Intercept the onExecuted to track last seed (auto-save happens in graphToPrompt before execution)
            const onExecuted = node.onExecuted;
            node.onExecuted = async function(message) {
                const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                
                // Store the seed that was actually used if available
                if (message && message.seed !== undefined) {
                    node._Eclipse_lastSeed = message.seed;
                    nodeLastSeeds[node.id] = message.seed;
                }
                
                // Handle pending template save/delete
                if (pendingTemplateSave) {
                    const savedTemplateName = pendingTemplateSave;
                    pendingTemplateSave = null;
                    
                    console.log(`✓ Template creation completed, refreshing template list...`);
                    await new Promise(resolve => setTimeout(resolve, 100));
                    await refreshTemplateList();
                    
                    setWidgetValue("template_action", "None");
                    setWidgetValue("template_name", savedTemplateName);
                    setWidgetValue("new_template_name", "");
                    updateVisibility();
                    console.log(`✓ Switched to None mode with template: ${savedTemplateName}`);
                }
                
                if (pendingTemplateDelete) {
                    pendingTemplateDelete = false;
                    
                    console.log(`✓ Delete completed, refreshing template list...`);
                    await new Promise(resolve => setTimeout(resolve, 100));
                    await refreshTemplateList();
                    
                    setWidgetValue("template_action", "None");
                    updateVisibility();
                    console.log(`✓ Template deleted, switched to None mode`);
                }
                
                return result;
            };
            
            // Listen for execution interrupts
            api.addEventListener("execution_interrupted", async (event) => {
                console.log('[SmartLM] execution_interrupted event:', event.detail);
                
                if (pendingTemplateSave || pendingTemplateDelete) {
                    console.log('[SmartLM] Processing pending template operation...');
                    
                    if (pendingTemplateSave) {
                        const savedTemplateName = pendingTemplateSave;
                        pendingTemplateSave = null;
                        
                        console.log(`✓ Template creation interrupted (as expected), refreshing template list...`);
                        await new Promise(resolve => setTimeout(resolve, 300));
                        await refreshTemplateList();
                        
                        setWidgetValue("template_action", "None");
                        setWidgetValue("template_name", savedTemplateName);
                        setWidgetValue("new_template_name", "");
                        updateVisibility();
                        console.log(`✓ Switched to None mode with template: ${savedTemplateName}`);
                    }
                    
                    if (pendingTemplateDelete) {
                        pendingTemplateDelete = false;
                        
                        console.log(`✓ Delete interrupted (as expected), refreshing template list...`);
                        await new Promise(resolve => setTimeout(resolve, 300));
                        await refreshTemplateList();
                        
                        setWidgetValue("template_action", "None");
                        updateVisibility();
                        console.log(`✓ Template deleted, switched to None mode`);
                    }
                }
            });
            
            // Initial setup - detect model type from default template and load values
            setTimeout(async () => {
                const templateName = getWidgetValue("template_name");
                
                if (templateName && templateName !== "None") {
                    const detectedType = await detectModelType(templateName);
                    currentModelType = detectedType;
                    
                    // Trigger template callback to load widget values
                    const templateWidget = node.widgets?.find(w => w.name === "template_name");
                    if (templateWidget && templateWidget.callback) {
                        await templateWidget.callback.call(templateWidget);
                    }
                } else {
                    // Default to QwenVL if no template selected
                    currentModelType = "qwenvl";
                }
                
                // Important: Call updateVisibility to ensure widgets are shown/hidden correctly on node creation
                updateVisibility();
            }, 10);
            
            // Hook into onConfigure to reload visibility when workflow is loaded
            const onConfigure = node.onConfigure;
            node.onConfigure = function(info) {
                // Set restoration flag to prevent template_action callback from resetting values
                node._Eclipse_isRestoring = true;
                
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                // After workflow is configured, detect model type and update visibility
                setTimeout(async () => {
                    const templateName = getWidgetValue("template_name");
                    
                    // Detect model type and load template defaults
                    if (templateName && templateName !== "None") {
                        const detectedType = await detectModelType(templateName);
                        currentModelType = detectedType;
                        
                        // Reload template defaults
                        const templateWidget = node.widgets?.find(w => w.name === "template_name");
                        if (templateWidget && templateWidget.callback) {
                            await templateWidget.callback.call(templateWidget);
                        }
                    } else {
                        currentModelType = "qwenvl";
                    }
                    
                    // Always call updateVisibility after configuration
                    updateVisibility();
                    
                    // Clear restoration flag after a delay to allow all widget callbacks to settle
                    setTimeout(() => {
                        node._Eclipse_isRestoring = false;
                    }, 100);
                }, 50);
            };
            
            // Hook into onConnectionsChange to detect when text input is connected/disconnected
            const onConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function(type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                // Check if the connection change is for the text input
                if (type === 1) { // 1 = input
                    const input = this.inputs[index];
                    if (input && input.name === "text") {
                        // Text input connection changed, update visibility
                        setTimeout(() => {
                            updateVisibility();
                        }, 10);
                    }
                }
            };
            
            return r;
        };
    },
    
    async setup() {
        // Hook into the graphToPrompt to modify seed values in the prompt data
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Call the original graphToPrompt first
            const result = await originalGraphToPrompt.apply(this, arguments);

            // Now modify the prompt data for all Smart LML nodes
            const nodes = app.graph._nodes;
            for (const node of nodes) {
                if (NODE_NAMES.includes(node.type)) {
                    // Auto-save template before execution if node has autoSaveTemplate function
                    if (node.autoSaveTemplate && typeof node.autoSaveTemplate === 'function') {
                        // Skip if node is muted or bypassed
                        if (node.mode !== 2 && node.mode !== 4) {
                            await node.autoSaveTemplate();
                        }
                    }
                }
                
                if (NODE_NAMES.includes(node.type) && node._Eclipse_seedWidget) {
                    // Skip if node is muted or bypassed
                    if (node.mode === 2 || node.mode === 4) {
                        continue;
                    }
                    
                    // Check if this node is in the prompt
                    const nodeId = String(node.id);
                    if (result.output && result.output[nodeId]) {
                        const seedToUse = node.getSeedToUse();
                        
                        // Update the seed in the prompt output (what gets sent to server)
                        if (result.output[nodeId].inputs && result.output[nodeId].inputs.seed !== undefined) {
                            const existing = result.output[nodeId].inputs.seed;
                            if (Number(existing) !== Number(seedToUse)) {
                                result.output[nodeId].inputs.seed = seedToUse;
                            }
                        }

                        // Update last seed tracking only when it actually changes
                        if (Number(node._Eclipse_lastSeed) !== Number(seedToUse)) {
                            node._Eclipse_lastSeed = seedToUse;
                            nodeLastSeeds[node.id] = seedToUse;
                        }
                        
                        // Clear the seed cache after use so next call generates fresh random seed
                        node._Eclipse_cachedInputSeed = null;
                        node._Eclipse_cachedResolvedSeed = null;
                        
                        // Update the last seed button - but DON'T change the widget value
                        if (node._Eclipse_lastSeedButton) {
                            const currentWidgetValue = node._Eclipse_seedWidget.value;
                            if (SPECIAL_SEEDS.includes(currentWidgetValue)) {
                                // Widget has special seed, show what was actually used
                                node._Eclipse_lastSeedButton.name = `♻️ ${seedToUse}`;
                                node._Eclipse_lastSeedButton.disabled = false;
                            } else {
                                // Widget has regular seed value
                                node._Eclipse_lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                                node._Eclipse_lastSeedButton.disabled = true;
                            }
                        }
                        
                        // Also update workflow data if present
                        if (result.workflow && result.workflow.nodes) {
                            const workflowNode = result.workflow.nodes.find(n => n.id === node.id);
                            if (workflowNode && workflowNode.widgets_values) {
                                const seedWidgetIndex = node.widgets.indexOf(node._Eclipse_seedWidget);
                                if (seedWidgetIndex >= 0) {
                                    // Only update workflow stored value if it differs
                                    if (workflowNode.widgets_values[seedWidgetIndex] !== seedToUse) {
                                        workflowNode.widgets_values[seedWidgetIndex] = seedToUse;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            return result;
        };
    }
});

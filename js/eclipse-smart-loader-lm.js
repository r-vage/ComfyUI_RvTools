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
                
                if (templateAction === "Save" && newTemplateName && newTemplateName.trim()) {
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
                const hasAction = (templateAction === "Save" || templateAction === "Delete");
                
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
                const templateAction = getWidgetValue("template_action");
                const isSaveMode = (templateAction === "Save");
                const isLoadMode = (templateAction === "Load");
                const isDeleteMode = (templateAction === "Delete");
                const isNoneMode = (templateAction === "None");
                
                // Check if text input is connected
                const textInput = node.inputs?.find(input => input.name === "text");
                const isTextConnected = textInput && textInput.link != null;
                
                // Determine current model type for widget visibility
                let widgetModelType = "qwenvl"; // Default
                let isGGUF = false;
                
                if (isLoadMode) {
                    // Load mode: use detected model type from template
                    widgetModelType = currentModelType || "qwenvl";
                    // currentIsGGUF is already set from template detection
                    isGGUF = currentIsGGUF;
                } else {
                    // None/Save modes: extract from combined model_type widget
                    const modelTypeValue = getWidgetValue("model_type") || "QwenVL";
                    isGGUF = modelTypeValue.includes("(GGUF)");
                    const baseType = modelTypeValue.replace(" (GGUF)", "").trim();
                    widgetModelType = baseType.toLowerCase();
                }
                
                const isQwenVL = (widgetModelType === "qwenvl");
                const isFlorence2 = (widgetModelType === "florence2");
                const isLLM = (widgetModelType === "llm");
                
                // Template management widgets
                setWidgetVisible("template_action", true);
                setWidgetVisible("template_name", isLoadMode || isDeleteMode);
                setWidgetVisible("new_template_name", isSaveMode);
                
                // Configuration widgets (visible in None and Save modes)
                const showConfig = isNoneMode || isSaveMode;
                setWidgetVisible("model_type", showConfig);
                setWidgetVisible("model_source", showConfig);
                
                // Model source-dependent widgets
                const modelSource = getWidgetValue("model_source") || "HuggingFace";
                setWidgetVisible("repo_id", showConfig && modelSource === "HuggingFace");
                setWidgetVisible("local_path", showConfig && modelSource === "HuggingFace");
                setWidgetVisible("local_model", showConfig && modelSource === "Local");
                
                // mmproj widgets (only for GGUF QwenVL models)
                const showMmproj = showConfig && isGGUF && isQwenVL;
                setWidgetVisible("mmproj_source", showMmproj);
                
                const mmprojSource = getWidgetValue("mmproj_source") || "Local";
                setWidgetVisible("mmproj_local", showMmproj && mmprojSource === "Local");
                setWidgetVisible("mmproj_url", showMmproj && mmprojSource === "HuggingFace");
                setWidgetVisible("mmproj_path", showMmproj && mmprojSource === "HuggingFace");
                
                // Update button visibility
                updateTemplateButton();
                
                // Generation/runtime parameters (visible in all modes except Delete)
                const showGeneration = !isDeleteMode;
                setWidgetVisible("quantization", showGeneration && !isGGUF);
                setWidgetVisible("attention_mode", showGeneration && !isGGUF);
                setWidgetVisible("context_size", showGeneration && isGGUF); // Only for GGUF - Transformers auto-handles context
                setWidgetVisible("max_tokens", showGeneration);
                setWidgetVisible("memory_cleanup", showGeneration);
                setWidgetVisible("keep_model_loaded", showGeneration && !isGGUF);
                setWidgetVisible("seed", showGeneration);
                
                // Model-specific widgets (visible in all modes except Delete)
                setWidgetVisible("qwen_preset_prompt", showGeneration && isQwenVL);
                setWidgetVisible("qwen_custom_prompt", showGeneration && isQwenVL && !isTextConnected);
                
                setWidgetVisible("florence_task", showGeneration && isFlorence2);
                
                // Florence text input visibility
                if (showGeneration && isFlorence2 && !isTextConnected) {
                    const florenceTask = getWidgetValue("florence_task") || "";
                    const taskNeedsTextInput = florenceTask.includes("grounding") || 
                                              florenceTask.includes("detection") || 
                                              florenceTask.includes("ocr") || 
                                              florenceTask.includes("region") ||
                                              florenceTask.includes("docvqa");
                    setWidgetVisible("florence_text_input", taskNeedsTextInput);
                } else {
                    setWidgetVisible("florence_text_input", false);
                }
                
                // Florence convert_to_bboxes visibility
                if (showGeneration && isFlorence2) {
                    const florenceTask = getWidgetValue("florence_task") || "";
                    const showConversion = florenceTask.includes("grounding") || 
                                          florenceTask.includes("detection") || 
                                          florenceTask.includes("ocr") || 
                                          florenceTask.includes("region");
                    setWidgetVisible("convert_to_bboxes", showConversion);
                } else {
                    setWidgetVisible("convert_to_bboxes", false);
                }
                
                setWidgetVisible("llm_instruction_mode", showGeneration && isLLM);
                setWidgetVisible("llm_prompt", showGeneration && isLLM && !isTextConnected);
                
                // LLM custom instruction visibility
                const llmMode = getWidgetValue("llm_instruction_mode") || "";
                setWidgetVisible("llm_custom_instruction", showGeneration && isLLM && llmMode === "Custom Instruction");
                
                // Show advanced parameters if they exist (for Advanced node)
                const advancedWidgets = ["device", "temperature", "top_p", "num_beams", "do_sample", "repetition_penalty", "frame_count", "use_torch_compile"];
                advancedWidgets.forEach(widgetName => {
                    const widget = node.widgets?.find(w => w.name === widgetName);
                    if (widget) {
                        setWidgetVisible(widgetName, showGeneration);
                    }
                });
                
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
                templateActionWidget.callback = async function() {
                    if (originalActionCallback) {
                        originalActionCallback.apply(this, arguments);
                    }
                    
                    const templateAction = getWidgetValue("template_action");
                    const templateName = getWidgetValue("template_name");
                    
                    // Auto-populate new_template_name when switching to Save mode (only if empty)
                    if (templateAction === "Save") {
                        const newTemplateNameWidget = node.widgets?.find(w => w.name === "new_template_name");
                        const currentValue = newTemplateNameWidget?.value || "";
                        
                        // Only auto-fill if the field is empty
                        if (newTemplateNameWidget && !currentValue.trim()) {
                            let suggestedName = "";
                            
                            // Priority 1: Use loaded template name if available
                            if (templateName && templateName !== "None") {
                                suggestedName = templateName;
                            } else {
                                // Priority 2: Use local_model if available
                                const modelSource = getWidgetValue("model_source");
                                if (modelSource === "Local") {
                                    const localModel = getWidgetValue("local_model");
                                    if (localModel && localModel !== "None") {
                                        suggestedName = localModel;
                                    }
                                } else {
                                    // Priority 3: Extract model name from repo_id
                                    const repoId = getWidgetValue("repo_id");
                                    if (repoId && repoId.trim()) {
                                        // Extract last part after / (e.g., "org/model-name" -> "model-name")
                                        const parts = repoId.trim().split('/');
                                        suggestedName = parts[parts.length - 1];
                                    }
                                }
                            }
                            
                            // Set the suggested name only if we found one
                            if (suggestedName) {
                                newTemplateNameWidget.value = suggestedName;
                            }
                        }
                    }
                    
                    // When switching to None or Save mode from Load mode, populate widgets with template values
                    if ((templateAction === "None" || templateAction === "Save") && templateName && templateName !== "None") {
                        const config = await loadTemplateConfig(templateName);
                        if (config) {
                            // Set model_type (combine base type with GGUF if needed)
                            const detectedType = await detectModelType(templateName);
                            const isGGUF = await detectIsGGUF(templateName);
                            let modelTypeValue = "";
                            if (detectedType === "qwenvl") {
                                modelTypeValue = isGGUF ? "QwenVL (GGUF)" : "QwenVL";
                            } else if (detectedType === "florence2") {
                                modelTypeValue = isGGUF ? "Florence2 (GGUF)" : "Florence2";
                            } else if (detectedType === "llm") {
                                modelTypeValue = isGGUF ? "LLM (GGUF)" : "LLM";
                            }
                            
                            const modelTypeWidget = node.widgets?.find(w => w.name === "model_type");
                            if (modelTypeWidget && modelTypeValue) {
                                modelTypeWidget.value = modelTypeValue;
                            }
                            
                            // Set model_source and populate all relevant fields from template
                            const hasRepoId = config.repo_id && config.repo_id.trim() !== "";
                            const hasLocalPath = config.local_path && config.local_path.trim() !== "";
                            const modelSourceWidget = node.widgets?.find(w => w.name === "model_source");
                            
                            if (modelSourceWidget) {
                                // Prefer Local if local_path exists (cleaner UI)
                                if (hasLocalPath) {
                                    modelSourceWidget.value = "Local";
                                    const localModelWidget = node.widgets?.find(w => w.name === "local_model");
                                    if (localModelWidget) {
                                        // Normalize local_path - add trailing slash for directories (not .gguf files)
                                        let normalizedPath = config.local_path;
                                        if (normalizedPath && !normalizedPath.endsWith('.gguf') && !normalizedPath.endsWith('/')) {
                                            normalizedPath = normalizedPath + '/';
                                        }
                                        localModelWidget.value = normalizedPath;
                                    }
                                } else if (hasRepoId) {
                                    // Fallback to HuggingFace if only repo_id exists
                                    modelSourceWidget.value = "HuggingFace";
                                    const repoIdWidget = node.widgets?.find(w => w.name === "repo_id");
                                    if (repoIdWidget) {
                                        repoIdWidget.value = config.repo_id;
                                    }
                                    const localPathWidget = node.widgets?.find(w => w.name === "local_path");
                                    if (localPathWidget) {
                                        localPathWidget.value = config.local_path || "";
                                    }
                                }
                            }
                            
                            // Update currentModelType and currentIsGGUF for visibility logic
                            currentModelType = detectedType;
                            currentIsGGUF = isGGUF;
                            
                            // For GGUF QwenVL models, load mmproj fields
                            if (isGGUF && detectedType === "qwenvl") {
                                const hasMmprojUrl = config.mmproj_url && config.mmproj_url.trim() !== "";
                                const hasMmprojPath = config.mmproj_path && config.mmproj_path.trim() !== "";
                                
                                const mmprojSourceWidget = node.widgets?.find(w => w.name === "mmproj_source");
                                if (mmprojSourceWidget) {
                                    // Prefer Local if mmproj_path exists (cleaner UI)
                                    if (hasMmprojPath) {
                                        mmprojSourceWidget.value = "Local";
                                        const mmprojLocalWidget = node.widgets?.find(w => w.name === "mmproj_local");
                                        if (mmprojLocalWidget) {
                                            mmprojLocalWidget.value = config.mmproj_path;
                                        }
                                    } else if (hasMmprojUrl) {
                                        // Fallback to HuggingFace if only mmproj_url exists
                                        mmprojSourceWidget.value = "HuggingFace";
                                        const mmprojUrlWidget = node.widgets?.find(w => w.name === "mmproj_url");
                                        if (mmprojUrlWidget) {
                                            mmprojUrlWidget.value = config.mmproj_url;
                                        }
                                        const mmprojPathWidget = node.widgets?.find(w => w.name === "mmproj_path");
                                        if (mmprojPathWidget) {
                                            mmprojPathWidget.value = config.mmproj_path || "";
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    updateVisibility();
                };
            }
            
            // Hook into model_type widget to update visibility
            const modelTypeWidget = node.widgets?.find(w => w.name === "model_type");
            if (modelTypeWidget) {
                const originalTypeCallback = modelTypeWidget.callback;
                modelTypeWidget.callback = function() {
                    if (originalTypeCallback) {
                        originalTypeCallback.apply(this, arguments);
                    }
                    updateVisibility();
                };
            }
            
            // Hook into model_source widget to show/hide model input fields
            const modelSourceWidget = node.widgets?.find(w => w.name === "model_source");
            if (modelSourceWidget) {
                const originalSourceCallback = modelSourceWidget.callback;
                modelSourceWidget.callback = function() {
                    if (originalSourceCallback) {
                        originalSourceCallback.apply(this, arguments);
                    }
                    updateVisibility();
                };
            }
            
            // Hook into mmproj_source widget to show/hide mmproj input fields
            const mmprojSourceWidget = node.widgets?.find(w => w.name === "mmproj_source");
            if (mmprojSourceWidget) {
                const originalMmprojCallback = mmprojSourceWidget.callback;
                mmprojSourceWidget.callback = function() {
                    if (originalMmprojCallback) {
                        originalMmprojCallback.apply(this, arguments);
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
                    const templateAction = getWidgetValue("template_action");
                    
                    // Only load template values when in Load mode
                    if (templateAction === "Load" && templateName && templateName !== "None") {
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
            
            // Intercept the onExecuted to track last seed
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
                    
                    console.log(`✓ Template save completed, refreshing template list...`);
                    await new Promise(resolve => setTimeout(resolve, 100));
                    await refreshTemplateList();
                    
                    setWidgetValue("template_action", "None");
                    setWidgetValue("new_template_name", "");
                    
                    updateVisibility();
                    console.log(`✓ Template saved: ${savedTemplateName}, switched to None mode`);
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
                        
                        console.log(`✓ Template save interrupted (as expected), refreshing template list...`);
                        await new Promise(resolve => setTimeout(resolve, 300));
                        await refreshTemplateList();
                        
                        setWidgetValue("template_action", "None");
                        setWidgetValue("new_template_name", "");
                        
                        updateVisibility();
                        console.log(`✓ Template saved: ${savedTemplateName}, switched to None mode`);
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
            
            // Initial setup - detect model type from template if in Load mode
            setTimeout(async () => {
                const templateAction = getWidgetValue("template_action");
                const templateName = getWidgetValue("template_name");
                
                if (templateAction === "Load" && templateName && templateName !== "None") {
                    const detectedType = await detectModelType(templateName);
                    const isGGUF = await detectIsGGUF(templateName);
                    currentModelType = detectedType;
                    currentIsGGUF = isGGUF;
                    
                    // Trigger template callback to load widget values
                    const templateWidget = node.widgets?.find(w => w.name === "template_name");
                    if (templateWidget && templateWidget.callback) {
                        await templateWidget.callback.call(templateWidget);
                    }
                } else {
                    // Default to QwenVL if in None mode
                    currentModelType = "qwenvl";
                    currentIsGGUF = false;
                }
                
                // Important: Call updateVisibility to ensure widgets are shown/hidden correctly on node creation
                updateVisibility();
            }, 10);
            
            // Hook into onConfigure to reload visibility when workflow is loaded
            const onConfigure = node.onConfigure;
            node.onConfigure = function(info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                // After workflow is configured, detect model type and update visibility
                setTimeout(async () => {
                    const templateName = getWidgetValue("template_name");
                    const templateAction = getWidgetValue("template_action");
                    
                    // Detect model type without loading template values (values come from saved workflow)
                    if (templateAction === "Load" && templateName && templateName !== "None") {
                        const detectedType = await detectModelType(templateName);
                        const isGGUF = await detectIsGGUF(templateName);
                        currentModelType = detectedType;
                        currentIsGGUF = isGGUF;
                    } else if (templateAction === "None") {
                        // Extract from model_type widget for None mode
                        const modelTypeValue = getWidgetValue("model_type") || "QwenVL";
                        currentIsGGUF = modelTypeValue.includes("(GGUF)");
                        const baseType = modelTypeValue.replace(" (GGUF)", "").trim();
                        currentModelType = baseType.toLowerCase();
                    } else {
                        currentModelType = "qwenvl";
                        currentIsGGUF = false;
                    }
                    
                    // Always call updateVisibility after configuration
                    updateVisibility();
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

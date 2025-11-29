import { app } from "../../scripts/app.js";

// Default parameter values for each model type
const DEFAULT_PARAMS = {
    "QwenVL": {
        "device": "cuda",
        "use_torch_compile": false,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "num_beams": 3,
        "do_sample": true,
        "repetition_penalty": 1.0,
        "frame_count": 8
    },
    "Florence2": {
        "device": "cuda",
        "use_torch_compile": false,
        "num_beams": 3,
        "do_sample": true,
        "convert_to_bboxes": false
    },
    "LLM": {
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2
    }
};

// Load defaults from config file
async function loadAdvancedDefaults() {
    try {
        const response = await fetch("/eclipse/smartlml_advanced_defaults");
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        console.warn("[Eclipse] Could not load advanced defaults, using built-in defaults:", e);
    }
    return DEFAULT_PARAMS;
}

// Language Model Advanced Options - Widget visibility based on model type
app.registerExtension({
    name: "Eclipse.SmartLoaderLMAdvancedOptions",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Pipe Out LM Advanced Options [Eclipse]") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Load default parameters from config
                const configDefaults = await loadAdvancedDefaults();
                
                // Parameter support matrix
                const parameterSupport = {
                    "QwenVL": {
                        "device": true,
                        "use_torch_compile": true,
                        "temperature": true,  // GGUF only, but show for simplicity
                        "top_p": true,
                        "top_k": true,  // GGUF only
                        "num_beams": true,
                        "do_sample": true,
                        "repetition_penalty": true,
                        "frame_count": true,
                        "convert_to_bboxes": false
                    },
                    "Florence2": {
                        "device": true,
                        "use_torch_compile": true,
                        "temperature": false,
                        "top_p": false,
                        "top_k": false,
                        "num_beams": true,
                        "do_sample": true,
                        "repetition_penalty": false,  // Not used by Florence2
                        "frame_count": false,
                        "convert_to_bboxes": true
                    },
                    "LLM": {
                        "device": false,  // LLM is GGUF-only, no device selection
                        "use_torch_compile": false,
                        "temperature": true,
                        "top_p": true,
                        "top_k": true,
                        "num_beams": false,
                        "do_sample": false,
                        "repetition_penalty": true,
                        "frame_count": false,
                        "convert_to_bboxes": false
                    }
                };
                
                // Helper to get widget by name
                const getWidget = (name) => {
                    return this.widgets?.find(w => w.name === name);
                };
                
                // Helper to set widget visibility
                const setWidgetVisible = (name, visible) => {
                    const widget = getWidget(name);
                    if (!widget) return;
                    
                    if (visible) {
                        // Show widget - restore original type
                        if (widget.origType) {
                            widget.type = widget.origType;
                            // Don't delete origType - keep it in case we hide again later
                        } else if (widget.type === "converted-widget") {
                            // Widget is hidden but origType wasn't saved - this shouldn't happen
                            // Default to restoring based on widget name patterns
                            widget.type = widget.name.includes("temperature") || widget.name.includes("top_p") || widget.name.includes("repetition_penalty") ? "number" : widget.type;
                            widget.origType = widget.type;
                        }
                        delete widget.computeSize;
                        widget.hidden = false; // Make sure widget is visible
                    } else {
                        // Hide widget - save original type first
                        if (widget.type !== "converted-widget" && !widget.origType) {
                            widget.origType = widget.type;
                        }
                        widget.type = "converted-widget";
                        widget.computeSize = () => [0, -4];
                        widget.hidden = true; // Hide widget from rendering
                    }
                };
                
                // Apply default parameters from config
                const applyDefaults = (modelType) => {
                    const defaults = configDefaults[modelType] || DEFAULT_PARAMS[modelType];
                    if (!defaults) return;
                    
                    // Apply each default value if widget exists
                    for (const [param, defaultValue] of Object.entries(defaults)) {
                        const widget = getWidget(param);
                        if (widget && widget.value !== undefined) {
                            // Only update if widget hasn't been modified from its previous default
                            // This is a simple approach - just update the value
                            widget.value = defaultValue;
                        }
                    }
                };
                
                // Update visibility based on model type
                const updateVisibility = () => {
                    const modelTypeWidget = getWidget("model_type");
                    if (!modelTypeWidget) return;
                    
                    const selectedType = modelTypeWidget.value || "QwenVL";
                    const supportMatrix = parameterSupport[selectedType] || parameterSupport["QwenVL"];
                    
                    // Apply default parameters for this model type
                    applyDefaults(selectedType);
                    
                    // Update visibility for all parameters except model_type
                    for (const [param, isSupported] of Object.entries(supportMatrix)) {
                        setWidgetVisible(param, isSupported);
                    }
                    
                    // Always show model_type itself
                    setWidgetVisible("model_type", true);
                    
                    // Smart resize after widget visibility changes
                    setTimeout(() => {
                        // Force canvas update before computing size
                        this.setDirtyCanvas(true, false);
                        
                        const computedSize = this.computeSize();
                        const currentSize = this.size;
                        
                        // Set minimum size
                        const minWidth = 259;
                        const minHeight = 100;
                        
                        // Preserve current width (only enforce minimum), adjust height to computed size
                        let newWidth = Math.max(currentSize[0], minWidth);
                        // Always add padding as computeSize doesn't account for all spacing
                        let padding = 5;
                        let newHeight = Math.max(computedSize[1] + padding, minHeight);
                        
                        // Always resize to match computed size to ensure proper widget display
                        this.setSize([newWidth, newHeight]);
                        
                        this.setDirtyCanvas(true, true);
                    }, 50);
                };
                
                // Override onResize to enforce minimum size based on computed size
                const originalOnResize = this.onResize;
                this.onResize = function(size) {
                    // Compute minimum size needed for all widgets
                    const computedSize = this.computeSize();
                    const minWidth = 259;
                    // Always add padding as computeSize doesn't account for all spacing
                    const padding = 5;
                    const minHeight = Math.max(computedSize[1] + padding, 100);
                    
                    // Enforce minimum dimensions
                    size[0] = Math.max(size[0], minWidth);
                    size[1] = Math.max(size[1], minHeight);
                    
                    if (originalOnResize) {
                        return originalOnResize.apply(this, [size]);
                    }
                };
                
                // Auto-save function - saves changed parameter values to config
                const autoSaveDefaults = async () => {
                    const modelTypeWidget = getWidget("model_type");
                    if (!modelTypeWidget) return;
                    
                    const modelType = modelTypeWidget.value || "QwenVL";
                    const supportMatrix = parameterSupport[modelType] || parameterSupport["QwenVL"];
                    
                    // Collect values for supported parameters
                    const updates = { model_type: modelType };
                    
                    for (const [param, isSupported] of Object.entries(supportMatrix)) {
                        if (!isSupported) continue;
                        
                        const widget = getWidget(param);
                        if (widget && widget.value !== undefined) {
                            updates[param] = widget.value;
                        }
                    }
                    
                    // Send updates to server
                    try {
                        const response = await fetch('/eclipse/smartlml_advanced_defaults', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(updates)
                        });
                        
                        if (!response.ok) {
                            console.error(`[SmartLM Advanced] Auto-save failed: ${response.status}`);
                        }
                    } catch (e) {
                        console.error(`[SmartLM Advanced] Auto-save error:`, e);
                    }
                };
                
                // Hook into all parameter widgets to trigger auto-save
                const paramWidgets = [
                    "device", "use_torch_compile", "temperature", "top_p", "top_k",
                    "num_beams", "do_sample", "repetition_penalty", "frame_count", "convert_to_bboxes"
                ];
                
                paramWidgets.forEach(widgetName => {
                    const widget = getWidget(widgetName);
                    if (widget) {
                        const originalCallback = widget.callback;
                        widget.callback = function() {
                            if (originalCallback) {
                                originalCallback.apply(this, arguments);
                            }
                            // Auto-save when parameter changes
                            autoSaveDefaults();
                        };
                    }
                });
                
                // Initial visibility update
                setTimeout(() => {
                    updateVisibility();
                }, 10);
                
                // Hook into model_type widget callback
                const modelTypeWidget = getWidget("model_type");
                if (modelTypeWidget) {
                    const originalCallback = modelTypeWidget.callback;
                    modelTypeWidget.callback = function() {
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        updateVisibility();
                        // Auto-save when model type changes
                        autoSaveDefaults();
                    };
                }
                
                return result;
            };
        }
    }
});

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Dynamic widget visibility for Smart Folder
// Shows/hides widgets based on generation mode selection and batch folder options
// Handles auto-increment for batch_number and skip_calculation

import { app } from "../../scripts/app.js";

const NODE_NAME = "Smart Folder [RvTools]";

app.registerExtension({
    name: "RvTools.SmartFolder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;

            // Initialize storage for last values
            node._rvtools_lastBatchNumber = null;
            node._rvtools_lastSkipFirstFramesCalc = null;

            // Helper function to hide/show a widget (using ComfyUI's proper method)
            const setWidgetVisible = (widgetName, visible) => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (!widget) return;

                if (visible) {
                    // Show widget - restore original type
                    if (widget.origType) {
                        widget.type = widget.origType;
                        // Don't delete origType - keep it in case we hide again later
                    } else if (widget.type === "converted-widget") {
                        // Widget is hidden but origType wasn't saved - this shouldn't happen
                        // Try to infer the original type based on widget name patterns
                        if (widgetName.includes("width") || widgetName.includes("height") || widgetName.includes("batch_number") || widgetName.includes("skip")) {
                            widget.type = "number";
                            widget.origType = "number";
                        } else {
                            widget.type = "combo";
                            widget.origType = "combo";
                        }
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

            // Get widget values safely
            const getWidgetValue = (name) => {
                const widget = node.widgets?.find(w => w.name === name);
                return widget ? widget.value : null;
            };

            // Main visibility update function
            const updateVisibility = () => {
                const generationMode = getWidgetValue("generation_mode");
                const createDateTimeFolder = getWidgetValue("create_date_time_folder");
                const createBatchFolder = getWidgetValue("create_batch_folder");
                const imageSize = getWidgetValue("image_size");
                const videoSize = getWidgetValue("video_size");

                const isImageMode = (generationMode === "Image Mode");
                const isVideoMode = (generationMode === "Video Mode");
                const isCustomImageSize = (imageSize === "Custom");
                const isCustomVideoSize = (videoSize === "Custom");

                // Date/Time configuration widgets
                setWidgetVisible("date_time_format", createDateTimeFolder);
                setWidgetVisible("date_time_position", createDateTimeFolder);

                // Image-specific widgets
                setWidgetVisible("root_folder_image", isImageMode);
                setWidgetVisible("image_size", isImageMode);
                setWidgetVisible("width", isImageMode && isCustomImageSize);
                setWidgetVisible("height", isImageMode && isCustomImageSize);

                // Video-specific widgets
                setWidgetVisible("root_folder_video", isVideoMode);
                setWidgetVisible("video_size", isVideoMode);
                setWidgetVisible("video_width", isVideoMode && isCustomVideoSize);
                setWidgetVisible("video_height", isVideoMode && isCustomVideoSize);
                setWidgetVisible("frame_rate", isVideoMode);
                setWidgetVisible("frame_load_cap", isVideoMode);
                setWidgetVisible("context_length", isVideoMode);
                setWidgetVisible("loop_count", isVideoMode);
                setWidgetVisible("overlap", isVideoMode);
                setWidgetVisible("skip_first_frames", isVideoMode);
                setWidgetVisible("skip_calculation", isVideoMode);
                setWidgetVisible("skip_calculation_control", isVideoMode);
                setWidgetVisible("select_every_nth", isVideoMode);

                // Batch folder widgets (only visible when create_batch_folder is enabled)
                setWidgetVisible("batch_folder_name", createBatchFolder);
                setWidgetVisible("batch_number", createBatchFolder);
                setWidgetVisible("batch_number_control", createBatchFolder);

                // Mode-specific common widgets
                setWidgetVisible("batch_size", isImageMode);

                // Common widgets (always visible)
                // generation_mode, project_root_name, create_date_time_folder,
                // create_batch_folder
                // Seed widgets are always visible (handled by rvtools-seed.js extension)
                
                // Smart resize - only resize height if needed, preserve width
                // Use setTimeout to ensure hidden widgets' computeSize is applied first
                setTimeout(() => {
                    // Force canvas update before computing size
                    node.setDirtyCanvas(true, false);
                    
                    const computedSize = node.computeSize();
                    const currentSize = node.size;
                    
                    // Set minimum size (width: 259, height: 100 for flexibility)
                    const minWidth = 259;
                    const minHeight = 100;
                    
                    // Preserve current width (only enforce minimum), only adjust height
                    let newWidth = Math.max(currentSize[0], minWidth);
                    let newHeight = Math.max(computedSize[1], minHeight);
                    
                    // Add padding to ensure all widgets fit properly
                    newHeight += 5;
                    
                    // Always resize height to match computed size
                    // This ensures proper sizing when widgets are hidden/shown
                    const heightDiff = Math.abs(currentSize[1] - newHeight);
                    const isGrowing = newHeight > currentSize[1];
                    
                    // Always resize when growing (showing widgets)
                    // Only resize when shrinking if difference is significant (> 10px)
                    if (isGrowing || heightDiff > 10) {
                        node.setSize([newWidth, newHeight]);
                    }
                    
                    node.setDirtyCanvas(true, true);
                }, 50);
            };

            // Hook into relevant widget changes
            const relevantWidgets = [
                "generation_mode",
                "create_date_time_folder",
                "create_batch_folder",
                "image_size",
                "video_size",
                "root_folder_image",
                "root_folder_video"
            ];

            relevantWidgets.forEach(widgetName => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (widget) {
                    const originalCallback = widget.callback;
                    widget.callback = function(value) {
                        updateVisibility();
                        if (originalCallback) {
                            originalCallback(value);
                        }
                    };
                }
            });

            // Initial visibility update
            updateVisibility();

            // Additional update after workflow loading (fixes visibility on page refresh)
            setTimeout(() => updateVisibility(), 50);

            return r;
        };
    },
    
    async setup() {
        // Hook into the graphToPrompt to handle auto-increment
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Call the original graphToPrompt first
            const result = await originalGraphToPrompt.apply(this, arguments);

            // Now modify the prompt data for Smart Folder nodes with increment controls
            const nodes = app.graph._nodes;
            for (const node of nodes) {
                if (node.type === NODE_NAME) {
                    // Skip if node is muted or bypassed
                    if (node.mode === 2 || node.mode === 4) {
                        continue;
                    }
                    
                    // Check if this node is in the prompt
                    const nodeId = String(node.id);
                    if (result.output && result.output[nodeId]) {
                        const inputs = result.output[nodeId].inputs;
                        
                        // Handle batch_number increment
                        const batchNumberWidget = node.widgets?.find(w => w.name === "batch_number");
                        const batchNumberControl = node.widgets?.find(w => w.name === "batch_number_control");
                        
                        if (batchNumberWidget && batchNumberControl && inputs) {
                            if (batchNumberControl.value === "increment") {
                                // If we have a last value, increment it
                                if (node._rvtools_lastBatchNumber != null) {
                                    const newValue = node._rvtools_lastBatchNumber + 1;
                                    // Only change inputs/workflow/UI if the new value differs
                                    try {
                                        const existing = inputs.batch_number;
                                        if (!inputs.batch_number || Number(existing) !== newValue) {
                                            inputs.batch_number = newValue;
                                        }
                                    } catch (e) {
                                        inputs.batch_number = newValue;
                                    }
                                    node._rvtools_lastBatchNumber = newValue;
                                    // Update the widget value displayed in the UI only if different
                                    try {
                                        if (Number(batchNumberWidget.value) !== newValue) {
                                            batchNumberWidget.value = newValue;
                                        }
                                    } catch (e) {}
                                    
                                    // Update workflow data if present
                                    if (result.workflow && result.workflow.nodes) {
                                        const workflowNode = result.workflow.nodes.find(n => n.id === node.id);
                                        if (workflowNode && workflowNode.widgets_values) {
                                            const widgetIndex = node.widgets.indexOf(batchNumberWidget);
                                            if (widgetIndex >= 0) {
                                                // Only update workflow stored value if it differs
                                                try {
                                                    const oldW = workflowNode.widgets_values[widgetIndex];
                                                    if (oldW !== newValue) {
                                                        workflowNode.widgets_values[widgetIndex] = newValue;
                                                    }
                                                } catch (e) {}
                                            }
                                        }
                                    }
                                } else {
                                    // First run, store the current value
                                    node._rvtools_lastBatchNumber = batchNumberWidget.value;
                                }
                            } else {
                                // Fixed mode, just store the current value
                                node._rvtools_lastBatchNumber = batchNumberWidget.value;
                            }
                        }
                        
                        // Handle skip_calculation increment
                        const skipCalcWidget = node.widgets?.find(w => w.name === "skip_calculation");
                        const skipCalcControl = node.widgets?.find(w => w.name === "skip_calculation_control");
                        
                        if (skipCalcWidget && skipCalcControl && inputs) {
                            if (skipCalcControl.value === "increment") {
                                // If we have a last value, increment it
                                if (node._rvtools_lastSkipFirstFramesCalc != null) {
                                    const newValue = node._rvtools_lastSkipFirstFramesCalc + 1;
                                    // Only change inputs/workflow/UI if the new value differs
                                    try {
                                        const existing = inputs.skip_calculation;
                                        if (!inputs.skip_calculation || Number(existing) !== newValue) {
                                            inputs.skip_calculation = newValue;
                                        }
                                    } catch (e) {
                                        inputs.skip_calculation = newValue;
                                    }
                                    node._rvtools_lastSkipFirstFramesCalc = newValue;

                                    // Update the widget value displayed in the UI only if different
                                    if (Number(skipCalcWidget.value) !== newValue) {
                                        skipCalcWidget.value = newValue;
                                    }
                                    
                                    // Update workflow data if present
                                    if (result.workflow && result.workflow.nodes) {
                                        const workflowNode = result.workflow.nodes.find(n => n.id === node.id);
                                        if (workflowNode && workflowNode.widgets_values) {
                                            const widgetIndex = node.widgets.indexOf(skipCalcWidget);
                                            if (widgetIndex >= 0) {
                                                try {
                                                    const oldW = workflowNode.widgets_values[widgetIndex];
                                                    if (oldW !== newValue) {
                                                        workflowNode.widgets_values[widgetIndex] = newValue;
                                                    }
                                                } catch (e) {}
                                            }
                                        }
                                    }
                                } else {
                                    // First run, store the current value
                                    node._rvtools_lastSkipFirstFramesCalc = skipCalcWidget.value;
                                }
                            } else {
                                // Fixed mode, just store the current value
                                node._rvtools_lastSkipFirstFramesCalc = skipCalcWidget.value;
                            }
                        }
                    }
                }
            }
            
            return result;
        };
    }
});

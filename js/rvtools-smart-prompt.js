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
* Dynamic widget visibility for Smart Prompt
* Shows/hides widgets based on selected folder
* Includes auto-sizing functionality
*/

import { app } from "../../scripts/app.js";

const NODE_NAME = "Smart Prompt [Eclipse]";

// Seed constants (from seed.js)
const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";
const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

app.registerExtension({
    name: "Eclipse.SmartPrompt",
    
    async setup() {
        // Hook into the graphToPrompt to modify seed values in the prompt data
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Call the original graphToPrompt first
            const result = await originalGraphToPrompt.apply(this, arguments);

            // Now modify the prompt data for Smart Prompt nodes
            const nodes = app.graph._nodes;
            for (const node of nodes) {
                if (node.type === NODE_NAME && node._Eclipse_seedWidget) {
                    // Skip if node is muted or bypassed
                    if (node.mode === 2 || node.mode === 4) {
                        continue;
                    }
                    
                    // Check if this node is in the prompt
                    const nodeId = String(node.id);
                    if (result.output && result.output[nodeId]) {
                        const seedToUse = node.getSeedToUse();
                        
                        // If seedToUse is null, it means seed_input is connected
                        // Skip seed modification and let the connection pass through
                        if (seedToUse === null) {
                            continue;
                        }
                        
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
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }
        
        // Method to generate random seed (from seed.js)
        nodeType.prototype.generateRandomSeed = function() {
            const step = this._Eclipse_seedWidget?.options?.step || 1;
            const randomMin = this._Eclipse_randomMin || 0;
            const randomMax = this._Eclipse_randomMax || 1125899906842624;
            const randomRange = (randomMax - randomMin) / (step / 10);
            let seed = Math.floor(Math.random() * randomRange) * (step / 10) + randomMin;
            
            // Avoid special seeds
            if (SPECIAL_SEEDS.includes(seed)) {
                seed = 0;
            }
            return seed;
        };
        
        // Method to determine seed to use (from seed.js)
        nodeType.prototype.getSeedToUse = function() {
            // Check if seed_input is connected - if so, skip seed resolution
            // Let the connection pass through naturally to the backend
            const seedInput = this.inputs?.find(input => input.name === "seed_input");
            if (seedInput && seedInput.link != null) {
                // Return null to indicate we should skip seed resolution
                // The backend will receive the seed from the connected node
                return null;
            }
            
            // Normal seed generation logic when seed_input is not connected
            const inputSeed = Number(this._Eclipse_seedWidget.value);
            
            // Check if we have a cached resolved seed for this input seed
            // This prevents generating different random seeds on multiple calls
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
            
            const finalSeed = seedToUse != null ? seedToUse : inputSeed;
            
            // Cache the resolved seed for this input seed
            this._Eclipse_cachedInputSeed = inputSeed;
            this._Eclipse_cachedResolvedSeed = finalSeed;
            
            return finalSeed;
        };
        
        // Intercept the prompt before it's sent to the server (from seed.js)
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            
            // Store the seed that was actually used if available
            if (message && message.seed !== undefined) {
                this._Eclipse_lastSeed = message.seed;
            }
            
            return result;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;
            
            // ===== SEED WIDGET SETUP ===== (from seed.js)
            // Find the seed widget and remove control_after_generate
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

            if (!seedWidget) {
                console.warn(`[Eclipse-SmartPrompt] Could not find Seed widget. Widgets:`, this.widgets.map(w => ({ name: w.name, label: w.label })));
            } else {
                // Store seed widget and initialize seed tracking properties
                this._Eclipse_seedWidget = seedWidget;
                this._Eclipse_lastSeed = undefined;
                this._Eclipse_randomMin = 0;
                this._Eclipse_randomMax = 1125899906842624;
                this._Eclipse_cachedInputSeed = null;
                this._Eclipse_cachedResolvedSeed = null;
                
                // Hook into the seed widget's value setter to clear cache when it changes
                const originalCallback = seedWidget.callback;
                seedWidget.callback = (value) => {
                    // Clear the seed cache when the seed value changes
                    this._Eclipse_cachedInputSeed = null;
                    this._Eclipse_cachedResolvedSeed = null;
                    // Call the original callback if it exists
                    if (originalCallback) {
                        return originalCallback.call(seedWidget, value);
                    }
                };
                
                // Add buttons after the seed widget
                const seedWidgetIndex = this.widgets.indexOf(seedWidget);
                
                // Button: Randomize Each Time
                const randomizeButton = this.addWidget(
                    "button",
                    "🎲 Randomize Each Time",
                    "",
                    () => {
                        seedWidget.value = SPECIAL_SEED_RANDOM;
                        // Trigger callback to notify listeners
                        if (seedWidget.callback) {
                            seedWidget.callback(SPECIAL_SEED_RANDOM);
                        }
                    },
                    { serialize: false }
                );
                
                // Button: New Fixed Random
                const newRandomButton = this.addWidget(
                    "button",
                    "🎲 New Fixed Random",
                    "",
                    () => {
                        const newSeed = this.generateRandomSeed();
                        seedWidget.value = newSeed;
                        // Trigger callback to notify listeners
                        if (seedWidget.callback) {
                            seedWidget.callback(newSeed);
                        }
                    },
                    { serialize: false }
                );
                
                // Button: Use Last Queued Seed
                const lastSeedButton = this.addWidget(
                    "button",
                    LAST_SEED_BUTTON_LABEL,
                    "",
                    () => {
                        if (this._Eclipse_lastSeed != null) {
                            seedWidget.value = this._Eclipse_lastSeed;
                            lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                            lastSeedButton.disabled = true;
                        }
                    },
                    { serialize: false }
                );
                lastSeedButton.disabled = true;
                this._Eclipse_lastSeedButton = lastSeedButton;
                
                // Store references to seed buttons for connection state handling
                this._Eclipse_randomizeButton = randomizeButton;
                this._Eclipse_newRandomButton = newRandomButton;
                
                // Move buttons to be right after the seed widget
                const buttonsToMove = [randomizeButton, newRandomButton, lastSeedButton];
                for (let i = buttonsToMove.length - 1; i >= 0; i--) {
                    const button = buttonsToMove[i];
                    const currentIndex = this.widgets.indexOf(button);
                    if (currentIndex !== seedWidgetIndex + 1) {
                        this.widgets.splice(currentIndex, 1);
                        this.widgets.splice(seedWidgetIndex + 1, 0, button);
                    }
                }
                
                // Function to update seed widget and buttons based on seed_input connection
                const updateSeedInputState = () => {
                    // Check if seed_input is connected
                    const seedInput = node.inputs?.find(input => input.name === "seed_input");
                    const seedInputConnected = seedInput && seedInput.link != null;
                    
                    // Store the size before any changes (for comparison)
                    const sizeBeforeUpdate = [...node.size];
                    
                    if (seedInputConnected) {
                        // Hide seed widget when connected
                        if (seedWidget.type !== "converted-widget") {
                            seedWidget.type = "converted-widget";
                            seedWidget.computeSize = () => [0, -4];
                        }
                        seedWidget.hidden = true;
                        
                        // Hide seed buttons completely
                        if (randomizeButton.type !== "converted-widget") {
                            randomizeButton.type = "converted-widget";
                            randomizeButton.computeSize = () => [0, -4];
                        }
                        if (newRandomButton.type !== "converted-widget") {
                            newRandomButton.type = "converted-widget";
                            newRandomButton.computeSize = () => [0, -4];
                        }
                        if (lastSeedButton.type !== "converted-widget") {
                            lastSeedButton.type = "converted-widget";
                            lastSeedButton.computeSize = () => [0, -4];
                        }
                        
                        // Also set them as hidden to prevent rendering
                        randomizeButton.hidden = true;
                        newRandomButton.hidden = true;
                        lastSeedButton.hidden = true;
                    } else {
                        // Show seed widget when not connected
                        if (seedWidget.type !== "number") {
                            seedWidget.type = "number";
                            delete seedWidget.computeSize;
                        }
                        seedWidget.hidden = false;
                        
                        // Show seed buttons by restoring their type
                        if (randomizeButton.type !== "button") {
                            randomizeButton.type = "button";
                            delete randomizeButton.computeSize;
                        }
                        if (newRandomButton.type !== "button") {
                            newRandomButton.type = "button";
                            delete newRandomButton.computeSize;
                        }
                        if (lastSeedButton.type !== "button") {
                            lastSeedButton.type = "button";
                            delete lastSeedButton.computeSize;
                        }
                        
                        // Make them visible again
                        randomizeButton.hidden = false;
                        newRandomButton.hidden = false;
                        lastSeedButton.hidden = false;
                    }
                    
                    // Trigger resize to update node size
                    setTimeout(() => {
                        const currentSize = node.size;
                        const computedSize = node.computeSize();
                        
                        if (seedInputConnected) {
                            // When seed_input is connected and widgets are hidden:
                            // Check if the widgets were already hidden (size didn't change much)
                            // If so, keep current size. Otherwise, allow resize to shrink.
                            const heightDifference = Math.abs(sizeBeforeUpdate[1] - currentSize[1]);
                            if (heightDifference < 10) {
                                // Size barely changed - widgets were already hidden, preserve current size
                                node.setSize([currentSize[0], currentSize[1]]);
                            } else {
                                // Size changed significantly - this is a fresh hide, allow computed size
                                node.setSize([currentSize[0], computedSize[1]]);
                            }
                        } else {
                            // When disconnected, use computed size to show all widgets
                            node.setSize([currentSize[0], computedSize[1]]);
                        }
                        
                        node.setDirtyCanvas(true, true);
                    }, 10);
                };
                
                // Check seed input state on node creation
                setTimeout(() => {
                    updateSeedInputState();
                }, 100);
                
                // Override onConnectionsChange to detect when seed_input is connected/disconnected
                const originalOnConnectionsChange = node.onConnectionsChange;
                node.onConnectionsChange = function(type, index, connected, link_info) {
                    if (originalOnConnectionsChange) {
                        originalOnConnectionsChange.apply(this, arguments);
                    }
                    setTimeout(() => {
                        updateSeedInputState();
                        // Also update visibility to recalculate node size
                        updateVisibility();
                    }, 10);
                };
            }
            
            // ===== FOLDER-BASED WIDGET VISIBILITY ===== (original smart-prompt.js logic)
            
            // Helper function to hide/show a widget
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
                        // Default to "combo" as that's what most widgets are
                        console.warn(`[SmartPrompt] Widget "${widgetName}" is converted-widget but has no origType, defaulting to combo`);
                        widget.type = "combo";
                        widget.origType = "combo"; // Save it for future hide/show cycles
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
            const updateVisibility = async () => {
                const selectedFolder = getWidgetValue("folder");
                
                // Always show folder widget
                setWidgetVisible("folder", true);
                
                // Note: seed widget visibility is handled by updateSeedInputState
                // We skip it here to avoid conflicts with the hidden property

                // Show/hide other widgets based on selected folder
                // Widget names are formatted as: "{folder_name} {widget_name}"
                // Extract folder from the widget name (first word before space)
                node.widgets?.forEach(widget => {
                    // Skip folder and seed widgets
                    if (widget.name === "folder" || widget.name === "seed") {
                        return;
                    }
                    
                    // Skip button widgets (seed control buttons are handled by updateSeedInputState)
                    if (widget.type === "button") {
                        return;
                    }
                    
                    // Skip seed control buttons (even if they're converted-widget)
                    if (widget === node._Eclipse_randomizeButton || 
                        widget === node._Eclipse_newRandomButton || 
                        widget === node._Eclipse_lastSeedButton) {
                        return;
                    }

                    // Extract folder from widget name (format: "foldername widgetname")
                    const widgetFolder = widget.name.split(' ')[0];
                    
                    // Show widget if folder is "All" or matches selected folder
                    const visible = (selectedFolder === "All" || widgetFolder === selectedFolder);
                    setWidgetVisible(widget.name, visible);
                });

                // Auto-resize logic
                setTimeout(() => {
                    // Force canvas update before computing size
                    node.setDirtyCanvas(true, false);

                    const computedSize = node.computeSize();
                    const currentSize = node.size;

                    // Set minimum size
                    const minWidth = 279;
                    const minHeight = 50;

                    // Preserve current width (only enforce minimum), always update height
                    let newWidth = Math.max(currentSize[0], minWidth);
                    let newHeight = Math.max(computedSize[1], minHeight);

                    // Always resize to match computed size for proper widget display
                    node.setSize([newWidth, newHeight]);

                    node.setDirtyCanvas(true, true);
                }, 50);
            };

            // Hook into the folder widget callback
            const folderWidget = node.widgets?.find(w => w.name === "folder");
            
            if (folderWidget) {
                const originalCallback = folderWidget.callback;
                folderWidget.callback = async function() {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    await updateVisibility();
                };
            }

            // Initial visibility update
            setTimeout(async () => {
                await updateVisibility();
            }, 10);

            // Set custom labels for widgets to hide folder prefix
            node.widgets?.forEach(widget => {
                // Skip folder, seed, and button widgets
                if (widget.name !== "folder" && widget.name !== "seed" && widget.type !== "button") {
                    const parts = widget.name.split(' ');
                    if (parts.length > 1) {
                        widget.label = parts.slice(1).join(' ');
                    }
                }
            });

            // Override onResize to enforce maximum width of 279
            const originalOnResize = node.onResize;
            node.onResize = function(size) {
                const maxWidth = 279;
                
                // Enforce maximum width
                if (size[0] > maxWidth) {
                    size[0] = maxWidth;
                }
                
                if (originalOnResize) {
                    return originalOnResize.apply(this, [size]);
                }
            };
            
            // Set initial size to max width if currently larger
            setTimeout(() => {
                const currentSize = node.size;
                if (currentSize[0] > 279) {
                    node.setSize([279, currentSize[1]]);
                }
            }, 10);

            return r;
        };
    },
});
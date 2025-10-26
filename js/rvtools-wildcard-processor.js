/**
 * RvTools Wildcard Processor - JavaScript Extension
 * 
 * Provides UI customization for the RvText_WildcardProcessor node:
 * - Dynamic wildcard combo loading from /rvtools/wildcards/list endpoint
 * - Mode-based UI state management (populate/fixed)
 * - Seed controls with special seed handling (-1, -2, -3)
 * - Real-time wildcard selection integration
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
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
 */

import { app } from "../../scripts/app.js";

// Seed constants
const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";
const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

// Wildcard list cache
let wildcardList = [];
let wildcardListLoading = false;

/**
 * Load available wildcards from server
 */
async function loadWildcardList() {
    if (wildcardListLoading) return;
    wildcardListLoading = true;

    try {
        const response = await fetch("/rvtools/wildcards/list");
        if (response.ok) {
            wildcardList = await response.json();
        }
    } catch (error) {
        console.warn("[RvTools Wildcard] Failed to load wildcard list:", error);
        wildcardList = [];
    } finally {
        wildcardListLoading = false;
    }
}

/**
 * Register the extension
 */
app.registerExtension({
    name: "RvTools.WildcardProcessor",
    
    async setup() {
        // Load wildcard list on startup
        await loadWildcardList();
        
        // Hook into the graphToPrompt to resolve seeds before sending to server
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Call the original graphToPrompt first
            const result = await originalGraphToPrompt.apply(this, arguments);

            // Now resolve seeds for Wildcard Processor nodes
            const nodes = app.graph._nodes;
            for (const node of nodes) {
                if (node.type === "Wildcard Processor [RvTools]" && node._rvtools_seedWidget) {
                    // Skip if node is muted or bypassed
                    if (node.mode === 2 || node.mode === 4) {
                        continue;
                    }
                    
                    // Check if this node is in the prompt
                    const nodeId = String(node.id);
                    if (result.output && result.output[nodeId]) {
                        // Check if seed widget has a connection
                        const seedWidget = node._rvtools_seedWidget;
                        const seedInput = node.inputs?.find(input => input.widget?.name === "seed");
                        const seedIsConnected = seedInput && seedInput.link != null;
                        
                        let seedToUse;
                        
                        // If seed is connected, skip resolution - let server-side handler process it
                        if (seedIsConnected) {
                            // For connected seeds, we don't resolve here, but we still need to update metadata
                            // Use the widget value (which may be a special seed or last used seed)
                            seedToUse = seedWidget.value;
                        } else {
                            // Get seed to use (with safety check)
                            seedToUse = (node.getSeedToUse && typeof node.getSeedToUse === 'function')
                                ? node.getSeedToUse()
                                : node._rvtools_seedWidget.value;
                            
                            // Update the seed in the prompt output (what gets sent to server)
                            if (result.output[nodeId].inputs && result.output[nodeId].inputs.seed !== undefined) {
                                result.output[nodeId].inputs.seed = seedToUse;
                            }

                            // Update last seed tracking
                            node._rvtools_lastSeed = seedToUse;
                            
                            // Clear the seed cache after use so next call generates fresh random seed
                            node._rvtools_cachedInputSeed = null;
                            node._rvtools_cachedResolvedSeed = null;
                            
                            // Update the last seed button
                            if (node._rvtools_lastSeedButton) {
                                const currentWidgetValue = node._rvtools_seedWidget.value;
                                if (SPECIAL_SEEDS.includes(currentWidgetValue)) {
                                    // Widget has special seed, show what was actually used
                                    node._rvtools_lastSeedButton.name = `♻️ ${seedToUse}`;
                                    node._rvtools_lastSeedButton.disabled = false;
                                } else {
                                    // Widget has regular seed value
                                    node._rvtools_lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                                    node._rvtools_lastSeedButton.disabled = true;
                                }
                            }
                        }
                        
                        // Update workflow metadata for reproducibility when loading from saved images
                        if (result.workflow && result.workflow.nodes) {
                            const workflowNode = result.workflow.nodes.find(n => n.id === node.id);
                            if (workflowNode && workflowNode.widgets_values) {
                                // Find widget indices
                                const modeWidget = node.widgets?.find(w => w.name === "mode");
                                const seedWidget = node.widgets?.find(w => w.name === "seed");
                                const populatedWidget = node.widgets?.find(w => w.name === "populated_text");
                                
                                const modeIndex = node.widgets.indexOf(modeWidget);
                                const seedIndex = node.widgets.indexOf(seedWidget);
                                const populatedIndex = node.widgets.indexOf(populatedWidget);
                                
                                // If in populate mode, save as fixed mode with actual values for reproducibility
                                if (modeIndex >= 0 && modeWidget?.value === "populate") {
                                    workflowNode.widgets_values[modeIndex] = "fixed";
                                }
                                
                                // Save the actual seed used (not the special seed value)
                                if (seedIndex >= 0) {
                                    workflowNode.widgets_values[seedIndex] = seedToUse;
                                }
                                
                                // Save the actual populated text that was generated
                                if (populatedIndex >= 0 && populatedWidget) {
                                    workflowNode.widgets_values[populatedIndex] = populatedWidget.value;
                                }
                            }
                        }
                    }
                }
            }
            
            return result;
        };
        
        // Hook into graph execution to update populated_text before queue runs
        const originalQueuePrompt = app.queuePrompt;
        app.queuePrompt = async function() {
            // Find all Wildcard Processor nodes
            for (const nodeId in app.graph._nodes) {
                const node = app.graph._nodes[nodeId];
                
                if (node.type === "Wildcard Processor [RvTools]") {
                    const wildcardTextWidget = node.widgets?.find(w => w.name === "wildcard_text");
                    const populatedTextWidget = node.widgets?.find(w => w.name === "populated_text");
                    const modeWidget = node.widgets?.find(w => w.name === "mode");
                    
                    if (!modeWidget || !wildcardTextWidget || !populatedTextWidget) {
                        continue;
                    }
                    
                    const mode = modeWidget.value;
                    const wildcardText = wildcardTextWidget.value;
                    
                    // In fixed mode: keep populated_text as-is, don't process
                    if (mode === "fixed") {
                        continue;
                    }
                    
                    // In populate mode: process the text with resolved seed
                    if (mode === "populate" && wildcardText) {
                        // Get seed to use (with fallback if method not available)
                        const seedWidget = node.widgets?.find(w => w.name === "seed");
                        const seedToUse = (node.getSeedToUse && typeof node.getSeedToUse === 'function') 
                            ? node.getSeedToUse() 
                            : (seedWidget?.value ?? 0);
                        
                        try {
                            const response = await fetch("/rvtools/wildcards/process", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    text: wildcardText,
                                    seed: seedToUse
                                })
                            });
                            
                            if (response.ok) {
                                const result = await response.json();
                                if (result.success) {
                                    populatedTextWidget.value = result.output;
                                    // Trigger callback to ensure widget updates properly
                                    if (populatedTextWidget.callback) {
                                        populatedTextWidget.callback(result.output);
                                    }
                                }
                            }
                        } catch (error) {
                            console.error("[RvTools Wildcard] queuePrompt error:", error);
                        }
                    }
                }
            }
            
            // Call original queuePrompt
            return originalQueuePrompt.call(this);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only process Wildcard Processor nodes
        if (nodeData.name !== "Wildcard Processor [RvTools]" && 
            nodeData.class_type !== "Wildcard Processor [RvTools]") {
            return;
        }

        // Add seed helper methods to prototype FIRST (before onNodeCreated)
        // Method to generate random seed
        nodeType.prototype.generateRandomSeed = function() {
            const step = this._rvtools_seedWidget?.options?.step || 1;
            const randomMin = this._rvtools_randomMin || 0;
            const randomMax = this._rvtools_randomMax || 1125899906842624;
            const randomRange = (randomMax - randomMin) / (step / 10);
            let seed = Math.floor(Math.random() * randomRange) * (step / 10) + randomMin;
            
            // Avoid special seeds
            if (SPECIAL_SEEDS.includes(seed)) {
                seed = 0;
            }
            return seed;
        };
        
        // Method to determine seed to use
        nodeType.prototype.getSeedToUse = function() {
            // Normal seed generation logic - seed widget can be connected directly
            const inputSeed = Number(this._rvtools_seedWidget.value);
            
            // Check if we have a cached resolved seed for this input seed
            // This prevents generating different random seeds on multiple calls
            if (this._rvtools_cachedInputSeed === inputSeed && this._rvtools_cachedResolvedSeed != null) {
                return this._rvtools_cachedResolvedSeed;
            }
            
            let seedToUse = null;
            
            // If our input seed was a special seed, then handle it
            if (SPECIAL_SEEDS.includes(inputSeed)) {
                // If the last seed was not a special seed and we have increment/decrement, then do that
                if (typeof this._rvtools_lastSeed === "number" && !SPECIAL_SEEDS.includes(this._rvtools_lastSeed)) {
                    if (inputSeed === SPECIAL_SEED_INCREMENT) {
                        seedToUse = this._rvtools_lastSeed + 1;
                    } else if (inputSeed === SPECIAL_SEED_DECREMENT) {
                        seedToUse = this._rvtools_lastSeed - 1;
                    }
                }
                
                // If we don't have a seed to use, or it's a special seed, randomize
                if (seedToUse == null || SPECIAL_SEEDS.includes(seedToUse)) {
                    seedToUse = this.generateRandomSeed();
                }
            }
            
            const finalSeed = seedToUse != null ? seedToUse : inputSeed;
            
            // Cache the resolved seed for this input seed
            this._rvtools_cachedInputSeed = inputSeed;
            this._rvtools_cachedResolvedSeed = finalSeed;
            
            return finalSeed;
        };
        
        // Intercept execution to track last seed and update populated_text
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            
            // Update populated_text widget with actual result from execution
            if (message && message.text && message.text.length > 0) {
                const populatedWidget = this.widgets?.find(w => w.name === "populated_text");
                if (populatedWidget) {
                    populatedWidget.value = message.text[0];
                }
            }
            
            // Store the seed that was actually used if available
            if (message && message.seed !== undefined) {
                // Handle both array and single value formats
                this._rvtools_lastSeed = Array.isArray(message.seed) ? message.seed[0] : message.seed;
            }
            
            return result;
        };

        // Helper function to check if seed widget has a connection
        nodeType.prototype.isSeedConnected = function() {
            const seedInput = this.inputs?.find(input => input.widget?.name === "seed");
            return seedInput && seedInput.link != null;
        };

        // Helper function to update seed button states based on mode and connection
        nodeType.prototype.updateSeedButtonStates = function() {
            const modeWidget = this.widgets?.find(w => w.name === "mode");
            const currentMode = modeWidget?.value || "populate";
            const seedIsConnected = this.isSeedConnected();
            
            if (currentMode === "populate" && !seedIsConnected) {
                // Enable buttons only in populate mode AND when seed is not connected
                if (this._rvtools_randomizeButton) {
                    this._rvtools_randomizeButton.disabled = false;
                }
                if (this._rvtools_newRandomButton) {
                    this._rvtools_newRandomButton.disabled = false;
                }
                if (this._rvtools_lastSeedButton && this._rvtools_lastSeed != null) {
                    const currentWidgetValue = this._rvtools_seedWidget?.value;
                    this._rvtools_lastSeedButton.disabled = !SPECIAL_SEEDS.includes(currentWidgetValue);
                }
            } else {
                // Disable buttons in fixed mode OR when seed is connected
                if (this._rvtools_randomizeButton) {
                    this._rvtools_randomizeButton.disabled = true;
                }
                if (this._rvtools_newRandomButton) {
                    this._rvtools_newRandomButton.disabled = true;
                }
                if (this._rvtools_lastSeedButton) {
                    this._rvtools_lastSeedButton.disabled = true;
                }
            }
        };

        // Store original onNodeCreated if it exists
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        /**
         * Custom onNodeCreated handler for wildcard processor nodes
         */
        nodeType.prototype.onNodeCreated = function() {
            // Set flag to prevent auto-population during initialization
            this._isInitializing = true;
            
            // Call original if exists
            if (originalOnNodeCreated) {
                originalOnNodeCreated.call(this);
            }

            // Capture node reference for use in callbacks
            const node = this;
            
            // ===== SEED WIDGET SETUP =====
            // Find the seed widget and remove control_after_generate
            // Important: Find first, remove later to avoid index shifting issues
            let seedWidget = null;
            let controlAfterGenerateIndex = -1;
            
            for (let i = 0; i < this.widgets.length; i++) {
                const widget = this.widgets[i];
                const wname = (widget.name || '').toString().toLowerCase();
                if (wname === 'seed') {
                    seedWidget = widget;
                } else if (wname === 'control_after_generate') {
                    controlAfterGenerateIndex = i;
                }
            }
            
            // Remove control_after_generate if found
            if (controlAfterGenerateIndex >= 0) {
                this.widgets.splice(controlAfterGenerateIndex, 1);
            }
            
            if (!seedWidget) {
                console.warn("[RvTools Wildcard] Seed widget not found! Available widgets:", this.widgets.map(w => w.name));
            }

            if (seedWidget) {
                // Initialize seed tracking properties
                this._rvtools_seedWidget = seedWidget;
                this._rvtools_lastSeed = undefined;
                this._rvtools_randomMin = 0;
                this._rvtools_randomMax = 1125899906842624;
                this._rvtools_cachedInputSeed = null;
                this._rvtools_cachedResolvedSeed = null;
                
                // Ensure seed widget is visible (not hidden)
                if (seedWidget.type) {
                    seedWidget.type = "number";  // Ensure it's displayed as a number input
                }
                seedWidget.hidden = false;
                if (seedWidget.options) {
                    seedWidget.options.hidden = false;
                }
                
                // Store original callback for later wrapping
                const originalSeedCallback = seedWidget.callback;
                
                // Add seed control buttons
                const seedWidgetIndex = this.widgets.indexOf(seedWidget);
                
                // Button: Randomize Each Time
                const randomizeButton = this.addWidget(
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
                const newRandomButton = this.addWidget(
                    "button",
                    "🎲 New Fixed Random",
                    "",
                    () => {
                        const newSeed = this.generateRandomSeed();
                        seedWidget.value = newSeed;
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
                        if (this._rvtools_lastSeed != null) {
                            seedWidget.value = this._rvtools_lastSeed;
                            lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                            lastSeedButton.disabled = true;
                        }
                    },
                    { serialize: false }
                );
                lastSeedButton.disabled = true;
                this._rvtools_lastSeedButton = lastSeedButton;
                
                // Store references to seed buttons for mode control
                this._rvtools_randomizeButton = randomizeButton;
                this._rvtools_newRandomButton = newRandomButton;
                
                // Find the wildcards combo and mode widget for positioning
                const selectCombo = this.widgets?.find(w => w.name === "wildcards");
                const modeWidget = this.widgets?.find(w => w.name === "mode");
                const modeWidgetIndex = modeWidget ? this.widgets.indexOf(modeWidget) : -1;
                
                // Move wildcards combo to be right after mode widget
                if (selectCombo && modeWidgetIndex >= 0) {
                    const selectComboIndex = this.widgets.indexOf(selectCombo);
                    if (selectComboIndex !== modeWidgetIndex + 1) {
                        this.widgets.splice(selectComboIndex, 1);
                        this.widgets.splice(modeWidgetIndex + 1, 0, selectCombo);
                    }
                }
                
                // Now move seed buttons to be right after seed widget (which is now after wildcards combo)
                const buttonsToMove = [randomizeButton, newRandomButton, lastSeedButton];
                for (let i = buttonsToMove.length - 1; i >= 0; i--) {
                    const button = buttonsToMove[i];
                    const currentIndex = this.widgets.indexOf(button);
                    const currentSeedIndex = this.widgets.indexOf(seedWidget);
                    if (currentIndex !== currentSeedIndex + 1) {
                        this.widgets.splice(currentIndex, 1);
                        this.widgets.splice(currentSeedIndex + 1, 0, button);
                    }
                }
                
                // Add invisible spacer for bottom padding
                const spacer = {
                    type: "SPACER",
                    name: "spacer",
                    computeSize: () => [0, 8],
                    draw: () => {},
                    mouse: () => {},
                    serialize: false
                };
                this.widgets.push(spacer);
            }
            
            // Override minimum node size to allow smaller width
            // The default min width of 259 is too wide for this node
            const originalOnResize = this.onResize;
            this.onResize = function(size) {
                // Set custom minimum dimensions
                const minWidth = 200;  // Reduced from default 259
                const minHeight = 100;
                
                // Enforce minimum size
                size[0] = Math.max(size[0], minWidth);
                size[1] = Math.max(size[1], minHeight);
                
                if (originalOnResize) {
                    return originalOnResize.apply(this, [size]);
                }
            };
            
            // Set initial size to smaller width if currently at default
            const currentSize = this.size;
            if (currentSize[0] >= 259) {
                this.size = [200, currentSize[1]];
            }

            // Find other widgets (seedWidget already defined above)
            const wildcardTextWidget = this.widgets?.find(w => w.name === "wildcard_text");
            const populatedTextWidget = this.widgets?.find(w => w.name === "populated_text");
            const modeWidget = this.widgets?.find(w => w.name === "mode");
            const wildcardCombo = this.widgets?.find(w => w.name === "wildcards");
            
            // Setup seed widget callback to clear cache and update preview
            if (seedWidget) {
                const originalSeedCallback = seedWidget.callback;
                seedWidget.callback = (value) => {
                    // Clear the seed cache when the seed value changes
                    node._rvtools_cachedInputSeed = null;
                    node._rvtools_cachedResolvedSeed = null;
                    
                    // Call the original callback if it exists
                    if (originalSeedCallback) {
                        originalSeedCallback.call(seedWidget, value);
                    }
                    
                    // Skip auto-population during initialization
                    if (node._isInitializing) {
                        return;
                    }
                    
                    // Update populated_text when seed changes (in populate mode)
                    if (modeWidget?.value === "populate" && wildcardTextWidget?.value && populatedTextWidget) {
                        const actualSeed = node.getSeedToUse();
                        updatePopulatedText(populatedTextWidget, wildcardTextWidget.value, actualSeed);
                    }
                };
            }
            
            // Hook wildcard_text for real-time processing
            if (wildcardTextWidget && populatedTextWidget) {
                const originalCallback = wildcardTextWidget.callback;
                wildcardTextWidget.callback = function(value) {
                    try {
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }
                        
                        // Skip auto-population during node initialization/loading
                        if (node._isInitializing) {
                            return;
                        }
                        
                        // Skip during graphToPrompt serialization - we don't want to regenerate during execution
                        // Check if we're in the call stack of graphToPrompt by looking for serializeValue
                        const stack = new Error().stack;
                        if (stack && stack.includes('serializeValue')) {
                            return;
                        }
                        
                        // Update populated_text when wildcard_text changes (in populate mode)
                        if (modeWidget?.value === "populate" && seedWidget) {
                            const actualSeed = node.getSeedToUse();
                            updatePopulatedText(populatedTextWidget, value, actualSeed);
                        }
                    } catch (e) {
                        console.error("[RvTools Wildcard] Error in wildcard_text callback:", e);
                    }
                };
            }

            // Hook mode widget to update populate behavior
            if (modeWidget) {
                const originalChange = modeWidget.callback;
                modeWidget.callback = function(value) {
                    try {
                        if (originalChange) {
                            originalChange.call(this, value);
                        }
                        
                        // Skip auto-population during node initialization/loading
                        if (node._isInitializing) {
                            return;
                        }
                        
                        // Update populated_text editability and content based on mode
                        if (value === "populate") {
                            // Populate mode: auto-generate and make read-only
                            if (wildcardTextWidget && populatedTextWidget && seedWidget) {
                                // Only auto-generate if populated_text is empty (don't overwrite loaded content)
                                if (!populatedTextWidget.value) {
                                    const actualSeed = node.getSeedToUse();
                                    updatePopulatedText(populatedTextWidget, wildcardTextWidget.value, actualSeed);
                                }
                                // Make read-only by disabling the widget
                                populatedTextWidget.disabled = true;
                                if (populatedTextWidget.element) {
                                    populatedTextWidget.element.style.opacity = "0.85";
                                    populatedTextWidget.element.style.cursor = "not-allowed";
                                    populatedTextWidget.element.title = "Auto-generated in populate mode. Change seed to generate new output, fix seed to keep same output.";
                                }
                            }
                            
                            // Update seed button states based on mode and connection
                            node.updateSeedButtonStates();
                        } else if (value === "fixed") {
                            // Fixed mode: make editable
                            if (populatedTextWidget) {
                                populatedTextWidget.disabled = false;
                                if (populatedTextWidget.element) {
                                    populatedTextWidget.element.style.opacity = "1.0";
                                    populatedTextWidget.element.style.cursor = "text";
                                    populatedTextWidget.element.title = "Edit to customize the output";
                                }
                            }
                            
                            // Update seed widget to show the actual seed that was used (if it's a special seed)
                            if (seedWidget) {
                                const currentSeedValue = seedWidget.value;
                                // If current seed is a special seed (-1, -2, -3), replace it with the actual last used seed
                                if (SPECIAL_SEEDS.includes(currentSeedValue)) {
                                    // Use last seed if available, otherwise fallback to 0
                                    seedWidget.value = node._rvtools_lastSeed != null ? node._rvtools_lastSeed : 0;
                                }
                            }
                            
                            // Update seed button states based on mode and connection
                            node.updateSeedButtonStates();
                        }
                        
                        updateUIForMode(node, value);
                    } catch (e) {
                        console.error("[RvTools Wildcard] Error in mode callback:", e);
                    }
                };
            }

            // Update combo widget
            if (wildcardCombo) {
                // Make combo update its options before drawing
                const originalDraw = wildcardCombo.draw;
                if (originalDraw) {
                    wildcardCombo.draw = function(ctx, node, widgetWidth, y, height) {
                        // Update options every time we draw (in case new wildcards were added)
                        updateWildcardCombo(this);
                        return originalDraw.call(this, ctx, node, widgetWidth, y, height);
                    };
                }
                
                // Add callback to insert wildcard into text when selected
                const originalCallback = wildcardCombo.callback;
                wildcardCombo.callback = function(value) {
                    // Call original callback if exists
                    if (originalCallback) {
                        originalCallback.call(this, value);
                    }
                    
                    // If a wildcard is selected (not "Select a Wildcard"), insert it into wildcard_text
                    if (value && value !== "Select a Wildcard") {
                        const wildcardTextWidget = node.widgets?.find(w => w.name === "wildcard_text");
                        if (wildcardTextWidget) {
                            let currentText = wildcardTextWidget.value || "";
                            
                            // Determine separator - add comma if text exists and doesn't already end with comma
                            let separator = "";
                            if (currentText) {
                                const trimmedText = currentText.trimEnd();
                                if (trimmedText && !trimmedText.endsWith(",")) {
                                    separator = ", ";
                                } else if (trimmedText.endsWith(",")) {
                                    separator = " ";
                                }
                            }
                            
                            // Add the wildcard
                            wildcardTextWidget.value = currentText + separator + value;
                            
                            // Clean up "., " patterns anywhere in the text (not just at the end)
                            wildcardTextWidget.value = wildcardTextWidget.value.replace(/\.,\s+/g, ', ');
                            
                            // Clean up multiple whitespaces
                            wildcardTextWidget.value = wildcardTextWidget.value.replace(/\s+/g, ' ').trim();
                            
                            // Trigger the wildcard_text callback to update populated_text
                            if (wildcardTextWidget.callback) {
                                wildcardTextWidget.callback(wildcardTextWidget.value);
                            }
                        }
                        
                        // Reset combo selection back to "Select a Wildcard"
                        setTimeout(() => {
                            wildcardCombo.value = "Select a Wildcard";
                        }, 10);
                    }
                };
                
                // Also update immediately
                updateWildcardCombo(wildcardCombo);
            }
            
            // Clear initialization flag and apply final UI state after setup is complete
            // Use setTimeout to ensure all widget values are loaded from workflow
            setTimeout(() => {
                this._isInitializing = false;
                
                // Now apply the correct UI state based on the actual loaded mode value
                if (modeWidget && populatedTextWidget) {
                    const currentMode = modeWidget.value;
                    
                    if (currentMode === "populate") {
                        populatedTextWidget.disabled = true;
                        if (populatedTextWidget.element) {
                            populatedTextWidget.element.style.opacity = "0.85";
                            populatedTextWidget.element.style.cursor = "not-allowed";
                            populatedTextWidget.element.title = "Auto-generated in populate mode. Change seed to generate new output, fix seed to keep same output.";
                        }
                        
                        // Update seed button states based on mode and connection
                        node.updateSeedButtonStates();
                    }
                    updateUIForMode(node, currentMode);
                }
            }, 0);
            
            // Hook into onConnectionsChange to update button states when seed connections change
            const originalOnConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function(type, index, connected, link_info) {
                // Call original if exists
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                
                // Check if this is the seed input
                if (type === 1) { // 1 = input connection
                    const input = this.inputs?.[index];
                    if (input && input.widget && input.widget.name === "seed") {
                        // Seed connection changed - update button states
                        this.updateSeedButtonStates();
                    }
                }
            };
        };
    },

    async nodeCreated(node, app) {
        // Additional node created handling if needed
        if (node.type !== "Wildcard Processor [RvTools]") {
            return;
        }
        
        // Force update the combo widget with current wildcard list
        // This ensures widgets created from saved graphs get the updated list
        const wildcardCombo = node.widgets?.find(w => w.name === "wildcards");
        if (wildcardCombo) {
            updateWildcardCombo(wildcardCombo);
        }
        
        // DON'T initialize populated_text here - it happens too early
        // The onNodeCreated handler with setTimeout will handle it properly after widget values load
    },

    async loadedGraphNode(node, app) {
        // Handle loaded graph nodes
        if (node.type !== "Wildcard Processor [RvTools]") {
            return;
        }

        // Refresh wildcard combo
        const wildcardCombo = node.widgets?.find(w => w.name === "wildcards");
        if (wildcardCombo) {
            updateWildcardCombo(wildcardCombo);
        }
        
        // DON'T process populated_text here - it happens too early and will overwrite loaded values
        // The onNodeCreated handler with setTimeout will handle initialization properly
    }
});

/**
 * Update wildcard combo with current wildcard list
 * @param {Widget} comboWidget - The combo widget to update
 */
function updateWildcardCombo(comboWidget) {
    if (!comboWidget) return;

    // Preserve "Select a Wildcard" as first option and add all loaded wildcards
    const options = ["Select a Wildcard", ...wildcardList];
    
    // Update combo options - support both array and object formats
    if (comboWidget.options) {
        if (typeof comboWidget.options === "object" && !Array.isArray(comboWidget.options)) {
            // ComfyUI widget format: { values: [...] }
            comboWidget.options.values = options;
        } else if (Array.isArray(comboWidget.options)) {
            // Direct array format
            Object.defineProperty(comboWidget, 'options', {
                value: options,
                writable: true
            });
        }
    } else {
        // Fallback: set options directly
        Object.defineProperty(comboWidget, 'options', {
            value: options,
            writable: true
        });
    }
    
    // Mark widget as needing redraw
    if (comboWidget.element) {
        comboWidget.element.style.setProperty('--changed', 'true', 'important');
    }
}

/**
 * Clean up text after wildcard processing
 * Removes unresolved wildcards and orphaned punctuation
 * @param {string} text - The text to clean
 * @returns {string} Cleaned text
 */
function cleanProcessedText(text) {
    if (!text) return text;
    
    // Remove unresolved wildcard patterns like __keyword__ or __*/actress__
    text = text.replace(/__[\w.\-+/*\\]+?__/g, '');
    
    // Clean up orphaned punctuation left behind
    text = text.replace(/[,\s]*,[,\s]*,/g, ',');  // Multiple commas with spaces -> single comma
    text = text.replace(/\.,\s*/g, ', ');         // "., " -> ", "
    text = text.replace(/,\s*\./g, '.');          // ", ." -> "."
    text = text.replace(/\s*,\s*,/g, ',');        // ", ," -> ","
    text = text.replace(/^\s*,\s*/g, '');         // Leading comma
    text = text.replace(/\s*,\s*$/g, '');         // Trailing comma
    
    // Clean up extra spaces
    text = text.replace(/\s+/g, ' ').trim();
    
    return text;
}

/**
 * Update the populated_text widget with processed wildcard text
 * @param {Widget} populatedWidget - The populated_text widget to update
 * @param {string} wildcardText - The wildcard text to process
 * @param {number} seed - The seed to use for processing
 */
async function updatePopulatedText(populatedWidget, wildcardText, seed) {
    if (!populatedWidget || !wildcardText) {
        return;
    }

    try {
        const response = await fetch("/rvtools/wildcards/process", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                text: wildcardText,
                seed: seed
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                // Clean up the output text
                const cleanedOutput = cleanProcessedText(result.output);
                populatedWidget.value = cleanedOutput;
                // Trigger callback to ensure widget updates properly
                if (populatedWidget.callback) {
                    populatedWidget.callback(cleanedOutput);
                }
            } else {
                console.warn("[RvTools Wildcard] Server error - success=false");
            }
        } else {
            console.warn("[RvTools Wildcard] Server returned status:", response.status);
        }
    } catch (error) {
        console.error("[RvTools Wildcard] Error updating preview:", error);
    }
}/**
 * Update UI state based on selected mode
 * @param {Node} node - The node instance
 * @param {string} mode - The selected mode (populate or fixed)
 */
function updateUIForMode(node, mode) {
    const seedWidget = node.widgets?.find(w => w.name === "seed");
    
    if (!seedWidget) return;

    // In "populate" mode, seed controls generation: change to create new output, fix to keep same
    // In "fixed" mode, seed is ignored

    switch (mode) {
        case "populate":
            // Seed is active - full opacity
            if (seedWidget.element) {
                seedWidget.element.style.opacity = "1.0";
                seedWidget.element.title = "Change seed to generate new output, fix seed to keep same output";
            }
            break;

        case "fixed":
            // Seed ignored - disable or gray out
            if (seedWidget.element) {
                seedWidget.element.style.opacity = "0.5";
                seedWidget.element.title = "Seed is ignored in 'fixed' mode";
            }
            break;
    }
}

/**
 * Setup periodic wildcard list refresh
 * Allows auto-loading of newly created wildcard files
 */
setInterval(async () => {
    try {
        const response = await fetch("/rvtools/wildcards/list");
        if (response.ok) {
            const newList = await response.json();
            
            // Check if list changed
            if (JSON.stringify(newList) !== JSON.stringify(wildcardList)) {
                wildcardList = newList;
                
                // Update all nodes with new list
                for (const nodeId in app.graph._nodes) {
                    const node = app.graph._nodes[nodeId];
                    if (node.type === "Wildcard Processor [RvTools]") {
                        const combo = node.widgets?.find(w => w.name === "wildcards");
                        if (combo) {
                            updateWildcardCombo(combo);
                        }
                    }
                }
            }
        }
    } catch (error) {
        // Silent fail - don't spam logs
    }
}, 5000); // Check every 5 seconds

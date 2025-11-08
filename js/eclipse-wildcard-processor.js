/**
 * Eclipse Wildcard Processor - JavaScript Extension
 * Simplified approach matching Impact Pack
 */

import { app, api } from './comfy/index.js';

const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";

const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

// Store last seeds per node ID
const nodeLastSeeds = {};

let wildcardsList = [];

async function loadWildcards() {
    try {
        const response = await api.fetchApi('/eclipse/wildcards/list');
        if (response.ok) {
            wildcardsList = await response.json();
        }
    } catch (error) {
        console.warn("[Eclipse Wildcard] Failed to load wildcard list:", error);
        wildcardsList = [];
    }
}

function handleNodeFeedback(event) {
    const { node_id, widget_name, value } = event.detail;
    const node = app.graph.getNodeById(parseInt(node_id));
    if (!node) return;
    const widget = node.widgets?.find(w => w.name === widget_name);
    if (!widget) return;
    widget.value = value;
    app.canvas.setDirty(true);
}

app.registerExtension({
    name: "Eclipse.WildcardProcessor",
    
    async setup() {
        await loadWildcards();
        api.addEventListener("eclipse-node-feedback", handleNodeFeedback);
        
        // Hook into graphToPrompt to resolve special seeds before server execution
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            const result = await originalGraphToPrompt.apply(this, arguments);
            
            // Process all Wildcard Processor nodes
            const nodes = app.graph._nodes;
            for (const node of nodes) {
                if (node.type === "Wildcard Processor [Eclipse]" && node._Eclipse_seedWidget) {
                    // Skip if node is muted or bypassed
                    if (node.mode === 2 || node.mode === 4) {
                        continue;
                    }
                    
                    const nodeId = String(node.id);
                    if (result.output && result.output[nodeId]) {
                        const seedToUse = node.getSeedToUse();
                        
                        // Update the seed in the prompt output (what gets sent to server)
                        if (result.output[nodeId].inputs && result.output[nodeId].inputs.seed !== undefined) {
                            result.output[nodeId].inputs.seed = seedToUse;
                        }
                        
                        // Update last seed tracking
                        if (Number(node._Eclipse_lastSeed) !== Number(seedToUse)) {
                            node._Eclipse_lastSeed = seedToUse;
                            nodeLastSeeds[node.id] = seedToUse;
                        }
                        
                        // Clear the seed cache after use
                        node._Eclipse_cachedInputSeed = null;
                        node._Eclipse_cachedResolvedSeed = null;
                        
                        // Update the last seed button
                        if (node._Eclipse_lastSeedButton) {
                            const currentWidgetValue = node._Eclipse_seedWidget.value;
                            if (SPECIAL_SEEDS.includes(currentWidgetValue)) {
                                node._Eclipse_lastSeedButton.name = `♻️ ${seedToUse}`;
                                node._Eclipse_lastSeedButton.disabled = false;
                            } else {
                                node._Eclipse_lastSeedButton.name = LAST_SEED_BUTTON_LABEL;
                                node._Eclipse_lastSeedButton.disabled = true;
                            }
                        }
                        
                        // Update workflow data if present
                        if (result.workflow && result.workflow.nodes) {
                            const workflowNode = result.workflow.nodes.find(n => n.id === node.id);
                            if (workflowNode && workflowNode.widgets_values) {
                                const seedWidgetIndex = node.widgets.indexOf(node._Eclipse_seedWidget);
                                if (seedWidgetIndex >= 0) {
                                    workflowNode.widgets_values[seedWidgetIndex] = seedToUse;
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
        if (nodeData.name !== "Wildcard Processor [Eclipse]" && 
            nodeData.class_type !== "Wildcard Processor [Eclipse]") {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            if (originalOnNodeCreated) {
                originalOnNodeCreated.call(this);
            }

            const node = this;
            const wildcardTextWidget = this.widgets?.find(w => w.name === "wildcard_text");
            const populatedTextWidget = this.widgets?.find(w => w.name === "populated_text");
            const modeWidget = this.widgets?.find(w => w.name === "mode");
            let seedWidget = null;
            let wildcardCombo = this.widgets?.find(w => w.name === "Select to add Wildcard");
            
            // Find the seed widget and remove control_after_generate
            for (const [i, widget] of this.widgets.entries()) {
                const wname = (widget.name || '').toString().toLowerCase();
                const wlabel = (widget.label || widget.options?.label || widget.options?.name || '').toString().toLowerCase();
                if (wname === 'seed' || wlabel === 'seed') {
                    seedWidget = widget;
                } else if (wname === 'control_after_generate') {
                    this.widgets.splice(i, 1);
                }
            }
            
            // Move wildcard combo before seed widget
            if (wildcardCombo && seedWidget) {
                const comboIndex = this.widgets.indexOf(wildcardCombo);
                const seedIndex = this.widgets.indexOf(seedWidget);
                
                // Only move if combo is after seed
                if (comboIndex > seedIndex) {
                    this.widgets.splice(comboIndex, 1);
                    const newSeedIndex = this.widgets.indexOf(seedWidget);
                    this.widgets.splice(newSeedIndex, 0, wildcardCombo);
                }
            }
            
            if (wildcardTextWidget && wildcardTextWidget.inputEl) {
                wildcardTextWidget.inputEl.placeholder = "Wildcard Prompt (User input)";
            }
            
            if (populatedTextWidget && populatedTextWidget.inputEl) {
                populatedTextWidget.inputEl.placeholder = "Populated Prompt (Will be generated automatically)";
                populatedTextWidget.inputEl.disabled = true;
            }

            if (modeWidget) {
                Object.defineProperty(modeWidget, "value", {
                    set: function(value) {
                        if (value === true) {
                            this._modeValue = "populate";
                        } else if (value === false) {
                            this._modeValue = "fixed";
                        } else {
                            this._modeValue = value;
                        }
                        
                        if (populatedTextWidget && populatedTextWidget.inputEl) {
                            const isPopulate = this._modeValue === 'populate';
                            populatedTextWidget.inputEl.disabled = isPopulate;
                        }
                    },
                    get: function() {
                        return this._modeValue !== undefined ? this._modeValue : 'populate';
                    }
                });
            }

            // Setup custom seed widget with buttons
            if (seedWidget) {
                // Store seed widget and tracking properties
                node._Eclipse_seedWidget = seedWidget;
                node._Eclipse_lastSeed = undefined;
                node._Eclipse_randomMin = 0;
                node._Eclipse_randomMax = 1125899906842624;
                node._Eclipse_cachedInputSeed = null;
                node._Eclipse_cachedResolvedSeed = null;
                
                // Hook into seed widget's callback to clear cache when it changes
                const originalCallback = seedWidget.callback;
                seedWidget.callback = (value) => {
                    node._Eclipse_cachedInputSeed = null;
                    node._Eclipse_cachedResolvedSeed = null;
                    if (originalCallback) {
                        return originalCallback.call(seedWidget, value);
                    }
                };
                
                // Button: Randomize Each Time (added at bottom)
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
                
                // Button: New Fixed Random (added at bottom)
                const newRandomButton = this.addWidget(
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
                
                // Button: Use Last Queued Seed (added at bottom)
                const lastSeedButton = this.addWidget(
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
                
                // Store button references for updating
                node._Eclipse_randomizeButton = randomizeButton;
                node._Eclipse_newRandomButton = newRandomButton;
                
                // Buttons are now at the bottom (no repositioning needed)
                // They'll stay at the end of the widget list
            }

            if (wildcardCombo) {
                // Initialize wildcard value storage on node (Impact Pack approach)
                node._wildcard_value = "Select the Wildcard to add to the text";
                
                wildcardCombo.serializeValue = () => {
                    return "Select the Wildcard to add to the text";
                };
                
                Object.defineProperty(wildcardCombo, "value", {
                    set: function(value) {
                        if (value !== "Select the Wildcard to add to the text") {
                            node._wildcard_value = value;
                        }
                    },
                    get: function() {
                        return "Select the Wildcard to add to the text";
                    }
                });

                Object.defineProperty(wildcardCombo.options, "values", {
                    set: function(x) {},
                    get: function() {
                        return ["Select the Wildcard to add to the text", ...wildcardsList];
                    }
                });
                
                wildcardCombo.callback = function(value, canvas, node, pos, e) {
                    if (node && wildcardTextWidget) {
                        // Add comma separator if text box is not empty
                        if (wildcardTextWidget.value != '') {
                            wildcardTextWidget.value += ', ';
                        }
                        
                        // Append the wildcard value
                        wildcardTextWidget.value += node._wildcard_value;
                        
                        // Trigger callback if exists to update internal state
                        if (wildcardTextWidget.callback) {
                            wildcardTextWidget.callback(wildcardTextWidget.value);
                        }
                    }
                };
            }
        };
        
        // Method to generate random seed
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
        
        // Method to determine seed to use
        nodeType.prototype.getSeedToUse = function() {
            const inputSeed = Number(this._Eclipse_seedWidget.value);
            
            // Check if we have a cached resolved seed for this input seed
            if (this._Eclipse_cachedInputSeed === inputSeed && this._Eclipse_cachedResolvedSeed != null) {
                return this._Eclipse_cachedResolvedSeed;
            }
            
            let seedToUse = null;
            
            // Handle special seeds
            if (SPECIAL_SEEDS.includes(inputSeed)) {
                if (typeof this._Eclipse_lastSeed === "number" && !SPECIAL_SEEDS.includes(this._Eclipse_lastSeed)) {
                    if (inputSeed === SPECIAL_SEED_INCREMENT) {
                        seedToUse = this._Eclipse_lastSeed + 1;
                    } else if (inputSeed === SPECIAL_SEED_DECREMENT) {
                        seedToUse = this._Eclipse_lastSeed - 1;
                    }
                }
                
                // If we don't have a seed to use, randomize
                if (seedToUse == null || SPECIAL_SEEDS.includes(seedToUse)) {
                    seedToUse = this.generateRandomSeed();
                }
            }
            
            const finalSeed = seedToUse != null ? seedToUse : inputSeed;
            
            // Cache the resolved seed
            this._Eclipse_cachedInputSeed = inputSeed;
            this._Eclipse_cachedResolvedSeed = finalSeed;
            
            return finalSeed;
        };
        
        // Method to update button states based on seed input connection
        nodeType.prototype.updateSeedButtonStates = function() {
            if (!this._Eclipse_seedWidget) return;
            
            // Check if seed widget has an input connection
            const seedInput = this.inputs?.find(input => {
                const inputName = input.name.toLowerCase();
                return inputName === 'seed';
            });
            
            const hasSeedConnection = seedInput && seedInput.link != null;
            
            // Disable/enable buttons based on connection state
            if (this._Eclipse_randomizeButton) {
                this._Eclipse_randomizeButton.disabled = hasSeedConnection;
            }
            if (this._Eclipse_newRandomButton) {
                this._Eclipse_newRandomButton.disabled = hasSeedConnection;
            }
            if (this._Eclipse_lastSeedButton) {
                // Last seed button is disabled if seed is connected OR if no last seed exists
                this._Eclipse_lastSeedButton.disabled = hasSeedConnection || this._Eclipse_lastSeed == null;
            }
        };
        
        // Hook into onConnectionsChange to update button states
        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }
            
            // Update button states when connections change
            if (this.updateSeedButtonStates) {
                this.updateSeedButtonStates();
            }
        };
    },

    async nodeCreated(node, app) {
        if (node.type !== "Wildcard Processor [Eclipse]") {
            return;
        }
        
        // Update button states after node is fully created
        if (node.updateSeedButtonStates) {
            node.updateSeedButtonStates();
        }
    },

    async loadedGraphNode(node, app) {
        if (node.type !== "Wildcard Processor [Eclipse]") {
            return;
        }

        const modeWidget = node.widgets?.find(w => w.name === "mode");
        const populatedWidget = node.widgets?.find(w => w.name === "populated_text");
        
        if (modeWidget && populatedWidget && populatedWidget.inputEl) {
            const currentMode = modeWidget.value;
            
            // Legacy: treat 'reproduce' as 'fixed'
            if (currentMode === "fixed" || currentMode === "reproduce") {
                populatedWidget.inputEl.disabled = false;
                populatedWidget.inputEl.style.opacity = "1.0";
            } else {
                populatedWidget.inputEl.disabled = true;
                populatedWidget.inputEl.style.opacity = "0.85";
            }
        }
        
        // Update button states after graph is loaded
        if (node.updateSeedButtonStates) {
            node.updateSeedButtonStates();
        }
    }
});

setInterval(async () => {
    try {
        const response = await api.fetchApi('/eclipse/wildcards/list');
        if (response.ok) {
            const newList = await response.json();
            if (JSON.stringify(newList) !== JSON.stringify(wildcardsList)) {
                wildcardsList = newList;
                app.canvas.setDirty(true);
            }
        }
    } catch (error) {}
}, 5000);

loadWildcards();

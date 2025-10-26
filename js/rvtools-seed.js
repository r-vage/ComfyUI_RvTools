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
*/

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const LAST_SEED_BUTTON_LABEL = "♻️ (Use Last Queued Seed)";

const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const SPECIAL_SEEDS = [SPECIAL_SEED_RANDOM, SPECIAL_SEED_INCREMENT, SPECIAL_SEED_DECREMENT];

// Store last seeds per node ID
const nodeLastSeeds = {};

// List of all seed-enabled node types (Wildcard Processor and Smart Prompt handle their own seed functionality)
const SEED_NODE_TYPES = [
    "Sampler Settings+Seed [RvTools]",
    "Sampler Settings Small+Seed [RvTools]",
    "Sampler Settings NI+Seed [RvTools]",
    "Smart Folder [RvTools]"
];

app.registerExtension({
    name: "RvTools.SamplerSettingsSeed",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (SEED_NODE_TYPES.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Find the seed widget and remove control_after_generate (case-insensitive and check label/options too)
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
                    console.warn(`RvTools: Could not find Seed widget in ${nodeData.name}. Widgets:`, this.widgets.map(w => ({ name: w.name, label: w.label, options: w.options })));
                    return result;
                }
                
                // Store original value
                this._rvtools_seedWidget = seedWidget;
                this._rvtools_lastSeed = undefined;
                this._rvtools_randomMin = 0;
                this._rvtools_randomMax = 1125899906842624;
                this._rvtools_cachedInputSeed = null;
                this._rvtools_cachedResolvedSeed = null;
                
                // Hook into the seed widget's value setter to clear cache when it changes
                const originalCallback = seedWidget.callback;
                seedWidget.callback = (value) => {
                    // Clear the seed cache when the seed value changes
                    this._rvtools_cachedInputSeed = null;
                    this._rvtools_cachedResolvedSeed = null;
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
                        // Trigger callback to notify listeners (e.g., wildcard processor preview)
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
                
                return result;
            };
            
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
            
            // Intercept the prompt before it's sent to the server
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                
                // Store the seed that was actually used if available
                if (message && message.seed !== undefined) {
                    this._rvtools_lastSeed = message.seed;
                    nodeLastSeeds[this.id] = message.seed;
                }
                
                return result;
            };
        }
    },
    
    async setup() {
        // Hook into the graphToPrompt to modify seed values in the prompt data
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            // Call the original graphToPrompt first
            const result = await originalGraphToPrompt.apply(this, arguments);

            // Now modify the prompt data for all seed-enabled Sampler Settings nodes
            const nodes = app.graph._nodes;
            for (const node of nodes) {
                if (SEED_NODE_TYPES.includes(node.type) && node._rvtools_seedWidget) {
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
                        if (Number(node._rvtools_lastSeed) !== Number(seedToUse)) {
                            node._rvtools_lastSeed = seedToUse;
                            nodeLastSeeds[node.id] = seedToUse;
                        }
                        
                        // Clear the seed cache after use so next call generates fresh random seed
                        node._rvtools_cachedInputSeed = null;
                        node._rvtools_cachedResolvedSeed = null;
                        
                        // Update the last seed button - but DON'T change the widget value
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
                        
                        // Also update workflow data if present
                        if (result.workflow && result.workflow.nodes) {
                            const workflowNode = result.workflow.nodes.find(n => n.id === node.id);
                            if (workflowNode && workflowNode.widgets_values) {
                                const seedWidgetIndex = node.widgets.indexOf(node._rvtools_seedWidget);
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

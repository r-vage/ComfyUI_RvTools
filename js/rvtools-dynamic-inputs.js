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

import { app } from "../../scripts/app.js";

// Robust dynamic inputs helper for Eclipse multi-switch nodes
// - Uses explicit name prefixes that match the Python optional input names (int_, float_, string_, pipe_, any_, basicpipe_)
// - Ensures widget-only declared prefixed inputs get real sockets created so they are linkable
// - Adds/removes the highest-numbered prefixed entries when inputcount changes
// - Avoids duplicating sockets when a node already exposes optional inputs as widgets

app.registerExtension({
    name: "Eclipse.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.name) return;

        const multiNodes = {
            "RvConversion_ConcatMulti": { type: "PIPE", prefix: "pipe" },
            "Concat Pipe Multi [Eclipse]": { type: "PIPE", prefix: "pipe" },

            "RvRouter_Any_MultiSwitch": { type: "*", prefix: "any" },
            "Any Multi-Switch [Eclipse]": { type: "*", prefix: "any" },

            "RvRouter_Any_MultiSwitch_purge": { type: "*", prefix: "any" },
            "Any Multi-Switch Purge [Eclipse]": { type: "*", prefix: "any" },

            "RvConversion_MergeStrings": { type: "STRING", prefix: "string" },
            "Merge Strings [Eclipse]": { type: "STRING", prefix: "string" },
            
            "RvConversion_Join": { type: "*", prefix: "input" },
            "Join [Eclipse]": { type: "*", prefix: "input" },
        };

        const baseName = nodeData.name && nodeData.name.includes('/') ? nodeData.name.split('/').pop() : nodeData.name;
        const info = multiNodes[baseName];
        if (!info) return;

        nodeType.prototype.onNodeCreated = function () {
            const node = this;
            
            // Track if this is an AnyType node that needs dynamic type handling
            const isAnyTypeNode = info.type === "*";

            // Helper to compute prefixed name
            const nameFor = (prefix, i) => `${prefix}_${i}`;
            
            // Setup dynamic type handling for multi-input AnyType nodes
            // ComfyUI's type system already prevents incompatible connections
            const setupMultiInputAnyTypeHandling = () => {
                if (!isAnyTypeNode) return;
                
                const prefix = info.prefix || 'any';
                
                // Override onConnectionsChange to handle dynamic typing for all any_X inputs
                const originalOnConnectionsChange = node.onConnectionsChange;
                node.onConnectionsChange = function(type, index, connected, link_info) {
                    if (originalOnConnectionsChange) {
                        originalOnConnectionsChange.apply(this, arguments);
                    }
                    
                    if (!link_info || !this.inputs || !this.outputs) return;
                    
                    const input = this.inputs[index];
                    if (!input || !input.name || !input.name.startsWith(prefix + '_')) return;
                    
                    if (connected && type === LiteGraph.INPUT) {
                        // Get the source node and type
                        const sourceNode = app.graph.getNodeById(link_info.origin_id);
                        if (!sourceNode) return;
                        
                        const sourceType = sourceNode.outputs[link_info.origin_slot].type;
                        const color = LGraphCanvas.link_type_colors[sourceType];
                        
                        // Update this input's type and color
                        input.type = sourceType;
                        if (link_info.id) {
                            app.graph.links[link_info.id].color = color;
                        }
                        
                        // Update ALL other any_X inputs to the same type to maintain consistency
                        // This allows ComfyUI's type system to prevent incompatible connections
                        this.inputs.forEach(inp => {
                            if (inp.name && inp.name.startsWith(prefix + '_') && inp !== input) {
                                inp.type = sourceType;
                            }
                        });
                        
                        // Update the output type to match
                        if (this.outputs[0]) {
                            this.outputs[0].type = sourceType;
                            this.outputs[0].name = sourceType;
                            
                            // NOTE: Auto-disconnect logic removed to prevent false disconnections
                            // when nodes are bypassed. Type validation now handled at execution
                            // time in Python, which provides better accuracy and prevents
                            // workflow breakage from temporary type mismatches.
                        }
                        
                    } else if (!connected && type === LiteGraph.INPUT) {
                        // When disconnecting, check if there are still connections
                        const remainingConnections = this.inputs.filter(inp => 
                            inp.name && 
                            inp.name.startsWith(prefix + '_') && 
                            inp.link !== null
                        );
                        
                        if (remainingConnections.length > 0) {
                            // Still have connections, maintain the type from remaining connections
                            const activeType = remainingConnections[0].type;
                            this.inputs.forEach(inp => {
                                if (inp.name && inp.name.startsWith(prefix + '_')) {
                                    inp.type = activeType;
                                }
                            });
                            if (this.outputs[0]) {
                                this.outputs[0].type = activeType;
                                this.outputs[0].name = activeType;
                                
                                // NOTE: Auto-disconnect logic removed - type validation in Python
                            }
                        } else {
                            // No connections left, reset everything to wildcard
                            this.inputs.forEach(inp => {
                                if (inp.name && inp.name.startsWith(prefix + '_')) {
                                    inp.type = "*";
                                }
                            });
                            if (this.outputs[0]) {
                                this.outputs[0].type = "*";
                                this.outputs[0].name = "";
                            }
                        }
                    }
                    
                    this.computeSize?.();
                };
            };
            
            // Setup AnyType handling immediately for this node type
            if (isAnyTypeNode) {
                setupMultiInputAnyTypeHandling();
            }

            const updateInputs = () => {
                if (!node.inputs) node.inputs = [];
                const w = node.widgets ? node.widgets.find(w => w.name === "inputcount") : null;
                const rawTarget = w ? w.value : 2;
                // Enforce minimum of 2 inputs (it's a switch, so need at least 2 options)
                const target = Math.max(2, rawTarget);
                
                // Update widget to reflect enforced minimum
                if (w && w.value < 2) {
                    w.value = 2;
                }
                
                const prefix = info.prefix || (typeof info.type === 'string' ? info.type.toLowerCase() : 'input');

                // collect existing names
                const socketNames = new Set(node.inputs.filter(i => typeof i.name === 'string').map(i => i.name));
                const widgetNames = new Set((node.widgets || []).map(w => w.name).filter(n => typeof n === 'string'));
                const allExisting = new Set([...socketNames, ...widgetNames].filter(n => n.startsWith(prefix + '_')));

                // If counts match, just ensure widget-only names have sockets
                if (allExisting.size === target) {
                    // For AnyType nodes, check if there's an active type
                    let activeType = info.type;
                    if (isAnyTypeNode && node.inputs && node.inputs.length > 0) {
                        const existingAnyInput = node.inputs.find(inp => 
                            inp.name && inp.name.startsWith(prefix + '_') && inp.type !== "*"
                        );
                        if (existingAnyInput) {
                            activeType = existingAnyInput.type;
                        } else {
                            // Fallback: check if any input is connected and get type from link
                            const existingConnectedInput = node.inputs.find(inp => 
                                inp.name && inp.name.startsWith(prefix + '_') && inp.link !== null
                            );
                            if (existingConnectedInput) {
                                const link = app.graph.links[existingConnectedInput.link];
                                if (link) {
                                    activeType = link.type;
                                }
                            }
                        }
                    }
                    
                    for (let i = 1; i <= target; ++i) {
                        const nm = nameFor(prefix, i);
                        if (widgetNames.has(nm) && !socketNames.has(nm)) {
                            node.addInput(nm, activeType, info.shape !== undefined ? { shape: info.shape } : undefined);
                        }
                    }
                    
                    // Smart resize after ensuring inputs are correct
                    setTimeout(() => {
                        node.setDirtyCanvas(true, false);
                        const computedSize = node.computeSize();
                        const currentSize = node.size;
                        const minWidth = 200;
                        const minHeight = 50;
                        let newWidth = Math.max(currentSize[0], minWidth);
                        let newHeight = Math.max(computedSize[1] + 5, minHeight);
                        const heightDiff = Math.abs(currentSize[1] - newHeight);
                        const isGrowing = newHeight > currentSize[1];
                        if (isGrowing || heightDiff > 10) {
                            node.setSize([newWidth, newHeight]);
                        }
                        node.setDirtyCanvas(true, true);
                    }, 50);
                    
                    return;
                }

                // If we need to reduce count, remove highest-numbered names first
                if (allExisting.size > target) {
                    const nums = Array.from(allExisting).map(n => {
                        const m = n.match(new RegExp(prefix + '_(\\d+)$'));
                        return m ? parseInt(m[1], 10) : null;
                    }).filter(Boolean).sort((a,b) => b - a);
                    for (const num of nums) {
                        if (allExisting.size <= target) break;
                        const nm = nameFor(prefix, num);
                        const si = node.inputs.findIndex(i => i.name === nm);
                        if (si !== -1) node.removeInput(si);
                        if (node.widgets) {
                            const wi = node.widgets.findIndex(w => w.name === nm);
                            if (wi !== -1) node.widgets.splice(wi, 1);
                        }
                        allExisting.delete(nm);
                    }
                    
                    // Smart resize after removing inputs
                    setTimeout(() => {
                        node.setDirtyCanvas(true, false);
                        const computedSize = node.computeSize();
                        const currentSize = node.size;
                        const minWidth = 200;
                        const minHeight = 50;
                        let newWidth = Math.max(currentSize[0], minWidth);
                        let newHeight = Math.max(computedSize[1] + 5, minHeight);
                        const heightDiff = Math.abs(currentSize[1] - newHeight);
                        const isGrowing = newHeight > currentSize[1];
                        if (isGrowing || heightDiff > 10) {
                            node.setSize([newWidth, newHeight]);
                        }
                        node.setDirtyCanvas(true, true);
                    }, 50);
                    
                    return;
                }

                // otherwise add missing sockets up to target
                // For AnyType nodes, check if there's an active type and use it for new inputs
                let activeType = info.type;
                if (isAnyTypeNode && node.inputs && node.inputs.length > 0) {
                    // Find the current active type from existing any_X inputs
                    const existingAnyInput = node.inputs.find(inp => 
                        inp.name && inp.name.startsWith(prefix + '_') && inp.type !== "*"
                    );
                    if (existingAnyInput) {
                        activeType = existingAnyInput.type;
                    } else {
                        // Fallback: check if any input is connected and get type from link
                        const existingConnectedInput = node.inputs.find(inp => 
                            inp.name && inp.name.startsWith(prefix + '_') && inp.link !== null
                        );
                        if (existingConnectedInput) {
                            const link = app.graph.links[existingConnectedInput.link];
                            if (link) {
                                activeType = link.type;
                            }
                        }
                    }
                }
                
                for (let i = 1; i <= target; ++i) {
                    const nm = nameFor(prefix, i);
                    if (allExisting.has(nm)) continue;
                    node.addInput(nm, activeType, info.shape !== undefined ? { shape: info.shape } : undefined);
                    allExisting.add(nm);
                }
                
                // Smart resize after adding inputs
                setTimeout(() => {
                    node.setDirtyCanvas(true, false);
                    const computedSize = node.computeSize();
                    const currentSize = node.size;
                    const minWidth = 200;
                    const minHeight = 50;
                    let newWidth = Math.max(currentSize[0], minWidth);
                    let newHeight = Math.max(computedSize[1] + 5, minHeight);
                    const heightDiff = Math.abs(currentSize[1] - newHeight);
                    const isGrowing = newHeight > currentSize[1];
                    if (isGrowing || heightDiff > 10) {
                        node.setSize([newWidth, newHeight]);
                    }
                    node.setDirtyCanvas(true, true);
                }, 50);
                
            };

            // Initial update after a short delay to allow widgets to initialize
            setTimeout(() => { try { updateInputs(); } catch (e) {} }, 80);

            let last = null;
            const pollId = setInterval(() => {
                if (!node.widgets) return;
                const w = node.widgets.find(w => w.name === "inputcount");
                if (!w) return;
                if (w.value !== last) { last = w.value; updateInputs(); }
            }, 200);

            const origRemoved = this.onRemoved || function(){};
            this.onRemoved = function() { clearInterval(pollId); origRemoved.apply(this, arguments); };
        };
    }
});

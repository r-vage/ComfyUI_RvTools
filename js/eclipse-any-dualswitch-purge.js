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
* Dynamic type handling for Dual Any Switch Purge node
*/

import { app } from './comfy/index.js';

app.registerExtension({
    name: "Eclipse.RouterAnyDualSwitchPurge",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Any Dual-Switch Purge [Eclipse]") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                const node = this;
                
                // Setup dynamic type handling for dual switch
                // All inputs must have the same type, output matches inputs
                const setupDualAnyTypeHandling = () => {
                    node.properties = node.properties || {};
                    
                    node.onGraphConfigured = function() {
                        this.configured = true;
                    };
                    
                    // Override onConnectionsChange to handle dynamic typing
                    const originalOnConnectionsChange = node.onConnectionsChange;
                    node.onConnectionsChange = function(type, index, connected, link_info) {
                        if (originalOnConnectionsChange) {
                            originalOnConnectionsChange.apply(this, arguments);
                        }
                        
                        if (!link_info || !this.inputs || !this.outputs) return;
                        
                        const input = this.inputs[index];
                        if (!input || (input.name !== "input1" && input.name !== "input2")) return;
                        
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
                            
                            // Update ALL other inputs to the same type
                            this.inputs.forEach(inp => {
                                if ((inp.name === "input1" || inp.name === "input2") && inp !== input) {
                                    inp.type = sourceType;
                                }
                            });
                            
                            // Update the output type to match
                            if (this.outputs[0]) {
                                this.outputs[0].type = sourceType;
                                this.outputs[0].name = sourceType;
                                
                                // NOTE: Auto-disconnect logic removed - ComfyUI handles bypassed nodes correctly NOW
                            }
                            
                        } else if (!connected && type === LiteGraph.INPUT) {
                            // When disconnecting, check if there are still connections
                            const remainingConnections = this.inputs.filter(inp => 
                                (inp.name === "input1" || inp.name === "input2") && inp.link !== null
                            );
                            
                            if (remainingConnections.length > 0) {
                                // Still have connections, maintain the type from remaining connections
                                const activeType = remainingConnections[0].type;
                                this.inputs.forEach(inp => {
                                    if (inp.name === "input1" || inp.name === "input2") {
                                        inp.type = activeType;
                                    }
                                });
                                if (this.outputs[0]) {
                                    this.outputs[0].type = activeType;
                                    this.outputs[0].name = activeType;
                                    
                                    // NOTE: Auto-disconnect logic removed - ComfyUI handles bypassed nodes correctly NOW
                                }
                            } else {
                                // No connections left, reset everything to wildcard
                                this.inputs.forEach(inp => {
                                    if (inp.name === "input1" || inp.name === "input2") {
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
                
                // Setup immediately
                setupDualAnyTypeHandling();
            };
        }
    }
});

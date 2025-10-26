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
* Centralized handler for AnyType nodes to support dynamic type changing and bypass
*/

import { app } from "../../scripts/app.js";

/**
 * Applies dynamic type changing to a node with AnyType inputs/outputs.
 * This enables proper bypass functionality and visual type feedback.
 * 
 * @param {Object} node - The LiteGraph node to enhance
 * @param {number} inputIndex - Index of the input slot (default: 0)
 * @param {number} outputIndex - Index of the output slot (default: 0)
 */
export function setupAnyTypeHandling(node, inputIndex = 0, outputIndex = 0) {
    node.properties = node.properties || {};

    // Track configuration state
    node.onGraphConfigured = function () {
        this.configured = true;
    }

    // Handle connection changes to update types dynamically
    node.onConnectionsChange = function (type, index, connected, link_info) {
        // Only process if we have link info
        if (!link_info) return;

        if (connected) {
            // INPUT connection: update both input and output to match source type
            if (type === LiteGraph.INPUT && index === inputIndex) {
                const sourceNode = app.graph.getNodeById(link_info.origin_id);
                const sourceType = sourceNode.outputs[link_info.origin_slot].type;
                const color = LGraphCanvas.link_type_colors[sourceType];
                
                // Update types
                this.outputs[outputIndex].type = sourceType;
                this.outputs[outputIndex].name = sourceType;
                this.inputs[inputIndex].type = sourceType;
                
                // Update link color
                if (link_info.id) {
                    app.graph.links[link_info.id].color = color;
                }
                
                // Disconnect incompatible output links
                if (this.outputs[outputIndex].links !== null) {
                    for (let i = this.outputs[outputIndex].links.length; i > 0; i--) {
                        const targetLinkId = this.outputs[outputIndex].links[i - 1];
                        const targetLink = app.graph.links[targetLinkId];
                        if (this.configured && sourceType !== targetLink.type) {
                            app.graph.getNodeById(targetLink.target_id).disconnectInput(targetLink.target_slot);
                        }
                    }
                }
            }
            
            // OUTPUT connection when input is not connected: update to match target type
            if (type === LiteGraph.OUTPUT && index === outputIndex && this.inputs[inputIndex].link === null) {
                this.inputs[inputIndex].type = link_info.type;
                this.outputs[outputIndex].type = link_info.type;
                this.outputs[outputIndex].name = link_info.type;
            }
        } else {
            // Disconnection: reset to wildcard if no connections remain
            const inputDisconnected = (type === LiteGraph.INPUT && index === inputIndex);
            const outputDisconnected = (type === LiteGraph.OUTPUT && index === outputIndex);
            
            const noInputLinks = (this.inputs[inputIndex].link === null);
            const noOutputLinks = (this.outputs[outputIndex].links === null || this.outputs[outputIndex].links.length === 0);
            
            if ((inputDisconnected && noOutputLinks) || (outputDisconnected && noInputLinks)) {
                this.inputs[inputIndex].type = "*";
                this.outputs[outputIndex].name = "";
                this.outputs[outputIndex].type = "*";
            }
        }
        
        // Recompute node size
        this.computeSize();
    };

    // Initialize to wildcard type
    node.onAdded = function () {
        this.inputs[inputIndex].type = "*";
        this.outputs[outputIndex].name = "";
        this.outputs[outputIndex].type = "*";
    };
}

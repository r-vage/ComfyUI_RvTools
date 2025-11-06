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
* Unified handler for all conversion nodes with widget-based type updates
*/

import { app } from './comfy/index.js';
import { setupAnyTypeHandling } from "./eclipse-any-type-handler.js";

// Type mappings for each conversion node
const CONVERSION_NODES = {
    "Convert Primitive [Eclipse]": {
        widgetName: "convert_to",
        typeMap: {
            "STRING": { type: "STRING", name: "STRING" },
            "INT": { type: "INT", name: "INT" },
            "FLOAT": { type: "FLOAT", name: "FLOAT" },
            "COMBO": { type: "COMBO", name: "COMBO" },
        },
        defaultType: "*",
        useAnyTypeHandling: true
    },
    "Convert To Batch [Eclipse]": {
        widgetName: "convert_to",
        typeMap: {
            "IMAGE_LIST_TO_BATCH": { type: "IMAGE", name: "IMAGE" },
            "MASK_LIST_TO_BATCH": { type: "MASK", name: "MASK" },
        },
        defaultType: "*",
        useAnyTypeHandling: true
    },
    "Convert to List [Eclipse]": {
        widgetName: "convert_to",
        typeMap: {
            "IMAGE_BATCH_TO_LIST": { type: "IMAGE", name: "IMAGE" },
            "MASK_BATCH_TO_LIST": { type: "MASK", name: "MASK" },
        },
        defaultType: "*",
        useAnyTypeHandling: false
    },
    "Image Convert [Eclipse]": {
        // No widget-based type changes - always IMAGE
        fixedType: { type: "IMAGE", name: "IMAGE" },
        useAnyTypeHandling: false
    }
};

// Shared function to update output type based on widget selection
function updateOutputTypeFromWidget(node, config) {
    // Handle fixed type nodes
    if (config.fixedType) {
        return;
    }
    
    const convertWidget = node.widgets?.find(w => w.name === config.widgetName);
    if (!convertWidget || !node.outputs || node.outputs.length === 0) return;
    
    const selectedType = convertWidget.value;
    const output = node.outputs[0];
    
    // Get target type from type map
    const typeInfo = config.typeMap[selectedType] || { type: config.defaultType || "*", name: "" };
    const targetType = typeInfo.type;
    const targetName = typeInfo.name;
    
    // Update output type if changed
    if (output.type !== targetType) {
        // Disconnect incompatible links
        if (output.links && output.links.length > 0) {
            const linksToRemove = [];
            for (const linkId of output.links) {
                const link = app.graph.links[linkId];
                if (link) {
                    const targetNode = app.graph.getNodeById(link.target_id);
                    if (targetNode && targetNode.inputs && targetNode.inputs[link.target_slot]) {
                        const targetInput = targetNode.inputs[link.target_slot];
                        // Check if types are incompatible
                        if (targetInput.type !== "*" && targetType !== "*" && targetInput.type !== targetType) {
                            linksToRemove.push(linkId);
                        }
                    }
                }
            }
            
            for (const linkId of linksToRemove) {
                app.graph.removeLink(linkId);
            }
        }
        
        // Update output
        output.type = targetType;
        output.name = targetName;
        
        // Update link colors
        if (output.links && output.links.length > 0) {
            const color = LGraphCanvas.link_type_colors[targetType];
            for (const linkId of output.links) {
                const link = app.graph.links[linkId];
                if (link && color) {
                    link.color = color;
                }
            }
        }
        
        node.setDirtyCanvas?.(true, true);
    }
}

app.registerExtension({
    name: "Eclipse.conversionNodes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const config = CONVERSION_NODES[nodeData.name];
        if (!config) return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            const node = this;
            
            // Apply base AnyType handling if specified
            if (config.useAnyTypeHandling) {
                setupAnyTypeHandling(this, 0, 0);
            }
            
            // Skip further setup for fixed-type nodes
            if (config.fixedType) {
                return result;
            }
            
            // Function to update output type based on widget
            const updateType = () => updateOutputTypeFromWidget(node, config);
            
            // Watch for widget changes
            const convertWidget = node.widgets?.find(w => w.name === config.widgetName);
            if (convertWidget) {
                const originalCallback = convertWidget.callback;
                convertWidget.callback = function() {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    updateType();
                };
            }
            
            // Override connection handler to also update based on widget
            const originalOnConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function(type, index, connected, link_info) {
                if (originalOnConnectionsChange) {
                    originalOnConnectionsChange.apply(this, arguments);
                }
                
                // After connection change, re-evaluate output type based on widget
                setTimeout(() => updateType(), 10);
            };
            
            // Initialize
            setTimeout(() => updateType(), 100);
            
            return result;
        };
    }
});

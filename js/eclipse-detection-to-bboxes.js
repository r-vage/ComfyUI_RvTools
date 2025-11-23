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
* Dynamic widget visibility for Detection to Bboxes node
* Hides CV2 detection widgets when get_mask_from_image is false
* Hides indices widget when combine_masks is true
*/

import { app } from './comfy/index.js';

const NODE_NAME = "Detection to Bboxes [Eclipse]";

app.registerExtension({
    name: "Eclipse.DetectionToBboxes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            const node = this;
            
            const setWidgetVisible = (widgetName, visible) => {
                const widget = node.widgets?.find(w => w.name === widgetName);
                if (!widget) return;
                
                if (visible) {
                    if (widget.origType) {
                        widget.type = widget.origType;
                    } else if (widget.type === "converted-widget") {
                        widget.type = widget.origComboType || "combo";
                        widget.origType = widget.origComboType || "combo";
                    }
                    delete widget.computeSize;
                    widget.hidden = false;
                } else {
                    if (widget.type !== "converted-widget" && !widget.origType) {
                        widget.origType = widget.type;
                        if (widget.type === "combo") {
                            widget.origComboType = "combo";
                        }
                    }
                    widget.type = "converted-widget";
                    widget.computeSize = () => [0, -4];
                    widget.hidden = true;
                }
            };
            
            const getWidgetValue = (name) => {
                const widget = node.widgets?.find(w => w.name === name);
                return widget ? widget.value : null;
            };
            
            const updateVisibility = () => {
                const getMaskFromImage = getWidgetValue("get_mask_from_image");
                const combineMasks = getWidgetValue("combine_masks");
                
                // CV2 detection widgets - only visible when get_mask_from_image is true
                setWidgetVisible("detect_color", getMaskFromImage);
                setWidgetVisible("threshold", getMaskFromImage);
                setWidgetVisible("min_area", getMaskFromImage);
                
                // indices widget - only visible when combine_masks is false
                setWidgetVisible("indices", !combineMasks);
                
                // Smart resize
                setTimeout(() => {
                    node.setDirtyCanvas(true, false);
                    
                    const computedSize = node.computeSize();
                    const currentSize = node.size;
                    
                    const minWidth = 259;
                    const minHeight = 100;
                    
                    let newWidth = Math.max(currentSize[0], minWidth);
                    let newHeight = Math.max(computedSize[1], minHeight);
                    
                    newHeight += 5;
                    
                    const heightDiff = Math.abs(currentSize[1] - newHeight);
                    const isGrowing = newHeight > currentSize[1];
                    
                    if (isGrowing || heightDiff > 10) {
                        node.setSize([newWidth, newHeight]);
                    }
                    
                    node.setDirtyCanvas(true, true);
                }, 50);
            };
            
            // Hook into get_mask_from_image widget
            const getMaskWidget = node.widgets?.find(w => w.name === "get_mask_from_image");
            if (getMaskWidget) {
                const originalCallback = getMaskWidget.callback;
                getMaskWidget.callback = function() {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    updateVisibility();
                };
            }
            
            // Hook into combine_masks widget
            const combineMasksWidget = node.widgets?.find(w => w.name === "combine_masks");
            if (combineMasksWidget) {
                const originalCallback = combineMasksWidget.callback;
                combineMasksWidget.callback = function() {
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    updateVisibility();
                };
            }
            
            // Initial visibility update
            setTimeout(() => {
                updateVisibility();
            }, 10);
            
            // Hook into onConfigure to update visibility when workflow is loaded
            const onConfigure = node.onConfigure;
            node.onConfigure = function(info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                setTimeout(() => {
                    updateVisibility();
                }, 50);
            };
            
            return r;
        };
    }
});

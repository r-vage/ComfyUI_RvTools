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
* Dynamically creates readonly text widgets to display any data type
*/

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_NAME = "Show Any [Eclipse]";

app.registerExtension({
    name: "Eclipse.showAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        /**
         * Populate the node with text widgets showing the data
         */
        function populate(text) {
            // Remove existing text widgets
            if (this.widgets) {
                const pos = this.widgets.findIndex((w) => w.name === "text");
                if (pos !== -1) {
                    for (let i = pos; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = pos;
                }
            }

            // Create new text widgets for each value
            for (const list of text) {
                const w = ComfyWidgets["STRING"](
                    this,
                    "text",
                    ["STRING", { multiline: true }],
                    app
                ).widget;
                
                // Make it readonly with visual feedback
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.inputEl.style.cursor = "default";
                w.value = list;
            }

            // Resize node to fit content
            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) {
                    sz[0] = this.size[0];
                }
                if (sz[1] < this.size[1]) {
                    sz[1] = this.size[1];
                }
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // Add callback to show_images widget to toggle image visibility
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            
            // Store reference to the node
            const node = this;
            
            // Find the show_images widget
            const showImagesWidget = this.widgets?.find(w => w.name === "show_images");
            if (showImagesWidget) {
                // Store original callback
                const originalCallback = showImagesWidget.callback;
                
                // Add custom callback to toggle image visibility
                showImagesWidget.callback = function(value) {
                    // Call original callback if it exists
                    if (originalCallback) {
                        originalCallback.apply(this, arguments);
                    }
                    
                    // Toggle image visibility flag
                    node.showImages = (value === "show");
                    app.graph.setDirtyCanvas(true, false);
                };
                
                // Initialize the flag based on current value
                node.showImages = (showImagesWidget.value === "show");
            }
            
            return r;
        };
        
        // Override onDrawBackground to conditionally hide images
        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function(ctx) {
            // If showImages is false, temporarily hide imgs
            const hadImages = this.imgs;
            if (this.showImages === false && this.imgs) {
                this.imgs = null;
            }
            
            // Call original draw function
            if (onDrawBackground) {
                onDrawBackground.apply(this, arguments);
            }
            
            // Restore imgs
            if (hadImages && this.imgs === null) {
                this.imgs = hadImages;
            }
        };

        // When the node is executed, display the output text
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            
            if (message.text) {
                populate.call(this, message.text);
            }
        };

        // When loading a workflow, restore the text widgets
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            // Set the showImages flag based on the widget value
            const showImagesWidget = this.widgets?.find(w => w.name === "show_images");
            if (showImagesWidget) {
                this.showImages = (showImagesWidget.value === "show");
            }
            // Note: widgets_values[0] is the show_images dropdown value
            // Text widgets are populated dynamically on execution, not from widgets_values
            // This prevents stale data from being displayed
        };

        // Reset node state when connections are removed
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            onConnectionsChange?.apply(this, arguments);
            
            // Clear text widgets and images when input connection is removed
            // Check if any input connection exists
            const hasInput = this.inputs?.some(input => input.link !== null && input.link !== undefined);
            
            if (!hasInput) {
                // Remove text widgets
                if (this.widgets) {
                    const pos = this.widgets.findIndex((w) => w.name === "text");
                    if (pos !== -1) {
                        for (let i = pos; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = pos;
                    }
                }
                
                // Clear images
                this.imgs = null;
                
                // Trigger redraw
                app.graph.setDirtyCanvas(true, false);
            }
        };
    },
});

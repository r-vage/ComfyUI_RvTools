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

import { app } from './comfy/index.js';

const canvasUtilsName = "Eclipse.canvasUtils";

// Track if the new API has been called
let newAPIAvailable = false;

// Define menu items in a reusable function
function getMenuItems() {
  return [
    null,
    // Arrange nodes
    {
      content: "Arrange (vertical)",
      callback: () =>
        app.graph.arrange(
          LiteGraph.CANVAS_GRID_SIZE * 4,
          LiteGraph.VERTICAL_LAYOUT
        ),
    },
    {
      content: "Arrange (horizontal)",
      callback: () => app.graph.arrange(LiteGraph.CANVAS_GRID_SIZE * 2),
    },
    null,
    // Pin/Unpin nodes
    {
      content: "Pin all Nodes",
      callback: () => {
        app.graph._nodes.forEach((node) => {
          node.flags.pinned = true;
        });
      },
    },
    {
      content: "Unpin all Nodes",
      callback: () => {
        app.graph._nodes.forEach((node) => {
          node.flags.pinned = false;
        });
      },
    }
  ];
}

app.registerExtension({
  name: canvasUtilsName,
  
  // New API: getCanvasMenuItems hook (ComfyUI v1.0+)
  getCanvasMenuItems(canvas) {
    // Mark that the new API is being used
    newAPIAvailable = true;
    return getMenuItems();
  },
  
  // Fallback for older ComfyUI versions (pre-v1.0)
  async setup(app) {
    // Wait a bit to see if the new API gets called during initialization
    // If it doesn't get called by the first user interaction, we need the fallback
    
    // Use a one-time event listener to check after first canvas interaction
    const originalOnContextMenu = LGraphCanvas.prototype.onContextMenu;
    let fallbackApplied = false;
    
    LGraphCanvas.prototype.onContextMenu = function(event, options) {
      // Restore original immediately to avoid multiple wraps
      LGraphCanvas.prototype.onContextMenu = originalOnContextMenu;
      
      // If new API wasn't called yet, apply fallback
      if (!newAPIAvailable && !fallbackApplied) {
        fallbackApplied = true;
        console.log(`[${canvasUtilsName}] Using legacy API fallback for older ComfyUI version`);
        
        // Apply monkey-patch for old API
        const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
          const menuOptions = getCanvasMenuOptions.apply(this, arguments);
          menuOptions.push(...getMenuItems());
          return menuOptions;
        };
      }
      
      // Call original handler
      return originalOnContextMenu.apply(this, arguments);
    };
  },
});

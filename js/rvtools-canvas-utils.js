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

const canvasUtilsName = "RvTools.canvasUtils";

// New API: Use getCanvasMenuItems hook (ComfyUI v1.0+)
app.registerExtension({
  name: canvasUtilsName,
  
  getCanvasMenuItems(canvas) {
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
  },
  
  // Old API fallback for older ComfyUI versions
  async setup(app) {
    const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
      const menuOptions = getCanvasMenuOptions.apply(this, arguments);

      menuOptions.push(
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
      );

      return menuOptions;
    };
  },
});

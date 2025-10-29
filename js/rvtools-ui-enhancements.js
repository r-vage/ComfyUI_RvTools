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
* Unified UI enhancements for RvTools
* Includes: Settings, Reroutes, Groups, and Colors
*/

import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

// ============================================================================
// COLOR UTILITIES
// ============================================================================


function rgbToHex(r, g, b) {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function shadeHexColor(hex, amount = -0.2) {
  if (hex.startsWith("#")) {
    hex = hex.slice(1);
  }

  let r = parseInt(hex.slice(0, 2), 16);
  let g = parseInt(hex.slice(2, 4), 16);
  let b = parseInt(hex.slice(4, 6), 16);

  r = Math.max(0, Math.min(255, r + amount * 100));
  g = Math.max(0, Math.min(255, g + amount * 100));
  b = Math.max(0, Math.min(255, b + amount * 100));

  return rgbToHex(r, g, b);
}

// ============================================================================
// COLOR MODES
// ============================================================================
let afterChange;

function invokeAfterChange() {
  return afterChange?.apply(this, arguments);
}

function setColorMode(value, app) {
  switch (value) {
    default:
      app.graph._nodes.forEach((node) => {
        node.bgcolor = node._bgcolor ?? node.bgcolor;
        node.color = node._color ?? node.color;
        node.setDirtyCanvas(true, true);
      });
      break;
  }
}

let loading = false;

// ============================================================================
// MAIN EXTENSION REGISTRATION
// ============================================================================

// Force Box Nodes Setting
app.registerExtension({
  name: "RvTools.forceBoxNodes",
  async init(app) {
    app.ui.settings.addSetting({
      id: "RvTools.forceBoxNodes",
      name: "📦 RvTools Force Box Nodes",
      type: "boolean",
      tooltip: "Remove rounded corners - nodes will always be boxes.",
      defaultValue: false,
      onChange(value) {
        app.canvas.round_radius = value ? 0 : 8;
        app.graph.setDirtyCanvas(true, true);
      },
    });
  },
});


// Colors Enhancement
app.registerExtension({
  name: "RvTools.colors",
  async setup(app) {
    const value = +(
      window.localStorage.getItem("Comfy.Settings.RvTools.colors") ?? "0"
    );
    app.graph._nodes.forEach((node) => {
      node._bgcolor = node._bgcolor ?? node.bgcolor;
      node._color = node._color ?? node.color;
    });
    setColorMode(value, app);
  },
  loadedGraphNode(node, app) {
    node._bgcolor = node._bgcolor ?? node.bgcolor;
    node._color = node._color ?? node.color;
    if (!loading) {
      loading = true;
      setTimeout(function () {
        loading = false;
        const value = +(
          window.localStorage.getItem("Comfy.Settings.RvTools.colors") ?? "0"
        );
        setColorMode(value, app);
      }, 500);
    }
  },
  async init(app) {
    afterChange = app.graph.afterChange;
    app.graph.afterChange = invokeAfterChange;
    
    // Add custom color pickers to context menu
    const onMenuNodeColors = LGraphCanvas.onMenuNodeColors;
    LGraphCanvas.onMenuNodeColors = function (value, options, e, menu, node) {
      const response = onMenuNodeColors.apply(this, arguments);
      const menuRoot = menu.current_submenu.root;
      const isGroup = node instanceof LGraphGroup;

      if (!isGroup) {
        menuRoot.append(
          $el("div.litemenu-entry.submenu", [
            $el(
              "label",
              {
                style: {
                  position: "relative",
                  overflow: "hidden",
                  display: "block",
                  paddingLeft: "4px",
                  borderLeft: "8px solid #222",
                },
              },
              [
                "Custom Title",
                $el("input", {
                  type: "color",
                  value: node.bgcolor,
                  style: {
                    position: "absolute",
                    right: "200%",
                  },
                  oninput(event) {
                    node.color = shadeHexColor(event.target.value);
                    node.setDirtyCanvas(true, true);
                  },
                }),
              ],
            ),
          ]),
        );
        menuRoot.append(
          $el("div.litemenu-entry.submenu", [
            $el(
              "label",
              {
                style: {
                  position: "relative",
                  overflow: "hidden",
                  display: "block",
                  paddingLeft: "4px",
                  borderLeft: "8px solid #222",
                },
              },
              [
                "Custom BG",
                $el("input", {
                  type: "color",
                  value: node.bgcolor,
                  style: {
                    position: "absolute",
                    right: "200%",
                  },
                  oninput(event) {
                    node.bgcolor = event.target.value;
                    node.setDirtyCanvas(true, true);
                  },
                }),
              ],
            ),
          ]),
        );
        menuRoot.append(
          $el("div.litemenu-entry.submenu", [
            $el(
              "label",
              {
                style: {
                  position: "relative",
                  overflow: "hidden",
                  display: "block",
                  paddingLeft: "4px",
                  borderLeft: "8px solid #222",
                },
              },
              [
                "Custom All",
                $el("input", {
                  type: "color",
                  value: node.bgcolor,
                  style: {
                    position: "absolute",
                    right: "200%",
                  },
                  oninput(event) {
                    node.bgcolor = event.target.value;
                    node.color = shadeHexColor(node.bgcolor);
                    node.setDirtyCanvas(true, true);
                  },
                }),
              ],
            ),
          ]),
        );
      }
      
      if (isGroup) {
        menuRoot.append(
          $el("div.litemenu-entry.submenu", [
            $el(
              "label",
              {
                style: {
                  position: "relative",
                  overflow: "hidden",
                  display: "block",
                  paddingLeft: "4px",
                  borderLeft: "8px solid #222",
                },
              },
              [
                "Color Group",
                $el("input", {
                  type: "color",
                  value: node.bgcolor,
                  style: {
                    position: "absolute",
                    right: "200%",
                  },
                  oninput(event) {
                    node.bgcolor = event.target.value;
                    node.color = shadeHexColor(node.bgcolor);
                    node.setDirtyCanvas(true, true);
                  },
                }),
              ],
            ),
          ]),
        );
        menuRoot.append(
          $el("div.litemenu-entry.submenu", [
            $el(
              "label",
              {
                style: {
                  position: "relative",
                  overflow: "hidden",
                  display: "block",
                  paddingLeft: "4px",
                  borderLeft: "8px solid #222",
                },
              },
              [
                "Color All Title",
                $el("input", {
                  type: "color",
                  value: node.bgcolor,
                  style: {
                    position: "absolute",
                    right: "200%",
                  },
                  oninput(event) {
                    node.recomputeInsideNodes();
                    node.color = shadeHexColor(event.target.value);
                    node._nodes.forEach((node_) => {
                      node_.color = shadeHexColor(event.target.value);
                      node_.setDirtyCanvas(true, true);
                    });
                    node.setDirtyCanvas(true, true);
                  },
                }),
              ],
            ),
          ]),
        );
        menuRoot.append(
          $el("div.litemenu-entry.submenu", [
            $el(
              "label",
              {
                style: {
                  position: "relative",
                  overflow: "hidden",
                  display: "block",
                  paddingLeft: "4px",
                  borderLeft: "8px solid #222",
                },
              },
              [
                "Color All BG",
                $el("input", {
                  type: "color",
                  value: node.bgcolor,
                  style: {
                    position: "absolute",
                    right: "200%",
                  },
                  oninput(event) {
                    node.recomputeInsideNodes();
                    node.bgcolor = event.target.value;
                    node._nodes.forEach((node_) => {
                      node_.bgcolor = event.target.value;
                      node_.setDirtyCanvas(true, true);
                    });
                    node.setDirtyCanvas(true, true);
                  },
                }),
              ],
            ),
          ]),
        );
        menuRoot.append(
          $el("div.litemenu-entry.submenu", [
            $el(
              "label",
              {
                style: {
                  position: "relative",
                  overflow: "hidden",
                  display: "block",
                  paddingLeft: "4px",
                  borderLeft: "8px solid #222",
                },
              },
              [
                "Color All",
                $el("input", {
                  type: "color",
                  value: node.bgcolor,
                  style: {
                    position: "absolute",
                    right: "200%",
                  },
                  oninput(event) {
                    node.recomputeInsideNodes();
                    node.bgcolor = event.target.value;
                    node.color = shadeHexColor(node.bgcolor);
                    node._nodes.forEach((node_) => {
                      node_.bgcolor = event.target.value;
                      node_.color = shadeHexColor(node.bgcolor);
                      node_.setDirtyCanvas(true, true);
                    });
                    node.setDirtyCanvas(true, true);
                  },
                }),
              ],
            ),
          ]),
        );
      }
      return response;
    };
    
  },
});


// Adds a "RvTools Node Dimensions" option to node right-click menu which opens
// a small dialog to set width/height for any node.
if (!LGraphCanvas.prototype.rvtoolsSetNodeDimension) {
  // Flag to track if new API is being used for node menu items
  window.rvtools_newNodeMenuAPIUsed = false;
  
  // Ensure rvtools dialog CSS exists so dialog is visible even without TinyTerRa
  if (!document.getElementById("rvtools-dialog-style")) {
    const style = document.createElement("style");
    style.id = "rvtools-dialog-style";
    style.innerHTML = `
    .rvtools-dialog {
      position: fixed;
      top: 10px;
      left: 10px;
      min-height: 1em;
      background-color: var(--comfy-menu-bg, #222);
      color: var(--descrip-text, #ddd);
      font-size: 1.0rem;
      box-shadow: 0 0 7px black !important;
      z-index: 10000;
      display: grid;
      border-radius: 7px;
      padding: 7px 7px;
    }
    .rvtools-dialog .name { display:inline-block; font-size:14px; padding:0; justify-self:center; }
    .rvtools-dialog input, .rvtools-dialog textarea, .rvtools-dialog select { margin:3px; min-width:60px; min-height:1.5em; background-color: var(--comfy-input-bg, #333); border:2px solid var(--border-color, #444); color: var(--input-text, #fff); border-radius:14px; padding-left:10px; outline:none; }
    .rvtools-dialog button { margin-top:3px; vertical-align:top; background-color:#999; border:0; padding:4px 18px; border-radius:20px; cursor:pointer; }
    `;
    document.head.appendChild(style);
  }

  LGraphCanvas.prototype.rvtoolsCreateDialog = function (htmlContent, onOK, onCancel) {
    var dialog = document.createElement("div");
    dialog.is_modified = false;
    dialog.className = "rvtools-dialog";
    dialog.innerHTML = htmlContent + "<button id='ok'>OK</button>";

    dialog.close = function () {
      if (dialog.parentNode) {
        dialog.parentNode.removeChild(dialog);
      }
    };

    var inputs = Array.from(dialog.querySelectorAll("input, select"));

    inputs.forEach((input) => {
      input.addEventListener("keydown", function (e) {
        dialog.is_modified = true;
        if (e.keyCode == 27) {
          onCancel && onCancel();
          dialog.close();
        } else if (e.keyCode == 13) {
          onOK && onOK(dialog, inputs.map((input) => input.value));
          dialog.close();
        } else if (e.keyCode != 13 && e.target.localName != "textarea") {
          return;
        }
        e.preventDefault();
        e.stopPropagation();
      });
    });

    var graphcanvas = LGraphCanvas.active_canvas;
    var canvas = graphcanvas.canvas;

    // Position near the mouse event when available; otherwise center on canvas
    try {
      var rect = canvas.getBoundingClientRect();
      var offsetx = -20;
      var offsety = -20;
      if (rect) {
        offsetx -= rect.left;
        offsety -= rect.top;
      }

      if (typeof event !== 'undefined' && event && event.clientX) {
        dialog.style.left = event.clientX + offsetx + "px";
        dialog.style.top = event.clientY + offsety + "px";
      } else {
        // fallback to canvas center in viewport coordinates
        const left = (rect?.left ?? 0) + (canvas.width * 0.5) + offsetx;
        const top = (rect?.top ?? 0) + (canvas.height * 0.5) + offsety;
        dialog.style.left = left + "px";
        dialog.style.top = top + "px";
      }
    } catch (e) {
      // best-effort positioning
      dialog.style.left = '20px';
      dialog.style.top = '20px';
    }

    var button = dialog.querySelector("#ok");
    button.addEventListener("click", function () {
      onOK && onOK(dialog, inputs.map((input) => input.value));
      dialog.close();
    });

    canvas.parentNode.appendChild(dialog);
    if (inputs) inputs[0].focus();

    var dialogCloseTimer = null;
    dialog.addEventListener("mouseleave", function (e) {
      if (LiteGraph.dialog_close_on_mouse_leave) if (!dialog.is_modified && LiteGraph.dialog_close_on_mouse_leave)
        dialogCloseTimer = setTimeout(dialog.close, LiteGraph.dialog_close_on_mouse_leave_delay);
    });
    dialog.addEventListener("mouseenter", function (e) {
      if (LiteGraph.dialog_close_on_mouse_leave) if (dialogCloseTimer) clearTimeout(dialogCloseTimer);
    });

    return dialog;
  };

  LGraphCanvas.prototype.rvtoolsSetNodeDimension = function (node) {
    const nodeWidth = node.size[0];
    const nodeHeight = node.size[1];

    let input_html = "<input type='text' class='width' value='" + nodeWidth + "'></input>";
    input_html += "<input type='text' class='height' value='" + nodeHeight + "'></input>";

    LGraphCanvas.prototype.rvtoolsCreateDialog("<span class='name'>Width/Height</span>" + input_html,
      function (dialog, values) {
        var widthValue = Number(values[0]) ? values[0] : nodeWidth;
        var heightValue = Number(values[1]) ? values[1] : nodeHeight;
        let sz = node.computeSize();
        node.setSize([Math.max(sz[0], widthValue), Math.max(sz[1], heightValue)]);
        if (dialog.parentNode) {
          dialog.parentNode.removeChild(dialog);
        }
        node.setDirtyCanvas(true, true);
      },
      null
    );
  };

  // Old API: Inject menu item into node right-click menu (non-destructive: keep existing options)
  // Use delayed detection to avoid triggering deprecation warning on new ComfyUI versions
  const origOnShowNodeMenu = LGraphCanvas.prototype.showContextMenu;
  let fallbackApplied = false;
  
  LGraphCanvas.prototype.showContextMenu = function(menu_info, options) {
    // Restore original immediately to avoid wrapping multiple times
    LGraphCanvas.prototype.showContextMenu = origOnShowNodeMenu;
    
    // If new API hasn't been called yet, apply the fallback
    if (!window.rvtools_newNodeMenuAPIUsed && !fallbackApplied) {
      fallbackApplied = true;
      console.log('[RvTools.nodeMenuItems] Using legacy API fallback for older ComfyUI version');
      
      const origGetNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
      LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
        const options = origGetNodeMenuOptions.apply(this, arguments);
        // Insert our item just before the last separator
        try {
          const item = {
            content: "RvTools: Node Dimensions",
            callback: () => {
              LGraphCanvas.prototype.rvtoolsSetNodeDimension(node);
            },
          };
          const reloadItem = {
            content: "RvTools: Reload Node",
            callback: () => {
              try {
                LGraphCanvas.prototype.rvtoolsReloadNode(node);
              } catch (e) {
                console.debug('rvtools: Reload Node failed', e);
              }
            },
          };
          // Insert both items adjacent with no separator.
          const hasNodeDimensions = options.some((o) => o && o.content && String(o.content).includes("RvTools: Node Dimensions"));
          const hasReloadNode = options.some((o) => o && o.content && String(o.content).includes("RvTools: Reload Node"));
          // If neither exist, insert both together. If one exists, insert the missing one next to it.
          if (!hasNodeDimensions && !hasReloadNode) {
            options.splice(options.length - 1, 0, item, reloadItem);
          } else if (!hasNodeDimensions && hasReloadNode) {
            options.splice(options.length - 1, 0, item);
          } else if (hasNodeDimensions && !hasReloadNode) {
            // find the index of the existing Node Dimensions and insert refresh after it
            const idx = options.findIndex((o) => o && o.content && String(o.content).includes("Node Dimensions"));
            if (idx >= 0) {
              options.splice(idx + 1, 0, reloadItem);
            } else {
              options.splice(options.length - 1, 0, reloadItem);
            }
          }
        } catch (e) {
          console.debug('rvtools: failed to inject Node Dimensions menu item', e);
        }
        return options;
      };
    }
    
    // Call original handler
    return origOnShowNodeMenu.apply(this, arguments);
  };

  // Full rvtoolsReloadNode implementation (adapted from TinyTerra tinyterraReloadNode)
  LGraphCanvas.prototype.rvtoolsReloadNode = function (node) {
    try {
      const CONVERTED_TYPE = "converted-widget";
      const GET_CONFIG = Symbol();

      function getConfig(widgetName, nodeRef) {
        const { nodeData } = nodeRef.constructor;
        return nodeData?.input?.required[widgetName] ?? nodeData?.input?.optional?.[widgetName];
      }

      function hideWidget(nodeRef, widget, suffix = "") {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.origSerializeValue = widget.serializeValue;
        widget.computeSize = () => [0, -4];
        widget.type = CONVERTED_TYPE + suffix;
        widget.serializeValue = () => {
          if (!nodeRef.inputs) return undefined;
          let node_input = nodeRef.inputs.find((i) => i.widget?.name === widget.name);
          if (!node_input || !node_input.link) return undefined;
          return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
        };
        if (widget.linkedWidgets) {
          for (const w of widget.linkedWidgets) {
            hideWidget(nodeRef, w, ":" + widget.name);
          }
        }
      }

      function getWidgetType(config) {
        let type = config[0];
        if (type instanceof Array) type = "COMBO";
        return { type };
      }

      function convertToInput(nodeRef, widget, config) {
        hideWidget(nodeRef, widget);
        const { type } = getWidgetType(config);
        const sz = nodeRef.size;
        nodeRef.addInput(widget.name, type, {
          widget: { name: widget.name, [GET_CONFIG]: () => config },
        });
        for (const w of nodeRef.widgets) {
          w.last_y += LiteGraph.NODE_SLOT_HEIGHT;
        }
        nodeRef.setSize([Math.max(sz[0], nodeRef.size[0]), Math.max(sz[1], nodeRef.size[1])]);
      }

      // Begin reload logic
      const { title: nodeTitle, color: nodeColor, bgcolor: bgColor } = node.properties.origVals || node;
      const options = {
        size: [...node.size],
        color: nodeColor,
        bgcolor: bgColor,
        pos: [...node.pos],
      };

      const oldNode = node;

      const inputConnections = [], outputConnections = [];
      if (node.inputs) {
        for (const input of node.inputs ?? []) {
          if (input.link) {
            const input_name = input.name;
            const input_slot = node.findInputSlot(input_name);
            const input_node = node.getInputNode(input_slot);
            const input_link = node.getInputLink(input_slot);
            inputConnections.push([input_link.origin_slot, input_node, input_name]);
          }
        }
      }
      if (node.outputs) {
        for (const output of node.outputs) {
          if (output.links) {
            const output_name = output.name;
            for (const linkID of output.links) {
              const output_link = app.graph.links[linkID];
              const output_node = app.graph._nodes_by_id[output_link.target_id];
              outputConnections.push([output_name, output_node, output_link.target_slot]);
            }
          }
        }
      }

  app.graph.remove(node);
      const newNode = app.graph.add(LiteGraph.createNode(oldNode.constructor.type, nodeTitle, options));
      if (newNode?.constructor?.hasOwnProperty("ttNnodeVersion")) {
        newNode.properties.ttNnodeVersion = newNode.constructor.ttNnodeVersion;
      }

      function handleLinks() {
        for (let ow of oldNode.widgets) {
          if (ow.type === CONVERTED_TYPE) {
            const config = getConfig(ow.name, oldNode);
            const WidgetToConvert = newNode.widgets.find((nw) => nw.name === ow.name);
            if (WidgetToConvert && !newNode?.inputs?.find((i) => i.name === ow.name)) {
              convertToInput(newNode, WidgetToConvert, config);
            }
          }
        }

        for (let input of inputConnections) {
          const [output_slot, output_node, input_name] = input;
          output_node.connect(output_slot, newNode.id, input_name);
        }
        for (let output of outputConnections) {
          const [output_name, input_node, input_slot] = output;
          newNode.connect(output_name, input_node, input_slot);
        }
      }

      // fix widget values
      let values = oldNode.widgets_values;
      if (!values) {
        console.log("NO VALUES");
        newNode.widgets.forEach((newWidget, index) => {
          let pass = false;
          while (index < oldNode.widgets.length && !pass) {
            const oldWidget = oldNode.widgets[index];
            if (newWidget.type === oldWidget.type) {
              newWidget.value = oldWidget.value;
              pass = true;
            }
            index++;
          }
        });
      } else {
        let isValid = false;
        const isIterateForwards = values.length <= newNode.widgets.length;
        let valueIndex = isIterateForwards ? 0 : values.length - 1;

        const parseWidgetValue = (value, widget) => {
          if (['', null].includes(value) && (widget.type === 'button' || widget.type === 'converted-widget')) {
            return { value, isValid: true };
          }
          if (typeof value === 'boolean' && widget.options?.on && widget.options?.off) {
            return { value, isValid: true };
          }
          if (widget.options?.values?.includes(value)) {
            return { value, isValid: true };
          }
          if (widget.inputEl) {
            if (typeof value === 'string' || value === widget.value) {
              return { value, isValid: true };
            }
          }
          if (!isNaN(value)) {
            value = parseFloat(value);
            if (widget.options?.min <= value && value <= widget.options?.max) {
              return { value, isValid: true };
            }
          }
          return { value: widget.value, isValid: false };
    };

        function updateValue(widgetIndex) {
          const oldWidget = oldNode.widgets[widgetIndex];
          let newWidget = newNode.widgets[widgetIndex];
          let newValueIndex = valueIndex;

          if (newWidget.name === oldWidget.name && (newWidget.type === oldWidget.type || oldWidget.type === 'ttNhidden' || newWidget.type === 'ttNhidden')) {
            while ((isIterateForwards ? newValueIndex < values.length : newValueIndex >= 0) && !isValid) {
              let parsed = parseWidgetValue(values[newValueIndex], newWidget);
              let value = parsed.value;
              isValid = parsed.isValid;
              if (isValid && value !== NaN) {
                newWidget.value = value;
                break;
              }
              newValueIndex += isIterateForwards ? 1 : -1;
            }

            if (isIterateForwards) {
              if (newValueIndex === valueIndex) {
                valueIndex++;
              }
              if (newValueIndex === valueIndex + 1) {
                valueIndex++;
                valueIndex++;
              }
            } else {
              if (newValueIndex === valueIndex) {
                valueIndex--;
              }
              if (newValueIndex === valueIndex - 1) {
                valueIndex--;
                valueIndex--;
              }
            }
          }
        }

        if (isIterateForwards) {
          for (let widgetIndex = 0; widgetIndex < newNode.widgets.length; widgetIndex++) {
            updateValue(widgetIndex);
          }
        } else {
          for (let widgetIndex = newNode.widgets.length - 1; widgetIndex >= 0; widgetIndex--) {
            updateValue(widgetIndex);
          }
        }
      }

      handleLinks();
      newNode.setSize(options.size);
      newNode.onResize([0, 0]);
      newNode.setDirtyCanvas(true, true);
    } catch (e) {
      console.debug('rvtools: rvtoolsReloadNode exception', e);
    }
  };
}

// New API: Register extension with getNodeMenuItems hook (ComfyUI v1.0+)
app.registerExtension({
  name: "RvTools.nodeMenuItems",
  
  getNodeMenuItems(node) {
    // Mark that new API is in use
    window.rvtools_newNodeMenuAPIUsed = true;
    
    return [
      {
        content: "RvTools: Node Dimensions",
        callback: () => {
          LGraphCanvas.prototype.rvtoolsSetNodeDimension(node);
        },
      },
      {
        content: "RvTools: Reload Node",
        callback: () => {
          try {
            LGraphCanvas.prototype.rvtoolsReloadNode(node);
          } catch (e) {
            console.debug('rvtools: Reload Node failed', e);
          }
        },
      },
    ];
  },
});

app.registerExtension({
  name: "RvTools.appearance",

  nodeCreated(node) {
    // Only apply appearance to RvTools nodes (nodes created by this extension)
    // Prefer matching the displayed NODE_NAME which commonly includes '[RvTools]'
    try {
      const title = node.title || node.constructor?.title || "";
      const comfy = node.comfyClass || "";
      const ctorType = node.constructor?.type || "";
      const nodeType = node.type || "";

      // Flexible detection: match if any identifier contains the RvTools marker or 'Rv' prefix
      const matchesTag = (s) => typeof s === 'string' && s.includes('[RvTools]');
      const matchesRv = (s) =>
        typeof s === 'string' &&
        (s.startsWith('Rv') || s.includes('Rv') || s.toLowerCase().includes('rv'));

      const isRvNode =
        matchesTag(title) ||
        matchesTag(comfy) ||
        matchesTag(node.constructor?.title) ||
        matchesRv(comfy) ||
        matchesRv(ctorType) ||
        matchesRv(nodeType);

      function applyAppearanceNow() {
        node.color = "#4e4e4e";
        node.bgcolor = "#3a3a3a";
        node.shape = "box";
        node.setDirtyCanvas?.(true, true);
        node._rvtools_appearance_applied = true;
      }

      if (isRvNode) {
        // Don't overwrite if we've already applied appearance
        if (node._rvtools_appearance_applied) {
          // already applied, nothing to do
        } else {
          // Record initial colors so we can detect if something/user changed them
          if (node._rvtools_initial_bgcolor === undefined)
            node._rvtools_initial_bgcolor = node.bgcolor;
          if (node._rvtools_initial_color === undefined)
            node._rvtools_initial_color = node.color;

          // Only apply if the node's current colors still equal the initial values (i.e., not user-customized)
          const stillDefault =
            node.bgcolor === node._rvtools_initial_bgcolor &&
            node.color === node._rvtools_initial_color;
          if (stillDefault) {
            applyAppearanceNow();
          }

          // One short re-check to handle cases where node construction finishes after this hook
          setTimeout(() => {
            if (node._rvtools_appearance_applied) return;
            const stillDefaultLater =
              node.bgcolor === node._rvtools_initial_bgcolor &&
              node.color === node._rvtools_initial_color;
            if (stillDefaultLater) {
              applyAppearanceNow();
            } else {
              // Someone changed the color already, don't overwrite
            }
          }, 50);
        }
      }
    } catch (e) {
      // if node doesn't expose expected properties or other unexpected error, don't modify it
      // swallow errors silently
    }
  },
});

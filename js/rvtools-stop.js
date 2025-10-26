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
* Based on: ComfyUI.mxToolkit.Stop v.0.9.7 - Max Smirnov 2024
*/

import { app } from "../../scripts/app.js";
import { setupAnyTypeHandling } from "./rvtools-any-type-handler.js";

class RvTools_Stop
{
    constructor(node)
    {
        this.node = node;
        
        // Apply centralized AnyType handling
        setupAnyTypeHandling(this.node, 0, 0);

        this.node.onMouseDown = function(e, pos, canvas)
        {
            let cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            if ( e.canvasY-this.pos[1] > 0 ) return false;
            if (this.flags.collapsed && (e.canvasX-this.pos[0] < LiteGraph.NODE_TITLE_HEIGHT)) return false;
            if (!this.flags.collapsed && ((e.canvasX-this.pos[0]) < (this.size[0]-cWidth+LiteGraph.NODE_TITLE_HEIGHT))) return false;
            this.updateThisNodeGraph?.();
            this.onTmpMouseUp(e, pos, canvas);
            return true;
        }

        this.node.onTmpMouseUp = function(e, pos, canvas)
        {
            app.queuePrompt(0);
        }

        this.node.onDrawForeground = function(ctx)
        {
            this.configured = true;
            if (this.size[1] > LiteGraph.NODE_SLOT_HEIGHT*1.3) this.size[1] = LiteGraph.NODE_SLOT_HEIGHT*1.3;
            let titleHeight = LiteGraph.NODE_TITLE_HEIGHT;
            let cWidth = this._collapsed_width || LiteGraph.NODE_COLLAPSED_WIDTH;
            let buttonWidth = cWidth-titleHeight-6;
            let cx = (this.flags.collapsed?cWidth:this.size[0])-buttonWidth-6;

            ctx.fillStyle = this.color || LiteGraph.NODE_DEFAULT_COLOR;
            ctx.beginPath();
            ctx.rect(cx, 2-titleHeight, buttonWidth, titleHeight-4);
            ctx.fill();

            cx += buttonWidth/2;

            ctx.lineWidth = 1;
            if (this.mouseOver)
            {
                ctx.fillStyle = LiteGraph.NODE_SELECTED_TITLE_COLOR
                ctx.beginPath(); ctx.moveTo(cx-8,-titleHeight/2-8); ctx.lineTo(cx+0,-titleHeight/2); ctx.lineTo(cx-8,-titleHeight/2+8); ctx.fill();
                ctx.beginPath(); ctx.moveTo(cx+1,-titleHeight/2-8); ctx.lineTo(cx+9,-titleHeight/2); ctx.lineTo(cx+1,-titleHeight/2+8); ctx.fill();
            }
            else
            {
                ctx.fillStyle = (this.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR);
                ctx.beginPath(); ctx.rect(cx-10,-titleHeight/2-8, 4, 16); ctx.fill();
                ctx.beginPath(); ctx.rect(cx-2,-titleHeight/2-8, 4, 16); ctx.fill();
            }
        }

        this.node.computeSize = function()
        {
            return [ (this.properties.showOutputText && this.outputs && this.outputs.length) ? LiteGraph.NODE_TEXT_SIZE * (this.outputs[0].name.length+5) * 0.6 + 140 : 140, LiteGraph.NODE_SLOT_HEIGHT*1.3 ];
        }
    }
}

app.registerExtension(
{
    name: "Stop [RvTools]",
    async beforeRegisterNodeDef(nodeType, nodeData, _app)
    {
        if (nodeData.name === "Stop [RvTools]")
        {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, []);
                this.Stop = new RvTools_Stop(this);
            }
        }
    }
});

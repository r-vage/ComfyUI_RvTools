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
* Dynamic type handling for Any Passer Purge node
*/

import { app } from "../../scripts/app.js";
import { setupAnyTypeHandling } from "./rvtools-any-type-handler.js";

app.registerExtension({
    name: "RvTools.RouterAnyPasserPurge",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Any Passer Purge [RvTools]") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // Apply centralized AnyType handling
                // Input index 0 (input), Output index 0 (output)
                setupAnyTypeHandling(this, 0, 0);
            };
        }
    }
});

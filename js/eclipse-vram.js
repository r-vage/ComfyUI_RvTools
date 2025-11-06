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

import { app, api } from './comfy/index.js';
app.registerExtension({
    name: "memory.cleanup",
    init() {
        api.addEventListener("memory_cleanup", ({ detail }) => {
            if (detail.type === "cleanup_request") {
                console.log("Memory cleanup request received");
                fetch("/free", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(detail.data)
                })
                .then(response => {
                    if (response.ok) {
                        console.log("Memory cleanup request sent");
                    } else {
                        console.error("Memory cleanup request failed");
                    }
                })
                .catch(error => {
                    console.error("Error sending memory cleanup request:", error);
                });
            }
        });
    }
});

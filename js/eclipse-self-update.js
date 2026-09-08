import { app, api } from './comfy/index.js';

const PACK_NAME = 'Eclipse';
const UPDATE_PREFIX = '/eclipse/update';
const SETTING_ID = 'Eclipse.SelfUpdate';
const SETTING_CATEGORY = ["Eclipse","General"];

function setVersionText(element, payload) {
    const running = payload?.running_version || 'unknown';
    const disk = payload?.disk_version || running;
    element.textContent = running === disk
        ? `Current version: ${running}`
        : `Running version: ${running} · Updated on disk: ${disk}`;
}

async function readPayload(response) {
    try {
        return await response.json();
    } catch {
        return {
            success: false,
            error: `Server returned HTTP ${response.status}`,
        };
    }
}

export function createSelfUpdateControl(apiRef = api, confirmUpdate = window.confirm.bind(window)) {
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.alignItems = 'flex-start';
    container.style.gap = '0.45rem';
    container.style.width = '100%';

    const version = document.createElement('div');
    version.textContent = 'Current version: loading…';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'p-button p-component';
    button.textContent = `Update ${PACK_NAME}`;
    button.disabled = true;
    let updateSupported = false;

    const status = document.createElement('div');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    status.style.fontSize = '0.85rem';
    status.style.opacity = '0.85';

    container.append(version, button, status);

    const refreshStatus = async () => {
        try {
            const response = await apiRef.fetchApi(`${UPDATE_PREFIX}/status`);
            const payload = await readPayload(response);
            setVersionText(version, payload);
            updateSupported = payload.git_supported === true;
            button.disabled = !updateSupported;
            status.textContent = updateSupported
                ? ''
                : (payload.message || payload.error || 'Self-update is unavailable.');
            return payload;
        } catch (error) {
            updateSupported = false;
            button.disabled = true;
            status.textContent = 'Could not read update status.';
            console.error('[Eclipse] Failed to read self-update status:', error);
            return null;
        }
    };

    button.addEventListener('click', async () => {
        const confirmed = confirmUpdate(
            `Update ${PACK_NAME} to the latest official main branch?\n\n` +
            'This permanently overwrites tracked changes, local commits, and fork changes. ' +
            'No backup is created. Untracked user files are preserved. Python requirements ' +
            'will be installed after the update. ComfyUI must be restarted afterward.'
        );
        if (!confirmed) return;

        button.disabled = true;
        button.textContent = 'Updating…';
        status.textContent = 'Fetching official main and installing requirements…';
        try {
            const response = await apiRef.fetchApi(UPDATE_PREFIX, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ confirmed: true }),
            });
            const payload = await readPayload(response);
            setVersionText(version, payload);
            if (payload.status === 'unsupported') updateSupported = false;
            if (payload.restart_required) {
                status.textContent = payload.success
                    ? 'Update installed. Restart ComfyUI to load it.'
                    : `${payload.error || 'Update incomplete.'} Restart ComfyUI before continuing.`;
            } else {
                status.textContent = payload.message || payload.error || (
                    response.ok ? 'Already on the latest official version.' : 'Update failed.'
                );
            }
        } catch (error) {
            status.textContent = 'Update request failed.';
            console.error('[Eclipse] Self-update failed:', error);
        } finally {
            button.textContent = `Update ${PACK_NAME}`;
            button.disabled = !updateSupported;
        }
    });

    void refreshStatus();
    return container;
}

app.registerExtension({
    name: 'Eclipse.SelfUpdate',
    async init(appRef) {
        appRef.ui.settings.addSetting({
            id: SETTING_ID,
            category: [...SETTING_CATEGORY, 'Update'],
            name: '⬆️ Software Update',
            type: () => createSelfUpdateControl(),
            tooltip: 'Show the installed version and update tracked files from the official main branch.',
            defaultValue: null,
            sortOrder: 1000,
        });
    },
});

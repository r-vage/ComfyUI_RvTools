/**
 * Eclipse Markdown Note — scroll-constrained workflow annotation.
 */

import { app } from './comfy/index.js';
import {
    captureScrollableWheelInVue,
    isConfiguringGraph,
    removeSocketlessInputs,
} from './eclipse-widget-performance-utils.js';

export const MARKDOWN_NOTE_NODE = 'Markdown Note [Eclipse]';
export const MARKDOWN_NOTE_WIDGET = 'text';
export const MARKDOWN_NOTE_MIN_HEIGHT = 80;
export const MARKDOWN_NOTE_DEFAULT_SIZE = [320, 220];

const STYLE_ID = 'eclipse-markdown-note-style';

function installStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
.eclipse-markdown-note-widget {
    position: relative;
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
    box-sizing: border-box;
}
.eclipse-markdown-note-widget > .eclipse-markdown-note-preview,
.eclipse-markdown-note-widget > .eclipse-markdown-note-editor {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
    max-height: 100%;
    overflow-y: auto;
    box-sizing: border-box;
    scrollbar-width: thin;
    scrollbar-color: var(--p-surface-500, #666) transparent;
}
.eclipse-markdown-note-widget > .eclipse-markdown-note-preview {
    padding: 8px 10px;
    color: var(--fg-color, #ddd);
    background: var(--comfy-input-bg, rgba(0, 0, 0, 0.2));
    border: 1px solid var(--border-color, #555);
    border-radius: 6px;
    cursor: text;
}
.eclipse-markdown-note-widget > .eclipse-markdown-note-preview.eclipse-markdown-note-fallback {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.eclipse-markdown-note-widget > .eclipse-markdown-note-editor {
    resize: none;
    padding: 8px 10px;
}
.eclipse-markdown-note-widget > [hidden] {
    display: none;
}
.eclipse-markdown-note-widget > .eclipse-markdown-note-preview::-webkit-scrollbar,
.eclipse-markdown-note-widget > .eclipse-markdown-note-editor::-webkit-scrollbar {
    width: 8px;
}
.eclipse-markdown-note-widget > .eclipse-markdown-note-preview::-webkit-scrollbar-thumb,
.eclipse-markdown-note-widget > .eclipse-markdown-note-editor::-webkit-scrollbar-thumb {
    background: var(--p-surface-500, #666);
    border: 2px solid transparent;
    border-radius: 999px;
    background-clip: padding-box;
}
.eclipse-markdown-note-preview > :first-child { margin-top: 0; }
.eclipse-markdown-note-preview > :last-child { margin-bottom: 0; }
.eclipse-markdown-note-preview pre,
.eclipse-markdown-note-preview code {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
.eclipse-markdown-note-preview img,
.eclipse-markdown-note-preview video {
    max-width: 100%;
    height: auto;
}
`;
    document.head.appendChild(style);
}

function normalizeText(value) {
    return value == null ? '' : String(value);
}

export function renderMarkdownPreview(preview, markdown, renderer) {
    const text = normalizeText(markdown);
    preview.classList.remove('eclipse-markdown-note-fallback');
    try {
        if (typeof renderer !== 'function') throw new Error('Markdown renderer unavailable');
        const html = renderer(text);
        if (typeof html !== 'string') throw new TypeError('Markdown renderer returned non-text');
        preview.innerHTML = html;
        return true;
    } catch {
        preview.replaceChildren(document.createTextNode(text));
        preview.classList.add('eclipse-markdown-note-fallback');
        return false;
    }
}

export function captureScrollableWheel(element) {
    const handleWheel = (event) => {
        if (!event.deltaY || element.scrollHeight <= element.clientHeight) return;
        const atTop = element.scrollTop <= 0;
        const atBottom = element.scrollTop + element.clientHeight >= element.scrollHeight - 1;
        if ((event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom)) return;
        event.stopPropagation();
    };
    element.addEventListener('wheel', handleWheel);
    const disposeVueCapture = captureScrollableWheelInVue(element);
    return () => {
        element.removeEventListener('wheel', handleWheel);
        disposeVueCapture();
    };
}

export function createMarkdownNoteController(node, initialValue = '') {
    installStyles();

    const wrapper = document.createElement('div');
    wrapper.className = 'eclipse-markdown-note-widget';
    wrapper.style.cssText = 'position:relative;width:100%;height:100%;min-height:0;overflow:hidden;box-sizing:border-box;';

    const preview = document.createElement('div');
    preview.className = 'eclipse-markdown-note-preview comfy-markdown-content';
    preview.tabIndex = 0;
    preview.setAttribute('role', 'textarea');
    preview.setAttribute('aria-label', 'Markdown Note');
    preview.setAttribute('aria-readonly', 'true');
    preview.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;min-height:0;max-height:100%;overflow-y:auto;box-sizing:border-box;';

    const editor = document.createElement('textarea');
    editor.className = 'eclipse-markdown-note-editor comfy-multiline-input';
    editor.setAttribute('aria-label', 'Edit Markdown Note');
    editor.hidden = true;
    editor.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;min-height:0;max-height:100%;overflow-y:auto;resize:none;box-sizing:border-box;';

    wrapper.append(preview, editor);

    let value = normalizeText(initialValue);
    let valueBeforeEditing = value;
    let disposed = false;
    let widget;
    const listeners = [];
    const listen = (element, type, callback, options) => {
        element.addEventListener(type, callback, options);
        listeners.push(() => element.removeEventListener(type, callback, options));
    };
    const renderer = (text) => {
        const render = app.extensionManager?.renderMarkdownToHtml;
        if (typeof render !== 'function') throw new Error('ComfyUI Markdown renderer unavailable');
        return render.call(app.extensionManager, text);
    };
    const refreshPreview = () => renderMarkdownPreview(preview, value, renderer);
    const startEditing = (event) => {
        if (disposed || !editor.hidden) return;
        event?.stopPropagation?.();
        valueBeforeEditing = value;
        editor.value = value;
        preview.hidden = true;
        editor.hidden = false;
        queueMicrotask(() => {
            if (!disposed && !editor.hidden) editor.focus({ preventScroll: true });
        });
    };
    const finishEditing = () => {
        if (disposed || editor.hidden) return;
        value = editor.value;
        editor.hidden = true;
        preview.hidden = false;
        refreshPreview();
        if (value !== valueBeforeEditing) {
            node.onWidgetChanged?.(MARKDOWN_NOTE_WIDGET, value, valueBeforeEditing, widget);
            if (node.graph) node.graph._version = (node.graph._version || 0) + 1;
            node.setDirtyCanvas?.(true, true);
        }
    };

    listen(preview, 'dblclick', startEditing);
    listen(editor, 'input', () => { value = editor.value; });
    listen(editor, 'blur', finishEditing);
    for (const element of [preview, editor]) {
        for (const type of ['pointerdown', 'pointermove', 'pointerup', 'click', 'keydown']) {
            listen(element, type, (event) => event.stopPropagation());
        }
    }

    const disposePreviewWheel = captureScrollableWheel(preview);
    const disposeEditorWheel = captureScrollableWheel(editor);

    editor.value = value;
    refreshPreview();

    widget = node.addDOMWidget(MARKDOWN_NOTE_WIDGET, 'custom', wrapper, {
        getMinHeight: () => MARKDOWN_NOTE_MIN_HEIGHT,
        getMaxHeight: () => undefined,
        getValue: () => value,
        hideOnZoom: false,
        serialize: true,
        setValue: (nextValue) => {
            value = normalizeText(nextValue);
            editor.value = value;
            refreshPreview();
        },
    });
    widget.serialize = true;
    widget.serializeValue = () => value;
    widget.computeLayoutSize = () => ({
        minHeight: MARKDOWN_NOTE_MIN_HEIGHT,
        maxHeight: undefined,
        minWidth: 180,
    });

    const originalWidgetRemove = widget.onRemove;
    const dispose = () => {
        if (disposed) return;
        disposed = true;
        for (const remove of listeners.splice(0)) remove();
        disposePreviewWheel();
        disposeEditorWheel();
    };
    widget.onRemove = function () {
        dispose();
        return originalWidgetRemove?.apply(this, arguments);
    };

    return {
        dispose,
        editor,
        finishEditing,
        preview,
        refreshPreview,
        startEditing,
        widget,
        wrapper,
    };
}

installStyles();

app.registerExtension({
    name: 'Eclipse.MarkdownNote',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== MARKDOWN_NOTE_NODE) return;

        nodeType.prototype.isVirtualNode = true;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            removeSocketlessInputs(this);
            const backingWidget = this.widgets?.find((item) => item.name === MARKDOWN_NOTE_WIDGET);
            if (!backingWidget) return result;

            const originalIndex = this.widgets.indexOf(backingWidget);
            const initialValue = backingWidget.value;
            backingWidget.onRemove?.();
            this.widgets.splice(originalIndex, 1);

            const controller = createMarkdownNoteController(this, initialValue);
            const addedIndex = this.widgets.indexOf(controller.widget);
            if (addedIndex >= 0 && addedIndex !== originalIndex) {
                this.widgets.splice(addedIndex, 1);
                this.widgets.splice(originalIndex, 0, controller.widget);
            }

            this.isVirtualNode = true;
            this.serialize_widgets = true;
            this._eclipseMarkdownNoteController = controller;

            const yellow = globalThis.LGraphCanvas?.node_colors?.yellow;
            if (yellow) {
                this.color = yellow.color;
                this.bgcolor = yellow.bgcolor;
                this.groupcolor = yellow.groupcolor;
            }

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                controller.dispose();
                return originalOnRemoved?.apply(this, arguments);
            };

            if (!isConfiguringGraph()) {
                this.setSize?.([...MARKDOWN_NOTE_DEFAULT_SIZE]);
            }
            this.setDirtyCanvas?.(true, true);
            return result;
        };
    },
});

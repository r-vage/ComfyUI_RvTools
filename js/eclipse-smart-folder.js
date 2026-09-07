import {
    app
} from './comfy/index.js';
import {
    debounce,
    smartResize,
    createWidgetVisibilityManager,
    isConfiguringGraph,
    isVueMode,
    onVueModeChange,
    notifyVue,
} from './eclipse-widget-performance-utils.js';
import {
    injectComboChipCSS,
    createComboChipWidget as _createComboChipWidget
} from './eclipse-combo-chip.js';
import { migrateSmartFolderWorkflow } from './eclipse-smart-folder-workflow-migration.js';
import { storeQueuedSeed, enterGraphToPromptHook, exitGraphToPromptHook, getGraphNodeList, clearNodeQueuedSeed, findWorkflowNode } from './eclipse-seed-utils.js';
const NODE_NAME = 'Smart Folder [Eclipse]';
const SPECIAL_SEEDS = [-1, -2, -3];
const LAST_SEED_BUTTON_LABEL = '🌘 (Use Last Queued Seed)';
const FEATURE_OPTIONS = [
    { label: 'image', tooltip: 'Image generation mode (radio with video)' },
    { label: 'video', tooltip: 'Video generation mode (radio with image)' },
    { label: 'date_time', tooltip: 'Append a date/time subfolder to the output path' },
    { label: 'batch', tooltip: 'Append a batch subfolder for grouped runs' },
    { label: 'image_size', tooltip: 'Include resolution settings in Image and Video modes' },
    { label: 'vhs', tooltip: 'Include VHS frame loading, skipping, and selection settings' },
    { label: 'loop', tooltip: 'Include loop count and overlap (requires Context)' },
    { label: 'context', tooltip: 'Include model context length' },
    { label: 'duration', tooltip: 'Include optional video duration metadata' },
    { label: 'seed', tooltip: 'Show the seed widgets' },
];
const DEFAULT_FEATURES = ['image', 'date_time', 'context'];
const RADIO_GROUPS = [
    ['image', 'video']
];
const FEATURE_LABELS = new Set(FEATURE_OPTIONS.map((option) => option.label));
const BACKING_WIDGETS = [
    'generation_mode', 'create_date_time_folder', 'create_batch_folder',
    'use_image_size', 'use_vhs', 'use_loop', 'use_context', 'use_duration', 'use_seed',
];
injectComboChipCSS('sf2');

function syncChipsToBacking(selectedSet, node) {
    const setW = (name, val) => {
        const w = node.widgets?.find((w) => w.name === name);
        if (w && w.value !== val) w.value = val;
    };
    setW('generation_mode', selectedSet.has('video') ? 'Video Mode' : 'Image Mode');
    setW('create_date_time_folder', selectedSet.has('date_time'));
    setW('create_batch_folder', selectedSet.has('batch'));
    setW('use_image_size', selectedSet.has('image_size'));
    setW('use_vhs', selectedSet.has('vhs'));
    setW('use_loop', selectedSet.has('loop'));
    setW('use_context', selectedSet.has('context'));
    setW('use_duration', selectedSet.has('duration'));
    setW('use_seed', selectedSet.has('seed'));
}

function readChipsFromBacking(node) {
    const gv = (name) => {
        const w = node.widgets?.find((w) => w.name === name);
        return w ? w.value : undefined;
    };
    const chips = new Set();
    if (gv('generation_mode') === 'Video Mode') chips.add('video');
    else chips.add('image');
    if (gv('create_date_time_folder')) chips.add('date_time');
    if (gv('create_batch_folder')) chips.add('batch');
    if (gv('use_image_size')) chips.add('image_size');
    if (gv('use_vhs')) chips.add('vhs');
    if (gv('use_loop')) chips.add('loop');
    if (gv('use_context')) chips.add('context');
    if (gv('use_duration')) chips.add('duration');
    if (gv('use_seed')) chips.add('seed');
    return chips;
}

function normalizeFeatureDependencies(selectedSet) {
    const normalized = new Set();
    for (const feature of selectedSet) {
        normalized.add(feature);
        if (feature === 'loop' && !selectedSet.has('context')) normalized.add('context');
    }
    return normalized;
}

function readConfiguredFeatures(value) {
    const values = Array.isArray(value)
        ? value
        : (typeof value === 'string' ? value.split(',') : []);
    return new Set(values.map((feature) => feature.trim()).filter(
        (feature) => FEATURE_LABELS.has(feature)
    ));
}

function normalizedDivisor(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return 8;
    return Math.min(512, Math.max(1, Math.round(parsed)));
}

function alignDimensionValue(value, divisor) {
    const numeric = Number(value);
    const safeValue = Number.isFinite(numeric) ? numeric : divisor;
    const minimum = Math.ceil(16 / divisor) * divisor;
    const maximum = Math.floor(32768 / divisor) * divisor;
    return Math.min(maximum, Math.max(minimum, Math.round(safeValue / divisor) * divisor));
}

function setDimensionStep(widget, divisor) {
    if (!widget) return;
    widget.options ??= {};
    widget.options.step = divisor * 10;
    widget.options.step2 = divisor;
    if (widget._state?.options && widget._state.options !== widget.options) {
        try {
            widget._state.options.step = divisor * 10;
            widget._state.options.step2 = divisor;
        } catch (_) {}
    }
}

function alignActiveCustomDimensions(node, selected) {
    const divisorWidget = node.widgets?.find((w) => w.name === 'divisible_by');
    const divisor = normalizedDivisor(divisorWidget?.value);
    if (divisorWidget && divisorWidget.value !== divisor) divisorWidget.value = divisor;
    for (const name of ['width', 'height', 'video_width', 'video_height']) {
        setDimensionStep(node.widgets?.find((w) => w.name === name), divisor);
    }
    if (!selected.has('image_size')) {
        if (isVueMode()) notifyVue(node);
        return;
    }
    const names = selected.has('video')
        ? (node.widgets?.find((w) => w.name === 'video_size')?.value === 'Custom'
            ? ['video_width', 'video_height'] : [])
        : (node.widgets?.find((w) => w.name === 'image_size')?.value === 'Custom'
            ? ['width', 'height'] : []);
    for (const name of names) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (!widget) continue;
        widget.value = alignDimensionValue(widget.value, divisor);
    }
    if (isVueMode()) notifyVue(node);
}

function createComboChipWidget(node, initialSet, origIdx) {
    return _createComboChipWidget({
        node,
        options: FEATURE_OPTIONS,
        savedValue: initialSet,
        origIdx,
        widgetName: '_sf_features',
        cssPrefix: 'sf2',
        radioGroups: RADIO_GROUPS,
        serialize: false,
    });
}

function updateVisibility(node, vis) {
    if (node.id === -1) return;
    const featW = node.widgets?.find((w) => w.name === '_sf_features');
    const selected = normalizeFeatureDependencies(
        featW ? new Set(featW.value) : readChipsFromBacking(node)
    );
    const isImage = selected.has('image');
    const isVideo = selected.has('video');
    const hasDateTime = selected.has('date_time');
    const hasBatch = selected.has('batch');
    const hasImageSize = selected.has('image_size');
    const hasVhs = selected.has('vhs');
    const hasLoop = selected.has('loop');
    const hasContext = selected.has('context');
    const hasDuration = selected.has('duration');
    const customImage = vis.getValue('image_size') === 'Custom';
    const customVideo = vis.getValue('video_size') === 'Custom';
    alignActiveCustomDimensions(node, selected);
    for (const name of BACKING_WIDGETS) vis.setVisible(name, false);
    vis.setVisible('date_time_format', hasDateTime);
    vis.setVisible('date_time_position', hasDateTime);
    vis.setVisible('batch_folder_name', hasBatch);
    vis.setVisible('batch_number', hasBatch);
    vis.setVisible('batch_number_control', hasBatch);
    vis.setVisible('root_folder_image', isImage);
    vis.setVisible('image_size', isImage && hasImageSize);
    vis.setVisible('width', isImage && hasImageSize && customImage);
    vis.setVisible('height', isImage && hasImageSize && customImage);
    vis.setVisible('latent_type', isImage && hasImageSize);
    vis.setVisible('batch_size', isImage);
    vis.setVisible('root_folder_video', isVideo);
    vis.setVisible('video_size', isVideo && hasImageSize);
    vis.setVisible('video_width', isVideo && hasImageSize && customVideo);
    vis.setVisible('video_height', isVideo && hasImageSize && customVideo);
    vis.setVisible('divisible_by', hasImageSize && ((isImage && customImage) || (isVideo && customVideo)));
    vis.setVisible('frame_rate', isVideo);
    vis.setVisible('frame_load_cap', isVideo && hasVhs);
    vis.setVisible('context_length', isVideo && hasContext);
    vis.setVisible('loop_count', isVideo && hasLoop);
    vis.setVisible('overlap', isVideo && hasLoop);
    vis.setVisible('skip_first_frames', isVideo && hasVhs);
    vis.setVisible('skip_calculation', isVideo && hasVhs && hasContext);
    vis.setVisible('skip_calculation_control', isVideo && hasVhs && hasContext);
    vis.setVisible('select_every_nth', isVideo && hasVhs);
    vis.setVisible('duration', isVideo && hasDuration);
    const hasSeed = selected.has('seed');
    vis.setVisible('seed', hasSeed);
    vis.setVisible('_btn_randomize', hasSeed);
    vis.setVisible('_btn_new_fixed', hasSeed);
    vis.setVisible('_btn_last_seed', hasSeed);
    smartResize(node);
}
app.registerExtension({
    name: 'Eclipse.SmartFolderV2',
    beforeConfigureGraph(graphData) {
        migrateSmartFolderWorkflow(graphData);
    },
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_NAME) return;
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : void 0;
            const node = this;
            const vis = createWidgetVisibilityManager(node);
            node._Eclipse_vis = vis;
            // Pre-hide widgets hidden at DEFAULT_FEATURES (image/date_time/context):
            // no batch, no image_size, no seed, no video.
            vis.hideInitially([
                ...BACKING_WIDGETS,
                'batch_folder_name', 'batch_number', 'batch_number_control',
                'image_size', 'width', 'height', 'latent_type',
                'root_folder_video', 'video_size', 'video_width', 'video_height',
                'divisible_by',
                'frame_rate', 'frame_load_cap', 'context_length', 'loop_count',
                'overlap', 'skip_first_frames', 'skip_calculation',
                'skip_calculation_control', 'select_every_nth', 'duration',
                'seed',
            ]);
            node._Eclipse_lastBatchNumber = null;
            node._Eclipse_lastSkipFirstFramesCalc = null;
            const initialSet = normalizeFeatureDependencies(readChipsFromBacking(node));
            const modeW = node.widgets?.find((w) => w.name === 'generation_mode');
            const origIdx = modeW ? node.widgets.indexOf(modeW) : 0;
            const featWidget = createComboChipWidget(node, initialSet, origIdx);
            const origFeatCb = featWidget.callback;
            featWidget.callback = function (value) {
                origFeatCb?.call(this, value);
                const currentFeatures = Array.isArray(featWidget.value) ? featWidget.value : [];
                const normalized = normalizeFeatureDependencies(new Set(currentFeatures));
                if (normalized.size !== currentFeatures.length) {
                    featWidget.value = [...normalized];
                }
                syncChipsToBacking(normalized, node);
                // Reset seed to stable value when seed chip is deselected
                const feats = Array.isArray(featWidget.value) ? featWidget.value : [];
                if (!feats.includes('seed') && node._Eclipse_seedWidget
                    && SPECIAL_SEEDS.includes(Number(node._Eclipse_seedWidget.value))) {
                    const fallback = (typeof node._Eclipse_lastSeed === 'number'
                        && !SPECIAL_SEEDS.includes(node._Eclipse_lastSeed))
                        ? node._Eclipse_lastSeed : 0;
                    node._Eclipse_seedWidget.value = fallback;
                }
                vis.markUserDriven();
                updateVisibility(node, vis);
            };
            for (let i = node.widgets.length - 1; i >= 0; i--) {
                const wName = (node.widgets[i].name || '').toString().toLowerCase();
                if (wName === 'control_after_generate') {
                    node.widgets.splice(i, 1);
                }
            }
            const seedWidget = node.widgets?.find((w) => w.name === 'seed');
            if (seedWidget) {
                node._Eclipse_seedWidget = seedWidget;
                node._Eclipse_lastSeed = undefined;
                node._Eclipse_randomMin = 0;
                node._Eclipse_randomMax = Number.MAX_SAFE_INTEGER;
                node._Eclipse_cachedInputSeed = null;
                node._Eclipse_cachedResolvedSeed = null;
                const origSeedCb = seedWidget.callback;
                seedWidget.callback = (v) => {
                    node._Eclipse_cachedInputSeed = null;
                    node._Eclipse_cachedResolvedSeed = null;
                    if (origSeedCb) origSeedCb.call(seedWidget, v);
                };
                const seedIdx = node.widgets.indexOf(seedWidget);
                const btnRandomize = node.addWidget('button', '_btn_randomize', '', () => {
                    seedWidget.value = -1;
                    seedWidget.callback && seedWidget.callback(-1);
                }, {
                    serialize: false
                });
                btnRandomize.label = '🌑 Randomize Each Time';
                const btnNewFixed = node.addWidget('button', '_btn_new_fixed', '', () => {
                    const s = node.generateRandomSeed();
                    seedWidget.value = s;
                    seedWidget.callback && seedWidget.callback(s);
                }, {
                    serialize: false
                });
                btnNewFixed.label = '🌕 New Fixed Random';
                const btnLastSeed = node.addWidget('button', '_btn_last_seed', '', () => {
                    if (node._Eclipse_lastSeed != null) {
                        seedWidget.value = node._Eclipse_lastSeed;
                        btnLastSeed.label = LAST_SEED_BUTTON_LABEL;
                        btnLastSeed.disabled = true;
                        if (isVueMode()) notifyVue(node);
                    }
                }, {
                    serialize: false
                });
                btnLastSeed.label = LAST_SEED_BUTTON_LABEL;
                btnLastSeed.disabled = true;
                node._Eclipse_lastSeedButton = btnLastSeed;
                const buttons = [btnRandomize, btnNewFixed, btnLastSeed];
                for (let i = buttons.length - 1; i >= 0; i--) {
                    const btn = buttons[i];
                    const idx = node.widgets.indexOf(btn);
                    if (idx !== seedIdx + 1) {
                        node.widgets.splice(idx, 1);
                        node.widgets.splice(seedIdx + 1, 0, btn);
                    }
                }
            }
            const debouncedUpdate = debounce(() => updateVisibility(node, vis), 100);
            for (const name of ['image_size', 'video_size']) {
                const w = node.widgets?.find((w) => w.name === name);
                if (w) {
                    const origCb = w.callback;
                    w.callback = function (v) {
                        vis.markUserDriven();
                        alignActiveCustomDimensions(
                            node,
                            normalizeFeatureDependencies(new Set(featWidget.value))
                        );
                        debouncedUpdate();
                        origCb?.call(this, v);
                    };
                }
            }
            const divisorWidget = node.widgets?.find((w) => w.name === 'divisible_by');
            if (divisorWidget) {
                const origDivisorCb = divisorWidget.callback;
                divisorWidget.callback = function (v) {
                    origDivisorCb?.call(divisorWidget, v);
                    divisorWidget.value = normalizedDivisor(divisorWidget.value);
                    alignActiveCustomDimensions(
                        node,
                        normalizeFeatureDependencies(new Set(featWidget.value))
                    );
                };
            }
            for (const name of ['width', 'height', 'video_width', 'video_height']) {
                const widget = node.widgets?.find((w) => w.name === name);
                if (!widget) continue;
                const origDimensionCb = widget.callback;
                widget.callback = function (v) {
                    const divisor = normalizedDivisor(divisorWidget?.value);
                    const aligned = alignDimensionValue(v, divisor);
                    origDimensionCb?.call(widget, aligned);
                    // ComfyUI's integer callback uses a min-offset step lattice;
                    // restore the required zero-based divisor lattice afterwards.
                    widget.value = aligned;
                    if (isVueMode()) notifyVue(node);
                };
            }
            syncChipsToBacking(initialSet, node);
            if (!node._Eclipse_initialized && !isConfiguringGraph()) {
                node._Eclipse_initialized = true;
                requestAnimationFrame(() => {
                    updateVisibility(node, vis);
                    const oldHeight = node.size?.[1];
                    if (!node.size || oldHeight === undefined) return;
                    node.size[1] = 0;
                    const computed = node.computeSize?.();
                    if (computed?.[1] !== oldHeight) node.setSize?.([node.size[0], computed[1]]);
                    else node.size[1] = oldHeight;
                });
            }
            const origConfigure = node.onConfigure;
            node.onConfigure = function (data) {
                origConfigure?.apply(this, arguments);
                node._Eclipse_initialized = true;
                vis.clearCache?.();
                const configuredFeatures = readConfiguredFeatures(featWidget.value);
                const chips = normalizeFeatureDependencies(
                    configuredFeatures.size ? configuredFeatures : readChipsFromBacking(node)
                );
                featWidget.value = [...chips];
                syncChipsToBacking(chips, node);
                updateVisibility(node, vis);
            };
            return ret;
        };
        nodeType.prototype.generateRandomSeed = function () {
            const step = this._Eclipse_seedWidget?.options?.step || 1;
            const min = this._Eclipse_randomMin || 0;
            const range = ((this._Eclipse_randomMax || 0xFFFFFFFF) - min) / (step / 10);
            let seed = Math.floor(Math.random() * range) * (step / 10) + min;
            if (SPECIAL_SEEDS.includes(seed)) seed = 0;
            return seed;
        };
        nodeType.prototype.getSeedToUse = function () {
            const input = Number(this._Eclipse_seedWidget.value);
            if (this._Eclipse_cachedInputSeed === input && this._Eclipse_cachedResolvedSeed != null)
                return this._Eclipse_cachedResolvedSeed;
            let resolved = null;
            if (SPECIAL_SEEDS.includes(input)) {
                if (typeof this._Eclipse_lastSeed === 'number' && !SPECIAL_SEEDS.includes(this._Eclipse_lastSeed)) {
                    if (input === -2) resolved = this._Eclipse_lastSeed + 1;
                    else if (input === -3) resolved = this._Eclipse_lastSeed - 1;
                }
                if (resolved == null || SPECIAL_SEEDS.includes(resolved))
                    resolved = this.generateRandomSeed();
            }
            const final = resolved != null ? resolved : input;
            this._Eclipse_cachedInputSeed = input;
            this._Eclipse_cachedResolvedSeed = final;
            return final;
        };
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (data) {
            const ret = origOnExecuted ? origOnExecuted.apply(this, arguments) : void 0;
            if (data && data.seed !== undefined) {
                this._Eclipse_lastSeed = data.seed;
            }
            return ret;
        };
    },
    async setup() {
        onVueModeChange(() => {
            app.graph?.setDirtyCanvas?.(true, true);
        });
        const origGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function () {
            // Shared node list across all chained hooks — one graph walk per queue call
            const nodeFilter = n => n.type === NODE_NAME && n._Eclipse_seedWidget;
            enterGraphToPromptHook();
            try {
                for (const { node } of getGraphNodeList(app.graph)) {
                    if (nodeFilter(node)) clearNodeQueuedSeed(node);
                }
                const result = await origGraphToPrompt.apply(this, arguments);
                // Use broad type filter so batch_number/skip_calc logic also fires for subgraph nodes
                for (const { node, outputKey } of getGraphNodeList(app.graph)) {
                    if (node.type !== NODE_NAME) continue;
                    if (node.mode === 2 || node.mode === 4) continue;
                    if (!result.output?.[outputKey]) continue;
                    const inputs = result.output[outputKey].inputs;
                    const batchW = node.widgets?.find((w) => w.name === 'batch_number');
                    const batchCtrl = node.widgets?.find((w) => w.name === 'batch_number_control');
                    if (batchW && batchCtrl && inputs) {
                        if (batchCtrl.value === 'increment') {
                            if (node._Eclipse_lastBatchNumber != null) {
                                const next = node._Eclipse_lastBatchNumber + 1;
                                inputs.batch_number = next;
                                node._Eclipse_lastBatchNumber = next;
                                if (Number(batchW.value) !== next) batchW.value = next;
                                const batchWfNode = findWorkflowNode(result.workflow, outputKey);
                                if (batchWfNode?.widgets_values) {
                                    const idx = node.widgets.indexOf(batchW);
                                    if (idx >= 0) batchWfNode.widgets_values[idx] = next;
                                }
                            } else {
                                node._Eclipse_lastBatchNumber = batchW.value;
                            }
                        } else {
                            node._Eclipse_lastBatchNumber = batchW.value;
                        }
                    }
                    const skipW = node.widgets?.find((w) => w.name === 'skip_calculation');
                    const skipCtrl = node.widgets?.find((w) => w.name === 'skip_calculation_control');
                    const modeW = node.widgets?.find((w) => w.name === 'generation_mode');
                    const vhsW = node.widgets?.find((w) => w.name === 'use_vhs');
                    const contextW = node.widgets?.find((w) => w.name === 'use_context');
                    const skipEnabled = modeW?.value === 'Video Mode'
                        && vhsW?.value === true
                        && contextW?.value === true;
                    if (skipEnabled && skipW && skipCtrl && inputs) {
                        if (skipCtrl.value === 'increment') {
                            if (node._Eclipse_lastSkipFirstFramesCalc != null) {
                                const next = node._Eclipse_lastSkipFirstFramesCalc + 1;
                                inputs.skip_calculation = next;
                                node._Eclipse_lastSkipFirstFramesCalc = next;
                                if (Number(skipW.value) !== next) skipW.value = next;
                                const skipWfNode = findWorkflowNode(result.workflow, outputKey);
                                if (skipWfNode?.widgets_values) {
                                    const idx = node.widgets.indexOf(skipW);
                                    if (idx >= 0) skipWfNode.widgets_values[idx] = next;
                                }
                            } else {
                                node._Eclipse_lastSkipFirstFramesCalc = skipW.value;
                            }
                        } else {
                            node._Eclipse_lastSkipFirstFramesCalc = skipW.value;
                        }
                    } else {
                        node._Eclipse_lastSkipFirstFramesCalc = null;
                    }
                    if (node._Eclipse_seedWidget) {
                        const resolved = node.getSeedToUse();
                        storeQueuedSeed(node, resolved);
                        if (inputs?.seed !== undefined) {
                            const current = inputs.seed;
                            if (Number(current) !== Number(resolved))
                                inputs.seed = resolved;
                        }
                        if (Number(node._Eclipse_lastSeed) !== Number(resolved)) {
                            node._Eclipse_lastSeed = resolved;
                        }
                        node._Eclipse_cachedInputSeed = null;
                        node._Eclipse_cachedResolvedSeed = null;
                        if (node._Eclipse_lastSeedButton) {
                            const seedVal = node._Eclipse_seedWidget.value;
                            if (SPECIAL_SEEDS.includes(seedVal)) {
                                node._Eclipse_lastSeedButton.label = `🌘 ${resolved}`;
                                node._Eclipse_lastSeedButton.disabled = false;
                            } else {
                                node._Eclipse_lastSeedButton.label = LAST_SEED_BUTTON_LABEL;
                                node._Eclipse_lastSeedButton.disabled = true;
                            }
                            if (isVueMode()) notifyVue(node);
                        }
                        if (result.workflow) {
                            const wfNode = findWorkflowNode(result.workflow, outputKey);
                            if (wfNode?.widgets_values) {
                                const idx = node.widgets.indexOf(node._Eclipse_seedWidget);
                                if (idx >= 0 && wfNode.widgets_values[idx] !== resolved)
                                    wfNode.widgets_values[idx] = resolved;
                            }
                        }
                    }
                }
                return result;
            } finally {
                exitGraphToPromptHook();
            }
        };
    },
});

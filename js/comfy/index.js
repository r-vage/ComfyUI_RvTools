/**
 * ComfyUI Import Hub - Eclipse Edition
 * 
 * This file centralizes all imports from ComfyUI core.
 * When ComfyUI updates their public API, we only need to update this file.
 * 
 * Import Status Legend:
 * ✅ STABLE - Part of ComfyUI's stable API, safe to use
 * ⚠️  INTERNAL - Internal module, may break in future updates
 * 🔶 DEPRECATED - Will be removed in future version (v1.34+)
 * 
 * Last Updated: November 6, 2025
 * Target ComfyUI Version: v1.33+
 */

// ============================================================================
// STABLE API - Safe to use, part of public API
// ============================================================================

/**
 * ✅ Main ComfyUI application object
 * @see https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking
 */
export { app } from '../../../scripts/app.js';

/**
 * ✅ ComfyUI API client for server communication
 * @see https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking
 */
export { api } from '../../../scripts/api.js';


// ============================================================================
// INTERNAL MODULES - Not part of public API, may break in future
// ============================================================================

/**
 * ⚠️  INTERNAL - ComfyUI widget utilities
 * WARNING: This is an internal module, not part of the public API
 * Future updates may break this import
 * 
 * Used for: Creating custom widgets (STRING, BOOLEAN, COMBO, etc.)
 * Alternative: Wait for official widget API or create custom widgets
 */
export { ComfyWidgets } from '../../../scripts/widgets.js';


// ============================================================================
// DEPRECATED - Find alternatives before v1.34
// ============================================================================

// No deprecated APIs currently in use!
// Previously exported $el but it was unused across all Eclipse files.
// Removed to maintain v1.34+ compatibility.


// ============================================================================
// OPTIONAL IMPORTS - Uncomment if needed
// ============================================================================

// /**
//  * ⚠️  INTERNAL - Utility functions
//  * WARNING: Internal module, not part of public API
//  */
// export * as utils from '../../../scripts/utils.js';

// /**
//  * 🔶 DEPRECATED - Will be removed in v1.34
//  * Use app.menu API or create custom components instead
//  */
// export { ComfyButtonGroup } from '../../../scripts/ui/components/buttonGroup.js';

// /**
//  * 🔶 DEPRECATED - Will be removed in v1.34
//  * Use native button elements or app.ui API
//  */
// export { ComfyButton } from '../../../scripts/ui/components/button.js';

// /**
//  * 🔶 DEPRECATED - Will be removed in v1.34
//  * Use native dialog elements or app.ui API
//  */
// export { ComfyPopup } from '../../../scripts/ui/components/popup.js';

// /**
//  * ⚠️  INTERNAL - PNG metadata handling
//  * WARNING: Internal module, not part of public API
//  */
// export * as pnginfo from '../../../scripts/pnginfo.js';

// /**
//  * ⚠️  INTERNAL - Clipspace functionality
//  * WARNING: Internal module, not part of public API
//  */
// export * as clipspace from '../../../extensions/core/clipspace.js';

// /**
//  * 🔶 DEPRECATED - Will be removed in v1.34
//  * Group node functionality
//  */
// export * as groupNode from '../../../extensions/core/groupNode.js';

// /**
//  * ⚠️  INTERNAL - Widget input conversion
//  * WARNING: Internal module, not part of public API
//  */
// export * as widgetInputs from '../../../extensions/core/widgetInputs.js';


// ============================================================================
// MIGRATION NOTES
// ============================================================================

/**
 * When ComfyUI releases their stable public API:
 * 
 * 1. Expected future import format (may change):
 *    import { app, api } from '@comfyui/api';
 * 
 * 2. Update this file to re-export from the new package:
 *    export { app, api } from '@comfyui/api';
 * 
 * 3. All extension files will automatically use the new API
 *    without needing individual updates!
 * 
 * References:
 * - ComfyUI Docs: https://docs.comfy.org/custom-nodes/js
 * - Migration Guide: https://docs.comfy.org/custom-nodes/js/context-menu-migration
 */

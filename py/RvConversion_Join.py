import re

import torch  # type: ignore
import torchaudio  # type: ignore
from comfy_api.latest import io  # type: ignore

from ..core import CATEGORY
from ..core.image_helpers import cat_and_fit_images, flatten_images, unwrap_value
from ..core.logger import log

# Inline pattern to avoid regex_patterns dependency
RE_NEWLINES = re.compile(r"[\r\n]+", re.IGNORECASE)

_LOG_PREFIX = "Join"


def _flatten_inputs(inputs):
    """Flatten ComfyUI list-wrapped inputs while preserving their order."""
    flattened = []

    def _append(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                _append(item)
        elif value is not None:
            flattened.append(value)

    for value in inputs:
        _append(value)
    return flattened


def _join_images(inputs):
    # Join IMAGE tensors into a batch, resizing to match first image dimensions
    images = flatten_images(inputs)
    return (cat_and_fit_images(images, _LOG_PREFIX),)


def _join_masks(inputs):
    # Join MASK tensors into a batch, resizing to match first mask dimensions
    tensors = []
    for mask in inputs:
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 4:
                mask = mask.squeeze(0)
            tensors.append(mask)

    if not tensors:
        return (None,)

    target_height = tensors[0].shape[1]
    target_width = tensors[0].shape[2]

    resized_tensors = []
    for tensor in tensors:
        if tensor.shape[1] != target_height or tensor.shape[2] != target_width:
            tensor_bchw = tensor.unsqueeze(1)
            resized = torch.nn.functional.interpolate(
                tensor_bchw,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            tensor = resized.squeeze(1)
        resized_tensors.append(tensor)

    result = torch.cat(resized_tensors, dim=0)
    return (result,)


def _is_audio(value) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("waveform"), torch.Tensor)
        and "sample_rate" in value
    )


def _audio_waveform(audio, index: int) -> tuple[torch.Tensor, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError(f"Audio input #{index} has an invalid sample rate.")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(
            f"Audio input #{index} must have [batch, channels, samples] shape."
        )
    return waveform, sample_rate


def _join_audio(inputs):
    # Append AUDIO clips in input order along their sample/time axis.
    clips = []
    for index, audio in enumerate(inputs, start=1):
        if not _is_audio(audio):
            raise ValueError(f"Join input #{index} is not a valid AUDIO value.")
        clips.append(_audio_waveform(audio, index))

    target_sample_rate = max(sample_rate for _, sample_rate in clips)
    target_batch = max(waveform.shape[0] for waveform, _ in clips)
    target_channels = max(waveform.shape[1] for waveform, _ in clips)
    reference = clips[0][0]
    prepared = []

    for index, (waveform, sample_rate) in enumerate(clips, start=1):
        batch, channels, _ = waveform.shape
        if batch not in (1, target_batch):
            raise ValueError(
                f"Audio input #{index} has {batch} batches; expected 1 or {target_batch}."
            )
        if channels not in (1, target_channels):
            raise ValueError(
                f"Audio input #{index} has {channels} channels; expected 1 or {target_channels}."
            )
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = waveform.to(device=reference.device, dtype=reference.dtype)
        if batch == 1 and target_batch > 1:
            waveform = waveform.expand(target_batch, -1, -1)
        if channels == 1 and target_channels > 1:
            waveform = waveform.expand(-1, target_channels, -1)
        prepared.append(waveform)

    return (
        {
            "waveform": torch.cat(prepared, dim=-1),
            "sample_rate": target_sample_rate,
        },
    )


def _join_strings(inputs, delimiter: str):
    # Join STRING values with delimiter
    if delimiter in ("\n", "\\n"):
        delimiter = "\n"

    text_inputs = []
    for v in inputs:
        if isinstance(v, str):
            v = v.strip()
            v = v.rstrip(".,;:!?")
            if v:
                text_inputs.append(v)

    if not text_inputs:
        return ("",)

    merged_text = delimiter.join(text_inputs)
    merged_text = RE_NEWLINES.sub(" ", merged_text)
    return (merged_text,)


def _join_primitives(inputs, delimiter: str):
    # Join INT/FLOAT/LIST values as comma-separated string
    if delimiter in ("\n", "\\n"):
        delimiter = "\n"

    text_inputs = []
    for v in inputs:
        if v is not None:
            text_inputs.append(str(v))

    if not text_inputs:
        return ("",)

    merged_text = delimiter.join(text_inputs)
    return (merged_text,)


class RvConversion_Join(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Join [Eclipse]",
            display_name="Join",
            category=CATEGORY.MAIN.value + CATEGORY.CONVERSION.value,
            is_input_list=True,
            inputs=[
                io.Int.Input(
                    "inputcount",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    socketless=True,
                    tooltip="Number of inputs to join. Only the first 'inputcount' input_X values will be used.",
                ),
                io.String.Input(
                    "delimiter",
                    default=", ",
                    optional=True,
                    tooltip="Delimiter for STRING types. Use \\n for newline. Ignored for IMAGE/MASK/AUDIO.",
                ),
                io.AnyType.Input("input_1", optional=True, tooltip="Input #1."),
                io.AnyType.Input("input_2", optional=True, tooltip="Input #2."),
            ],
            outputs=[
                io.AnyType.Output("output"),
            ],
        )

    @classmethod
    def execute(cls, inputcount: int, delimiter: str = ", ", **kwargs) -> io.NodeOutput:
        inputcount = int(unwrap_value(inputcount, 2))
        delimiter = str(unwrap_value(delimiter, ", "))
        inputs = []

        for i in range(1, min(inputcount, 64) + 1):
            key = f"input_{i}"
            v = kwargs.get(key)
            if v is not None:
                inputs.append(v)

        inputs = _flatten_inputs(inputs)

        if not inputs:
            return io.NodeOutput(None)

        first_input = inputs[0]

        if _is_audio(first_input):
            return io.NodeOutput(*_join_audio(inputs))
        if isinstance(first_input, torch.Tensor) and first_input.ndim == 4:
            return io.NodeOutput(*_join_images(inputs))
        if isinstance(first_input, torch.Tensor) and first_input.ndim in (2, 3):
            return io.NodeOutput(*_join_masks(inputs))
        if isinstance(first_input, str):
            return io.NodeOutput(*_join_strings(inputs, delimiter))
        if isinstance(first_input, (int, float, list, tuple)):
            return io.NodeOutput(*_join_primitives(inputs, delimiter))

        log.warning(
            _LOG_PREFIX, f"Unknown type: {type(first_input)}, returning first input"
        )
        return io.NodeOutput(first_input)

from comfy_api.latest import io  # type: ignore

from ..core import CATEGORY


class RvText_Multiline_List(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="String Multiline List [Eclipse]",
            display_name="String Multiline List",
            category=CATEGORY.MAIN.value + CATEGORY.TEXT.value,
            inputs=[
                io.String.Input(
                    "input_string",
                    optional=True,
                    force_input=True,
                    tooltip="Optional string to prepend to the full output and each list item.",
                ),
                io.String.Input(
                    "string",
                    multiline=True,
                    default="",
                    tooltip="Multiline input split into non-empty list items and joined into one full string.",
                ),
            ],
            outputs=[
                io.String.Output("string"),
                io.String.Output("string_list", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, string=None, input_string=None):
        # Outputs the input multiline string as a single joined string and as a list of lines.
        input_prefix = (
            input_string.strip()
            if isinstance(input_string, str) and input_string.strip()
            else ""
        )

        # Process multiline content
        content_lines = []
        if isinstance(string, str) and string.strip():
            content_lines = [
                line.strip() for line in string.strip().split("\n") if line.strip()
            ]

        joined_parts = ([input_prefix] if input_prefix else []) + content_lines

        # If no valid lines found, return empty
        if not joined_parts:
            return io.NodeOutput("", [""])

        joined_string = " ".join(joined_parts)
        if input_prefix and content_lines:
            list_items = [f"{input_prefix} {line}" for line in content_lines]
        else:
            list_items = content_lines or [input_prefix]

        return io.NodeOutput(joined_string, list_items)

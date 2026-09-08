from comfy_api.latest import io  # type: ignore

from ..core import CATEGORY


class RvText_MarkdownNote(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Markdown Note [Eclipse]",
            display_name="Markdown Note",
            category=CATEGORY.MAIN.value + CATEGORY.TEXT.value,
            description=(
                "Frontend-only Markdown annotation with a resizable, scrollable "
                "preview. Double-click to edit and leave the editor to preview."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    default="",
                    multiline=True,
                    socketless=True,
                    tooltip="Markdown annotation text stored with the workflow.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, text=""):
        # The frontend marks this as virtual, so prompt execution never reaches here.
        return io.NodeOutput()

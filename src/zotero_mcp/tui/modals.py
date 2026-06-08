"""Reusable modal screens for the Zotero TUI: a generic form and a result viewer."""

from __future__ import annotations

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Markdown, Select, Static


@dataclass
class Field:
    """A single field in a :class:`FormModal`."""

    key: str
    label: str
    placeholder: str = ""
    value: str = ""
    kind: str = "input"  # "input" | "select"
    options: list[tuple[str, str]] | None = None
    required: bool = False


class FormModal(ModalScreen[dict | None]):
    """A modal that collects a set of fields and returns them as a dict.

    Dismisses with the collected ``{key: value}`` dict on submit, or ``None``
    on cancel / escape.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, fields: list[Field], submit_label: str = "Submit") -> None:
        super().__init__()
        self._title = title
        self._fields = fields
        self._submit_label = submit_label

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(self._title, id="modal-title")
            for f in self._fields:
                yield Static(f.label + (" *" if f.required else ""), classes="modal-label")
                if f.kind == "select" and f.options:
                    yield Select(
                        f.options, value=f.value or Select.BLANK, id=f"field-{f.key}",
                        allow_blank=not f.required,
                    )
                else:
                    yield Input(
                        value=f.value, placeholder=f.placeholder, id=f"field-{f.key}"
                    )
            with Vertical(id="modal-buttons"):
                yield Button(self._submit_label, variant="primary", id="submit")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        # Focus the first input for immediate typing.
        for f in self._fields:
            if f.kind != "select":
                self.query_one(f"#field-{f.key}").focus()
                break

    def _collect(self) -> dict | None:
        values: dict[str, str] = {}
        for f in self._fields:
            widget = self.query_one(f"#field-{f.key}")
            if isinstance(widget, Select):
                val = widget.value
                val = "" if val is Select.BLANK else str(val)
            else:
                val = widget.value.strip()
            if f.required and not val:
                self.app.bell()
                self.query_one(f"#field-{f.key}").focus()
                return None
            values[f.key] = val
        return values

    @on(Button.Pressed, "#submit")
    def _submit(self) -> None:
        values = self._collect()
        if values is not None:
            self.dismiss(values)

    @on(Input.Submitted)
    def _input_submitted(self) -> None:
        values = self._collect()
        if values is not None:
            self.dismiss(values)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class ResultModal(ModalScreen[None]):
    """Scrollable markdown viewer for action output."""

    BINDINGS = [("escape", "close", "Close"), ("q", "close", "Close")]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(self._title, id="modal-title")
            with VerticalScroll(id="result-body"):
                yield Markdown(self._body)
            with Vertical(id="modal-buttons"):
                yield Button("Close", variant="primary", id="close")

    @on(Button.Pressed, "#close")
    def action_close(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """Yes/No confirmation modal."""

    BINDINGS = [("escape", "no", "No")]

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(self._title, id="modal-title")
            yield Static(self._message, classes="modal-label")
            with Vertical(id="modal-buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", id="no")

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def action_no(self) -> None:
        self.dismiss(False)

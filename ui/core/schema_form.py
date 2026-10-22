"""JSON Schema widgets with recursive, data-dependent object branches."""

import json
from copy import deepcopy

from jsonschema import validators
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


_MISSING = object()

from PySide6.QtWidgets import QDoubleSpinBox


class FlexibleDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setDecimals(15)
        self.setRange(-1e10, 1e10)

    def textFromValue(self, value):
        # 小数点以下4桁で表現できる場合
        if abs(value - round(value, 2)) < 1e-14:
            return f"{value:.2f}"

        # それ以外は必要な桁数を表示
        return f"{value:.15f}".rstrip("0").rstrip(".")

def merge_schema(left, right):
    """Combine presentation constraints; final validation uses the original schema."""
    result = deepcopy(left)
    for key, value in right.items():
        if key == "properties":
            props = result.setdefault(key, {})
            for name, prop in value.items():
                props[name] = merge_schema(props.get(name, {}), prop)
        elif key == "required":
            result[key] = list(dict.fromkeys(result.get(key, []) + value))
        elif key == "allOf":
            result[key] = result.get(key, []) + deepcopy(value)
        else:
            result[key] = deepcopy(value)
    return result


def active_schema(schema, data, validator):
    """Resolve branches at this instance location, including else-if and allOf."""
    if not isinstance(schema, dict):
        return {}
    result = {k: v for k, v in schema.items()
              if k not in ("if", "then", "else", "allOf")}
    for part in schema.get("allOf", []):
        result = merge_schema(result, active_schema(part, data, validator))
    if "if" in schema:
        branch = "then" if validator.evolve(schema=schema["if"]).is_valid(data) else "else"
        result = merge_schema(result, active_schema(schema.get(branch, {}), data, validator))
    return result


class JsonValueEdit(QLineEdit):
    """An unconstrained item is JSON, not implicitly a string."""

    def __init__(self, value=None):
        super().__init__()
        self.setPlaceholderText('JSON値 (例: 1.2, "文字列", null)')
        self.set_value(value)

    def set_value(self, value):
        self.setText(json.dumps(value, ensure_ascii=False, allow_nan=False))

    def get_data(self):
        try:
            return json.loads(self.text(), parse_constant=self._invalid_constant)
        except ValueError as exc:
            raise ValueError(f"JSON値が不正です: {self.text()}") from exc

    @staticmethod
    def _invalid_constant(value):
        raise ValueError(value)


class OptionalWidget(QWidget):
    """An unchecked optional field is absent from JSON, rather than null."""
    changed = Signal()

    def __init__(self, schema, validator=None):
        super().__init__()
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("この項目を出力する")
        layout.addWidget(self.enabled)
        inner = dict(schema)
        inner.pop("x-ui-optional", None)
        self.widget = SchemaForm.create_field(inner, validator)
        self.widget.setEnabled(False)
        layout.addWidget(self.widget)
        self.enabled.toggled.connect(self.widget.setEnabled)
        self.enabled.toggled.connect(lambda *_: self.changed.emit())
        SchemaForm.connect_changed(self.widget, self.changed.emit)

    def get_data(self):
        return SchemaForm.get_field_value(self.widget) if self.enabled.isChecked() else _MISSING

    def set_data(self, value):
        if value is _MISSING:
            self.enabled.setChecked(False)
            return
        SchemaForm.set_field_value(self.widget, value)
        self.enabled.setChecked(True)


class ChoiceWidget(QWidget):
    """Select a oneOf alternative without adding a discriminator to the JSON."""
    changed = Signal()

    def __init__(self, schema, validator=None):
        super().__init__()
        self.validator = validator or validators.validator_for(schema)(schema)
        base = {key: value for key, value in schema.items() if key not in ("oneOf", "default")}
        self.schemas = [merge_schema(base, alternative) for alternative in schema["oneOf"]]
        self.widgets = []
        layout = QVBoxLayout(self)
        self.selector = QComboBox()
        layout.addWidget(self.selector)
        for index, alternative in enumerate(self.schemas):
            self.selector.addItem(alternative.get("title", f"選択肢 {index + 1}"))
            widget = SchemaForm.create_field(alternative, self.validator)
            layout.addWidget(widget)
            widget.setVisible(index == 0)
            SchemaForm.connect_changed(widget, lambda w=widget: self._child_changed(w))
            self.widgets.append(widget)
        self.selector.currentIndexChanged.connect(self._select)

    @property
    def current_widget(self):
        return self.widgets[self.selector.currentIndex()]

    def _child_changed(self, widget):
        if widget is self.current_widget:
            self.changed.emit()

    def _select(self, index):
        for i, widget in enumerate(self.widgets):
            widget.setVisible(i == index)
        self.changed.emit()

    def get_data(self):
        return SchemaForm.get_field_value(self.current_widget)

    def set_data(self, value):
        matches = [i for i, schema in enumerate(self.schemas)
                   if self.validator.evolve(schema=schema).is_valid(value)]
        if len(matches) != 1:
            raise ValueError("oneOfの選択肢のいずれか1つに一致する値が必要です")
        index = matches[0]
        previous = self.blockSignals(True)
        try:
            SchemaForm.set_field_value(self.widgets[index], value)
            self.selector.setCurrentIndex(index)
        finally:
            self.blockSignals(previous)
        self.changed.emit()


class ArrayWidget(QWidget):
    changed = Signal()

    def __init__(self, schema, parent=None, validator=None):
        super().__init__(parent)
        self.schema = schema
        self.validator = validator
        self.item_widgets = []
        if schema.get("minItems", 0) > schema.get("maxItems", float("inf")):
            raise ValueError("minItemsがmaxItemsを超えています")
        layout = QVBoxLayout(self)
        self.items_layout = QVBoxLayout()
        layout.addLayout(self.items_layout)
        self.add_button = QPushButton("+ 追加")
        self.add_button.clicked.connect(lambda: self.add_item())
        layout.addWidget(self.add_button)
        self.set_data(schema.get("default", []))

    def add_item(self, value=_MISSING):
        limit = self.schema.get("maxItems")
        if limit is not None and len(self.item_widgets) >= limit:
            return
        row = QWidget()
        layout = QHBoxLayout(row)
        index = len(self.item_widgets)
        prefix = self.schema.get("prefixItems", [])
        item_schema = prefix[index] if index < len(prefix) else self.schema.get("items", {})
        widget = SchemaForm.create_field(item_schema, self.validator)
        layout.addWidget(widget, 1)
        button = QPushButton("削除")
        layout.addWidget(button)
        button.clicked.connect(lambda: self.remove_item(row, widget))
        self.items_layout.addWidget(row)
        self.item_widgets.append((row, widget))
        if value is not _MISSING:
            SchemaForm.set_field_value(widget, value)
        SchemaForm.connect_changed(widget, self.changed.emit)
        self.update_buttons()
        self.changed.emit()

    def remove_item(self, row, widget):
        if len(self.item_widgets) <= self.schema.get("minItems", 0):
            return
        # Recreate positional schemas after removal (prefixItems can differ).
        values = [SchemaForm.get_field_value(w) for r, w in self.item_widgets if r is not row]
        self.set_data(values)

    def update_buttons(self):
        limit = self.schema.get("maxItems")
        self.add_button.setEnabled(limit is None or len(self.item_widgets) < limit)
        for row, _ in self.item_widgets:
            row.findChild(QPushButton).setEnabled(len(self.item_widgets) > self.schema.get("minItems", 0))

    def get_data(self):
        return [SchemaForm.get_field_value(widget) for _, widget in self.item_widgets]

    def set_data(self, values):
        if not isinstance(values, list):
            raise ValueError("配列が必要です")
        limit = self.schema.get("maxItems")
        if limit is not None and len(values) > limit:
            raise ValueError(f"配列は {limit} 要素以下にしてください")
        previous = self.blockSignals(True)
        try:
            for row, _ in self.item_widgets:
                self.items_layout.removeWidget(row)
                row.setParent(None)
                row.deleteLater()
            self.item_widgets.clear()
            for value in values:
                self.add_item(value)
            while len(self.item_widgets) < self.schema.get("minItems", 0):
                self.add_item()
            self.update_buttons()
        finally:
            self.blockSignals(previous)
        self.changed.emit()


class SchemaForm(QWidget):
    changed = Signal()

    def __init__(self, schema, parent=None, validator=None):
        super().__init__(parent)
        if schema.get("type", "object") != "object":
            raise ValueError("object型が必要です")
        self.schema = schema
        self.validator = validator or validators.validator_for(schema)(schema)
        self.layout = QFormLayout(self)
        self.fields = {}
        self.rows = {}
        self._specs = {}
        self._cache = {}
        self._required = set()
        self._updating = False
        self._pending = {}
        self._condition_error = None
        self.update_conditions()
        if "default" in schema:
            self.set_data(schema["default"])

    @staticmethod
    def create_field(prop, validator=None):
        if prop is False:
            raise ValueError("falseスキーマの入力欄は作成できません")
        if prop is True:
            prop = {}
        default = prop.get("default", _MISSING)
        field_type = prop.get("type")
        if prop.get("x-ui-optional"):
            widget = OptionalWidget(prop, validator)
        elif "oneOf" in prop:
            widget = ChoiceWidget(prop, validator)
        elif "enum" in prop or "const" in prop:
            widget = QComboBox()
            for value in prop.get("enum", [prop.get("const")]):
                widget.addItem(json.dumps(value, ensure_ascii=False), value)
            widget.setEnabled("const" not in prop)
        elif field_type == "array":
            widget = ArrayWidget(prop, validator=validator)
        elif field_type == "object" or "properties" in prop:
            widget = QGroupBox()
            QVBoxLayout(widget).addWidget(SchemaForm(prop, validator=validator))
        elif field_type == "string":
            widget = QLineEdit()
        elif field_type == "integer" and prop.get("minimum", -2147483648) >= -2147483648 and prop.get("maximum", 2147483647) <= 2147483647:
            widget = QSpinBox()
            widget.setRange(prop.get("minimum", -2147483648), prop.get("maximum", 2147483647))
        elif field_type == "number":
            widget = FlexibleDoubleSpinBox()
        elif field_type == "boolean":
            widget = QCheckBox()
        else:
            widget = JsonValueEdit(None if default is _MISSING else default)
        if default is not _MISSING:
            SchemaForm.set_field_value(widget, default)
        if prop.get("description"):
            widget.setToolTip(prop["description"])
        return widget

    @staticmethod
    def connect_changed(widget, callback):
        # Qt's value signals carry arguments, but our changed signal does not.
        def notify(*args):
            callback()

        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(notify)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(notify)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(notify)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(notify)
        elif isinstance(widget, (ArrayWidget, ChoiceWidget, OptionalWidget)):
            widget.changed.connect(notify)
        elif isinstance(widget, QGroupBox):
            widget.findChild(SchemaForm).changed.connect(notify)

    @staticmethod
    def get_field_value(widget):
        if isinstance(widget, (ArrayWidget, JsonValueEdit, ChoiceWidget, OptionalWidget)):
            return widget.get_data()
        if isinstance(widget, QGroupBox):
            return widget.findChild(SchemaForm).get_data()
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        raise TypeError(type(widget))

    @staticmethod
    def set_field_value(widget, value):
        if isinstance(widget, (ChoiceWidget, OptionalWidget)):
            widget.set_data(value)
        elif isinstance(widget, JsonValueEdit):
            widget.set_value(value)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
        elif isinstance(widget, QComboBox):
            # JSON equality distinguishes booleans from numbers, unlike findData.
            encoded = json.dumps(value, sort_keys=True)
            for index in range(widget.count()):
                if json.dumps(widget.itemData(index), sort_keys=True) == encoded:
                    widget.setCurrentIndex(index)
                    break
            else:
                raise ValueError(f"選択肢にない値です: {value}")
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(value)
        elif isinstance(widget, QGroupBox):
            widget.findChild(SchemaForm).set_data(value)
        elif isinstance(widget, ArrayWidget):
            widget.set_data(value)

    def on_field_changed(self, *args):
        if self._updating:
            return
        try:
            self.update_conditions()
            self._condition_error = None
        except ValueError as exc:
            # A partially typed JSON value must not escape a Qt signal handler.
            self._condition_error = str(exc)
        self.changed.emit()

    def get_raw_data(self, skip=()):
        result = {}
        for key, widget in self.fields.items():
            if key in skip:
                continue
            value = self.get_field_value(widget)
            if value is _MISSING:
                continue
            if isinstance(widget, QLineEdit) and not isinstance(widget, JsonValueEdit) and value == "" and key not in self._required:
                continue
            result[key] = value
        return result

    def update_conditions(self):
        if self._updating:
            return
        self._updating = True
        try:
            seen = set()
            while True:
                data = self.get_raw_data(skip=self._pending)
                data.update(self._pending)
                effective = active_schema(self.schema, data, self.validator)
                specs = {k: v for k, v in effective.get("properties", {}).items() if v is not False}
                required = set(effective.get("required", []))
                signature = json.dumps([specs, sorted(required)], sort_keys=True)
                if specs == self._specs and required == self._required:
                    break
                if signature in seen:
                    raise ValueError("条件分岐が安定しません。ifの条件と既定値を確認してください")
                seen.add(signature)
                # Keep widgets by full property schema so branch-specific values survive.
                for key, row in self.rows.items():
                    self.layout.takeRow(row)
                    row.hide()
                self.fields, self.rows = {}, {}
                self._required = required
                for key, prop in specs.items():
                    cache_key = (key, json.dumps(prop, sort_keys=True))
                    if cache_key not in self._cache:
                        widget = self.create_field(prop, self.validator)
                        row = QWidget(self)
                        layout = QVBoxLayout(row)
                        layout.setContentsMargins(0, 0, 0, 0)
                        label = QLabel(prop.get("title", key) if isinstance(prop, dict) else key)
                        layout.addWidget(label)
                        layout.addWidget(widget)
                        self.connect_changed(widget, self.on_field_changed)
                        self._cache[cache_key] = (row, widget)
                    row, widget = self._cache[cache_key]
                    if key in self.schema.get("properties", {}) and key in data:
                        if self.validator.evolve(schema=prop).is_valid(data[key]):
                            self.set_field_value(widget, data[key])
                    self.fields[key], self.rows[key] = widget, row
                    if key in self._pending:
                        self.set_field_value(widget, self._pending[key])
                    self.layout.addRow(row)
                    row.show()
                self._specs = deepcopy(specs)
        finally:
            self._updating = False

    def get_data(self):
        self.update_conditions()
        return self.get_raw_data()

    def set_data(self, data):
        if not isinstance(data, dict):
            raise ValueError("object型が必要です")
        self._pending = dict(data)
        try:
            # Resolve the incoming discriminators before applying dependent
            # values: old array limits/types may not accept the new branch.
            self.update_conditions()
            self._updating = True
            try:
                for key, widget in self.fields.items():
                    if key in data:
                        self.set_field_value(widget, data[key])
            finally:
                self._updating = False
            self.update_conditions()
        finally:
            self._pending = {}
        self.changed.emit()

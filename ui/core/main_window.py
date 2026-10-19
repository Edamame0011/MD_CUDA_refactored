import json
from pathlib import Path

from jsonschema import validators
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .schema_form import SchemaForm
from .schema_loader import prepare_schema


DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "configs" / "schema.json"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JSON Schema Editor")
        self.resize(720, 600)
        self.form = None
        central = QWidget()
        self.setCentralWidget(central)
        self.layout = QVBoxLayout(central)
        buttons = QHBoxLayout()
        self.schema_path_edit = QLineEdit(str(DEFAULT_SCHEMA))
        self.schema_path_edit.setPlaceholderText("JSON Schemaのパスを入力")
        self.schema_path_edit.returnPressed.connect(
            lambda: self.load_schema(self.schema_path_edit.text())
        )
        save_button = QPushButton("JSONを保存")
        save_button.clicked.connect(self.save_json)
        buttons.addWidget(self.schema_path_edit)
        buttons.addWidget(save_button)
        self.layout.addLayout(buttons)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.layout.addWidget(self.scroll)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: red;")
        self.error_label.hide()
        self.layout.addWidget(self.error_label)
        self.load_schema(str(DEFAULT_SCHEMA))

    def load_schema(self, schema_path):
        self.error_label.clear()
        self.error_label.hide()
        try:
            path = Path(schema_path).expanduser()
            schema, repairs = prepare_schema(json.loads(path.read_text(encoding="utf-8")))
            validator_class = validators.validator_for(schema)
            validator_class.check_schema(schema)
            form = SchemaForm(schema)
        except Exception as exc:
            self.show_error(f"Schemaの読み込みに失敗しました: {exc}")
            return
        self.schema = schema
        self.form = form
        self.scroll.setWidget(form)
        if repairs:
            self.show_error("Schemaのrequiredの配置を読み込み時に補正しました（元ファイルは変更しません）: " + ", ".join(repairs))

    def save_json(self):
        if self.form is None:
            QMessageBox.warning(self, "警告", "Schemaを読み込んでください")
            return
        try:
            data = self.form.get_data()
            validator = validators.validator_for(self.schema)(self.schema)
            errors = sorted(validator.iter_errors(data), key=lambda e: tuple(str(p) for p in e.absolute_path))
            if errors:
                messages = []
                for error in errors:
                    path = ".".join(str(p) for p in error.absolute_path) or "(root)"
                    messages.append(f"{path}: {error.message}")
                QMessageBox.warning(self, "入力エラー", "\n".join(messages))
                return
            serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "入力エラー", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "JSONを保存", "data.json", "JSON Files (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(serialized, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "保存エラー", str(exc))
            return
        QMessageBox.information(self, "保存完了", "JSONファイルを保存しました")

    def show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()

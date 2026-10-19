"""Run from the repository root: QT_QPA_PLATFORM=offscreen python -m unittest discover -s ui/tests -v"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
from jsonschema import validators
from ui.core.schema_form import ArrayWidget, ChoiceWidget, OptionalWidget, SchemaForm
from ui.core.schema_loader import prepare_schema
from ui.core.main_window import DEFAULT_SCHEMA, MainWindow

APP = QApplication.instance() or QApplication([])


def child(form, key):
    return form.fields[key].findChild(SchemaForm)


def multiway_schema(modes):
    return {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": modes}},
        "required": ["mode"],
        "allOf": [
            {
                "if": {"properties": {"mode": {"const": mode}}, "required": ["mode"]},
                "then": {
                    "properties": {mode + "_value": {"type": "integer", "default": index + 1}},
                    "required": [mode + "_value"],
                },
            }
            for index, mode in enumerate(modes)
        ],
    }


class SchemaFormTests(unittest.TestCase):
    def setUp(self):
        self.forms = []

    def form(self, schema):
        form = SchemaForm(schema)
        self.forms.append(form)
        return form

    def tearDown(self):
        for form in self.forms:
            form.close()
            form.deleteLater()
        APP.processEvents()

    def test_shipped_nested_branch_and_array_types(self):
        schema, repairs = prepare_schema(json.loads(DEFAULT_SCHEMA.read_text()))
        self.assertEqual(repairs, [])
        validators.validator_for(schema).check_schema(schema)
        form = self.form(schema)
        initial = form.get_data()
        self.assertEqual(initial["common_settings"]["atoms"]["ratio"], [0.8, 0.2])
        self.assertIs(type(initial["common_settings"]["interactions"]["potentials"]["numbers"][0]), int)
        steps = form.fields["steps"]
        steps.add_item()
        step = steps.item_widgets[0][1].findChild(SchemaForm)
        ensemble = child(child(step, "simulation"), "ensemble")
        self.assertEqual(ensemble.get_data(), {"type": "NVE", "temperature": 2.0})
        ensemble.fields["type"].setCurrentIndex(1)
        data = ensemble.get_data()
        self.assertEqual(data["thermostat"], "Bussi")
        self.assertEqual(data["tau"], 0.02)
        ensemble.fields["tau"].setValue(0.123)
        ensemble.fields["thermostat"].setCurrentIndex(1)
        self.assertNotIn("seed", ensemble.get_data())
        self.assertEqual(ensemble.get_data()["tau"], 1.0)
        ensemble.fields["thermostat"].setCurrentIndex(0)
        self.assertEqual(ensemble.get_data()["tau"], 0.123)
        ensemble.fields["type"].setCurrentIndex(0)
        self.assertNotIn("thermostat", ensemble.get_data())
        self.assertTrue(validators.validator_for(schema)(schema).is_valid(form.get_data()))

    def test_completed_schema_step_and_fire_parameter_choices(self):
        schema = json.loads(DEFAULT_SCHEMA.read_text())
        validator = validators.validator_for(schema)(schema)
        form = self.form(schema)
        steps = form.fields["steps"]
        choice = steps.item_widgets[0][1]
        self.assertIsInstance(choice, ChoiceWidget)
        validator.validate(form.get_data())
        md_step = choice.current_widget.findChild(SchemaForm)
        optional = md_step.fields["save_last_structure"]
        self.assertIsInstance(optional, OptionalWidget)
        self.assertNotIn("save_last_structure", md_step.get_data())
        optional.enabled.setChecked(True)
        self.assertEqual(md_step.get_data()["save_last_structure"], {"path": "last_structure.xyz", "is_unwrap": False})
        validator.validate(form.get_data())
        choice.selector.setCurrentIndex(1)
        minimum_step = choice.current_widget.findChild(SchemaForm)
        self.assertNotIn("simulation", minimum_step.get_data())
        minimizer = child(minimum_step, "minimize")
        params = minimizer.fields["params"]
        self.assertEqual(minimizer.get_data()["params"], "default")
        validator.validate(form.get_data())
        params.selector.setCurrentIndex(1)
        custom = params.current_widget.findChild(SchemaForm)
        self.assertEqual(len(custom.fields), 11)
        custom.fields["n_max"].setValue(123)
        self.assertEqual(minimizer.get_data()["params"]["n_max"], 123)
        validator.validate(form.get_data())
        params.selector.setCurrentIndex(0)
        self.assertEqual(minimizer.get_data()["params"], "default")
        params.selector.setCurrentIndex(1)
        self.assertEqual(minimizer.get_data()["params"]["n_max"], 123)
        choice.selector.setCurrentIndex(0)
        self.assertIn("simulation", choice.get_data())
        self.assertNotIn("minimize", choice.get_data())
        self.assertIn("save_last_structure", choice.get_data())
        optional.enabled.setChecked(False)
        self.assertNotIn("save_last_structure", choice.get_data())

    def test_completed_schema_observers_thermostats_and_scheduler(self):
        schema = json.loads(DEFAULT_SCHEMA.read_text())
        validator = validators.validator_for(schema)(schema)
        form = self.form(schema)
        choice = form.fields["steps"].item_widgets[0][1]
        step = choice.current_widget.findChild(SchemaForm)
        observer = child(step, "observer")
        expected = [
            {"type", "interval", "output_path"},
            {"type", "divisions", "output_path"},
            {"type", "interval", "is_unwrap", "output_path"},
            {"type", "divisions", "is_unwrap", "output_path", "temp_path"},
        ]
        for index, keys in enumerate(expected):
            observer.fields["type"].setCurrentIndex(index)
            self.assertEqual(set(observer.get_data()), keys)
            validator.validate(form.get_data())
        ensemble = child(child(step, "simulation"), "ensemble")
        ensemble.fields["type"].setCurrentIndex(1)
        for index in range(3):
            ensemble.fields["thermostat"].setCurrentIndex(index)
            self.assertIn("tau", ensemble.get_data())
            self.assertEqual("seed" in ensemble.get_data(), index != 1)
            ensemble.fields["scheduler"].setCurrentIndex(1)
            self.assertIn("rate_per_unit_time", ensemble.get_data())
            validator.validate(form.get_data())
            ensemble.fields["scheduler"].setCurrentIndex(0)
            self.assertNotIn("rate_per_unit_time", ensemble.get_data())

    def test_completed_schema_roundtrips_runner_examples_without_losing_keys(self):
        # Unknown legacy keys (cell/sort/cell_type) are not form fields, but all
        # values consumed by SimulationRunner must survive assigning a workflow.
        schema = json.loads(DEFAULT_SCHEMA.read_text())
        validator = validators.validator_for(schema)(schema)
        for name in ("example_workflow_Fire.json", "example_workflow_NVE_NNP_traj.json", "example_workflow_NVE_no_graph.json"):
            with self.subTest(name=name):
                data = json.loads((DEFAULT_SCHEMA.parent / name).read_text())
                form = self.form(schema)
                form.set_data(data)
                result = form.get_data()
                validator.validate(result)
                self.assertEqual(len(result["steps"]), len(data["steps"]))
                self.assertEqual(result["common_settings"]["atoms"]["symbol"], data["common_settings"]["atoms"]["symbol"])
                self.assertEqual(result["common_settings"]["interactions"]["potentials"], data["common_settings"]["interactions"]["potentials"])
                for before, after in zip(data["steps"], result["steps"]):
                    for kind in ("simulation", "minimize"):
                        if kind in before:
                            expected = {k: v for k, v in before[kind].items() if k != "cell_type"}
                            for key, value in expected.items():
                                self.assertEqual(after[kind][key], value)
                    self.assertEqual("save_last_structure" in after, "save_last_structure" in before)

    def test_deep_else_if_and_set_data(self):
        schema = {"type": "object", "properties": {"mode": {"enum": ["a", "b", "c"]}},
                  "if": {"properties": {"mode": {"const": "a"}}},
                  "then": {"properties": {"a": {"type": "integer", "default": 1}}},
                  "else": {"if": {"properties": {"mode": {"const": "b"}}},
                           "then": {"properties": {"b": {"type": "integer", "default": 2}}},
                           "else": {"properties": {"c": {"type": "integer", "default": 3}}}}}
        form = self.form(schema)
        for mode in "abc":
            form.set_data({"mode": mode, mode: 9})
            self.assertEqual(form.get_data(), {"mode": mode, mode: 9})

    def test_allof_duplicate_property_and_required_empty(self):
        form = self.form({"type": "object", "properties": {"mode": {"type": "boolean"}, "value": {"type": "string"}},
                          "allOf": [{"if": {"properties": {"mode": {"const": True}}},
                                     "then": {"required": ["value"], "properties": {"value": {"title": "Required"}}}}]})
        self.assertNotIn("value", form.get_data())
        form.fields["mode"].setChecked(True)
        self.assertEqual(form.get_data()["value"], "")
        self.assertEqual(len(form.fields), 2)
        self.assertEqual(form.layout.rowCount(), 2)
        form.fields["value"].setText("preserved")
        form.fields["mode"].setChecked(False)
        self.assertEqual(form.get_data()["value"], "preserved")
        form.fields["mode"].setChecked(True)
        self.assertEqual(form.get_data()["value"], "preserved")

    def test_allof_three_and_five_way_switching_and_value_restoration(self):
        for modes in (["a", "b", "c"], ["a", "b", "c", "d", "e"]):
            with self.subTest(modes=modes):
                schema = multiway_schema(modes)
                validator = validators.validator_for(schema)(schema)
                form = self.form(schema)
                for index, mode in enumerate(modes):
                    form.fields["mode"].setCurrentIndex(index)
                    key = mode + "_value"
                    self.assertEqual(set(form.fields), {"mode", key})
                    self.assertEqual(form._required, {"mode", key})
                    self.assertEqual(form.layout.rowCount(), 2)
                    self.assertFalse(form.rows[key].isHidden())
                    self.assertEqual(form.get_data(), {"mode": mode, key: index + 1})
                    form.fields[key].setValue(100 + index)
                    validator.validate(form.get_data())
                for index in reversed(range(len(modes))):
                    mode = modes[index]
                    form.fields["mode"].setCurrentIndex(index)
                    self.assertEqual(form.get_data(), {"mode": mode, mode + "_value": 100 + index})
                    validator.validate(form.get_data())

    def test_allof_multiway_in_array_and_set_data(self):
        modes = ["a", "b", "c"]
        schema = {"type": "object", "properties": {
            "steps": {"type": "array", "items": multiway_schema(modes)},
        }}
        form = self.form(schema)
        data = {"steps": [{"mode": mode, mode + "_value": index + 10}
                          for index, mode in enumerate(modes)]}
        form.set_data(data)
        self.assertEqual(form.get_data(), data)
        steps = form.fields["steps"]
        first = steps.item_widgets[0][1].findChild(SchemaForm)
        second = steps.item_widgets[1][1].findChild(SchemaForm)
        first.fields["mode"].setCurrentIndex(2)
        self.assertEqual(first.get_data(), {"mode": "c", "c_value": 3})
        self.assertEqual(second.get_data(), data["steps"][1])
        validators.validator_for(schema)(schema).validate(form.get_data())

    def test_allof_applies_all_matching_branches(self):
        schema = multiway_schema(["a", "b", "c"])
        schema["allOf"].append({
            "if": {"properties": {"mode": {"enum": ["b", "c"]}}, "required": ["mode"]},
            "then": {"properties": {"shared": {"type": "integer", "default": 7}}, "required": ["shared"]},
            "else": {"properties": {"fallback": {"type": "boolean", "default": True}}},
        })
        form = self.form(schema)
        validator = validators.validator_for(schema)(schema)
        self.assertEqual(form.get_data(), {"mode": "a", "a_value": 1, "fallback": True})
        for index in (1, 2):
            form.fields["mode"].setCurrentIndex(index)
            mode = "abc"[index]
            self.assertEqual(form.get_data(), {"mode": mode, mode + "_value": index + 1, "shared": 7})
            validator.validate(form.get_data())

    def test_unconditional_then_is_ignored(self):
        form = self.form({"type": "object", "then": {"properties": {"x": {"type": "integer"}}}})
        self.assertEqual(form.get_data(), {})

    def test_object_default_and_nested_array_notifications(self):
        form = self.form({"type": "object", "properties": {"obj": {"type": "object", "default": {"x": 7}, "properties": {"x": {"type": "integer"}}},
                         "values": {"type": "array", "items": {"type": "array", "items": {"type": "boolean"}}, "default": [[True, False]]}}})
        calls = []
        form.changed.connect(lambda: calls.append(True))
        inner = form.fields["values"].item_widgets[0][1]
        inner.item_widgets[0][1].setChecked(False)
        self.assertTrue(calls)
        self.assertEqual(form.get_data(), {"obj": {"x": 7}, "values": [[False, False]]})

    def test_array_limits_nulls_and_replacement(self):
        array = ArrayWidget({"type": "array", "minItems": 1, "maxItems": 2, "default": [None]})
        self.forms.append(array)
        self.assertEqual(array.get_data(), [None])
        array.add_item(2.5)
        array.add_item(3)
        self.assertEqual(array.get_data(), [None, 2.5])
        self.assertFalse(array.add_button.isEnabled())
        with self.assertRaises(ValueError):
            array.set_data([1, 2, 3])
        self.assertEqual(array.get_data(), [None, 2.5])
        array.remove_item(*array.item_widgets[0])
        self.assertEqual(array.get_data(), [2.5])
        array.remove_item(*array.item_widgets[0])
        self.assertEqual(array.get_data(), [2.5])

    def test_invalid_json_can_be_edited_without_signal_exception(self):
        form = self.form({"type": "object", "properties": {"x": {}}})
        form.fields["x"].setText("[")
        with self.assertRaises(ValueError):
            form.get_data()
        form.fields["x"].setText("[1, null]")
        self.assertEqual(form.get_data(), {"x": [1, None]})

    def test_impossible_array_limits_fail_without_hanging(self):
        with self.assertRaises(ValueError):
            ArrayWidget({"minItems": 2, "maxItems": 1})

    def test_required_empty_string_controls_same_condition_as_output(self):
        form = self.form({"type": "object", "properties": {"name": {"type": "string"}},
                          "if": {"required": ["name"]},
                          "then": {"properties": {"yes": {"const": True}}},
                          "else": {"properties": {"no": {"const": True}}}})
        self.assertEqual(form.get_data(), {"no": True})
        form.fields["name"].setText("a")
        self.assertEqual(form.get_data(), {"name": "a", "yes": True})
        form.fields["name"].setText("")
        self.assertEqual(form.get_data(), {"no": True})

    def test_prepare_schema_does_not_modify_input_or_instance_data(self):
        schema = {"type": "object", "properties": {"required": {"type": "string"}},
                  "default": {"properties": {"required": ["untouched"]}}}
        prepared, repairs = prepare_schema(schema)
        self.assertEqual(schema, prepared)
        self.assertEqual(repairs, [])
        schema["else"] = {"properties": {"required": ["x"]}}
        prepared, repairs = prepare_schema(schema)
        self.assertIn("required", schema["else"]["properties"])
        self.assertEqual(prepared["else"]["required"], ["x"])

    def test_mainwindow_default_path_and_save_errors(self):
        with patch("os.getcwd", return_value="/tmp"):
            window = MainWindow()
        self.forms.append(window)
        self.assertIsNotNone(window.form)
        previous = window.form
        window.load_schema("/does/not/exist.json")
        self.assertIs(window.form, previous)
        with patch.object(QMessageBox, "warning") as warning, patch.object(QFileDialog, "getSaveFileName", return_value=("", "")):
            window.form = self.form({"type": "object", "properties": {"x": {}}})
            window.form.fields["x"].setText("NaN")
            window.save_json()
            warning.assert_called_once()

    def test_save_writes_valid_json_with_numeric_types(self):
        window = MainWindow()
        self.forms.append(window)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.json")
            with patch.object(QFileDialog, "getSaveFileName", return_value=(path, "")), patch.object(QMessageBox, "information") as success:
                window.save_json()
                success.assert_called_once()
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            self.assertEqual(data["common_settings"]["atoms"]["ratio"], [0.8, 0.2])
            validators.validator_for(window.schema)(window.schema).validate(data)

    def test_entrypoints_load_schema_and_shut_down_from_other_directory(self):
        root = DEFAULT_SCHEMA.parent.parent
        for module_mode in (True, False):
            with self.subTest(module_mode=module_mode), tempfile.TemporaryDirectory() as directory:
                search_path = str(root if module_mode else root / "ui")
                entrypoint = 'runpy.run_module("ui.main", run_name="__main__")' if module_mode else f'runpy.run_path({str(root / "ui" / "main.py")!r}, run_name="__main__")'
                script = f'''
import runpy, sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow
sys.path.insert(0, {search_path!r})
original_exec = QApplication.exec
def timed_exec(self):
    windows = [w for w in self.topLevelWidgets() if isinstance(w, QMainWindow)]
    assert len(windows) == 1 and windows[0].form is not None
    QTimer.singleShot(10, self.quit)
    return original_exec()
QApplication.exec = timed_exec
{entrypoint}
'''
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                result = subprocess.run([sys.executable, "-c", script], cwd=directory, env=env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Compatibility for the legacy schema shipped with the editor."""
from copy import deepcopy


def prepare_schema(schema):
    """Repair only misplaced required arrays; never touch defaults or enum data."""
    result = deepcopy(schema)
    repairs = []

    def visit(node, path):
        if not isinstance(node, dict):
            return
        properties = node.get("properties", {})
        if isinstance(properties, dict):
            misplaced = properties.get("required")
            if isinstance(misplaced, list) and all(isinstance(x, str) for x in misplaced):
                node["required"] = list(dict.fromkeys(node.get("required", []) + properties.pop("required")))
                repairs.append(path + "/properties/required")
        for keyword in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
            for name, child in node.get(keyword, {}).items():
                visit(child, f"{path}/{keyword}/{name}")
        for keyword in ("if", "then", "else", "not", "items", "contains", "additionalProperties", "propertyNames", "unevaluatedProperties", "unevaluatedItems"):
            visit(node.get(keyword), f"{path}/{keyword}")
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
            for index, child in enumerate(node.get(keyword, [])):
                visit(child, f"{path}/{keyword}/{index}")

    visit(result, "#")
    return result, repairs

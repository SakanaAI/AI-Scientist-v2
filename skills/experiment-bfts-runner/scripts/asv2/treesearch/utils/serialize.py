import copy
import json
from pathlib import Path
from typing import Type, TypeVar
import re

import dataclasses_json
from ..journal import Journal, Node


def dumps_json(obj: dataclasses_json.DataClassJsonMixin):
    """Serialize dataclasses (such as Journals) to JSON."""
    if isinstance(obj, Journal):
        obj = copy.deepcopy(obj)
        node2parent = {}
        for n in obj.nodes:
            if n.parent is not None:
                # Handle both Node objects and string IDs
                parent_id = n.parent.id if isinstance(n.parent, Node) else n.parent
                node2parent[n.id] = parent_id
        for n in obj.nodes:
            n.parent = None
            n.children = set()

    obj_dict = obj.to_dict()

    if isinstance(obj, Journal):
        obj_dict["node2parent"] = node2parent
        obj_dict["__version"] = "2"

    return json.dumps(obj_dict, separators=(",", ":"))


def dump_json(obj: dataclasses_json.DataClassJsonMixin, path: Path):
    with open(path, "w") as f:
        f.write(dumps_json(obj))


G = TypeVar("G", bound=dataclasses_json.DataClassJsonMixin)


def loads_json(s: str, cls: Type[G]) -> G:
    """Deserialize JSON to AIDE dataclasses."""
    obj_dict = json.loads(s)
    obj = cls.from_dict(obj_dict)

    if isinstance(obj, Journal):
        id2nodes = {n.id: n for n in obj.nodes}
        for child_id, parent_id in obj_dict["node2parent"].items():
            id2nodes[child_id].parent = id2nodes[parent_id]
            id2nodes[child_id].__post_init__()
    return obj


def load_json(path: Path, cls: Type[G]) -> G:
    with open(path, "r") as f:
        return loads_json(f.read(), cls)


def parse_markdown_to_dict(content: str):
    """
    Reads a file that contains lines of the form:

        "Key": "Value",
        "Another Key": "Another Value",
        ...

    including possible multi-line values, and returns a Python dictionary.
    """

    pattern = r'"([^"]+)"\s*:\s*"([^"]*?)"(?:,\s*|\s*$)'

    matches = re.findall(pattern, content, flags=re.DOTALL)

    data_dict = {}
    for key, value in matches:
        data_dict[key] = value

    return data_dict

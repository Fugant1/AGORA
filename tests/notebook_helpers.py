"""Load current notebook functions without GPU/Drive execution or cell offsets."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_cells():
    return json.loads((ROOT / "ser_grpo_v2.ipynb").read_text())["cells"]


def cell_source(marker):
    matches = ["".join(c["source"]) for c in notebook_cells()
               if c["cell_type"] == "code" and marker in "".join(c["source"])]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one code cell containing {marker!r}, got {len(matches)}")
    return matches[0]


def load_reward_environment():
    env = {"display": lambda *args: None}
    for marker in ("VALID_EMOTIONS =  {", "LABEL_SYNONYMS =", "REQUIRED_INVENTORY_KEYS =",
                   "import re\nfrom typing import Any", "def extract_acoustic_inventory(",
                   "def extract_reasoning(", "def extract_answer_content(",
                   "def calibrate_emotion_payoffs(", "def reward_acoustic_inventory(",
                   "def compute_format_reward_from_parsed(", "REWARD_WEIGHTS ="):
        exec(compile(cell_source(marker), f"<notebook:{marker}>", "exec"), env)
    for marker, names in (
        ("def evaluate_grpo_validation(", {"INVENTORY_DIMENSIONS", "evaluate_single_prompt_acoustics"}),
        ("def validate_grpo_reward_mathematics(", {"build_sample_completion", "validate_grpo_reward_mathematics"}),
    ):
        nodes = []
        for node in ast.parse(cell_source(marker)).body:
            name = getattr(node, "name", None)
            if isinstance(node, ast.AnnAssign):
                name = getattr(node.target, "id", None)
            if name in names:
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), f"<notebook:{marker}>", "exec"), env)
    return env

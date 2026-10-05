
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

source, root = map(Path, sys.argv[1:3])
sys.path.insert(0, str(source))

from autoddi.search_algorithm import utils
from autoddi.search_space.search_space_config import SearchSpace

result = json.loads(
    (root / "candidate_05/result.json").read_text()
)
assert result["completed_epochs"] == 5
assert result["test_evaluated"] is False

controller_path = root / "controller_after_candidate_05.pt"
pending_path = root / "candidate_06_selection.json"

# Preserve a selection already made: never resample on rerun.
if pending_path.exists():
    assert controller_path.is_file()
    print("Using saved candidate 06 selection:", flush=True)
    print(pending_path.read_text(), flush=True)
    sys.exit(0)

tree = ast.parse((
    source / "autoddi/search_algorithm/graphnas_search_algorithm.py"
).read_text())
classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name in {"SearchController", "MoveAverageOperator", "Search"}
]
assert len(classes) == 3
exec(compile(
    ast.Module(body=classes, type_ignores=[]),
    "original_controller", "exec"
), globals())

assert torch.cuda.is_available()
device = "cuda:0"

parameters = {
    "cuda": True,
    "device": device,
    "softmax_temperature": 5.0,
    "tanh_c": 2.5,
    "controller_lr": 3.5e-4,
    "controller_train_parallel_num": 1,
    "entropy_coeff": 1e-4,
    "discount": 1.0,
    "ema_baseline_decay": 0.95,
    "controller_grad_clip": 0.0,
}

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

search = Search(
    data=None,
    search_parameter=parameters,
    gnn_parameter={},
    search_space=SearchSpace(),
)

def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)

if controller_path.exists():
    state = torch.load(
        controller_path, map_location="cpu", weights_only=False
    )
    assert state["candidate_05_result"] == result
    search.controller.load_state_dict(state["controller"])
    search.controller_optim.load_state_dict(state["optimizer"])
    search.baseline = state["baseline"]
    search.history = state["history"]
    search.move_average_reward_operator.scores = state["scores"]
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
else:
    previous_state = torch.load(
        root / "controller_after_candidate_04.pt",
        map_location="cpu",
        weights_only=False,
    )
    selected = json.loads(
        (root / "candidate_05_selection.json").read_text()
    )
    assert selected["architecture"] == result["architecture"]

    search.controller.load_state_dict(previous_state["controller"])
    search.controller_optim.load_state_dict(previous_state["optimizer"])
    search.baseline = previous_state["baseline"]
    search.history = previous_state["history"]
    search.move_average_reward_operator.scores = previous_state["scores"]

    torch.set_rng_state(previous_state["torch_rng"])
    torch.cuda.set_rng_state_all(previous_state["cuda_rng"])

    # Reproduce the saved sample and advance the sampling RNG.
    with torch.no_grad():
        reproduced, _, _ = search.controller.sample(
            batch_size=1, device=device
        )
    assert reproduced[0] == result["architecture"], (
        "Saved candidate 05 sampling could not be reproduced."
    )

    # Evaluate the known architecture's action probabilities.
    # This is explicitly forced replay, not a recovered random draw.
    def replay_sample(batch_size=1, device="cuda:0"):
        assert batch_size == 1
        controller = search.controller
        x = torch.zeros(1, controller.controller_hid, device=device)
        hidden = (
            torch.zeros_like(x),
            torch.zeros_like(x),
        )
        log_probs, entropies = [], []

        for block, (name, value) in enumerate(zip(
            controller.action_list, result["architecture"]
        )):
            logits, hidden = controller.forward(
                x, hidden, name, is_embed=(block == 0)
            )
            probabilities = F.softmax(logits, dim=-1)
            logs = F.log_softmax(logits, dim=-1)
            index = controller.search_space[name].index(value)

            log_probs.append(logs[:, index])
            entropies.append(-(logs * probabilities).sum(-1))

            offset = sum(controller.num_tokens[
                :controller.action_index(name)
            ])
            token = torch.tensor([offset + index], device=device)
            x = controller.encoder(token)

        return (
            [result["architecture"]],
            torch.cat(log_probs),
            torch.cat(entropies),
        )

    def Estimation(
        gnn_architecture, graph_data, gnn_parameter, device
    ):
        assert list(gnn_architecture) == result["architecture"]
        return result["best_cal_macro84"]

    original_sample = search.controller.sample
    search.controller.sample = replay_sample
    search.train_controller()
    search.controller.sample = original_sample

    atomic_save({
        "candidate_05_result": result,
        "reconstruction": "candidate 05 replay from saved controller state",
        "controller": search.controller.state_dict(),
        "optimizer": search.controller_optim.state_dict(),
        "baseline": search.baseline,
        "history": search.history,
        "scores": search.move_average_reward_operator.scores,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, controller_path)

# One sample only; no choosing between draws.
with torch.no_grad():
    architectures, _, _ = search.controller.sample(
        batch_size=1, device=device
    )

selection = {
    "candidate": 6,
    "architecture": architectures[0],
    "candidate_05_reward": result["best_cal_macro84"],
    "controller_initialization": "reconstructed_with_forced_replay",
}
temporary = pending_path.with_suffix(".tmp")
temporary.write_text(json.dumps(selection, indent=2))
temporary.replace(pending_path)

print(" Controller state saved on Drive.", flush=True)
print("Candidate 06:", selection["architecture"], flush=True)

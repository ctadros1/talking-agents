"""Guards the pre-registered v1 paths: any change that alters their numbers fails here.

GOLDEN (training) was recorded on 2026-09-30 from the code frozen for the benchmark, before the
exploratory vocal channel was added. GOLDEN_EVAL (evaluation: episode sampling, observations, channel
routing, policy sampling, per-episode records, replay traces, and the scripted team) was recorded on
2026-10-05 from the v1.0 code, before any v2 change. If either fails, the v1 results no longer reproduce.
"""

import hashlib
import json

import pytest

from emcomm.benchmark import EVAL_SEED, episode_sets
from emcomm.config import EnvConfig, PPOConfig
from emcomm.evaluate import LearnedTeam
from emcomm.heuristics import ScriptedTeam
from emcomm.rollout import run_episodes
from emcomm.train import load_checkpoint, train

GOLDEN = "e6c0229b32db257103f798cef31812d40adede33069152d174c26f48bf15de28"
GOLDEN_EVAL = "8788bf2844c467e6bee7759787f59891c8291b4cae72828b8814c5170bdf02b0"
TINY = PPOConfig(
    n_envs=8, rollout_len=16, total_env_steps=8 * 16 * 3, hidden=32, n_minibatches=2, checkpoint_every=10, sig_coef=0.2
)


@pytest.fixture(scope="module")
def tiny_comm(tmp_path_factory):
    return load_checkpoint(train(EnvConfig(), TINY, "comm", seed=0, out_dir=tmp_path_factory.mktemp("tiny")))[0]


def weights_hash(nets) -> str:
    digest = hashlib.sha256()
    for net in nets:
        for name, tensor in net.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def test_discrete_training_path_is_unchanged(tiny_comm):
    assert weights_hash(tiny_comm) == GOLDEN


def test_discrete_evaluation_path_is_unchanged(tiny_comm):
    cfg = EnvConfig()
    specs = episode_sets(cfg)["held_out"][:20]
    digest = hashlib.sha256()
    for mode in ("intact", "permuted", "silenced"):
        team = LearnedTeam(tiny_comm, len(specs), EVAL_SEED)
        out = run_episodes(cfg, specs, team, "comm", mode, seed=EVAL_SEED, record=(0, 7))
        digest.update(json.dumps(out, sort_keys=True).encode())
    for condition, mode in (("comm", "intact"), ("random_msg", "random"), ("oracle", "silenced")):
        team = ScriptedTeam(len(specs), runner_sees_cue=condition == "oracle")
        out = run_episodes(cfg, specs, team, condition, mode, seed=EVAL_SEED, record=(3,))
        digest.update(json.dumps(out, sort_keys=True).encode())
    assert digest.hexdigest() == GOLDEN_EVAL

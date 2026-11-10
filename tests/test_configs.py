from dataclasses import replace
from pathlib import Path

from emcomm.config import EnvConfig, PPOConfig, load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def test_base_config_is_explicit_and_matches_defaults():
    env_cfg, ppo_cfg = load_config(CONFIGS / "base.toml")
    assert env_cfg == EnvConfig()
    assert ppo_cfg.total_env_steps % ppo_cfg.steps_per_update == 0
    assert ppo_cfg == PPOConfig(total_env_steps=ppo_cfg.total_env_steps)


def test_v2_config_matches_the_frozen_rung():
    assert load_config(CONFIGS / "v2.toml") == load_config(CONFIGS / "v2_pilot_rung1.toml")


def test_v2_pilot_rungs():
    env1, ppo1 = load_config(CONFIGS / "v2_pilot_rung1.toml")
    assert env1 == EnvConfig(task="compositional", msg_words=2) and ppo1 == PPOConfig(total_env_steps=20_004_864)
    env2, ppo2 = load_config(CONFIGS / "v2_pilot_rung2.toml")
    assert env2 == env1 and ppo2 == replace(ppo1, sig_coef=0.5)


def test_submission_ablation_changes_only_signalling_weight():
    """The post-hoc original-method comparison must retain the full benchmark budget."""
    env, ppo = load_config(CONFIGS / "base.toml")
    ablation_env, ablation_ppo = load_config(CONFIGS / "ablation_unbiased.toml")
    assert ablation_env == env
    assert ablation_ppo == replace(ppo, sig_coef=0.0)

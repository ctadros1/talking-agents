import pytest

from emcomm.config import EnvConfig, PPOConfig, load_config


def test_obs_dim_matches_feature_layout():
    cfg = EnvConfig()
    # 7x7 view * 6 channels + pos 2 + carrying 5 + cue 4 + received 9 + time 1
    assert cfg.obs_dim == 49 * 6 + 2 + 5 + 4 + 9 + 1 == 315
    assert cfg.silence == cfg.vocab_size == 8
    assert cfg.cue_slice == slice(301, 305)


def test_ppo_budget_arithmetic():
    ppo = PPOConfig(n_envs=64, rollout_len=128, total_env_steps=10_000_000)
    assert ppo.steps_per_update == 8192
    assert ppo.n_updates == 1220


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[env]\ngrid_sise = 10\n")
    with pytest.raises(ValueError, match="grid_sise"):
        load_config(path)


def test_load_config_overrides(tmp_path):
    path = tmp_path / "ok.toml"
    path.write_text("[env]\nmax_steps = 40\n[ppo]\nlr = 0.001\n")
    env_cfg, ppo_cfg = load_config(path)
    assert env_cfg.max_steps == 40 and ppo_cfg.lr == 0.001


V2 = EnvConfig(task="compositional", msg_words=2)


def test_v2_obs_layout():
    assert V2.n_features == 10 and V2.n_view_channels == 12
    # 7x7 view * 12 channels + pos 2 + carrying 11 + cue 10 + two received words 2 * 9 + time 1
    assert V2.obs_dim == 49 * 12 + 2 + 11 + 10 + 18 + 1 == 630
    assert V2.cue_slice == slice(601, 611)


def test_v1_layout_is_the_types_special_case():
    cfg = EnvConfig()
    assert cfg.task == "types" and cfg.msg_words == 1 and cfg.n_features == cfg.n_item_types == 4


def test_config_validation():
    with pytest.raises(ValueError, match="task"):
        EnvConfig(task="towers")
    with pytest.raises(ValueError, match="4 items"):
        EnvConfig(task="compositional", n_item_types=3)
    with pytest.raises(ValueError, match="vocal"):
        EnvConfig(channel="vocal", msg_words=2)

"""Config: serializacao, merge e sobreposicao por caminho."""

import json

import pytest

from yargen.config import DEFAULT, Config


def test_round_trips_through_json():
    assert Config.from_dict(json.loads(DEFAULT.to_json())).to_dict() == DEFAULT.to_dict()


def test_override_path_reaches_nested_values():
    cfg = DEFAULT.override_path("mapper.max_jump", 3)
    assert cfg.mapper.max_jump == 3
    assert DEFAULT.mapper.max_jump == 2, "a config original nao pode ser mutada"


def test_override_path_reaches_difficulty_tiers():
    cfg = DEFAULT.override_path("difficulty.easy.max_notes_per_second", 2.5)
    assert cfg.difficulty.easy.max_notes_per_second == 2.5
    assert type(cfg.difficulty.easy).__name__ == "DifficultyTier"


def test_unknown_parameter_is_rejected():
    """Um typo em --set tem que falhar alto, nao virar um parametro fantasma
    que e silenciosamente ignorado enquanto voce acha que testou algo."""
    with pytest.raises(ValueError, match="max_jumps"):
        DEFAULT.override_path("mapper.max_jumps", 3)


def test_merge_is_deep():
    cfg = DEFAULT.merged({"mapper": {"max_jump": 4}})
    assert cfg.mapper.max_jump == 4
    assert cfg.mapper.grid_division == DEFAULT.mapper.grid_division


def test_load_from_file(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(DEFAULT.override_path("mapper.seed", 42).to_json())
    assert Config.load(path).mapper.seed == 42

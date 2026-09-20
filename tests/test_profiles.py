"""Perfis de genero: roteamento fonte -> trilha."""

import pytest

from yargen import profiles
from yargen.pipeline import resolve_plans


def test_rock_routes_the_lead_from_the_guitar_stem():
    plan = next(p for p in profiles.get("rock").plans if p.instrument == "guitar")
    assert plan.sources[0] == "guitar"
    assert plan.strategy == profiles.PITCH


def test_trap_routes_the_lead_from_the_drums_percussively():
    """O ponto todo deste modulo: em trap nao ha guitarra, e o que se quer
    tocar e a batida."""
    plan = next(p for p in profiles.get("trap").plans if p.instrument == "guitar")
    assert plan.sources[0] == "drums"
    assert plan.strategy == profiles.PERCUSSIVE


def test_pop_prefers_keys_over_guitar():
    plan = next(p for p in profiles.get("pop").plans if p.instrument == "guitar")
    assert plan.sources[0] == "other"
    assert plan.sources.index("other") < plan.sources.index("guitar")


def test_every_profile_falls_back_to_the_mix():
    """Sem Demucs nao ha stem nenhum; todo plano precisa continuar produzindo
    alguma coisa."""
    for profile in profiles.canonical():
        for plan in profile.plans:
            assert plan.sources[-1] == "mix"


def test_every_profile_has_a_lead_track():
    for profile in profiles.canonical():
        assert any(p.track == profiles.GUITAR_TRACK for p in profile.plans)


def test_aliases_resolve_to_the_same_profile():
    assert profiles.get("hiphop") is profiles.get("trap")
    assert profiles.get("edm") is profiles.get("electronic")
    assert profiles.get("TRAP") is profiles.get("trap")


def test_unknown_genre_lists_the_valid_ones():
    with pytest.raises(KeyError, match="desconhecido"):
        profiles.get("reggaeton-espacial")


def test_resolve_plans_filters_by_instrument():
    plans = resolve_plans("trap", ["guitar", "vocals"])
    assert {p.instrument for p in plans} == {"guitar", "vocals"}


def test_resolve_plans_rejects_an_impossible_request():
    """`percussion` nao tem voz; pedir vocals dele tem que falhar com uma
    mensagem util em vez de gerar uma pasta vazia."""
    with pytest.raises(ValueError, match="vocals"):
        resolve_plans("percussion", ["vocals"])


def test_track_overrides_are_dotted_config_paths():
    from yargen.config import DEFAULT

    plan = next(p for p in profiles.get("trap").plans if p.instrument == "guitar")
    cfg = DEFAULT
    for key, value in plan.overrides.items():
        cfg = cfg.override_path(key, value)   # falha alto se a chave nao existe
    assert cfg.mapper.grid_division == 32
    assert cfg.mapper.chord_adjacent_only is False

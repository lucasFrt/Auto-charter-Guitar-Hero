"""Reducao Expert -> Hard/Medium/Easy."""

import pytest

from yargen.chart.difficulty import ORDER, reduce_all, reduce_to
from yargen.chart.ir import AnalyzedNote, ChartNote, InstrumentChart, TempoMap
from yargen.chart.mapper import map_track
from yargen.config import DEFAULT

TIERS = {"expert": DEFAULT.difficulty.expert, "hard": DEFAULT.difficulty.hard,
         "medium": DEFAULT.difficulty.medium, "easy": DEFAULT.difficulty.easy}


@pytest.fixture
def dense_chart():
    """Chart Expert denso o bastante para todos os tetos morderem."""
    import random
    rng = random.Random(7)
    tm = TempoMap.constant(140.0, 60.0)
    raw = [AnalyzedNote(i * 0.107, rng.uniform(0.3, 1.0),
                        220 * 2 ** (rng.randrange(0, 20) / 12)) for i in range(500)]
    return map_track(raw, tm, DEFAULT), tm


def nps(notes, tm):
    span = tm.tick_to_time(notes[-1].tick) - tm.tick_to_time(notes[0].tick)
    return len(notes) / span


@pytest.mark.parametrize("diff", ["expert", "hard", "medium", "easy"])
def test_density_ceiling_is_respected(dense_chart, diff):
    chart, tm = dense_chart
    reduce_all(chart, tm, DEFAULT)
    assert nps(chart.notes[diff], tm) <= TIERS[diff].max_notes_per_second * 1.05


@pytest.mark.parametrize("diff", ["hard", "medium", "easy"])
def test_density_budget_is_actually_used(dense_chart, diff):
    """Regressao: `int(1.5 * 1.0)` truncava o teto do Easy para 1 nota/s, e o
    Easy saia com 40% menos notas do que a config pedia."""
    chart, tm = dense_chart
    reduce_all(chart, tm, DEFAULT)
    ceiling = TIERS[diff].max_notes_per_second
    assert nps(chart.notes[diff], tm) >= ceiling * 0.7


def test_each_difficulty_is_sparser_than_the_one_above(dense_chart):
    chart, tm = dense_chart
    reduce_all(chart, tm, DEFAULT)
    counts = [len(chart.notes[d]) for d in ("expert", "hard", "medium", "easy")]
    assert counts == sorted(counts, reverse=True)


def test_medium_and_easy_use_only_three_colors(dense_chart):
    chart, tm = dense_chart
    reduce_all(chart, tm, DEFAULT)
    for diff in ("medium", "easy"):
        assert {f for n in chart.notes[diff] for f in n.frets} <= {0, 1, 2}


def test_medium_and_easy_have_no_chords():
    tm = TempoMap.constant(120.0, 30.0)
    expert = [ChartNote(i * 480, [0, 1], strength=0.9) for i in range(20)]
    for diff in ("medium", "easy"):
        notes = reduce_to(expert, tm, TIERS[diff], DEFAULT)
        assert all(len(n.frets) == 1 for n in notes)


def test_color_compression_preserves_order():
    """GRYBO -> GGRRY: a nota mais aguda nunca vira uma cor mais grave."""
    tm = TempoMap.constant(60.0, 60.0)
    expert = [ChartNote(i * 480, [i % 5], strength=0.9) for i in range(5)]
    notes = reduce_to(expert, tm, TIERS["medium"], DEFAULT)
    picked = [n.frets[0] for n in notes]
    assert picked == sorted(picked)


def test_reduced_notes_land_on_the_coarser_grid():
    tm = TempoMap.constant(120.0, 30.0)
    expert = [ChartNote(i * 120, [i % 5], strength=0.9) for i in range(40)]
    notes = reduce_to(expert, tm, TIERS["hard"], DEFAULT)   # grade 1/8 = 240 ticks
    assert all(n.tick % 240 == 0 for n in notes)


def test_merged_collisions_keep_the_strongest_note():
    """Duas semicolcheias caem no mesmo tick da grade 1/8. Sobrevive a mais
    forte, e ela nao pode virar um acorde acidental com a perdedora."""
    tm = TempoMap.constant(120.0, 30.0)
    expert = [ChartNote(0, [0], strength=0.2), ChartNote(120, [4], strength=0.9)]
    notes = reduce_to(expert, tm, TIERS["hard"], DEFAULT)
    assert len(notes) == 1 and notes[0].frets == [4]


def test_sustains_are_rebuilt_and_never_overlap(dense_chart):
    """Obrigatorio depois dos cortes: um sustain herdado do Expert passaria
    por cima da proxima nota que sobrou."""
    chart, tm = dense_chart
    reduce_all(chart, tm, DEFAULT)
    for diff in ORDER:
        notes = chart.notes[diff]
        for a, b in zip(notes, notes[1:]):
            assert a.tick + a.sustain_ticks <= b.tick, f"{diff}: sustain invade a proxima"


def test_can_generate_a_subset_of_difficulties(dense_chart):
    chart, tm = dense_chart
    reduce_all(chart, tm, DEFAULT, difficulties=["expert", "easy"])
    assert set(chart.notes) == {"expert", "easy"}


def test_empty_expert_produces_nothing():
    chart = InstrumentChart("PART GUITAR", notes={"expert": []})
    reduce_all(chart, TempoMap.constant(120.0, 10.0), DEFAULT)
    assert not chart.notes.get("hard")

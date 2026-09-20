"""TempoMap: conversao tempo <-> tick, inclusive com andamento variavel."""

import pytest

from yargen.chart.ir import TempoMap


def test_constant_grid_round_trips():
    tm = TempoMap.constant(120.0, 10.0)
    for t in (0.0, 0.25, 0.5, 1.0, 3.7):
        assert tm.tick_to_time(tm.time_to_tick(t)) == pytest.approx(t, abs=1e-9)


def test_beat_lands_on_whole_resolution():
    tm = TempoMap.constant(120.0, 10.0)
    assert tm.time_to_tick(0.5) == pytest.approx(480)
    assert tm.time_to_tick(2.0) == pytest.approx(1920)


def test_variable_tempo_keeps_beats_on_the_grid():
    """O ponto central do desenho: mesmo acelerando, o beat N vale N*resolution.

    E isto que faz a grade musical continuar valendo em gravacao com rubato -
    a semicolcheia 3 do compasso 12 e sempre o mesmo tick.
    """
    beats, step = [0.0], 0.5
    for _ in range(32):
        beats.append(beats[-1] + step)
        step *= 0.97
    tm = TempoMap(beats)
    for i in (0, 5, 17, 32):
        assert tm.time_to_tick(beats[i]) == pytest.approx(i * 480, abs=1e-6)


def test_extrapolates_before_first_and_after_last_beat():
    tm = TempoMap([1.0, 1.5, 2.0])
    assert tm.time_to_tick(0.5) == pytest.approx(-480)
    # 2.5s e um beat inteiro depois do ultimo beat (2.0s), logo tick 1440
    assert tm.time_to_tick(2.5) == pytest.approx(1440)


def test_quantize_snaps_to_subdivision():
    tm = TempoMap.constant(120.0, 10.0)
    assert tm.quantize(0.51, 16) == 480       # 1/16 = 120 ticks
    assert tm.quantize(0.58, 16) == 600
    assert tm.quantize(0.56, 4) == 480        # 1/4 = 480 ticks


def test_tempo_events_deduplicate_constant_tempo():
    """Sem deduplicacao, um BPM constante geraria um set_tempo por beat."""
    assert len(TempoMap.constant(174.0, 60.0).tempo_events()) == 1


def test_tempo_events_follow_real_drift():
    beats, step = [0.0], 0.5
    for _ in range(40):
        beats.append(beats[-1] + step)
        step *= 0.95
    assert len(TempoMap(beats).tempo_events(min_drift_bpm=0.5)) > 10


def test_downbeat_detection_respects_offset():
    tm = TempoMap.constant(120.0, 20.0, downbeat_index=2)
    assert tm.is_downbeat(2 * 480)
    assert not tm.is_downbeat(3 * 480)


def test_rejects_degenerate_grid():
    with pytest.raises(ValueError):
        TempoMap([0.0])

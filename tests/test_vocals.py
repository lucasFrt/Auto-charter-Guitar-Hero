"""Vocals: notas, frases e marcacao de silaba nao-tonal."""

import pytest

from yargen.chart.ir import AnalyzedWord, TempoMap
from yargen.chart.vocals import build, build_phrases
from yargen.config import DEFAULT


def words(spec):
    return [AnalyzedWord(t, d, text, hz, hz is not None) for t, d, text, hz in spec]


def test_words_become_notes_with_lyrics():
    tm = TempoMap.constant(120.0, 30.0)
    chart = build(words([(0.0, 0.4, "ho", 261.6), (0.5, 0.4, "la", 329.6)]), tm, DEFAULT)
    assert [n.text for n in chart.notes] == ["ho", "la"]
    assert [n.midi for n in chart.notes] == [60, 64]


def test_word_without_pitch_becomes_unpitched():
    tm = TempoMap.constant(120.0, 30.0)
    chart = build(words([(0.0, 0.4, "sh", None)]), tm, DEFAULT)
    assert chart.notes[0].pitched is False
    assert chart.notes[0].midi == DEFAULT.vocals.unpitched_fallback_midi


def test_pitch_outside_range_is_folded_by_octaves_not_clamped():
    """Erro de oitava do pyin e comum. Dobrar por oitava preserva a classe de
    altura - que e o que o jogador canta - em vez de colar a nota no teto."""
    tm = TempoMap.constant(120.0, 30.0)
    chart = build(words([(0.0, 0.4, "hi", 2093.0)]), tm, DEFAULT)  # C7, acima de C6
    assert 36 <= chart.notes[0].midi <= 84
    assert chart.notes[0].midi % 12 == 0, "C7 devia continuar sendo um do"


def test_notes_never_overlap():
    tm = TempoMap.constant(120.0, 30.0)
    chart = build(words([(0.0, 5.0, "aa", 261.6), (0.3, 0.4, "bb", 261.6)]), tm, DEFAULT)
    a, b = chart.notes
    assert a.tick + a.duration_ticks <= b.tick


def test_phrases_split_on_silence():
    tm = TempoMap.constant(120.0, 60.0)
    spec = [(0.0, 0.3, "a", 261.6), (0.4, 0.3, "b", 261.6),
            (5.0, 0.3, "c", 261.6), (5.4, 0.3, "d", 261.6)]
    chart = build(words(spec), tm, DEFAULT)
    assert len(chart.phrases) == 2


def test_long_phrase_is_force_split():
    tm = TempoMap.constant(120.0, 120.0)
    spec = [(i * 0.4, 0.3, f"w{i}", 261.6) for i in range(60)]  # 24s contiguos
    chart = build(words(spec), tm, DEFAULT)
    assert len(chart.phrases) >= 2


def test_every_note_falls_inside_some_phrase():
    """Sem frase o YARG nao pontua: a nota existe mas nao pertence a nada."""
    tm = TempoMap.constant(120.0, 60.0)
    spec = [(i * 0.5, 0.3, f"w{i}", 220.0 * 2 ** (i / 12)) for i in range(20)]
    chart = build(words(spec), tm, DEFAULT)
    for note in chart.notes:
        assert any(p.start_tick <= note.tick <= p.end_tick for p in chart.phrases)


def test_no_words_gives_empty_chart():
    assert build([], TempoMap.constant(120.0, 10.0), DEFAULT).notes == []


# ------------------------------------------- degradacao quando falta whisper

class _Contour:
    """Contorno sintetico, para testar a segmentacao sem tocar em audio."""

    def __init__(self, f0, hop=0.02):
        import numpy as np
        self.f0 = np.asarray(f0, dtype=float)
        self.times = np.arange(len(self.f0)) * hop

    def median_in(self, a, b):
        import numpy as np
        lo, hi = np.searchsorted(self.times, (a, b))
        w = self.f0[lo:max(hi, lo + 1)]
        w = w[np.isfinite(w)]
        return float(np.median(w)) if w.size else None

    def coverage(self, a, b):
        import numpy as np
        lo, hi = np.searchsorted(self.times, (a, b))
        w = self.f0[lo:max(hi, lo + 1)]
        return float(np.mean(np.isfinite(w))) if w.size else 0.0


def smoothed_step(a_hz, b_hz, n_each=40, ramp=5):
    """Duas notas com a transicao borrada, como o filtro de mediana deixa."""
    import numpy as np
    ramp_vals = list(np.linspace(a_hz, b_hz, ramp + 2)[1:-1])
    return [a_hz] * n_each + ramp_vals + [b_hz] * n_each


def test_contour_segmentation_splits_on_a_smoothed_transition():
    """Regressao: comparando frame a frame, um salto de 3 semitons espalhado
    pelo filtro de mediana vira 0.6 semitom por frame e nunca era detectado -
    as duas notas saiam grudadas como uma so."""
    from yargen.chart.vocals import words_from_contour

    words = words_from_contour(_Contour(smoothed_step(329.63, 392.00)), DEFAULT.vocals)
    assert len(words) == 2
    assert words[0].pitch_hz == pytest.approx(329.63, rel=0.02)
    assert words[1].pitch_hz == pytest.approx(392.00, rel=0.02)


def test_contour_segmentation_does_not_split_on_vibrato():
    """Vibrato de ~0.2 semitom nao pode picotar uma nota longa em varias."""
    import numpy as np
    from yargen.chart.vocals import words_from_contour

    t = np.arange(120) * 0.02
    f0 = 440.0 * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
    assert len(words_from_contour(_Contour(f0), DEFAULT.vocals)) == 1


def test_contour_segmentation_splits_on_silence():
    import numpy as np
    from yargen.chart.vocals import words_from_contour

    f0 = [440.0] * 40 + [np.nan] * 20 + [440.0] * 40
    assert len(words_from_contour(_Contour(f0), DEFAULT.vocals)) == 2


def test_missing_whisper_raises_a_typed_error_not_a_crash(monkeypatch):
    """O download do modelo falha atras de proxy, sem rede ou com disco cheio.
    Uma rodada de varios minutos nao pode morrer no ultimo passo por isso."""
    import builtins

    from yargen.chart.vocals import TranscriptionUnavailable, transcribe

    real_import = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    with pytest.raises(TranscriptionUnavailable):
        transcribe("qualquer.wav", DEFAULT.vocals)


def test_pipeline_falls_back_to_pitch_when_transcription_is_unavailable(monkeypatch):
    """O caminho completo: sem letra, vocals ainda tem que sair jogavel."""
    from yargen.chart import vocals as vocals_mod
    from yargen import pipeline

    monkeypatch.setattr(vocals_mod, "transcribe", lambda *a, **k: (_ for _ in ()).throw(
        vocals_mod.TranscriptionUnavailable("proxy bloqueou o download")))

    class Stub:
        path = None
        duration = 4.0
        samples = None
        sample_rate = 22050

    monkeypatch.setattr(pipeline.pitch_mod, "contour",
                        lambda *a, **k: _Contour(smoothed_step(329.63, 392.00, 60)))
    track = pipeline._analyze_vocals(None, Stub(), DEFAULT, lambda m: None)
    assert len(track.words) == 2
    assert all(w.text == "" for w in track.words)

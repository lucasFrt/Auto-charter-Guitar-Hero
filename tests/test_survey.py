"""Levantamento dos stems e recomendacao de genero."""

import numpy as np
import pytest

from yargen.analysis.survey import (StemSurvey, Survey, active_mask, recommend,
                                    render, rms_peak, weighted_energy)
from yargen.audio.loader import Audio


def stem(name, activity=0.9, onsets=200, pitched=0.9, centroid=1000.0,
         energy=0.3, rate=3.0):
    return StemSurvey(name, energy, rate, pitched, centroid, onsets, activity)


def absent(name):
    """Como o Demucs devolve um instrumento que a musica nao tem: nao vazio,
    mas so vazamento."""
    return stem(name, activity=0.01, onsets=0, pitched=0.0, energy=0.0, rate=0.0)


def survey_of(**stems) -> Survey:
    return recommend(Survey(stems=stems, duration=180.0))


# ---------------------------------------------------------------- presenca

def test_leakage_is_not_counted_as_present():
    assert not absent("guitar").present


def test_an_active_stem_with_onsets_is_present():
    assert stem("guitar").present


def test_percussive_stem_is_not_melodic():
    assert not stem("drums", pitched=0.05).melodic


# ------------------------------------------------------------ recomendacao

def test_guitar_driven_song_recommends_rock():
    s = survey_of(guitar=stem("guitar", rate=4.0), drums=stem("drums", pitched=0.05),
                  bass=stem("bass"), vocals=stem("vocals", activity=0.4),
                  other=absent("other"))
    assert s.recommended == "rock"


def test_beat_with_no_lead_and_a_voice_recommends_trap():
    """O caso que motivou este modulo inteiro: nao ha guitarra, o que se quer
    tocar e a batida, e a voz merece a propria trilha."""
    s = survey_of(guitar=absent("guitar"), other=absent("other"),
                  drums=stem("drums", pitched=0.05, rate=5.0),
                  bass=stem("bass"), vocals=stem("vocals", activity=0.4))
    assert s.recommended == "trap"
    assert any("bateria conduz" in r for r in s.reasons)


def test_instrumental_beat_recommends_percussion():
    s = survey_of(guitar=absent("guitar"), other=absent("other"),
                  drums=stem("drums", pitched=0.05, rate=5.0),
                  bass=stem("bass"), vocals=absent("vocals"))
    assert s.recommended == "percussion"


def test_synth_lead_with_voice_recommends_pop():
    s = survey_of(guitar=absent("guitar"), other=stem("other", rate=3.0),
                  drums=stem("drums", pitched=0.05), bass=stem("bass"),
                  vocals=stem("vocals", activity=0.4))
    assert s.recommended == "pop"


def test_synth_lead_without_voice_recommends_electronic():
    s = survey_of(guitar=absent("guitar"), other=stem("other"),
                  drums=stem("drums", pitched=0.05), bass=stem("bass"),
                  vocals=absent("vocals"))
    assert s.recommended == "electronic"


def test_guitar_without_drums_recommends_acoustic():
    s = survey_of(guitar=stem("guitar", rate=3.0), drums=absent("drums"),
                  vocals=stem("vocals", activity=0.4), other=absent("other"))
    assert s.recommended == "acoustic"


def test_nothing_present_falls_back_to_rock():
    s = survey_of(guitar=absent("guitar"), other=absent("other"),
                  drums=absent("drums"), vocals=absent("vocals"))
    assert s.recommended == "rock"


def test_recommendation_always_explains_itself():
    """Um classificador opaco seria pior aqui mesmo se fosse mais preciso: a
    pessoa precisa poder discordar com conhecimento de causa."""
    s = survey_of(guitar=absent("guitar"), other=absent("other"),
                  drums=stem("drums", pitched=0.05, rate=5.0),
                  vocals=stem("vocals", activity=0.4))
    assert s.reasons
    assert "--genre trap" in render(s)
    assert "PART GUITAR" in render(s)


# ------------------------------------------------------------------ medidas

def audio_of(samples, sr=22050):
    return Audio(np.asarray(samples, dtype=np.float32), sr, None,
                 len(samples) / sr)


def test_activity_uses_an_external_reference():
    """Contra o proprio pico, um stem que so tem vazamento inaudivel parece
    100% ativo - o vazamento e o pico dele. E exatamente o caso que
    precisamos distinguir."""
    loud = audio_of(np.sin(np.linspace(0, 400, 22050)) * 0.9)
    quiet = audio_of(np.sin(np.linspace(0, 400, 22050)) * 0.0005)
    reference = rms_peak(loud)
    assert active_mask(quiet, 512).mean() > 0.9           # sem referencia: mente
    assert active_mask(quiet, 512, reference_peak=reference).mean() < 0.1


def test_weighted_energy_does_not_let_sub_bass_swallow_everything():
    """Com energia bruta um 808 leva quase tudo e a bateria some."""
    sr = 22050
    t = np.arange(sr) / sr
    sub = audio_of(np.sin(2 * np.pi * 50 * t) * 0.9)
    mid = audio_of(np.sin(2 * np.pi * 2000 * t) * 0.15)
    raw_ratio = float(np.sum(sub.samples ** 2) / np.sum(mid.samples ** 2))
    weighted_ratio = weighted_energy(sub) / weighted_energy(mid)
    assert raw_ratio > 20
    assert weighted_ratio < raw_ratio / 10


def test_measure_survives_a_silent_stem():
    from yargen.analysis.survey import measure
    from yargen.config import DEFAULT

    result = measure("guitar", audio_of(np.zeros(22050)), 1.0, DEFAULT)
    assert result.onset_count == 0 and not result.present

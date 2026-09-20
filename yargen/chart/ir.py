"""Representacao intermediaria (IR) do YARGen.

Duas IRs, nao uma:

  AnalysisIR  - o que saiu do DSP. Tempos em segundos, pitch em Hz. Nao sabe
                nada sobre trastes, ticks ou YARG.
  ChartIR     - o que o mapper produziu. Tempos em ticks, notas em trastes.
                Nao sabe nada sobre audio.

O mapper e a unica coisa entre as duas, e e justamente por isso que ele pode
ser testado com JSON fixo, sem tocar em audio.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

IR_VERSION = 1

# Trastes. 0..4 sao as cinco cores; OPEN e um valor sentinela fora da faixa.
GREEN, RED, YELLOW, BLUE, ORANGE = 0, 1, 2, 3, 4
OPEN = 7
FRET_NAMES = {GREEN: "G", RED: "R", YELLOW: "Y", BLUE: "B", ORANGE: "O", OPEN: "open"}

DIFFICULTIES = ("easy", "medium", "hard", "expert")


# ============================================================ mapa de tempo

@dataclass
class TempoMap:
    """Grade de beats + conversao tempo <-> tick.

    Guardamos a lista de beats em segundos em vez de um unico BPM porque o
    beat tracking em musica real anda. Cada beat vale exatamente uma seminima
    (`resolution` ticks), entao a grade musical continua valendo mesmo quando
    o andamento oscila: a semicolcheia 3 do compasso 12 e sempre o mesmo tick,
    tenha o baterista acelerado ou nao.

    O preco e um evento set_tempo por beat no MIDI, que e barato e o que os
    editores da cena ja esperam.
    """

    beats: list[float]
    resolution: int = 480
    beats_per_bar: int = 4
    downbeat_index: int = 0
    """Indice do primeiro tempo forte dentro de `beats`."""

    def __post_init__(self) -> None:
        if len(self.beats) < 2:
            raise ValueError("TempoMap precisa de pelo menos 2 beats")
        self.beats = [float(b) for b in self.beats]

    # -- construtores

    @classmethod
    def constant(cls, bpm: float, duration: float, resolution: int = 480,
                 beats_per_bar: int = 4, start: float = 0.0,
                 downbeat_index: int = 0) -> "TempoMap":
        """Grade de BPM fixo. Usada pelo M0, pelo `--bpm` e pelos testes."""
        step = 60.0 / bpm
        n = max(2, int((duration - start) / step) + 2)
        return cls([start + i * step for i in range(n)], resolution,
                   beats_per_bar, downbeat_index)

    # -- propriedades

    @property
    def bpm_at_start(self) -> float:
        return 60.0 / (self.beats[1] - self.beats[0])

    @property
    def average_bpm(self) -> float:
        span = self.beats[-1] - self.beats[0]
        return 60.0 * (len(self.beats) - 1) / span if span > 0 else 0.0

    @property
    def ticks_per_bar(self) -> int:
        return self.resolution * self.beats_per_bar

    # -- conversao

    def _interval(self, index: int) -> float:
        """Duracao do beat `index`, extrapolando fora dos limites."""
        if index < 0:
            return self.beats[1] - self.beats[0]
        if index >= len(self.beats) - 1:
            return self.beats[-1] - self.beats[-2]
        return self.beats[index + 1] - self.beats[index]

    def time_to_tick(self, t: float) -> float:
        """Segundos -> ticks (fracionario; quantize depois)."""
        i = bisect.bisect_right(self.beats, t) - 1
        if i < 0:
            # antes do primeiro beat: extrapola com o primeiro intervalo
            return (t - self.beats[0]) / self._interval(0) * self.resolution
        if i >= len(self.beats) - 1:
            step = self._interval(len(self.beats) - 1)
            extra = (t - self.beats[-1]) / step
            return (len(self.beats) - 1 + extra) * self.resolution
        frac = (t - self.beats[i]) / self._interval(i)
        return (i + frac) * self.resolution

    def tick_to_time(self, tick: float) -> float:
        beat_pos = tick / self.resolution
        i = int(beat_pos // 1)
        frac = beat_pos - i
        if i < 0:
            return self.beats[0] + beat_pos * self._interval(0)
        if i >= len(self.beats) - 1:
            over = beat_pos - (len(self.beats) - 1)
            return self.beats[-1] + over * self._interval(len(self.beats) - 1)
        return self.beats[i] + frac * self._interval(i)

    def quantize(self, t: float, division: int) -> int:
        """Segundos -> tick alinhado a uma subdivisao (16 = semicolcheia)."""
        step = self.resolution * 4 / division
        return int(round(self.time_to_tick(t) / step) * step)

    def snap_tick(self, tick: float, division: int) -> int:
        step = self.resolution * 4 / division
        return int(round(tick / step) * step)

    # -- posicao metrica

    def is_on_beat(self, tick: int, tolerance: int = 2) -> bool:
        return (tick % self.resolution) <= tolerance or \
               (self.resolution - (tick % self.resolution)) <= tolerance

    def is_downbeat(self, tick: int, tolerance: int = 2) -> bool:
        bar = self.ticks_per_bar
        offset = (tick - self.downbeat_index * self.resolution) % bar
        return offset <= tolerance or (bar - offset) <= tolerance

    def tempo_events(self, min_drift_bpm: float = 0.5) -> list[tuple[int, int]]:
        """[(tick, microsegundos_por_seminima)] ja deduplicado.

        Sem a deduplicacao o MIDI ganha um set_tempo por beat mesmo quando o
        andamento e constante e o jitter e de 0.01 BPM.
        """
        events: list[tuple[int, int]] = []
        last_bpm: float | None = None
        for i in range(len(self.beats) - 1):
            interval = self.beats[i + 1] - self.beats[i]
            if interval <= 0:
                continue
            bpm = 60.0 / interval
            if last_bpm is None or abs(bpm - last_bpm) >= min_drift_bpm:
                events.append((i * self.resolution, int(round(interval * 1_000_000))))
                last_bpm = bpm
        if not events:
            events.append((0, 500_000))
        return events

    # -- json
    def to_dict(self) -> dict[str, Any]:
        return {
            "beats": [round(b, 6) for b in self.beats],
            "resolution": self.resolution,
            "beats_per_bar": self.beats_per_bar,
            "downbeat_index": self.downbeat_index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TempoMap":
        return cls(list(d["beats"]), int(d.get("resolution", 480)),
                   int(d.get("beats_per_bar", 4)), int(d.get("downbeat_index", 0)))


# ====================================================== IR de analise (DSP)

@dataclass
class AnalyzedNote:
    """Um evento detectado no audio. Ainda nao e uma nota de jogo."""

    time: float
    """Segundos desde o inicio do arquivo."""

    strength: float = 1.0
    """0..1, forca do onset. Usada para cortar notas na reducao de dificuldade
    e para decidir o que vira acorde."""

    pitch_hz: float | None = None
    duration: float = 0.0
    """Ate o proximo onset, ou ate o fim da regiao tonal. 0 = desconhecido."""

    centroid_hz: float | None = None
    """Brilho (centroide espectral) no ataque. E o escalar que substitui a
    altura em conteudo percussivo, que nao tem f0."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"time": round(self.time, 6),
                             "strength": round(self.strength, 4),
                             "duration": round(self.duration, 6)}
        if self.pitch_hz is not None:
            d["pitch_hz"] = round(self.pitch_hz, 3)
        if self.centroid_hz is not None:
            d["centroid_hz"] = round(self.centroid_hz, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalyzedNote":
        return cls(float(d["time"]), float(d.get("strength", 1.0)),
                   (float(d["pitch_hz"]) if d.get("pitch_hz") is not None else None),
                   float(d.get("duration", 0.0)),
                   (float(d["centroid_hz"]) if d.get("centroid_hz") is not None else None))

    @property
    def pitch_midi(self) -> float | None:
        return hz_to_midi(self.pitch_hz)


@dataclass
class AnalyzedWord:
    """Uma palavra transcrita, com tempo e (talvez) altura."""

    time: float
    duration: float
    text: str
    pitch_hz: float | None = None
    pitched: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"time": round(self.time, 6),
                             "duration": round(self.duration, 6),
                             "text": self.text, "pitched": self.pitched}
        if self.pitch_hz is not None:
            d["pitch_hz"] = round(self.pitch_hz, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalyzedWord":
        return cls(float(d["time"]), float(d["duration"]), str(d["text"]),
                   (float(d["pitch_hz"]) if d.get("pitch_hz") is not None else None),
                   bool(d.get("pitched", True)))


@dataclass
class AnalyzedTrack:
    """Tudo que o DSP achou em um stem."""

    name: str
    """"guitar", "bass", "vocals", ..."""

    notes: list[AnalyzedNote] = field(default_factory=list)
    words: list[AnalyzedWord] = field(default_factory=list)
    source_stem: str = ""
    """Qual stem do Demucs gerou isso. Util para depurar a fonte de notas
    espurias sem re-rodar a separacao."""

    track_name: str = ""
    """Trilha do jogo que esta faixa alimenta: "PART GUITAR", "PART BASS"..."""

    strategy: str = "pitch"
    """"pitch", "percussive" ou "vocal". Fica gravado na IR para que
    `yargen build` reproduza o mesmo chart sem precisar do genero de novo."""

    overrides: dict[str, Any] = field(default_factory=dict)
    """Patch de config especifico desta trilha, vindo do perfil de genero."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "source_stem": self.source_stem,
                             "track_name": self.track_name,
                             "strategy": self.strategy,
                             "notes": [n.to_dict() for n in self.notes]}
        if self.overrides:
            d["overrides"] = dict(self.overrides)
        if self.words:
            d["words"] = [w.to_dict() for w in self.words]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalyzedTrack":
        return cls(str(d["name"]),
                   [AnalyzedNote.from_dict(n) for n in d.get("notes", [])],
                   [AnalyzedWord.from_dict(w) for w in d.get("words", [])],
                   str(d.get("source_stem", "")),
                   str(d.get("track_name", "")),
                   str(d.get("strategy", "pitch")),
                   dict(d.get("overrides", {})))


@dataclass
class SongMeta:
    name: str = "Unknown Song"
    artist: str = "Unknown Artist"
    album: str = ""
    genre: str = ""
    year: str = ""
    charter: str = "YARGen"
    duration: float = 0.0
    """Segundos."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SongMeta":
        return cls(**{k: v for k, v in d.items() if k in
                      {f for f in ("name", "artist", "album", "genre", "year",
                                   "charter", "duration")}})


@dataclass
class AnalysisIR:
    """Saida completa do estagio de analise. Serializa para JSON."""

    meta: SongMeta = field(default_factory=SongMeta)
    tempo: TempoMap | None = None
    tracks: dict[str, AnalyzedTrack] = field(default_factory=dict)
    genre: str = "rock"
    """Perfil usado na analise. Gravado para a rodada ser reproduzivel e para
    `yargen build` saber o que foi decidido."""

    version: int = IR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "genre": self.genre,
            "meta": self.meta.to_dict(),
            "tempo": self.tempo.to_dict() if self.tempo else None,
            "tracks": {k: v.to_dict() for k, v in self.tracks.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnalysisIR":
        return cls(
            SongMeta.from_dict(d.get("meta", {})),
            TempoMap.from_dict(d["tempo"]) if d.get("tempo") else None,
            {k: AnalyzedTrack.from_dict(v) for k, v in d.get("tracks", {}).items()},
            str(d.get("genre", "rock")),
            int(d.get("version", IR_VERSION)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2,
                                         ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisIR":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ======================================================= IR de chart (jogo)

@dataclass
class ChartNote:
    """Uma nota jogavel, ja em ticks."""

    tick: int
    frets: list[int]
    """0..4 para as cores, [OPEN] para nota aberta. Mais de um = acorde."""

    sustain_ticks: int = 0
    hopo: bool = False
    """Estado DESEJADO. O writer compara com o estado natural que o jogo
    calcularia e so grava marcador de forcing quando os dois divergem."""

    tap: bool = False
    strength: float = 1.0
    """Carregado da analise para a reducao de dificuldade poder cortar."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tick": self.tick, "frets": list(self.frets),
                             "sustain_ticks": self.sustain_ticks}
        if self.hopo:
            d["hopo"] = True
        if self.tap:
            d["tap"] = True
        d["strength"] = round(self.strength, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChartNote":
        return cls(int(d["tick"]), [int(f) for f in d["frets"]],
                   int(d.get("sustain_ticks", 0)), bool(d.get("hopo", False)),
                   bool(d.get("tap", False)), float(d.get("strength", 1.0)))

    @property
    def is_chord(self) -> bool:
        return len(self.frets) > 1

    @property
    def is_open(self) -> bool:
        return OPEN in self.frets


@dataclass
class Phrase:
    """Um marcador com duracao: star power, solo, frase de vocal."""

    start_tick: int
    end_tick: int

    def to_dict(self) -> dict[str, int]:
        return {"start_tick": self.start_tick, "end_tick": self.end_tick}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Phrase":
        return cls(int(d["start_tick"]), int(d["end_tick"]))


@dataclass
class VocalNote:
    tick: int
    duration_ticks: int
    midi: int
    text: str = ""
    pitched: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"tick": self.tick, "duration_ticks": self.duration_ticks,
                "midi": self.midi, "text": self.text, "pitched": self.pitched}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VocalNote":
        return cls(int(d["tick"]), int(d["duration_ticks"]), int(d["midi"]),
                   str(d.get("text", "")), bool(d.get("pitched", True)))


@dataclass
class InstrumentChart:
    """Um instrumento de 5 trastes, com suas quatro dificuldades."""

    track_name: str
    """"PART GUITAR", "PART BASS", ..."""

    notes: dict[str, list[ChartNote]] = field(default_factory=dict)
    star_power: list[Phrase] = field(default_factory=list)
    solos: list[Phrase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_name": self.track_name,
            "notes": {k: [n.to_dict() for n in v] for k, v in self.notes.items()},
            "star_power": [p.to_dict() for p in self.star_power],
            "solos": [p.to_dict() for p in self.solos],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstrumentChart":
        return cls(str(d["track_name"]),
                   {k: [ChartNote.from_dict(n) for n in v]
                    for k, v in d.get("notes", {}).items()},
                   [Phrase.from_dict(p) for p in d.get("star_power", [])],
                   [Phrase.from_dict(p) for p in d.get("solos", [])])


@dataclass
class VocalChart:
    track_name: str = "PART VOCALS"
    notes: list[VocalNote] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)
    star_power: list[Phrase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"track_name": self.track_name,
                "notes": [n.to_dict() for n in self.notes],
                "phrases": [p.to_dict() for p in self.phrases],
                "star_power": [p.to_dict() for p in self.star_power]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VocalChart":
        return cls(str(d.get("track_name", "PART VOCALS")),
                   [VocalNote.from_dict(n) for n in d.get("notes", [])],
                   [Phrase.from_dict(p) for p in d.get("phrases", [])],
                   [Phrase.from_dict(p) for p in d.get("star_power", [])])


@dataclass
class ChartIR:
    """O chart inteiro, pronto para o writer."""

    meta: SongMeta = field(default_factory=SongMeta)
    tempo: TempoMap | None = None
    instruments: dict[str, InstrumentChart] = field(default_factory=dict)
    vocals: VocalChart | None = None
    sections: list[tuple[int, str]] = field(default_factory=list)
    """[(tick, nome)] para a trilha EVENTS."""

    version: int = IR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "meta": self.meta.to_dict(),
            "tempo": self.tempo.to_dict() if self.tempo else None,
            "instruments": {k: v.to_dict() for k, v in self.instruments.items()},
            "vocals": self.vocals.to_dict() if self.vocals else None,
            "sections": [list(s) for s in self.sections],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChartIR":
        return cls(
            SongMeta.from_dict(d.get("meta", {})),
            TempoMap.from_dict(d["tempo"]) if d.get("tempo") else None,
            {k: InstrumentChart.from_dict(v)
             for k, v in d.get("instruments", {}).items()},
            VocalChart.from_dict(d["vocals"]) if d.get("vocals") else None,
            [(int(t), str(n)) for t, n in d.get("sections", [])],
            int(d.get("version", IR_VERSION)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2,
                                         ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChartIR":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def note_count(self) -> dict[str, int]:
        out = {f"{inst}:{diff}": len(notes)
               for inst, chart in self.instruments.items()
               for diff, notes in chart.notes.items()}
        if self.vocals:
            out["vocals"] = len(self.vocals.notes)
        return out


# ------------------------------------------------------------------ helpers

def hz_to_midi(hz: float | None) -> float | None:
    """Hz -> numero MIDI fracionario. None passa direto."""
    if hz is None or hz <= 0:
        return None
    import math
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

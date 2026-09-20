"""Harness de validacao objetiva.

Sem isto, ajustar heuristica vira achismo: voce muda um parametro, acha que
melhorou, e nao tem como saber. A regua e comparar os onsets gerados com os de
um chart humano bem avaliado, com tolerancia de +-50ms, e olhar
precisao/recall.

Nao vai bater 100% e nem precisa. O que importa e o SINAL: o numero subiu ou
desceu depois da mudanca?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

# Mesma tabela do writer; repetida aqui de proposito para que o harness leia
# charts de terceiros mesmo que a nossa escrita mude.
_DIFF_BASE = {"easy": 60, "medium": 72, "hard": 84, "expert": 96}
_GUITAR_TRACKS = {"PART GUITAR", "T1 GEMS", "PART GUITAR COOP"}


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    matched: int
    generated: int
    reference: int
    mean_abs_error_ms: float

    def __str__(self) -> str:
        return (f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}  "
                f"({self.matched}/{self.generated} geradas, "
                f"{self.matched}/{self.reference} de referencia, "
                f"erro medio {self.mean_abs_error_ms:.1f}ms)")


def note_times(midi_path: str | Path, difficulty: str = "expert",
               tracks: set[str] | None = None) -> list[float]:
    """Instantes (em segundos) das notas de guitarra de um notes.mid.

    Colapsa acordes num instante so: estamos medindo "acertamos onde tem
    nota", nao "acertamos quantas casas tem na nota".
    """
    tracks = tracks or _GUITAR_TRACKS
    base = _DIFF_BASE[difficulty]
    lanes = set(range(base, base + 5))

    mid = mido.MidiFile(str(midi_path))
    times: set[float] = set()
    for track in mid.tracks:
        name = next((m.name for m in track if m.type == "track_name"), "")
        if name not in tracks:
            continue
        # mido resolve o tempo map do arquivo inteiro apenas iterando o
        # MidiFile, nao uma trilha solta; por isso convertemos manualmente.
        for time, msg in _absolute(mid, track):
            if msg.type == "note_on" and msg.velocity > 0 and msg.note in lanes:
                times.add(round(time, 4))
    return sorted(times)


def _absolute(mid: mido.MidiFile, target: mido.MidiTrack):
    """Eventos de `target` em segundos, usando o tempo map do arquivo."""
    tempo_changes: list[tuple[int, int]] = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_changes.append((tick, msg.tempo))
    tempo_changes.sort()
    if not tempo_changes or tempo_changes[0][0] > 0:
        tempo_changes.insert(0, (0, 500_000))

    tick = 0
    for msg in target:
        tick += msg.time
        yield _tick_to_seconds(tick, tempo_changes, mid.ticks_per_beat), msg


def _tick_to_seconds(tick: int, changes: list[tuple[int, int]], tpb: int) -> float:
    seconds = 0.0
    prev_tick, prev_tempo = changes[0]
    for change_tick, tempo in changes[1:]:
        if change_tick >= tick:
            break
        seconds += (change_tick - prev_tick) / tpb * prev_tempo / 1e6
        prev_tick, prev_tempo = change_tick, tempo
    seconds += (tick - prev_tick) / tpb * prev_tempo / 1e6
    return seconds


def compare(generated: list[float], reference: list[float],
            tolerance_ms: float = 50.0) -> Scores:
    """Casa os dois conjuntos de instantes e calcula precisao/recall.

    O casamento e guloso sobre as duas listas ordenadas e cada instante so
    pode ser usado uma vez, dos dois lados. Sem a exclusividade, um trecho
    denso do chart gerado "acertaria" a mesma nota humana varias vezes e o
    recall mentiria para cima.
    """
    tolerance = tolerance_ms / 1000.0
    i = j = 0
    matched = 0
    errors: list[float] = []

    while i < len(generated) and j < len(reference):
        delta = generated[i] - reference[j]
        if abs(delta) <= tolerance:
            matched += 1
            errors.append(abs(delta))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1

    precision = matched / len(generated) if generated else 0.0
    recall = matched / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mae = sum(errors) / len(errors) * 1000 if errors else 0.0
    return Scores(precision, recall, f1, matched, len(generated),
                  len(reference), mae)


def compare_files(generated_mid: str | Path, reference_mid: str | Path,
                  difficulty: str = "expert", tolerance_ms: float = 50.0) -> Scores:
    return compare(note_times(generated_mid, difficulty),
                   note_times(reference_mid, difficulty), tolerance_ms)

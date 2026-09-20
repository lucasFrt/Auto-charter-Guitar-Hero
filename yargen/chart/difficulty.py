"""Reducao Expert -> Hard / Medium / Easy.

Derivar do Expert, e nao gerar cada dificuldade do zero, por dois motivos: as
dificuldades ficam coerentes entre si (o Easy e reconhecivelmente a mesma
musica que o Expert) e o custo e proximo de zero, porque nada aqui toca em
audio.

Criterio de corte, na ordem em que e aplicado:
  1. re-quantizar para a grade mais grossa e fundir colisoes
  2. limitar as cores disponiveis
  3. desmontar acordes onde a dificuldade nao os permite
  4. cortar por densidade, protegendo posicoes metricas fortes
"""

from __future__ import annotations

from typing import Sequence

from . import density
from ..config import DEFAULT, Config, DifficultyTier
from .ir import ChartNote, InstrumentChart, TempoMap

ORDER = ("expert", "hard", "medium", "easy")


def reduce_all(chart: InstrumentChart, tempo: TempoMap, cfg: Config = DEFAULT,
               *, difficulties: Sequence[str] = ORDER) -> InstrumentChart:
    """Preenche as dificuldades pedidas a partir do Expert."""
    expert = chart.notes.get("expert", [])
    if not expert:
        return chart
    tiers = {"expert": cfg.difficulty.expert, "hard": cfg.difficulty.hard,
             "medium": cfg.difficulty.medium, "easy": cfg.difficulty.easy}
    for name in difficulties:
        if name == "expert":
            continue
        chart.notes[name] = reduce_to(expert, tempo, tiers[name], cfg)
    if "expert" not in difficulties:
        chart.notes.pop("expert", None)
    return chart


def reduce_to(expert: Sequence[ChartNote], tempo: TempoMap, tier: DifficultyTier,
              cfg: Config = DEFAULT) -> list[ChartNote]:
    notes = _requantize(expert, tempo, tier.grid_division, cfg.difficulty.merge_window_ticks)
    notes = _limit_frets(notes, tier)
    notes = _cap_density(notes, tempo, tier, cfg)
    notes = _rebuild_sustains(notes, tempo, tier, cfg)
    return notes


def _requantize(notes: Sequence[ChartNote], tempo: TempoMap, division: int,
                merge_ticks: int) -> list[ChartNote]:
    """Move as notas para a grade mais grossa e funde as que colidem.

    Ao passar de 1/16 para 1/8, pares de semicolcheias caem no mesmo tick. A
    nota sobrevivente e a mais forte; a perdedora some em vez de virar um
    acorde acidental, que e o que aconteceria se simplesmente juntassemos as
    casas.
    """
    merged: dict[int, ChartNote] = {}
    for note in sorted(notes, key=lambda n: n.tick):
        tick = tempo.snap_tick(note.tick, division)
        existing = merged.get(tick)
        if existing is None:
            for offset in range(-merge_ticks, merge_ticks + 1):
                if tick + offset in merged:
                    existing = merged[tick + offset]
                    tick = tick + offset
                    break
        if existing is None:
            merged[tick] = ChartNote(tick, list(note.frets), note.sustain_ticks,
                                     note.hopo, note.tap, note.strength)
        elif note.strength > existing.strength:
            existing.frets = list(note.frets)
            existing.strength = note.strength
    return [merged[t] for t in sorted(merged)]


def _limit_frets(notes: Sequence[ChartNote], tier: DifficultyTier) -> list[ChartNote]:
    """Comprime 5 cores em `max_frets` e desmonta acordes se preciso.

    Com 3 cores usamos fret // 2, que mapeia GRYBO -> GGRRY. Preserva a ordem
    (nota mais aguda nunca vira cor mais grave) e mantem o contorno melodico
    legivel, que e o que faz o Easy parecer a mesma musica.
    """
    out: list[ChartNote] = []
    scale = 5 // tier.max_frets if tier.max_frets < 5 else 1
    for note in notes:
        frets = sorted({min(tier.max_frets - 1, f // scale) if f < 5 else f
                        for f in note.frets})
        if not tier.allow_chords and len(frets) > 1:
            frets = [frets[0]]
        out.append(ChartNote(note.tick, frets,
                             note.sustain_ticks if tier.allow_sustains else 0,
                             note.hopo, note.tap, note.strength))
    return out


def _cap_density(notes: Sequence[ChartNote], tempo: TempoMap, tier: DifficultyTier,
                 cfg: Config) -> list[ChartNote]:
    """Corta ate o teto de notas/s, favorecendo forca e posicao metrica forte.

    O bonus metrico e o que impede o Easy de virar uma sequencia de notas
    sincopadas aleatorias: entre duas notas de forca parecida, sobrevive a que
    cai no tempo forte do compasso, que e onde o jogador iniciante espera.
    """
    if tier.max_notes_per_second <= 0 or not notes:
        return list(notes)
    items = []
    for note in notes:
        score = note.strength
        if tempo.is_downbeat(note.tick):
            score += cfg.difficulty.bar_bonus
        elif tempo.is_on_beat(note.tick):
            score += cfg.difficulty.beat_bonus
        items.append((tempo.tick_to_time(note.tick), score))
    keep = density.select(items, tier.max_notes_per_second, cfg.mapper.density_window_sec)
    return [notes[i] for i in keep]


def _rebuild_sustains(notes: list[ChartNote], tempo: TempoMap, tier: DifficultyTier,
                      cfg: Config) -> list[ChartNote]:
    """Recalcula sustains e HOPOs depois dos cortes.

    Obrigatorio: quando uma nota e removida, a anterior passa a ter um buraco
    maior depois dela. Sem recalcular, ou a nota anterior fica com um sustain
    curto no meio de um vazio, ou - pior - um sustain herdado do Expert passa
    por cima da proxima nota que sobrou.
    """
    if not notes:
        return notes
    res = tempo.resolution
    cutoff = res // 3
    for i, note in enumerate(notes):
        if not tier.allow_sustains:
            note.sustain_ticks = 0
            continue
        gap = (notes[i + 1].tick - note.tick) if i + 1 < len(notes) else \
            int(cfg.mapper.sustain_min_beats * res)
        if gap >= cfg.mapper.sustain_min_beats * res:
            sustain = int(min(gap * cfg.mapper.sustain_ratio,
                              cfg.mapper.sustain_max_beats * res))
            note.sustain_ticks = sustain if sustain >= cutoff else 0
        else:
            note.sustain_ticks = 0

    if cfg.mapper.hopo_enabled:
        from .writer_mid import compute_natural_hopo
        threshold = cfg.mapper.hopo_threshold_ticks if cfg.mapper.hopo_threshold_ticks \
            is not None else res // 3 + 1
        for note, hopo in zip(notes, compute_natural_hopo(notes, threshold)):
            note.hopo = hopo
    return notes

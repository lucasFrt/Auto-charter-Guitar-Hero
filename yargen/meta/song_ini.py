"""Geracao do song.ini.

O .mid nao carrega metadado nenhum: titulo, artista e duracao vivem aqui.
Tags e semantica conferidas contra
docs/Chart-File-Formats/song-ini/Standard-Tags.md.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config, DEFAULT
from ..chart.ir import ChartIR, DIFFICULTIES

# Ordem em que as tags saem no arquivo. Puramente cosmetica, mas um song.ini
# com ordem estavel e muito mais facil de comparar em diff entre rodadas.
_ORDER = [
    "name", "artist", "album", "genre", "year", "charter",
    "song_length", "preview_start_time", "preview_end_time",
    "diff_band", "diff_guitar", "diff_bass", "diff_vocals",
    "icon", "delay", "hopo_frequency", "sustain_cutoff_threshold",
    "loading_phrase",
]


def estimate_difficulty(ir: ChartIR, instrument: str) -> int:
    """Nota de dificuldade 0..6 a partir da densidade do Expert.

    E so para o jogo ter o que mostrar na lista de musicas; a escala da cena e
    frouxa mesmo. Densidade e o unico sinal barato e honesto que temos.
    """
    chart = ir.instruments.get(instrument)
    if chart is None:
        return -1
    notes = chart.notes.get("expert") or chart.notes.get("hard") or []
    if not notes or ir.meta.duration <= 0:
        return -1
    nps = len(notes) / ir.meta.duration
    for threshold, value in ((1.0, 0), (2.0, 1), (3.5, 2), (5.0, 3), (6.5, 4), (8.0, 5)):
        if nps < threshold:
            return value
    return 6


def build_tags(ir: ChartIR, cfg: Config = DEFAULT, *,
               extra: dict[str, object] | None = None) -> dict[str, object]:
    meta = ir.meta
    duration_ms = int(round(meta.duration * 1000))

    tags: dict[str, object] = {
        "name": meta.name,
        "artist": meta.artist,
        "album": meta.album,
        "genre": meta.genre,
        "year": meta.year,
        "charter": meta.charter or cfg.output.charter,
        "song_length": duration_ms,
        "preview_start_time": int(duration_ms * cfg.output.preview_start_frac),
        "icon": cfg.output.icon,
        "delay": cfg.output.offset_ms,
        "loading_phrase": "Chart gerado automaticamente pelo YARGen - revise antes de jogar a serio.",
    }

    if "PART GUITAR" in ir.instruments:
        tags["diff_guitar"] = estimate_difficulty(ir, "PART GUITAR")
    if "PART BASS" in ir.instruments:
        tags["diff_bass"] = estimate_difficulty(ir, "PART BASS")
    if ir.vocals is not None and ir.vocals.notes:
        # Vocals nao tem densidade comparavel a guitarra; 3 e o meio da escala
        # e nao mente mais do que qualquer outro palpite.
        tags["diff_vocals"] = 3

    diffs = [v for k, v in tags.items() if k.startswith("diff_") and isinstance(v, int) and v >= 0]
    tags["diff_band"] = max(diffs) if diffs else -1

    # So gravamos hopo_frequency quando sobrepomos o limiar padrao. Gravar o
    # valor padrao seria redundante e mudaria o hash do chart a toa.
    if cfg.mapper.hopo_threshold_ticks is not None:
        tags["hopo_frequency"] = cfg.mapper.hopo_threshold_ticks

    if extra:
        tags.update(extra)
    return tags


def render(tags: dict[str, object]) -> str:
    lines = ["[song]"]
    seen = set()
    for key in _ORDER:
        if key in tags:
            lines.append(f"{key} = {_fmt(tags[key])}")
            seen.add(key)
    for key in sorted(k for k in tags if k not in seen):
        lines.append(f"{key} = {_fmt(tags[key])}")
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def write_song_ini(ir: ChartIR, path: str | Path, cfg: Config = DEFAULT,
                   *, extra: dict[str, object] | None = None) -> dict[str, object]:
    tags = build_tags(ir, cfg, extra=extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: varios parsers da cena esperam BOM em song.ini com acento.
    path.write_text(render(tags), encoding="utf-8-sig")
    return tags

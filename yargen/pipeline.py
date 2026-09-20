"""Orquestracao: audio -> IR de analise -> IR de chart -> pasta da musica.

Os tres estagios sao funcoes separadas de proposito. `analyze` e o caro
(Demucs, pyin, whisper); `build_chart` e barato e puro. Quem esta ajustando o
mapper roda `analyze` uma vez, guarda a IR, e depois so re-roda `build_chart`.
E o que a flag --from-ir da CLI faz.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .analysis import onsets as onsets_mod
from .analysis import pitch as pitch_mod
from .analysis import tempo as tempo_mod
from .audio import loader, stems as stems_mod
from .chart import difficulty as difficulty_mod
from .chart import mapper as mapper_mod
from .chart import vocals as vocals_mod
from .chart.ir import (AnalysisIR, AnalyzedTrack, ChartIR, SongMeta, TempoMap)
from .chart.writer_mid import write_midi
from .config import DEFAULT, Config
from .meta.song_ini import write_song_ini

INSTRUMENT_TRACKS = {"guitar": "PART GUITAR", "bass": "PART BASS"}

Log = Callable[[str], None]


def _noop(msg: str) -> None:
    pass


@dataclass
class ChartResult:
    out_dir: Path
    analysis: AnalysisIR
    chart: ChartIR
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ analise

def analyze(audio_path: str | Path, cfg: Config = DEFAULT, *,
            instruments: Sequence[str] = ("guitar",),
            bpm: float | None = None, use_stems: bool = True,
            meta: SongMeta | None = None, log: Log = _noop) -> AnalysisIR:
    """Roda o DSP e devolve a IR de analise."""
    audio_path = Path(audio_path)
    log(f"[audio] carregando {audio_path.name}")
    mix = loader.load(audio_path, cfg.audio)

    song_meta = meta or SongMeta()
    song_meta.duration = mix.duration
    song_meta.charter = song_meta.charter or cfg.output.charter

    log("[tempo] beat tracking")
    grid = tempo_mod.detect(mix, cfg.tempo, bpm_override=bpm)
    log(f"[tempo] {grid.average_bpm:.1f} BPM, {len(grid.beats)} beats"
        + (" (BPM fornecido)" if bpm else ""))

    stem_set: stems_mod.StemSet | None = None
    if use_stems and cfg.stems.enabled:
        try:
            stem_set = stems_mod.separate(audio_path, cfg.stems, verbose=False)
            log(f"[stems] {stem_set.model}: {', '.join(stem_set.names)}"
                + (" (cache)" if stem_set.from_cache else ""))
        except stems_mod.DemucsUnavailable as exc:
            log(f"[stems] indisponivel: {exc}")
        except Exception as exc:
            log(f"[stems] falhou ({exc}); usando a mixagem completa")

    ir = AnalysisIR(meta=song_meta, tempo=grid)

    for instrument in instruments:
        if instrument == "vocals":
            continue
        source, audio = _source_for(instrument, stem_set, mix, cfg, log)
        log(f"[{instrument}] onsets em '{source}'")
        notes = onsets_mod.detect(audio, cfg.onset)
        log(f"[{instrument}] {len(notes)} onsets")
        if notes:
            log(f"[{instrument}] contorno de f0 ({cfg.pitch.method})")
            contour = pitch_mod.contour(audio, cfg.pitch)
            pitch_mod.annotate(notes, contour, cfg.pitch)
            got = sum(1 for n in notes if n.pitch_hz)
            log(f"[{instrument}] altura em {got}/{len(notes)} notas")
        ir.tracks[instrument] = AnalyzedTrack(instrument, notes, source_stem=source)

    if "vocals" in instruments and cfg.vocals.enabled:
        ir.tracks["vocals"] = _analyze_vocals(stem_set, mix, cfg, log)

    return ir


def _source_for(instrument: str, stem_set, mix, cfg: Config, log: Log):
    """Escolhe o stem de onde o instrumento vai ser charteado."""
    if stem_set is None:
        return "mix", mix
    if instrument == "guitar":
        name = stem_set.pick_guitar(cfg.stems.guitar_source)
        if name == "other":
            log("[guitar] modelo sem stem de guitarra: usando 'other', "
                "que mistura guitarra com teclado e sintetizador")
    else:
        name = instrument if instrument in stem_set else None
    if name is None:
        return "mix", mix
    return name, stem_set.load(name, cfg.audio.sample_rate)


def _analyze_vocals(stem_set, mix, cfg: Config, log: Log) -> AnalyzedTrack:
    if stem_set is not None and "vocals" in stem_set:
        source = "vocals"
        audio = stem_set.load("vocals", cfg.audio.sample_rate)
        transcribe_path = str(stem_set.paths["vocals"])
    else:
        source = "mix"
        audio = mix
        transcribe_path = str(mix.path) if mix.path else ""
        log("[vocals] sem stem vocal: transcrevendo a mixagem, a letra vai sair pior")

    log("[vocals] contorno de f0")
    contour = pitch_mod.contour(audio, cfg.pitch)

    words = []
    if transcribe_path:
        log(f"[vocals] transcrevendo com faster-whisper ({cfg.vocals.whisper_model})")
        try:
            words = vocals_mod.transcribe(transcribe_path, cfg.vocals)
        except vocals_mod.TranscriptionUnavailable as exc:
            log(f"[vocals] sem transcricao ({exc})")
    if words:
        vocals_mod.attach_pitch(words, contour, cfg.vocals)
        log(f"[vocals] {len(words)} palavras")
    else:
        log("[vocals] sem letra: segmentando o contorno de f0 em notas")
        words = vocals_mod.words_from_contour(contour, cfg.vocals)
        log(f"[vocals] {len(words)} notas sem letra")

    return AnalyzedTrack("vocals", [], words, source_stem=source)


# ------------------------------------------------------------------- chart

def build_chart(ir: AnalysisIR, cfg: Config = DEFAULT, *,
                difficulties: Sequence[str] = difficulty_mod.ORDER,
                log: Log = _noop) -> ChartIR:
    """Analise -> chart. Puro: nenhum audio e tocado aqui.

    E esta funcao, e so ela, que voce re-roda ao ajustar um parametro do
    mapper.
    """
    if ir.tempo is None:
        raise ValueError("IR de analise sem grade de tempo")
    grid = ir.tempo
    chart = ChartIR(meta=ir.meta, tempo=grid)

    for name, track in ir.tracks.items():
        if name == "vocals":
            continue
        track_name = INSTRUMENT_TRACKS.get(name, f"PART {name.upper()}")
        use_pitch = any(n.pitch_hz for n in track.notes)
        if not use_pitch and track.notes:
            log(f"[{name}] nenhuma altura detectada: trastes pseudoaleatorios "
                "respeitando o limite de salto")
        instrument = mapper_mod.map_track(track.notes, grid, cfg, use_pitch=use_pitch)
        instrument.track_name = track_name
        difficulty_mod.reduce_all(instrument, grid, cfg, difficulties=difficulties)
        chart.instruments[track_name] = instrument
        log(f"[{name}] " + ", ".join(f"{d}={len(v)}"
                                     for d, v in instrument.notes.items()))

    vocal_track = ir.tracks.get("vocals")
    if vocal_track is not None and vocal_track.words:
        chart.vocals = vocals_mod.build(vocal_track.words, grid, cfg)
        log(f"[vocals] {len(chart.vocals.notes)} notas, "
            f"{len(chart.vocals.phrases)} frases")

    return chart


# ------------------------------------------------------------------- saida

def write_song_folder(chart: ChartIR, out_dir: str | Path, audio_path: str | Path,
                      cfg: Config = DEFAULT, *, log: Log = _noop) -> Path:
    """Escreve notes.mid + song.ini + audio numa pasta que o YARG reconhece."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_midi(chart, out / "notes.mid", cfg)
    write_song_ini(chart, out / "song.ini", cfg)

    if cfg.output.copy_audio:
        src = Path(audio_path)
        # O formato reserva nomes especificos de arquivo de audio (song,
        # guitar, bass, drums, vocals...). Um nome arbitrario simplesmente nao
        # carrega, entao renomeamos na copia.
        dest = out / f"{cfg.output.audio_stem_name}{src.suffix.lower()}"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        if src.suffix.lower() == ".mp3":
            log("[saida] audio em .mp3: o padding do encoder desloca a musica "
                "em alguns ms. Se as notas soarem adiantadas ou atrasadas, "
                "ajuste com --offset-ms (gravado em `delay` no song.ini), "
                "ou converta para .ogg.")
    log(f"[saida] {out}")
    return out


def run(audio_path: str | Path, out_dir: str | Path, cfg: Config = DEFAULT, *,
        instruments: Sequence[str] = ("guitar",),
        difficulties: Sequence[str] = difficulty_mod.ORDER,
        bpm: float | None = None, use_stems: bool = True,
        meta: SongMeta | None = None, keep_ir: bool = False,
        log: Log = _noop) -> ChartResult:
    """Pipeline completo, do arquivo de audio a pasta pronta."""
    ir = analyze(audio_path, cfg, instruments=instruments, bpm=bpm,
                 use_stems=use_stems, meta=meta, log=log)
    chart = build_chart(ir, cfg, difficulties=difficulties, log=log)
    out = write_song_folder(chart, out_dir, audio_path, cfg, log=log)

    if keep_ir:
        ir.save(out / "yargen.analysis.json")
        chart.save(out / "yargen.chart.json")
        (out / "yargen.config.json").write_text(cfg.to_json(), encoding="utf-8")
        log("[saida] IR e config gravadas para inspecao")

    return ChartResult(out, ir, chart)

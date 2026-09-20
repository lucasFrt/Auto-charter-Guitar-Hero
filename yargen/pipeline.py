"""Orquestracao: audio -> IR de analise -> IR de chart -> pasta da musica.

O pipeline NAO assume mais que a trilha de 5 trastes vem da guitarra. Ele
segue um PLANO DE ROTEAMENTO (yargen/profiles.py) que diz, para cada trilha do
jogo, de qual stem ela sai e com qual estrategia. Rock charteia a guitarra;
trap charteia a batida; pop charteia o teclado. E a mesma maquinaria.

Os estagios sao funcoes separadas de proposito. `analyze` e o caro (Demucs,
pyin, whisper); `build_chart` e barato e puro. Quem esta ajustando o mapper
roda `analyze` uma vez e depois so `build_chart`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from . import profiles
from .analysis import onsets as onsets_mod
from .analysis import pitch as pitch_mod
from .analysis import survey as survey_mod
from .analysis import tempo as tempo_mod
from .analysis import timbre as timbre_mod
from .audio import loader, stems as stems_mod
from .chart import difficulty as difficulty_mod
from .chart import mapper as mapper_mod
from .chart import vocals as vocals_mod
from .chart.ir import AnalysisIR, AnalyzedTrack, ChartIR, SongMeta, TempoMap
from .chart.writer_mid import write_midi
from .config import DEFAULT, Config
from .meta.song_ini import write_song_ini
from .profiles import PERCUSSIVE, PITCH, VOCAL, TrackPlan

Log = Callable[[str], None]


def _noop(msg: str) -> None:
    pass


@dataclass
class ChartResult:
    out_dir: Path
    analysis: AnalysisIR
    chart: ChartIR
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ plano

def resolve_plans(genre: str, instruments: Sequence[str]) -> list[TrackPlan]:
    """Perfil de genero + instrumentos pedidos -> plano de roteamento."""
    profile = profiles.get(genre)
    plans = profile.for_instruments(instruments)
    if not plans:
        raise ValueError(
            f"o perfil '{profile.name}' nao produz nenhuma das trilhas pedidas "
            f"({', '.join(instruments)}). Ele cobre: "
            f"{', '.join(p.instrument for p in profile.plans)}")
    return plans


def _apply(cfg: Config, overrides: dict) -> Config:
    for key, value in overrides.items():
        cfg = cfg.override_path(key, value)
    return cfg


# ------------------------------------------------------------------ analise

def inspect(audio_path: str | Path, cfg: Config = DEFAULT,
            log: Log = _noop) -> survey_mod.Survey:
    """Mede cada stem e recomenda um plano, sem gerar chart nenhum.

    Existe porque a escolha da fonte nao pode ser automatizada com honestidade:
    ela e musical. O que da para fazer e mostrar o que a musica tem e explicar
    a recomendacao, para quem decide decidir sabendo.
    """
    import numpy as np

    audio_path = Path(audio_path)
    mix = loader.load(audio_path, cfg.audio)
    result = survey_mod.Survey(duration=mix.duration)

    stem_set = None
    if cfg.stems.enabled:
        try:
            stem_set = stems_mod.separate(audio_path, cfg.stems, verbose=False)
            log(f"[stems] {stem_set.model}: {', '.join(stem_set.names)}"
                + (" (cache)" if stem_set.from_cache else ""))
        except Exception as exc:
            log(f"[stems] indisponivel ({exc}); medindo so a mixagem")

    if stem_set is None:
        # Sem stems da para dizer pouco: fica o levantamento da mixagem e o
        # padrao conservador. Melhor do que recomendar com confianca falsa.
        result.stems["mix"] = survey_mod.measure(
            "mix", mix, survey_mod.weighted_energy(mix), cfg)
        result.recommended = "rock"
        result.reasons = ["sem separacao de stems: nao da para medir os "
                          "instrumentos separadamente; rock e o padrao"]
        return result

    # normalize=False: normalizar cada stem apaga o balanco da mixagem, que
    # e exatamente o que precisamos medir aqui.
    loaded = {name: stem_set.load(name, cfg.audio.sample_rate, normalize=False)
              for name in stem_set.names}
    total = sum(survey_mod.weighted_energy(a) for a in loaded.values()) or 1.0
    reference = max((survey_mod.rms_peak(a) for a in loaded.values()), default=0.0)
    for name, audio in loaded.items():
        log(f"[inspect] medindo '{name}'")
        result.stems[name] = survey_mod.measure(name, audio, total, cfg,
                                                reference_peak=reference)

    return survey_mod.recommend(result)


def analyze(audio_path: str | Path, cfg: Config = DEFAULT, *,
            genre: str = "rock", instruments: Sequence[str] = ("guitar",),
            bpm: float | None = None, use_stems: bool = True,
            meta: SongMeta | None = None, log: Log = _noop) -> AnalysisIR:
    """Roda o DSP seguindo o plano do genero e devolve a IR de analise."""
    audio_path = Path(audio_path)
    plans = resolve_plans(genre, instruments)

    log(f"[audio] carregando {audio_path.name}")
    mix = loader.load(audio_path, cfg.audio)

    song_meta = meta or SongMeta()
    song_meta.duration = mix.duration
    song_meta.charter = song_meta.charter or cfg.output.charter
    song_meta.genre = song_meta.genre or profiles.get(genre).name

    log("[tempo] beat tracking")
    grid = tempo_mod.detect(mix, cfg.tempo, bpm_override=bpm)
    log(f"[tempo] {grid.average_bpm:.1f} BPM, {len(grid.beats)} beats"
        + (" (BPM fornecido)" if bpm else ""))

    stem_set = None
    if use_stems and cfg.stems.enabled:
        try:
            stem_set = stems_mod.separate(audio_path, cfg.stems, verbose=False)
            log(f"[stems] {stem_set.model}: {', '.join(stem_set.names)}"
                + (" (cache)" if stem_set.from_cache else ""))
        except stems_mod.DemucsUnavailable as exc:
            log(f"[stems] indisponivel: {exc}")
        except Exception as exc:
            log(f"[stems] falhou ({exc}); usando a mixagem completa")

    ir = AnalysisIR(meta=song_meta, tempo=grid, genre=profiles.get(genre).name)

    for plan in plans:
        source, audio = _pick_source(plan, stem_set, mix, log)
        log(f"[{plan.track}] <- '{source}' [{plan.strategy}]  {plan.note}")

        if plan.strategy == VOCAL:
            ir.tracks[plan.instrument] = _analyze_vocals(
                plan, source, audio, stem_set, mix, cfg, log)
            continue

        track_cfg = _apply(cfg, plan.overrides)

        if plan.strategy == PERCUSSIVE:
            # Deteccao por banda: bumbo e chimbal simultaneos precisam ser
            # dois eventos, nao um evento borrado. Ver onsets.detect_percussive.
            notes = onsets_mod.detect_percussive(audio, track_cfg.onset)
            log(f"[{plan.track}] {len(notes)} onsets em "
                f"{len(onsets_mod.PERCUSSIVE_BANDS)} bandas de frequencia")
        else:
            notes = onsets_mod.detect(audio, track_cfg.onset)
            log(f"[{plan.track}] {len(notes)} onsets")

        if notes and plan.strategy == PERCUSSIVE:
            pass  # o brilho ja veio da banda em que o onset foi detectado
        elif notes:
            log(f"[{plan.track}] contorno de f0 ({track_cfg.pitch.method})")
            contour = pitch_mod.contour(audio, track_cfg.pitch)
            pitch_mod.annotate(notes, contour, track_cfg.pitch)
            got = sum(1 for n in notes if n.pitch_hz)
            log(f"[{plan.track}] altura em {got}/{len(notes)} notas")
            if got < len(notes) * 0.25:
                log(f"[{plan.track}] pouca altura detectada: esta fonte pode ser "
                    "percussiva. Considere --genre trap/percussion, ou "
                    "--set-track para forcar a estrategia percussiva.")

        ir.tracks[plan.instrument] = AnalyzedTrack(
            plan.instrument, notes, source_stem=source, track_name=plan.track,
            strategy=plan.strategy, overrides=dict(plan.overrides))

    return ir


def _pick_source(plan: TrackPlan, stem_set, mix, log: Log):
    """Primeiro stem disponivel da lista de preferencia do plano."""
    if stem_set is not None:
        for name in plan.sources:
            if name == "mix":
                break
            if name in stem_set:
                return name, stem_set.load(name, DEFAULT.audio.sample_rate)
        log(f"[{plan.track}] nenhum stem de {'/'.join(plan.sources)} disponivel")
    return "mix", mix


def _analyze_vocals(plan: TrackPlan, source: str, audio, stem_set, mix,
                    cfg: Config, log: Log) -> AnalyzedTrack:
    transcribe_path = ""
    if stem_set is not None and source in stem_set:
        transcribe_path = str(stem_set.paths[source])
    elif mix.path:
        transcribe_path = str(mix.path)
        log("[PART VOCALS] sem stem vocal: transcrevendo a mixagem, "
            "a letra vai sair pior")

    log("[PART VOCALS] contorno de f0")
    contour = pitch_mod.contour(audio, cfg.pitch)

    words = []
    if transcribe_path:
        log(f"[PART VOCALS] transcrevendo com faster-whisper "
            f"({cfg.vocals.whisper_model})")
        try:
            words = vocals_mod.transcribe(transcribe_path, cfg.vocals)
        except vocals_mod.TranscriptionUnavailable as exc:
            log(f"[PART VOCALS] sem transcricao ({exc})")
    if words:
        vocals_mod.attach_pitch(words, contour, cfg.vocals)
        log(f"[PART VOCALS] {len(words)} palavras")
    else:
        log("[PART VOCALS] sem letra: segmentando o contorno de f0 em notas")
        words = vocals_mod.words_from_contour(contour, cfg.vocals)
        log(f"[PART VOCALS] {len(words)} notas sem letra")

    return AnalyzedTrack(plan.instrument, [], words, source_stem=source,
                         track_name=plan.track, strategy=VOCAL,
                         overrides=dict(plan.overrides))


# ------------------------------------------------------------------- chart

def build_chart(ir: AnalysisIR, cfg: Config = DEFAULT, *,
                difficulties: Sequence[str] = difficulty_mod.ORDER,
                log: Log = _noop) -> ChartIR:
    """Analise -> chart. Puro: nenhum audio e tocado aqui.

    E esta funcao, e so ela, que voce re-roda ao ajustar um parametro.
    """
    if ir.tempo is None:
        raise ValueError("IR de analise sem grade de tempo")
    grid = ir.tempo
    chart = ChartIR(meta=ir.meta, tempo=grid)

    for track in ir.tracks.values():
        if track.strategy == VOCAL:
            continue
        track_cfg = _apply(cfg, track.overrides)
        track_name = track.track_name or profiles.INSTRUMENT_TRACKS.get(
            track.name, f"PART {track.name.upper()}")

        has_tone = any(n.centroid_hz for n in track.notes) \
            if track.strategy == PERCUSSIVE else any(n.pitch_hz for n in track.notes)
        if not has_tone and track.notes:
            log(f"[{track_name}] nenhum tom detectado: trastes pseudoaleatorios "
                "respeitando o limite de salto")

        instrument = mapper_mod.map_track(track.notes, grid, track_cfg,
                                          use_pitch=has_tone,
                                          strategy=track.strategy)
        instrument.track_name = track_name
        difficulty_mod.reduce_all(instrument, grid, track_cfg,
                                  difficulties=difficulties)
        chart.instruments[track_name] = instrument
        log(f"[{track_name}] ({track.strategy} <- {track.source_stem}) "
            + ", ".join(f"{d}={len(v)}" for d, v in instrument.notes.items()))

    for track in ir.tracks.values():
        if track.strategy == VOCAL and track.words:
            chart.vocals = vocals_mod.build(track.words, grid, cfg)
            log(f"[PART VOCALS] {len(chart.vocals.notes)} notas, "
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

    if cfg.output.copy_audio and audio_path:
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
        genre: str = "rock", instruments: Sequence[str] = ("guitar",),
        difficulties: Sequence[str] = difficulty_mod.ORDER,
        bpm: float | None = None, use_stems: bool = True,
        meta: SongMeta | None = None, keep_ir: bool = False,
        log: Log = _noop) -> ChartResult:
    """Pipeline completo, do arquivo de audio a pasta pronta."""
    ir = analyze(audio_path, cfg, genre=genre, instruments=instruments, bpm=bpm,
                 use_stems=use_stems, meta=meta, log=log)
    chart = build_chart(ir, cfg, difficulties=difficulties, log=log)
    out = write_song_folder(chart, out_dir, audio_path, cfg, log=log)

    if keep_ir:
        ir.save(out / "yargen.analysis.json")
        chart.save(out / "yargen.chart.json")
        (out / "yargen.config.json").write_text(cfg.to_json(), encoding="utf-8")
        log("[saida] IR e config gravadas para inspecao")

    return ChartResult(out, ir, chart)

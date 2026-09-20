"""CLI do YARGen.

argparse e nao click de proposito: uma dependencia a menos num projeto cuja
maior friccao de instalacao ja e o PyTorch.

Os subcomandos existem para separar o caro do barato. `chart` faz tudo;
`analyze` + `build` fazem o mesmo em dois passos, e e assim que se ajusta o
mapper - analisa uma vez, depois reconstroi o chart quantas vezes quiser em
segundos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import profiles
from .chart.ir import AnalysisIR, SongMeta
from .chart.difficulty import ORDER as DIFF_ORDER
from .config import DEFAULT, Config

KNOWN_INSTRUMENTS = ("guitar", "bass", "vocals")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ------------------------------------------------------------------ helpers

def _parse_list(value: str, known: tuple[str, ...], what: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(known)
    items = [x.strip().lower() for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in known]
    if bad:
        raise SystemExit(f"{what} desconhecido(s): {', '.join(bad)}. "
                         f"Validos: {', '.join(known)}")
    return items


def _load_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else DEFAULT
    for assignment in getattr(args, "set", None) or []:
        if "=" not in assignment:
            raise SystemExit(f"--set espera chave=valor, recebi {assignment!r}")
        key, raw = assignment.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw  # string simples, sem aspas
        try:
            cfg = cfg.override_path(key.strip(), value)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if getattr(args, "offset_ms", None) is not None:
        cfg = cfg.override_path("output.offset_ms", args.offset_ms)
    if getattr(args, "charter", None):
        cfg = cfg.override_path("output.charter", args.charter)
    return cfg


def _apply_track_overrides(plans, assignments: list[str] | None):
    """`--set-track guitar.source=other` / `guitar.strategy=percussive`.

    Escape para quando o perfil de genero quase acerta. Uma musica de rock com
    um solo de teclado, um beat com guitarra de amostra - casos reais que nao
    cabem num rotulo de genero e nao deveriam exigir um perfil novo.
    """
    if not assignments:
        return plans
    by_instrument = {p.instrument: p for p in plans}
    for assignment in assignments:
        if "=" not in assignment or "." not in assignment.split("=", 1)[0]:
            raise SystemExit(f"--set-track espera instrumento.campo=valor, "
                             f"recebi {assignment!r}")
        path, value = assignment.split("=", 1)
        instrument, field = path.split(".", 1)
        plan = by_instrument.get(instrument.strip().lower())
        if plan is None:
            raise SystemExit(f"--set-track: o plano atual nao tem a trilha "
                             f"{instrument!r}. Tem: "
                             f"{', '.join(sorted(by_instrument))}")
        field = field.strip().lower()
        if field == "source":
            plan.sources = [value.strip()] + [s for s in plan.sources
                                              if s != value.strip()]
            plan.note = f"fonte forcada para '{value.strip()}'"
        elif field == "strategy":
            if value.strip() not in profiles.STRATEGIES:
                raise SystemExit(f"--set-track: estrategia invalida "
                                 f"{value.strip()!r}. Validas: "
                                 f"{', '.join(profiles.STRATEGIES)}")
            plan.strategy = value.strip()
            plan.note = f"estrategia forcada para '{value.strip()}'"
        else:
            raise SystemExit(f"--set-track: campo desconhecido {field!r}. "
                             "Use 'source' ou 'strategy'.")
    return plans


def _meta_from_args(args: argparse.Namespace, audio: Path) -> SongMeta:
    """Metadados do song.ini, com um palpite a partir do nome do arquivo.

    "Artista - Musica.mp3" e a convencao mais comum em biblioteca de musica, e
    adivinhar acerta o suficiente para nao obrigar o usuario a digitar os dois
    campos toda vez. Quando erra, --name e --artist sobrepoem.
    """
    stem = audio.stem
    artist, name = "", stem
    if " - " in stem:
        artist, name = (p.strip() for p in stem.split(" - ", 1))
    return SongMeta(name=args.name or name, artist=args.artist or artist,
                    album=args.album or "",
                    genre=getattr(args, "genre_tag", None) or "",
                    year=args.year or "")


def _default_out(meta: SongMeta) -> str:
    safe = lambda s: "".join(c for c in s if c not in '<>:"/\\|?*').strip()
    return f"{safe(meta.artist)} - {safe(meta.name)}" if meta.artist else safe(meta.name)


# ----------------------------------------------------------------- comandos

def cmd_chart(args: argparse.Namespace) -> int:
    from . import pipeline

    cfg = _load_config(args)
    audio = Path(args.audio)
    if not audio.is_file():
        raise SystemExit(f"arquivo nao encontrado: {audio}")

    meta = _meta_from_args(args, audio)
    out = Path(args.out) if args.out else Path(_default_out(meta))
    instruments = _parse_list(args.instruments, KNOWN_INSTRUMENTS, "instrumento")
    difficulties = _parse_list(args.difficulties, DIFF_ORDER, "dificuldade")

    try:
        plans = pipeline.resolve_plans(args.genre, instruments)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _apply_track_overrides(plans, args.set_track)

    ir = pipeline.analyze(audio, cfg, genre=args.genre, instruments=instruments,
                          bpm=args.bpm, use_stems=not args.no_stems, meta=meta,
                          log=log)
    # Reaplica os overrides de --set-track, que a analise ja consumiu, para
    # que a IR gravada reproduza exatamente esta rodada.
    chart = pipeline.build_chart(ir, cfg, difficulties=difficulties, log=log)
    out_dir = pipeline.write_song_folder(chart, out, audio, cfg, log=log)
    if args.keep_ir:
        ir.save(out_dir / "yargen.analysis.json")
        chart.save(out_dir / "yargen.chart.json")
        (out_dir / "yargen.config.json").write_text(cfg.to_json(), encoding="utf-8")
        log("[saida] IR e config gravadas para inspecao")

    for key, count in chart.note_count().items():
        log(f"  {key}: {count}")
    print(out_dir)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Mede os stems e recomenda um plano, sem gerar chart.

    A escolha de qual fonte conduz o chart e musical e nao da para
    automatizar com honestidade. Este comando existe para a escolha ser
    barata e informada em vez de um chute sobre o genero.
    """
    from . import pipeline
    from .analysis.survey import render

    cfg = _load_config(args)
    audio = Path(args.audio)
    if not audio.is_file():
        raise SystemExit(f"arquivo nao encontrado: {audio}")
    result = pipeline.inspect(audio, cfg, log=log)
    print(render(result))
    print(f"\n  yargen chart {audio.name!r} --genre {result.recommended} "
          "--instruments all")
    return 0


def cmd_genres(args: argparse.Namespace) -> int:
    for profile in profiles.canonical():
        aliases = f"  (tambem: {', '.join(profile.aliases)})" if profile.aliases else ""
        print(f"{profile.name}{aliases}")
        print(f"  {profile.description}")
        for plan in profile.plans:
            print(f"    {plan.track:13} <- {'/'.join(plan.sources):22} "
                  f"[{plan.strategy}]  {plan.note}")
        print()
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from . import pipeline

    cfg = _load_config(args)
    audio = Path(args.audio)
    if not audio.is_file():
        raise SystemExit(f"arquivo nao encontrado: {audio}")
    instruments = _parse_list(args.instruments, KNOWN_INSTRUMENTS, "instrumento")
    try:
        pipeline.resolve_plans(args.genre, instruments)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    ir = pipeline.analyze(audio, cfg, genre=args.genre, instruments=instruments,
                          bpm=args.bpm, use_stems=not args.no_stems,
                          meta=_meta_from_args(args, audio), log=log)
    out = Path(args.out or "yargen.analysis.json")
    ir.save(out)
    print(out)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """IR de analise -> pasta da musica, sem tocar em audio.

    O comando que voce roda dezenas de vezes seguidas ao ajustar o mapper.
    """
    from . import pipeline

    cfg = _load_config(args)
    ir = AnalysisIR.load(args.ir)
    difficulties = _parse_list(args.difficulties, DIFF_ORDER, "dificuldade")
    chart = pipeline.build_chart(ir, cfg, difficulties=difficulties, log=log)

    out = Path(args.out) if args.out else Path(_default_out(ir.meta))
    if args.audio:
        pipeline.write_song_folder(chart, out, args.audio, cfg, log=log)
    else:
        cfg = cfg.override_path("output.copy_audio", False)
        pipeline.write_song_folder(chart, out, "", cfg, log=log)
        log("[saida] sem --audio: copie o arquivo de audio para a pasta "
            f"como '{cfg.output.audio_stem_name}.ogg' antes de jogar")
    for key, count in chart.note_count().items():
        log(f"  {key}: {count}")
    print(out)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .validate.compare import compare_files

    scores = compare_files(args.generated, args.reference, args.difficulty,
                           args.tolerance_ms)
    print(scores)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    print(cfg.to_json())
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    from .audio.stems import clear_cache
    from .config import DEFAULT as D

    if args.action == "clear":
        n = clear_cache(D.stems)
        print(f"{n} entradas removidas de {D.stems.cache_dir}")
    else:
        root = Path(D.stems.cache_dir).expanduser()
        entries = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
        print(f"{root}: {len(entries)} entradas")
        for entry in entries:
            print(f"  {entry}")
    return 0


# -------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yargen",
        description="Gera charts para YARG / Clone Hero a partir de um arquivo de audio.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="arquivo JSON de config (veja `yargen config`)")
        p.add_argument("--set", action="append", metavar="CHAVE=VALOR",
                       help="sobrepoe um parametro, ex: --set mapper.max_jump=3")

    def add_meta(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name"), p.add_argument("--artist")
        p.add_argument("--album")
        # `--genre` e o perfil de roteamento; a tag de metadado do song.ini e
        # outra coisa e ganha nome proprio. Quando nao for dada, o nome do
        # perfil preenche a tag, que e o que se quer em quase todo caso.
        p.add_argument("--genre-tag", dest="genre_tag",
                       help="valor da tag `genre` no song.ini "
                            "(padrao: o nome do perfil de --genre)")
        p.add_argument("--year")

    c = sub.add_parser("chart", help="audio -> pasta pronta para o YARG")
    c.add_argument("audio")
    c.add_argument("--out", help='pasta de saida (padrao: "Artista - Musica/")')
    c.add_argument("--genre", default="rock",
                   help="perfil de roteamento fonte->trilha (padrao: rock). "
                        "Veja `yargen genres` e `yargen inspect`")
    c.add_argument("--instruments", default="all",
                   help="guitar,bass,vocals ou all (padrao: all)")
    c.add_argument("--set-track", action="append", metavar="TRILHA.CAMPO=VALOR",
                   help="sobrepoe o roteamento, ex: --set-track "
                        "guitar.source=other --set-track guitar.strategy=percussive")
    c.add_argument("--difficulties", default="all",
                   help="expert,hard,medium,easy ou all (padrao: all)")
    c.add_argument("--bpm", type=float, help="sobrepoe o BPM detectado")
    c.add_argument("--offset-ms", type=int,
                   help="atraso em ms gravado em `delay` no song.ini")
    c.add_argument("--charter")
    c.add_argument("--no-stems", action="store_true",
                   help="charteia a mixagem completa, sem Demucs")
    c.add_argument("--keep-ir", action="store_true",
                   help="grava as IRs e a config em JSON na pasta de saida")
    add_meta(c), add_common(c)
    c.set_defaults(func=cmd_chart)

    a = sub.add_parser("analyze", help="audio -> IR de analise (o passo caro)")
    a.add_argument("audio")
    a.add_argument("--out", help="padrao: yargen.analysis.json")
    a.add_argument("--genre", default="rock")
    a.add_argument("--instruments", default="all")
    a.add_argument("--bpm", type=float)
    a.add_argument("--no-stems", action="store_true")
    add_meta(a), add_common(a)
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("build", help="IR de analise -> chart (rapido, sem audio)")
    b.add_argument("ir")
    b.add_argument("--out")
    b.add_argument("--audio", help="audio para copiar para a pasta")
    b.add_argument("--difficulties", default="all")
    b.add_argument("--offset-ms", type=int)
    b.add_argument("--charter")
    add_common(b)
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate",
                       help="precisao/recall dos onsets contra um chart humano")
    v.add_argument("generated"), v.add_argument("reference")
    v.add_argument("--difficulty", default="expert", choices=list(DIFF_ORDER))
    v.add_argument("--tolerance-ms", type=float, default=50.0)
    v.set_defaults(func=cmd_validate)

    i = sub.add_parser("inspect",
                       help="mede os stems e recomenda um genero/roteamento")
    i.add_argument("audio")
    add_common(i)
    i.set_defaults(func=cmd_inspect)

    n = sub.add_parser("genres", help="lista os perfis e o que cada um roteia")
    n.set_defaults(func=cmd_genres)

    g = sub.add_parser("config", help="imprime a config efetiva em JSON")
    add_common(g)
    g.set_defaults(func=cmd_config)

    k = sub.add_parser("cache", help="cache de stems do Demucs")
    k.add_argument("action", choices=["list", "clear"], nargs="?", default="list")
    k.set_defaults(func=cmd_cache)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

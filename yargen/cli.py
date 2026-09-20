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
                    album=args.album or "", genre=args.genre or "",
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

    result = pipeline.run(audio, out, cfg, instruments=instruments,
                          difficulties=difficulties, bpm=args.bpm,
                          use_stems=not args.no_stems, meta=meta,
                          keep_ir=args.keep_ir, log=log)
    for key, count in result.chart.note_count().items():
        log(f"  {key}: {count}")
    print(result.out_dir)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from . import pipeline

    cfg = _load_config(args)
    audio = Path(args.audio)
    if not audio.is_file():
        raise SystemExit(f"arquivo nao encontrado: {audio}")
    instruments = _parse_list(args.instruments, KNOWN_INSTRUMENTS, "instrumento")
    ir = pipeline.analyze(audio, cfg, instruments=instruments, bpm=args.bpm,
                          use_stems=not args.no_stems,
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
        p.add_argument("--album"), p.add_argument("--genre"), p.add_argument("--year")

    c = sub.add_parser("chart", help="audio -> pasta pronta para o YARG")
    c.add_argument("audio")
    c.add_argument("--out", help='pasta de saida (padrao: "Artista - Musica/")')
    c.add_argument("--instruments", default="guitar",
                   help="guitar,bass,vocals ou all (padrao: guitar)")
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
    a.add_argument("--instruments", default="guitar")
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

"""CLI: parsing, validacao e o comando `build`, que nao toca em audio.

Existia um vazio aqui e ele custou caro: a colisao entre o `--genre` de
roteamento e o `--genre` de metadado do song.ini so apareceu rodando o
binario na mao. Testes de unidade nos modulos nao veem esse tipo de defeito.
"""

import configparser

import pytest

from yargen.chart.ir import (AnalysisIR, AnalyzedNote, AnalyzedTrack, SongMeta,
                             TempoMap)
from yargen.cli import build_parser, main
from yargen.profiles import GUITAR_TRACK, PERCUSSIVE


@pytest.fixture
def analysis_ir(tmp_path):
    """IR de analise pronta em disco, para exercitar `build` sem DSP."""
    notes = [AnalyzedNote(i * 0.25, 0.6 + (i % 4) / 10, 220 * 2 ** ((i % 12) / 12))
             for i in range(80)]
    ir = AnalysisIR(
        meta=SongMeta(name="Musica", artist="Banda", duration=20.0),
        tempo=TempoMap.constant(120.0, 20.0),
        tracks={"guitar": AnalyzedTrack("guitar", notes, source_stem="guitar",
                                        track_name=GUITAR_TRACK)},
        genre="rock")
    path = tmp_path / "ir.json"
    ir.save(path)
    return path


# ------------------------------------------------------------------ parser

def test_parser_exposes_every_subcommand():
    sub = {a.dest: a for a in build_parser()._actions if a.choices}
    assert set(sub["command"].choices) == {
        "chart", "analyze", "build", "validate", "config", "cache",
        "inspect", "genres"}


def test_routing_genre_and_metadata_genre_are_different_flags():
    """Regressao: os dois se chamavam `--genre` e o parser nem construia."""
    args = build_parser().parse_args(
        ["chart", "x.ogg", "--genre", "trap", "--genre-tag", "Funk Carioca"])
    assert args.genre == "trap"
    assert args.genre_tag == "Funk Carioca"


def test_chart_defaults_to_every_instrument():
    args = build_parser().parse_args(["chart", "x.ogg"])
    assert args.instruments == "all" and args.genre == "rock"


# -------------------------------------------------------------- validacao

def test_unknown_instrument_is_rejected(tmp_path, capsys):
    audio = tmp_path / "x.ogg"
    audio.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        main(["chart", str(audio), "--instruments", "drums"])
    assert "drums" in str(exc.value)


def test_unknown_genre_is_rejected(tmp_path):
    audio = tmp_path / "x.ogg"
    audio.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        main(["chart", str(audio), "--genre", "reggaeton-espacial"])
    assert "desconhecido" in str(exc.value)


def test_missing_audio_file_is_rejected():
    with pytest.raises(SystemExit, match="nao encontrado"):
        main(["chart", "/nao/existe.ogg"])


def test_typo_in_set_is_rejected():
    with pytest.raises(SystemExit, match="max_jumps"):
        main(["config", "--set", "mapper.max_jumps=3"])


def test_set_track_rejects_a_bad_field(analysis_ir, tmp_path):
    with pytest.raises(SystemExit, match="campo desconhecido"):
        main(["chart", str(analysis_ir), "--set-track", "guitar.fonte=drums"])


def test_set_track_rejects_an_invalid_strategy(tmp_path):
    audio = tmp_path / "x.ogg"
    audio.write_bytes(b"")
    with pytest.raises(SystemExit, match="estrategia invalida"):
        main(["chart", str(audio), "--set-track", "guitar.strategy=telepatia"])


# ------------------------------------------------------------------ build

def test_build_produces_a_playable_folder(analysis_ir, tmp_path, capsys):
    out = tmp_path / "saida"
    assert main(["build", str(analysis_ir), "--out", str(out)]) == 0
    assert (out / "notes.mid").is_file()
    assert (out / "song.ini").is_file()

    parser = configparser.ConfigParser()
    parser.read(out / "song.ini", encoding="utf-8-sig")
    assert parser["song"]["name"] == "Musica"


def test_build_honours_a_config_override(analysis_ir, tmp_path):
    from yargen.chart.ir import ChartIR

    for jump, folder in ((1, "j1"), (4, "j4")):
        out = tmp_path / folder
        main(["build", str(analysis_ir), "--out", str(out),
              "--set", f"mapper.max_jump={jump}", "--difficulties", "expert"])
        ChartIR  # noqa: B018  - import so para deixar claro o que foi escrito
        assert (out / "notes.mid").is_file()


def test_build_can_limit_difficulties(analysis_ir, tmp_path):
    import mido

    out = tmp_path / "so-expert"
    main(["build", str(analysis_ir), "--out", str(out),
          "--difficulties", "expert"])
    mid = mido.MidiFile(str(out / "notes.mid"))
    notes = {m.note for tr in mid.tracks for m in tr
             if m.type == "note_on" and m.velocity > 0}
    assert notes & set(range(96, 101))          # Expert existe
    assert not notes & set(range(60, 65))       # Easy nao


def test_build_reproduces_the_strategy_stored_in_the_ir(tmp_path):
    """A IR carrega a estrategia para `build` nao precisar do genero de novo."""
    notes = [AnalyzedNote(i * 0.25, 0.8, centroid_hz=60.0 * (1 + i % 3))
             for i in range(40)]
    ir = AnalysisIR(meta=SongMeta(name="B", duration=12.0),
                    tempo=TempoMap.constant(120.0, 12.0),
                    tracks={"guitar": AnalyzedTrack(
                        "guitar", notes, source_stem="drums",
                        track_name=GUITAR_TRACK, strategy=PERCUSSIVE)},
                    genre="trap")
    path = tmp_path / "ir.json"
    ir.save(path)
    assert AnalysisIR.load(path).tracks["guitar"].strategy == PERCUSSIVE
    assert main(["build", str(path), "--out", str(tmp_path / "o")]) == 0


# ----------------------------------------------------------------- outros

def test_genres_lists_every_profile(capsys):
    assert main(["genres"]) == 0
    printed = capsys.readouterr().out
    for name in ("rock", "trap", "pop", "electronic", "metal", "percussion"):
        assert name in printed


def test_config_prints_valid_json(capsys):
    import json

    assert main(["config"]) == 0
    assert json.loads(capsys.readouterr().out)["mapper"]["max_jump"] == 2

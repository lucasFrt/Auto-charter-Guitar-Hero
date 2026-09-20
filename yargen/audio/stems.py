"""Separacao de stems via Demucs, com cache agressivo.

O cache nao e otimizacao, e requisito. Uma musica de 4 minutos leva varios
minutos em CPU, e o loop de trabalho previsto para este projeto e "mexer num
parametro do mapper e rodar de novo" - dezenas de vezes. Sem cache, cada
iteracao paga a separacao inteira e o projeto fica inutilizavel.

Chave do cache: SHA-1 do conteudo do arquivo + nome do modelo. Conteudo e nao
caminho, para renomear a musica nao custar 6 minutos.

O Demucs e um import tardio e opcional: sem ele, o pipeline cai para a
mixagem completa e avisa.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DEFAULT, StemsConfig
from .loader import Audio, file_hash, from_samples

# Stems do htdemucs (4) e do htdemucs_6s (6).
FOUR_STEMS = ("drums", "bass", "other", "vocals")
SIX_STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")


class DemucsUnavailable(RuntimeError):
    pass


@dataclass
class StemSet:
    """Stems em disco, prontos para carregar sob demanda."""

    paths: dict[str, Path] = field(default_factory=dict)
    model: str = ""
    sample_rate: int = 44100
    from_cache: bool = False

    def __contains__(self, name: str) -> bool:
        return name in self.paths

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.paths))

    def load(self, name: str, target_sr: int, *, normalize: bool = True) -> Audio:
        """Carrega um stem.

        `normalize=False` e obrigatorio para QUALQUER medicao comparativa
        entre stems: normalizar cada um para pico 1.0 apaga o balanco da
        mixagem, que e exatamente a informacao que diz qual instrumento
        conduz a musica. Para analise de um stem isolado (onset, f0) a
        normalizacao ajuda e continua sendo o padrao.
        """
        import librosa

        from ..config import AudioConfig

        y, sr = librosa.load(str(self.paths[name]), sr=target_sr, mono=True)
        return from_samples(y, int(sr), AudioConfig(sample_rate=target_sr,
                                                    normalize=normalize))

    def pick_guitar(self, preference: str = "auto") -> str | None:
        """Qual stem usar como fonte da guitarra.

        O htdemucs padrao nao tem stem de guitarra: ela cai em `other` junto
        com teclado, orgao e sintetizador - exatamente o risco previsto no
        projeto. O modelo de 6 stems separa guitarra de verdade, e e por isso
        que ele e o padrao aqui, apesar de ser mais lento.
        """
        if preference not in ("auto", ""):
            return preference if preference in self.paths else None
        for candidate in ("guitar", "other"):
            if candidate in self.paths:
                return candidate
        return None


def cache_key(audio_path: str | Path, model: str) -> str:
    return f"{file_hash(audio_path)}-{model}"


def separate(audio_path: str | Path, cfg: StemsConfig = DEFAULT.stems,
             *, force: bool = False, verbose: bool = True) -> StemSet:
    """Separa `audio_path` em stems, reaproveitando o cache quando possivel."""
    audio_path = Path(audio_path).resolve()
    root = Path(cfg.cache_dir).expanduser()

    model = cfg.model
    key = cache_key(audio_path, model)
    target = root / key

    if target.is_dir() and not force:
        found = _collect(target)
        if found:
            if verbose:
                print(f"[stems] cache: {target} ({', '.join(sorted(found))})")
            return StemSet(found, model, from_cache=True)

    try:
        _run_demucs(audio_path, target, model, cfg, verbose=verbose)
    except DemucsUnavailable:
        raise
    except Exception as exc:
        if model != cfg.fallback_model:
            if verbose:
                print(f"[stems] {model} falhou ({exc}); tentando {cfg.fallback_model}")
            model = cfg.fallback_model
            target = root / cache_key(audio_path, model)
            _run_demucs(audio_path, target, model, cfg, verbose=verbose)
        else:
            raise

    found = _collect(target)
    if not found:
        raise RuntimeError(f"Demucs nao produziu stems em {target}")
    return StemSet(found, model, from_cache=False)


def _run_demucs(audio_path: Path, target: Path, model: str, cfg: StemsConfig,
                *, verbose: bool) -> None:
    try:
        import demucs.separate  # noqa: F401
    except ImportError as exc:
        raise DemucsUnavailable(
            "demucs nao esta instalado; rode `pip install yargen[stems]` "
            "ou use --no-stems para chartear a mixagem completa") from exc

    import demucs.separate

    # Escreve num diretorio temporario e so promove para o cache no fim: um
    # Ctrl-C no meio da separacao nao pode deixar um cache pela metade que
    # seria reaproveitado na proxima rodada como se estivesse completo.
    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    args = ["-n", model, "-o", str(staging), "--filename", "{stem}.{ext}",
            "-d", cfg.device, "-j", str(cfg.jobs)]
    if cfg.shifts:
        args += ["--shifts", str(cfg.shifts)]
    if cfg.segment:
        args += ["--segment", str(cfg.segment)]
    args.append(str(audio_path))

    if verbose:
        print(f"[stems] rodando demucs {model} em {cfg.device} "
              f"(isto e o passo lento; o resultado fica em cache)")
    demucs.separate.main(args)

    produced = _collect(staging)
    if not produced:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("demucs rodou mas nao gerou arquivos")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    for name, path in produced.items():
        shutil.move(str(path), str(target / path.name))
    shutil.rmtree(staging, ignore_errors=True)


def _collect(root: Path) -> dict[str, Path]:
    """Encontra os stems em qualquer profundidade abaixo de `root`.

    O demucs aninha a saida em <out>/<modelo>/<musica>/, e o layout mudou
    entre versoes. Procurar por nome de stem e mais robusto do que montar o
    caminho esperado.
    """
    known = set(SIX_STEMS)
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
            if path.stem in known:
                found[path.stem] = path
    return found


def clear_cache(cfg: StemsConfig = DEFAULT.stems) -> int:
    root = Path(cfg.cache_dir).expanduser()
    if not root.is_dir():
        return 0
    n = sum(1 for _ in root.iterdir())
    shutil.rmtree(root, ignore_errors=True)
    return n

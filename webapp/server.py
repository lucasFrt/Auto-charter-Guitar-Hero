"""Servidor web do YARGen.

Desenho em uma frase: a requisicao HTTP NUNCA espera o chart ficar pronto.

Isso nao e capricho. Separar stems com Demucs leva minutos em CPU, e qualquer
coisa entre o navegador e o Python - IIS, nginx, o proprio navegador - vai
cortar uma requisicao que demora tanto. Entao o upload cria um JOB e devolve
na hora; o navegador pergunta o andamento de tempos em tempos. Nenhuma
requisicao passa de alguns segundos, e o servidor pode ficar atras de
qualquer proxy sem ajuste de timeout.

Um worker so, de proposito: o pipeline e limitado por CPU e satura os nucleos
sozinho. Dois jobs em paralelo nao terminam mais rapido, so competem por
memoria - e o Demucs e guloso.
"""

from __future__ import annotations

import os
import shutil
import threading
import traceback
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from yargen import pipeline, profiles
from yargen.analysis.survey import render as render_survey
from yargen.chart.difficulty import ORDER as DIFF_ORDER
from yargen.chart.ir import SongMeta
from yargen.cli import KNOWN_INSTRUMENTS
from yargen.config import DEFAULT, Config

HERE = Path(__file__).resolve().parent

# Tudo configuravel por variavel de ambiente porque no servidor quem manda e
# quem publica, nao o codigo. Em IIS isso vem do web.config.
WORKSPACE = Path(os.environ.get(
    "YARGEN_WORKSPACE", HERE / "workspace")).expanduser().resolve()
MAX_UPLOAD_MB = int(os.environ.get("YARGEN_MAX_UPLOAD_MB", "80"))
KEEP_JOBS = int(os.environ.get("YARGEN_KEEP_JOBS", "40"))

ALLOWED_SUFFIXES = {".mp3", ".ogg", ".wav", ".flac", ".opus", ".m4a"}

app = FastAPI(title="YARGen", docs_url=None, redoc_url=None)


# ==================================================================== jobs

JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    mode: Literal["chart", "inspect"]
    song: str
    status: JobStatus = "queued"
    log: list[str] = field(default_factory=list)
    error: str = ""
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    result: dict[str, Any] = field(default_factory=dict)
    dir: Path | None = None

    # Separado de `result` de proposito: `result` vai inteiro para JSON na
    # resposta da API, e as opcoes carregam objetos que nao serializam (a
    # Config, o SongMeta). Misturar os dois derrubava a criacao do job.
    options: dict[str, Any] = field(default_factory=dict, repr=False)

    def say(self, line: str) -> None:
        # O pipeline registra caminhos absolutos, uteis na CLI e indevidos na
        # web: expor a arvore de diretorios do servidor para quem envia um
        # arquivo nao ajuda ninguem e conta mais do que deveria.
        self.log.append(line.replace(str(WORKSPACE), "<workspace>"))

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "mode": self.mode, "song": self.song,
                "status": self.status, "log": self.log, "error": self.error,
                "created": self.created, "result": self.result,
                "download": (f"/api/jobs/{self.id}/download"
                             if self.status == "done" and self.mode == "chart"
                             else None)}


_jobs: dict[str, Job] = {}
_order: deque[str] = deque()
_queue: "Queue[str]" = Queue()
_lock = threading.Lock()


def _register(job: Job) -> None:
    with _lock:
        _jobs[job.id] = job
        _order.append(job.id)
        # Poda os antigos: sem isto o disco do servidor enche em silencio,
        # porque cada job guarda o audio enviado mais a pasta gerada.
        while len(_order) > KEEP_JOBS:
            old = _jobs.pop(_order.popleft(), None)
            if old and old.dir and old.dir.is_dir():
                shutil.rmtree(old.dir, ignore_errors=True)


def _worker() -> None:
    while True:
        job_id = _queue.get()
        job = _jobs.get(job_id)
        if job is None:
            _queue.task_done()
            continue
        try:
            job.status = "running"
            (_run_chart if job.mode == "chart" else _run_inspect)(job)
            job.status = "done"
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.say(f"[erro] {job.error}")
            job.say(traceback.format_exc(limit=3))
        finally:
            _queue.task_done()


threading.Thread(target=_worker, daemon=True, name="yargen-worker").start()


# ================================================================ execucao

def _run_chart(job: Job) -> None:
    assert job.dir is not None
    options = job.options
    audio = Path(options["audio"])
    cfg: Config = options["cfg"]
    out = job.dir / "out" / options["folder"]

    result = pipeline.run(audio, out, cfg, genre=options["genre"],
                          instruments=options["instruments"],
                          difficulties=options["difficulties"],
                          bpm=options["bpm"], use_stems=options["use_stems"],
                          meta=options["meta"], keep_ir=options["keep_ir"],
                          log=job.say)

    archive = job.dir / f"{options['folder']}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(result.out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, Path(options["folder"]) / path.relative_to(result.out_dir))

    job.result = {
        "folder": options["folder"],
        "genre": result.analysis.genre,
        "bpm": round(result.chart.tempo.average_bpm, 1) if result.chart.tempo else None,
        "duration": round(result.chart.meta.duration, 1),
        "tracks": [
            {"track": name,
             "source": next((t.source_stem for t in result.analysis.tracks.values()
                             if t.track_name == name), ""),
             "strategy": next((t.strategy for t in result.analysis.tracks.values()
                               if t.track_name == name), ""),
             "counts": {d: len(v) for d, v in chart.notes.items()}}
            for name, chart in result.chart.instruments.items()],
        "vocals": (len(result.chart.vocals.notes) if result.chart.vocals else 0),
        "zip": archive.name,
    }
    job.say(f"[pronto] {archive.name}")


def _run_inspect(job: Job) -> None:
    survey = pipeline.inspect(Path(job.options["audio"]), job.options["cfg"],
                              log=job.say)
    job.result = {
        "recommended": survey.recommended,
        "reasons": survey.reasons,
        "report": render_survey(survey),
        "stems": [{"name": s.name, "activity": round(s.activity, 3),
                   "energy": round(s.energy_share, 3),
                   "onset_rate": round(s.onset_rate, 2),
                   "pitched": round(s.pitched_fraction, 3),
                   "centroid": round(s.centroid_hz), "present": s.present}
                  for s in sorted(survey.stems.values(), key=lambda x: -x.activity)],
    }


# ================================================================ entradas

def _safe_name(value: str, fallback: str = "musica") -> str:
    """Nome de pasta seguro a partir de texto vindo do navegador.

    Sem isto, um titulo com '../' escreveria fora do workspace. Tambem
    derrubamos os caracteres que o Windows recusa em nome de arquivo, ja que
    o servidor de destino e Windows.
    """
    cleaned = "".join(c for c in value if c not in '<>:"/\\|?*').strip().strip(".")
    cleaned = cleaned.replace("\x00", "")
    return cleaned[:120] or fallback


async def _store_upload(upload: UploadFile, target_dir: Path) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"formato nao suportado: {suffix or '(sem extensao)'}. "
                                 f"Use: {', '.join(sorted(ALLOWED_SUFFIXES))}")
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"input{suffix}"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with open(destination, "wb") as fh:
        while chunk := await upload.read(1 << 20):
            written += len(chunk)
            if written > limit:
                fh.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, f"arquivo maior que {MAX_UPLOAD_MB} MB")
            fh.write(chunk)
    if written == 0:
        raise HTTPException(400, "arquivo vazio")
    return destination


def _parse_config(config_sets: str) -> Config:
    """`mapper.max_jump=3` por linha, igual ao --set da CLI."""
    import json

    cfg = DEFAULT
    for line in (config_sets or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise HTTPException(400, f"esperado chave=valor, recebi: {line!r}")
        key, raw = line.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        try:
            cfg = cfg.override_path(key.strip(), value)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    return cfg


def _parse_list(value: str, known: tuple[str, ...], what: str) -> list[str]:
    if not value or value.strip().lower() == "all":
        return list(known)
    items = [x.strip().lower() for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in known]
    if bad:
        raise HTTPException(400, f"{what} invalido(s): {', '.join(bad)}")
    return items


# =============================================================== endpoints

@app.get("/api/genres")
def api_genres() -> list[dict[str, Any]]:
    return [{"name": p.name, "aliases": list(p.aliases),
             "description": p.description,
             "plans": [{"track": pl.track, "sources": pl.sources,
                        "strategy": pl.strategy, "note": pl.note}
                       for pl in p.plans]}
            for p in profiles.canonical()]


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Diz o que esta disponivel de verdade nesta maquina.

    Demucs e faster-whisper sao opcionais e pesados; sem esta rota, a
    ausencia deles so apareceria como um chart pior, sem explicacao.
    """
    def has(module: str) -> bool:
        import importlib.util
        return importlib.util.find_spec(module) is not None

    return {"ok": True, "workspace": str(WORKSPACE),
            "max_upload_mb": MAX_UPLOAD_MB,
            "stems_available": has("demucs") and has("torch"),
            "lyrics_available": has("faster_whisper"),
            "queue": _queue.qsize(),
            "jobs": len(_jobs)}


@app.post("/api/jobs")
async def api_create_job(
    audio: UploadFile = File(...),
    mode: str = Form("chart"),
    genre: str = Form("rock"),
    instruments: str = Form("all"),
    difficulties: str = Form("all"),
    bpm: str = Form(""),
    offset_ms: str = Form("0"),
    use_stems: str = Form("true"),
    keep_ir: str = Form("false"),
    name: str = Form(""),
    artist: str = Form(""),
    charter: str = Form(""),
    config_sets: str = Form(""),
) -> JSONResponse:
    if mode not in ("chart", "inspect"):
        raise HTTPException(400, "mode deve ser 'chart' ou 'inspect'")
    try:
        profiles.get(genre)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORKSPACE / "jobs" / job_id
    stored = await _store_upload(audio, job_dir)

    stem = Path(audio.filename or "musica").stem
    guess_artist, guess_name = ("", stem)
    if " - " in stem:
        guess_artist, guess_name = (p.strip() for p in stem.split(" - ", 1))

    meta = SongMeta(name=(name or guess_name), artist=(artist or guess_artist),
                    charter=(charter or DEFAULT.output.charter))
    folder = _safe_name(f"{meta.artist} - {meta.name}" if meta.artist else meta.name)

    cfg = _parse_config(config_sets)
    try:
        cfg = cfg.override_path("output.offset_ms", int(offset_ms or 0))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = Job(id=job_id, mode=mode, song=folder, dir=job_dir)
    job.options = {
        "audio": str(stored), "cfg": cfg, "folder": folder, "genre": genre,
        "instruments": _parse_list(instruments, KNOWN_INSTRUMENTS, "instrumento"),
        "difficulties": _parse_list(difficulties, DIFF_ORDER, "dificuldade"),
        "bpm": (float(bpm) if bpm.strip() else None),
        "use_stems": use_stems.lower() in ("1", "true", "on", "yes"),
        "keep_ir": keep_ir.lower() in ("1", "true", "on", "yes"),
        "meta": meta,
    }
    _register(job)
    _queue.put(job_id)
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs")
def api_list_jobs() -> list[dict[str, Any]]:
    with _lock:
        ids = list(_order)[::-1]
    return [{k: v for k, v in _jobs[i].public().items() if k != "log"}
            for i in ids if i in _jobs]


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str, since: int = 0) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job nao encontrado")
    data = job.public()
    # `since` deixa o navegador buscar so as linhas novas do log: em musica
    # longa com Demucs o log passa de centenas de linhas e reenviar tudo a
    # cada segundo e desperdicio puro.
    data["log"] = job.log[since:]
    data["log_total"] = len(job.log)
    return data


@app.get("/api/jobs/{job_id}/download")
def api_download(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if job is None or job.status != "done" or job.mode != "chart":
        raise HTTPException(404, "nada para baixar")
    assert job.dir is not None
    archive = job.dir / job.result.get("zip", "")
    if not archive.is_file():
        raise HTTPException(404, "arquivo expirou")
    return FileResponse(archive, filename=archive.name,
                        media_type="application/zip")


@app.delete("/api/jobs/{job_id}")
def api_delete(job_id: str) -> dict[str, bool]:
    with _lock:
        job = _jobs.pop(job_id, None)
        if job_id in _order:
            _order.remove(job_id)
    if job is None:
        raise HTTPException(404, "job nao encontrado")
    if job.dir and job.dir.is_dir():
        shutil.rmtree(job.dir, ignore_errors=True)
    return {"deleted": True}


# A interface fica montada por ultimo: montar "/" antes engoliria as rotas
# de API registradas depois dela.
(WORKSPACE / "jobs").mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")


def main() -> None:
    """Sobe o servidor. `python -m webapp.server` ou o script `yargen-web`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Interface web do YARGen")
    # 127.0.0.1 por padrao: expor para a rede tem que ser uma escolha
    # explicita, nao o que acontece quando voce nao pensa no assunto.
    parser.add_argument("--host", default=os.environ.get("YARGEN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT",
                                    os.environ.get("YARGEN_PORT", "8000"))))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"YARGen web em http://{args.host}:{args.port}")
    print(f"  workspace: {WORKSPACE}")
    uvicorn.run("webapp.server:app" if args.reload else app,
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

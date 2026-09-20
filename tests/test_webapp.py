"""API web.

Usa o worker de verdade: os jobs sao executados, nao simulados. O audio de
teste tem 3 segundos e roda sem Demucs, entao a suite continua rapida.
"""

import io
import time

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from webapp import server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


def wav_bytes(seconds: float = 3.0, sr: int = 22050) -> bytes:
    """Trecho curto com ataques claros, para gerar onsets de verdade."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    for i in range(int(seconds * 4)):
        t = np.arange(int(0.25 * sr)) / sr
        hz = 220.0 * 2 ** ((i % 7) / 12)
        start = int(i * 0.25 * sr)
        y[start:start + len(t)] += (np.sin(2 * np.pi * hz * t)
                                    * np.exp(-6 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y / max(np.max(np.abs(y)), 1e-9), sr, format="WAV")
    return buf.getvalue()


def submit(client, **fields):
    data = {"mode": "chart", "genre": "rock", "instruments": "guitar",
            "difficulties": "expert", "use_stems": "false"}
    data.update({k: str(v) for k, v in fields.items()})
    name = data.pop("filename", "Banda - Musica.wav")
    return client.post("/api/jobs", data=data,
                       files={"audio": (name, wav_bytes(), "audio/wav")})


def wait(client, job_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.4)
    raise AssertionError(f"job {job_id} nao terminou em {timeout}s")


# ------------------------------------------------------------------ basico

def test_health_reports_what_is_installed(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert isinstance(body["stems_available"], bool)
    assert isinstance(body["lyrics_available"], bool)


def test_genres_endpoint_lists_the_routing(client):
    genres = {g["name"]: g for g in client.get("/api/genres").json()}
    assert "trap" in genres and "rock" in genres
    lead = next(p for p in genres["trap"]["plans"] if p["track"] == "PART GUITAR")
    assert lead["sources"][0] == "drums" and lead["strategy"] == "percussive"


def test_index_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "YARGen" in page.text


# --------------------------------------------------------------- validacao

def test_rejects_an_unsupported_format(client):
    res = client.post("/api/jobs", data={"mode": "chart"},
                      files={"audio": ("x.txt", b"nao sou audio", "text/plain")})
    assert res.status_code == 400 and "formato" in res.json()["detail"]


def test_rejects_an_empty_file(client):
    res = client.post("/api/jobs", data={"mode": "chart"},
                      files={"audio": ("x.wav", b"", "audio/wav")})
    assert res.status_code == 400


def test_rejects_an_unknown_genre(client):
    assert submit(client, genre="reggaeton-espacial").status_code == 400


def test_rejects_a_bad_config_line(client):
    res = submit(client, config_sets="mapper.max_jumps=3")
    assert res.status_code == 400 and "max_jumps" in res.json()["detail"]


def test_rejects_an_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 0)
    assert submit(client).status_code == 413


@pytest.mark.parametrize("nasty,forbidden", [
    ("../../etc/passwd", ".."),
    ('C:\\Windows\\sys "x"', "\\"),
    ("nome/com/barra", "/"),
])
def test_folder_name_cannot_escape_the_workspace(nasty, forbidden):
    """Titulo vem do navegador; sem higienizar, escreveria fora do workspace."""
    safe = server._safe_name(nasty)
    assert forbidden not in safe
    assert safe and not safe.startswith(".")


def test_empty_name_gets_a_fallback():
    assert server._safe_name("...") == "musica"


# ------------------------------------------------------------ ciclo do job

def test_full_job_produces_a_downloadable_zip(client):
    import zipfile

    created = submit(client)
    assert created.status_code == 202
    job = wait(client, created.json()["id"])
    assert job["status"] == "done", job["error"]

    assert job["result"]["folder"] == "Banda - Musica"
    assert job["result"]["tracks"][0]["track"] == "PART GUITAR"
    assert job["result"]["tracks"][0]["counts"]["expert"] > 0

    zipped = client.get(job["download"])
    assert zipped.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(zipped.content)).namelist()
    assert any(n.endswith("notes.mid") for n in names)
    assert any(n.endswith("song.ini") for n in names)
    assert any(n.endswith(".wav") for n in names)


def test_log_does_not_leak_server_paths(client):
    job = wait(client, submit(client).json()["id"])
    assert str(server.WORKSPACE) not in "\n".join(job["log"])
    assert any("<workspace>" in line for line in job["log"])


def test_since_returns_only_new_log_lines(client):
    job = wait(client, submit(client).json()["id"])
    total = job["log_total"]
    tail = client.get(f"/api/jobs/{job['id']}?since={total - 1}").json()
    assert len(tail["log"]) == 1 and tail["log_total"] == total


def test_bpm_override_is_honoured(client):
    job = wait(client, submit(client, bpm=174).json()["id"])
    assert job["result"]["bpm"] == pytest.approx(174, abs=0.5)


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/naoexiste").status_code == 404
    assert client.get("/api/jobs/naoexiste/download").status_code == 404


def test_job_can_be_deleted(client):
    job_id = wait(client, submit(client).json()["id"])["id"]
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_history_lists_jobs_without_their_logs(client):
    wait(client, submit(client).json()["id"])
    listed = client.get("/api/jobs").json()
    assert listed and "log" not in listed[0]
    assert {"id", "status", "song", "mode"} <= set(listed[0])

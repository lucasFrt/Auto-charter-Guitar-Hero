#!/usr/bin/env bash
# Equivalente do start.ps1 para Linux e macOS.
#   ./start.sh            interface web em http://127.0.0.1:8000
#   ./start.sh --full     instala tambem Demucs e faster-whisper
set -euo pipefail
cd "$(dirname "$0")"

FULL=0; PORT=8000
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "argumento desconhecido: $1"; exit 1 ;;
  esac
done

PY=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
  command -v "$candidate" >/dev/null 2>&1 || continue
  if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
[ -n "$PY" ] || { echo "Nenhum Python 3.10+ encontrado."; exit 1; }
echo "  $PY $("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"

[ -x .venv/bin/python ] || { echo "  criando .venv ..."; "$PY" -m venv .venv; }
VENV=.venv/bin/python

if ! $VENV -c 'import yargen, fastapi' >/dev/null 2>&1; then
  echo "  instalando dependencias (a primeira vez demora) ..."
  $VENV -m pip install --upgrade pip --quiet
  $VENV -m pip install -e ".[web]"
fi

if [ "$FULL" = "1" ] && ! $VENV -c 'import demucs, faster_whisper' >/dev/null 2>&1; then
  echo "  instalando Demucs e faster-whisper (~1 GB) ..."
  $VENV -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  $VENV -m pip install -e ".[all,web]"
fi

echo; echo "  http://127.0.0.1:$PORT"; echo
exec $VENV -m webapp.server --port "$PORT"

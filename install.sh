#!/usr/bin/env bash
# Instala las dependencias en un entorno virtual propio (.venv), sin tocar el Python del
# sistema. Hace falta en las distribuciones que bloquean `pip install` global (PEP 668:
# "externally-managed-environment"), como Ubuntu 23.04+ o Debian 12+.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "No se encontró $PY. Instálalo (p. ej.: sudo apt install python3)." >&2
    exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Se necesita Python 3.10 o más nuevo; tienes $("$PY" --version 2>&1)." >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "Creando el entorno virtual en .venv…"
    if ! "$PY" -m venv .venv; then
        rm -rf .venv
        ver="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        echo >&2
        echo "No se pudo crear el entorno virtual: falta el módulo venv. Instálalo con:" >&2
        echo "    sudo apt install python${ver}-venv     # Debian / Ubuntu" >&2
        echo "y vuelve a ejecutar ./install.sh" >&2
        exit 1
    fi
fi

echo "Instalando dependencias…"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

if [ ! -f config.json ]; then
    cp config.example.json config.json
    echo "Creado config.json a partir de config.example.json"
fi

echo
echo "✅ Listo. Siguientes pasos:"
echo "    ./wikichat.sh setup                       # prepara Ollama y los modelos para este equipo"
echo "    ./wikichat.sh import-zim <archivo>.zim    # importa la Wikipedia"
echo "    ./wikichat.sh serve                       # abre el chat en http://127.0.0.1:8800"

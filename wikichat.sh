#!/usr/bin/env bash
# Ejecuta wikichat con el Python del entorno virtual (.venv) creado por install.sh.
# Uso: ./wikichat.sh <comando> [opciones]      p. ej.: ./wikichat.sh serve
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "Falta el entorno virtual: ejecuta primero ./install.sh" >&2
    exit 1
fi
exec .venv/bin/python -m wikichat "$@"

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

has_pip() { [ -x .venv/bin/python ] && .venv/bin/python -m pip --version >/dev/null 2>&1; }

# Pone pip dentro de .venv. Primero con ensurepip (viene con Python); en Debian/Ubuntu sin el
# paquete python3-venv no está, y entonces se usa el instalador oficial get-pip.py (no pide sudo).
install_pip() {
    .venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 && return 0
    echo "Instalando pip en el entorno (get-pip.py)…"
    .venv/bin/python - <<'PY' || return 1
import os, runpy, sys, tempfile, urllib.request
fd, path = tempfile.mkstemp(suffix="-get-pip.py")
os.close(fd)
urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", path)
sys.argv = [path, "--quiet"]
runpy.run_path(path, run_name="__main__")
PY
    has_pip
}

if [ -x .venv/bin/python ] && ! has_pip; then
    echo "El entorno .venv existe pero no tiene pip (pasa si se creó sin python3-venv); reparándolo…"
    install_pip || true
fi

if ! has_pip; then
    if [ ! -x .venv/bin/python ]; then
        echo "Creando el entorno virtual en .venv…"
        # Sin python3-venv, Debian/Ubuntu crean el entorno pero sin pip y devuelven error:
        # en ese caso se sigue y se instala pip aparte.
        "$PY" -m venv .venv >/dev/null 2>&1 || "$PY" -m venv --without-pip .venv >/dev/null 2>&1 || true
    fi
    if [ ! -x .venv/bin/python ] || ! { has_pip || install_pip; }; then
        rm -rf .venv
        ver="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        echo >&2
        echo "No se pudo preparar el entorno virtual con pip. Instala el soporte de entornos" >&2
        echo "virtuales de Python y vuelve a ejecutar ./install.sh:" >&2
        echo "    sudo apt install python${ver}-venv      # Debian / Ubuntu" >&2
        echo "    sudo dnf install python3-pip           # Fedora" >&2
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

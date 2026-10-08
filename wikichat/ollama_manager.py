"""Ollama administrado: si no está instalado se descarga en data/ollama y se ejecuta como
proceso propio, en un puerto aparte, solo mientras el chat está abierto."""
import atexit
import json
import logging
import os
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

log = logging.getLogger("wikichat.ollama")

DOWNLOAD = "https://ollama.com/download/ollama-linux-{arch}.tar.zst"
SYSTEM_URL = "http://127.0.0.1:11434"
MANAGED_PORT = 11435


def version(url, timeout=3):
    """Versión de un Ollama que responde en `url`, o None."""
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=timeout) as r:
            return json.load(r).get("version")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def managed_binary(ollama_dir):
    path = os.path.join(ollama_dir, "bin", "ollama")
    return path if os.access(path, os.X_OK) else None


def _keep(member, gpus):
    """Qué archivos del paquete extraer: las librerías de GPU solo si hay esa GPU."""
    name = member.name
    vendors = {g["vendor"] for g in gpus}
    if "mlx" in name:
        return "nvidia" in vendors  # motor MLX con CUDA: solo sirve con NVIDIA
    if "cuda" in name or "jetpack" in name:
        return "nvidia" in vendors
    if "rocm" in name:
        return "amd" in vendors
    if "vulkan" in name:
        return bool(vendors & {"amd", "intel"})
    return True


def _zstd_reader(fileobj):
    try:
        import zstandard
    except ImportError:
        raise SystemExit("Para descomprimir Ollama hace falta:  pip install zstandard")
    return zstandard.ZstdDecompressor().stream_reader(fileobj)


def install(ollama_dir, arch, gpus, progress=print):
    """Descarga Ollama para Linux y extrae solo lo necesario para este equipo."""
    os.makedirs(ollama_dir, exist_ok=True)
    url = DOWNLOAD.format(arch=arch)
    tmp = os.path.join(ollama_dir, "ollama.tar.zst.part")
    progress(f"Descargando Ollama desde {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done - last > total / 20:
                last = done
                progress(f"  {done / 2**20:.0f} / {total / 2**20:.0f} MB")
    progress("Extrayendo…")
    with open(tmp, "rb") as f, tarfile.open(fileobj=_zstd_reader(f), mode="r|") as tar:
        for member in tar:
            if _keep(member, gpus):
                tar.extract(member, ollama_dir, filter="data")
    os.remove(tmp)
    binary = managed_binary(ollama_dir)
    if not binary:
        raise SystemExit("La descarga de Ollama no trajo el ejecutable esperado (bin/ollama)")
    return binary


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError):
        return False


def _stop_group(pid, wait=10):
    """Detiene un Ollama y los `llama-server` que lanzó (comparten grupo de procesos)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.time() + wait
        while time.time() < deadline and _alive(pid):
            time.sleep(0.2)
            try:
                os.waitpid(pid, os.WNOHANG)  # recoge al proceso si es hijo nuestro
            except ChildProcessError:
                pass
        if not _alive(pid):
            return


class ManagedOllama:
    """Arranca y detiene un `ollama serve` propio.

    Guarda en ollama.pid el proceso de Ollama y el del chat que lo inició. Si el chat se
    cerró de golpe y Ollama quedó huérfano, el siguiente `serve` lo adopta y lo detiene al
    salir; si el chat que lo inició sigue abierto, lo comparte sin detenerlo.
    """

    def __init__(self, binary, ollama_dir, port=MANAGED_PORT):
        self.binary, self.dir, self.port = binary, ollama_dir, port
        self.url = f"http://127.0.0.1:{port}"
        self.pidfile = os.path.join(ollama_dir, "ollama.pid")
        self.proc = None
        self.owned_pid = None

    def _read_pidfile(self):
        try:
            with open(self.pidfile) as f:
                ollama_pid, owner_pid = (int(x) for x in f.read().split())
            return ollama_pid, owner_pid
        except (OSError, ValueError):
            return None, None

    def _write_pidfile(self, ollama_pid):
        with open(self.pidfile, "w") as f:
            f.write(f"{ollama_pid} {os.getpid()}")

    def start(self, wait=60):
        if version(self.url):
            ollama_pid, owner_pid = self._read_pidfile()
            if ollama_pid and _alive(ollama_pid) and not _alive(owner_pid):
                self.owned_pid = ollama_pid  # huérfano de un `serve` anterior: lo adoptamos
                self._write_pidfile(ollama_pid)
                atexit.register(self.stop)
            return self.url
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{self.port}",
                   OLLAMA_MODELS=os.path.join(self.dir, "models"))
        log_file = open(os.path.join(self.dir, "serve.log"), "ab")
        self.proc = subprocess.Popen([self.binary, "serve"], env=env, stdout=log_file,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.owned_pid = self.proc.pid
        self._write_pidfile(self.proc.pid)
        atexit.register(self.stop)
        deadline = time.time() + wait
        while time.time() < deadline:
            if version(self.url):
                return self.url
            if self.proc.poll() is not None:
                break
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"Ollama no arrancó; revisa {os.path.join(self.dir, 'serve.log')}")

    def stop(self):
        if self.owned_pid:
            _stop_group(self.owned_pid)
            if self._read_pidfile()[0] == self.owned_pid:
                try:
                    os.remove(self.pidfile)
                except OSError:
                    pass
        self.proc = None
        self.owned_pid = None


def has_model(url, name):
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=10) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return name in names or f"{name}:latest" in names


def pull(url, name, progress=print):
    """Descarga un modelo mostrando el avance."""
    if has_model(url, name):
        progress(f"Modelo {name}: ya descargado")
        return
    req = urllib.request.Request(f"{url}/api/pull", data=json.dumps({"model": name}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = -10
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            ev = json.loads(line)
            if ev.get("error"):
                raise RuntimeError(f"No se pudo descargar {name}: {ev['error']}")
            total, done = ev.get("total"), ev.get("completed")
            if total and total > 50 * 2**20 and done is not None:  # solo las capas grandes
                pct = 100 * done / total
                if pct - last >= 10 or pct >= 100:
                    last = pct
                    progress(f"  {name}: {pct:.0f}% de {total / 2**30:.1f} GB")
            if ev.get("status") == "success":
                progress(f"Modelo {name}: listo")


def ensure_running(cfg):
    """Para `serve`: si la configuración pide Ollama administrado, lo arranca."""
    if not cfg.get("manage_ollama"):
        return None
    binary = managed_binary(cfg["ollama_dir"]) or shutil.which("ollama")
    if not binary:
        log.warning("manage_ollama está activo pero no hay Ollama; ejecuta `python -m wikichat setup`")
        return None
    manager = ManagedOllama(binary, cfg["ollama_dir"])
    try:
        cfg["llm_url"] = manager.start()
        log.info("Ollama administrado en %s", cfg["llm_url"])
    except RuntimeError as e:
        log.warning("%s; el chat seguirá sin modelos", e)
    return manager

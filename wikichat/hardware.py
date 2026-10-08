"""Detección del hardware (Linux) y elección de modelos y ajustes según lo que hay."""
import os
import platform
import re
import shutil
import subprocess

GB = 1024 ** 3


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_meminfo(text):
    """(RAM total, RAM disponible) en bytes a partir de /proc/meminfo."""
    values = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+) kB", line)
        if m:
            values[m.group(1)] = int(m.group(2)) * 1024
    return values.get("MemTotal", 0), values.get("MemAvailable", values.get("MemFree", 0))


def parse_nvidia_smi(text):
    """[{vendor, name, vram}] a partir de `nvidia-smi --query-gpu=name,memory.total
    --format=csv,noheader,nounits` (memoria en MiB)."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.rsplit(",", 1)]
        if len(parts) == 2 and parts[1].isdigit():
            gpus.append({"vendor": "nvidia", "name": parts[0], "vram": int(parts[1]) * 1024 ** 2})
    return gpus


def parse_rocm_smi(text):
    """[{vendor, name, vram}] a partir de `rocm-smi --showmeminfo vram --csv`."""
    gpus = []
    for line in text.strip().splitlines()[1:]:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 2 and cells[1].isdigit():
            gpus.append({"vendor": "amd", "name": cells[0], "vram": int(cells[1])})
    return gpus


def parse_lspci(text):
    """Tarjetas de video que no se pudieron medir con nvidia-smi/rocm-smi (sin VRAM)."""
    gpus = []
    for line in text.splitlines():
        if re.search(r"VGA compatible|3D controller|Display controller", line):
            low = line.lower()
            vendor = ("nvidia" if "nvidia" in low else "amd" if ("amd" in low or "ati " in low)
                      else "intel" if "intel" in low else "otra")
            gpus.append({"vendor": vendor, "name": line.split(": ", 1)[-1].strip(), "vram": 0})
    return gpus


def detect(data_dir="data"):
    cpuinfo = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except OSError:
        pass
    model = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    flags = re.search(r"flags\s*:\s*(.+)", cpuinfo)
    flags = set(flags.group(1).split()) if flags else set()
    try:
        with open("/proc/meminfo") as f:
            ram_total, ram_free = parse_meminfo(f.read())
    except OSError:
        ram_total = ram_free = 0

    gpus = []
    if shutil.which("nvidia-smi"):
        gpus = parse_nvidia_smi(_run(["nvidia-smi", "--query-gpu=name,memory.total",
                                      "--format=csv,noheader,nounits"]))
    if not gpus and shutil.which("rocm-smi"):
        gpus = parse_rocm_smi(_run(["rocm-smi", "--showmeminfo", "vram", "--csv"]))
    if not gpus and shutil.which("lspci"):
        gpus = [g for g in parse_lspci(_run(["lspci"])) if g["vendor"] != "otra"]

    os.makedirs(data_dir, exist_ok=True)
    return {
        "arch": {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine(), platform.machine()),
        "cpu": model.group(1).strip() if model else platform.processor() or "desconocida",
        "cores": os.cpu_count() or 1,
        "avx2": "avx2" in flags,
        "avx512": any(f.startswith("avx512") for f in flags),
        "ram_total": ram_total,
        "ram_free": ram_free,
        "disk_free": shutil.disk_usage(data_dir).free,
        "gpus": gpus,
    }


def usable_vram(hw):
    """VRAM de la mejor GPU que Ollama puede usar (NVIDIA o AMD con su memoria medida)."""
    return max((g["vram"] for g in hw["gpus"] if g["vendor"] in ("nvidia", "amd")), default=0)


# Modelos de chat de menor a mayor: (nombre, GB que ocupa cargado, aprox.)
CHAT_MODELS = [("qwen2.5:1.5b", 1.2), ("qwen2.5:3b", 2.3), ("qwen2.5:7b", 5.0), ("qwen2.5:14b", 9.5)]
EMBED_MODEL = ("embeddinggemma", 0.7)
# Tamaño de descarga de cada modelo, en GB.
DOWNLOAD_GB = {"qwen2.5:1.5b": 1.0, "qwen2.5:3b": 1.9, "qwen2.5:7b": 4.7, "qwen2.5:14b": 9.0,
               "embeddinggemma": 0.6}


def choose_profile(hw):
    """Primera elección a partir del hardware; `setup` la ajusta después de medir."""
    vram = usable_vram(hw) / GB
    ram = hw["ram_total"] / GB
    if vram >= 6:
        # Con GPU: el modelo más grande que entre en la VRAM junto al de embeddings.
        fits = [m for m, size in CHAT_MODELS if size + EMBED_MODEL[1] + 0.5 <= vram]
        return {"name": "gpu", "chat_model": fits[-1] if fits else "qwen2.5:3b",
                "top_k": 6, "rewrite_followups": True, "embed_dims": 384}
    # Sin GPU grande: el modelo debe caber en RAM dejando sitio al sistema, al índice y al servidor.
    budget = ram - 3.0 - EMBED_MODEL[1] - 1.0
    fits = [m for m, size in CHAT_MODELS[:3] if size <= budget]
    chat = fits[-1] if fits else CHAT_MODELS[0][0]
    small_gpu = vram >= 3  # Ollama reparte el modelo entre la GPU y la CPU
    if chat == "qwen2.5:7b" and hw["cores"] < 8 and not small_gpu:
        chat = "qwen2.5:3b"  # con pocos núcleos y sin GPU un 7B tarda más de un minuto por respuesta
    return {"name": "cpu+gpu" if small_gpu else "cpu", "chat_model": chat,
            "top_k": 4 if ram >= 8 else 3, "rewrite_followups": False, "embed_dims": 384}


def smaller_chat_model(name):
    names = [m for m, _ in CHAT_MODELS]
    i = names.index(name) if name in names else 1
    return names[max(0, i - 1)]


def describe(hw):
    gpu = ", ".join(f"{g['name']}" + (f" ({g['vram'] / GB:.0f} GB)" if g["vram"] else "")
                    for g in hw["gpus"]) or "ninguna"
    return (f"CPU: {hw['cpu']} · {hw['cores']} núcleos"
            f"{' · AVX-512' if hw['avx512'] else ' · AVX2' if hw['avx2'] else ''}\n"
            f"RAM: {hw['ram_total'] / GB:.1f} GB ({hw['ram_free'] / GB:.1f} GB libres)\n"
            f"GPU: {gpu}\n"
            f"Disco libre: {hw['disk_free'] / GB:.0f} GB")

from pathlib import Path

import json
import subprocess
import time
import urllib.request
import urllib.error


RAIZ = Path(__file__).resolve().parents[2]


def completion(cuerpo: dict, puerto: int = 8090) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{puerto}/completion",
        data=json.dumps(cuerpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def medir_punto(prefijo: list[int], nuevo: list[int], puerto: int = 8090, generar: int = 128) -> dict:
    # Petición 1: si prefijo no está vacío
    if prefijo:
        completion({"prompt": prefijo, "n_predict": 0, "cache_prompt": False}, puerto)

    # Petición 2
    cache_prompt_val = True if prefijo else False
    body = {
        "prompt": prefijo + nuevo,
        "n_predict": generar,
        "temperature": 0,
        "ignore_eos": True,
        "cache_prompt": cache_prompt_val,
    }
    body["stream"] = True

    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"http://127.0.0.1:{puerto}/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        chunks = []
        ttft_ms = None
        for line in resp:
            line = line.decode("utf-8")
            if line.startswith("data:"):
                chunk = json.loads(line[len("data:"):].strip())
                chunks.append(chunk)
                if ttft_ms is None and "tokens" in chunk and isinstance(chunk["tokens"], list) and chunk["tokens"]:
                    ttft_ms = (time.perf_counter() - t0) * 1000

    timings = chunks[-1].get("timings", {}) if chunks else {}

    if ttft_ms is None:
        raise RuntimeError("ttft_ms no calculado")
    timings["ttft_ms"] = ttft_ms

    expected = {
        "cache_n": len(prefijo),
        "prompt_n": len(nuevo),
        "predicted_n": generar,
    }

    for key in expected:
        if timings.get(key) != expected[key]:
            raise RuntimeError(
                f"{key}: se esperó {expected[key]} pero obtuvo {timings.get(key)}"
            )

    return timings


def tokenizar(texto: str, puerto: int = 8090) -> list[int]:
    payload = {
        "content": texto,
        "add_special": False,
        "parse_special": False,
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{puerto}/tokenize",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["tokens"]


def cortar(tokens: list[int], profundidad: int, prompt: int = 512) -> tuple[list[int], list[int]]:
    needed = profundidad + prompt
    available = len(tokens)

    if available < needed:
        raise ValueError(
            f"La lista tiene {available} tokens, pero se necesitan {needed} "
            f"({profundidad} + {prompt})."
        )

    return tokens[:profundidad], tokens[profundidad:profundidad + prompt]


def memoria_proceso(pid: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.strip().splitlines():
        if int(line.split(",")[0]) == pid:
            return int(line.split(",")[1])
    raise RuntimeError("No se encontró el proceso con el ID proporcionado")


def vram_libre() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    print("Hello from arnes!")


def argumentos_servidor(modelo: str, ctx: int, puerto: int) -> list[str]:
    return [
        "-m", modelo,
        "-ngl", "99",
        "--ctx-size", str(ctx),
        "-fit", "off",
        "-np", "1",
        "-fa", "on",
        "--host", "127.0.0.1",
        "--port", str(puerto),
    ]


def sha256_fichero(ruta: str) -> str:
    import hashlib
    hash_sha256 = hashlib.sha256()
    with open(ruta, "rb") as f:
        block_size = 1024 * 1024
        while True:
            data = f.read(block_size)
            if not data:
                break
            hash_sha256.update(data)
    return hash_sha256.hexdigest()


def arrancar_servidor(modelo: str, ctx: int, puerto: int = 8090, log: str = "server.log"):
    import socket
    import subprocess

    try:
        socket.create_connection(("127.0.0.1", puerto), timeout=1)
    except ConnectionRefusedError:
        pass
    else:
        raise RuntimeError(f"puerto {puerto} ocupado")

    import time

    cmd = ["llama-server"] + argumentos_servidor(modelo, ctx, puerto)

    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    timeout = 120
    deadline = timeout
    start = time.time()

    while (time.time() - start) < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{puerto}/health", timeout=5) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(1)

        if proc.poll() is not None:
            detener_servidor(proc)
            raise RuntimeError(get_last_lines(log))

    if proc.poll() is not None:
        detener_servidor(proc)
        raise RuntimeError(get_last_lines(log))
    else:
        detener_servidor(proc)
        raise RuntimeError(get_last_lines(log))


def detener_servidor(proceso: subprocess.Popen) -> None:
    proceso.terminate()
    try:
        proceso.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proceso.kill()
        proceso.wait()


def get_last_lines(log: str) -> str:
    with open(log, "r") as f:
        lines = f.readlines()
    return "".join(lines[-3:])


def estado_arnes() -> tuple[str, bool]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = result.stdout.strip()

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=True,
    )
    has_uncommitted = bool(result.stdout.strip())

    return commit, has_uncommitted



def configuracion(modelo: str, ctx: int, puerto: int, permitir_cambios: bool) -> dict:
    import datetime
    from pathlib import Path

    estado = estado_arnes()
    if estado[1] and not permitir_cambios:
        raise RuntimeError("hay cambios sin commit en el arnés: haz commit antes de medir")

    return {
        "fecha": datetime.datetime.now().astimezone().isoformat(),
        "modelo": str(Path(modelo).resolve()),
        "modelo_sha256": sha256_fichero(modelo),
        "ctx": ctx,
        "argumentos": argumentos_servidor(modelo, ctx, puerto),
        "llama_cpp_version": (
            subprocess.run(
                ["llama-server", "--version"],
                cwd=RAIZ,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            + subprocess.run(
                ["llama-server", "--version"],
                cwd=RAIZ,
                capture_output=True,
                text=True,
                check=True,
            ).stderr
        ).strip(),
        "arnes_commit": estado[0],
        "arnes_con_cambios": estado[1],
        "disco": (
            subprocess.run(
                ["findmnt", "-n", "-o", "SOURCE", "--target", str(Path(modelo).resolve())],
                cwd=RAIZ,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        ).strip(),
        "gpu": (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                cwd=RAIZ,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        ).strip(),
        "vram_libre_mib": vram_libre(),
        "carga_sha256": sha256_fichero(str(RAIZ / "carga" / "llama-vocab.cpp")),
    }


def medir_tanda(modelo: str, salida: str, ctx: int = 17408, profundidades: tuple = (0, 4096, 16384), repeticiones: int = 5, puerto: int = 8090, permitir_cambios: bool = False) -> dict:
    from pathlib import Path
    if Path(salida).resolve().is_relative_to(RAIZ):
        raise ValueError("la salida no puede estar dentro del repo del arnés")

    config = configuracion(modelo, ctx, puerto, permitir_cambios)
    config["profundidades"] = profundidades
    config["repeticiones"] = repeticiones
    config["prompt"] = 512
    config["generar"] = 128

    proceso = arrancar_servidor(modelo, ctx, puerto)
    try:
        tokens = tokenizar((RAIZ / "carga" / "llama-vocab.cpp").read_text(), puerto)
        mediciones = []
        for d in profundidades:
            prefijo, nuevo = cortar(tokens, d)
            for j in range(repeticiones + 1):
                t = medir_punto(prefijo, nuevo, puerto)
                t["profundidad"] = d
                t["repeticion"] = j
                t["calentamiento"] = j == 0
                mediciones.append(t)
        memoria = memoria_proceso(proceso.pid)
    finally:
        detener_servidor(proceso)

    resultado = {
        "configuracion": config,
        "memoria_servidor_mib": memoria,
        "mediciones": mediciones,
    }
    with open(salida, "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    return resultado

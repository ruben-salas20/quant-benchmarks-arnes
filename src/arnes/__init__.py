import json
import subprocess
import time
import urllib.request
import urllib.error


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

    cmd = ["llama-server", "-m", modelo, "-ngl", "99", "--ctx-size", str(ctx), "-fit", "off", "-np", "1", "-fa", "on", "--host", "127.0.0.1", "--port", str(puerto)]

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

import json
import subprocess
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
    resp2 = completion(
        {
            "prompt": prefijo + nuevo,
            "n_predict": generar,
            "temperature": 0,
            "ignore_eos": True,
            "cache_prompt": cache_prompt_val,
        },
        puerto,
    )

    timings = resp2.get("timings", {})

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

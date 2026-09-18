import subprocess
import urllib.request
import urllib.error


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

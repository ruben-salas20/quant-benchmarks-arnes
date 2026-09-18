# arnes

## Qué es

Una herramienta para medir el rendimiento de modelos GGUF servidos con llama-server. Realiza prefill, decode, tiempo al primer token, memoria y guarda resultados en JSON.

## Qué mide

- **Prefill**: 512 tokens al inicio de cada punto de profundidad (0, 4096, 16384).
- **Decode**: 128 tokens generados, con temperature: 0, ignore_eos: true, stream: true.
- **Tiempo al primer token**: medido a nivel de `ttft_ms` en cada repetición.
- **Memoria**: la memoria de GPU que ocupa el servidor (`memoria_servidor_mib`) y la VRAM libre al empezar la tanda (`vram_libre_mib`), las dos con `nvidia-smi`.
- **Profundidades**: 0 (Estándar), 4096 (Tarea), 16384 (Límite).
- **Carga**: `carga/llama-vocab.cpp` (src/llama-vocab.cpp de llama.cpp, MIT, tag `b11028`).

## Requisitos

- Linux
- GPU NVIDIA con `nvidia-smi`
- `llama-server` en el PATH
- `uv` y Python 3.13

## Uso

```python
from arnes import medir_tanda

resultado = medir_tanda(
    "/ruta/al/modelo.gguf",
    "/ruta/fuera/del/repo/tanda.json",
)
```

La salida debe estar fuera del repo del arnés. Se ejecuta con `uv run python` desde la carpeta del repo.

Si el repo del arnés tiene cambios sin commit, `medir_tanda` se niega a medir. Con `permitir_cambios=True` mide igual, pero la tanda queda marcada con `arnes_con_cambios: true`.

## Qué guarda

Un JSON con tres partes:

- **`configuracion`**: fecha, modelo y sha256, argumentos, versión de llama.cpp, commit del arnés, disco, GPU, VRAM libre al inicio, sha256 de la carga.
- **`memoria_servidor_mib`**: memoria de GPU ocupada por el proceso del servidor al terminar las mediciones (mide con `memoria_proceso`).
- **`mediciones`**: una lista de diccionarios con `profundidad`, `repeticion`, `calentamiento`, `ttft_ms`, `prompt_per_second`, `predicted_per_second`, `prompt_ms`, `predicted_ms`, `cache_n`, `prompt_n` y `predicted_n` para cada punto de profundidad.

## Licencia

- **arnés**: MIT.
- **carga/llama-vocab.cpp**: de llama.cpp, MIT. Licencia en `carga/LICENSE-llama.cpp`.

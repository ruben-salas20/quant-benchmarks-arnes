# Carga de trabajo

Fija para todas las mediciones. Cambiar cualquier cosa de aquí obliga a medir todo de nuevo.

## Puntos de medida

| Punto | Profundidad | Prefill | Decode |
|---|---|---|---|
| Estándar | 0 | 512 | 128 |
| Tarea | 4096 | 512 | 128 |
| Límite | 16384 | 512 | 128 |

- **Profundidad**: tokens que ya están en la caché KV antes de medir. Cada token generado lee esa caché
  entera, así que el decode se frena al crecer, y más en los escalones pequeños, porque la caché no
  se cuantiza con los pesos.
- **Estándar** se puede comparar con las cifras que publica la comunidad. **Tarea** equivale a un
  fichero de código más una instrucción. **Límite** es lo más lleno que admite el `bf16` de
  MiniCPM5-2B en la RTX 3050 con margen: contexto y escalera tienen que ser los mismos para todos.
- Prefill y decode no cambian entre puntos: así, lo único que cambia es la profundidad.

## Herramientas

- **`llama-bench`**, la referencia del motor: `-p 512 -n 128 -d 0,4096,16384`. Usa tokens al azar
  y no lee el texto de abajo.
- **El arnés**, la experiencia real: peticiones a `llama-server` con el texto de abajo. En el mismo
  punto debe dar cifras cercanas a las de `llama-bench` y algo más bajas.

## Texto del arnés

Fuente: `src/llama-vocab.cpp` de llama.cpp en el commit `972d2313b` (tag `b11028`), con licencia MIT,
Copyright (c) 2023-2026 The ggml authors.

Copiado en `carga/llama-vocab.cpp` (183 255 bytes, sha256
`b9588d7116c11573b378c43bf3c85f87249ad5eb9626324abded4abc7c91e6dc`), con la licencia en
`carga/LICENSE-llama.cpp`. Va dentro del repo para que la carga no cambie al actualizar llama.cpp.

- **Continuación, sin plantilla de chat.** El modelo recibe los tokens del fichero tal cual y sigue
  escribiendo el código. Se parece a lo que hace `llama-bench`, así que las dos se comparan
  directamente. La plantilla de chat se mide aparte.
- **Cortes:** el fichero se tokeniza con el tokenizador del modelo que se mide. Los primeros *d* tokens
  son la profundidad y los 512 siguientes, el prompt. Todo se cuenta en tokens de ese modelo, no en
  líneas ni en palabras.
- **Decode voraz** (`temperature: 0`), con exactamente 128 tokens y el fin de texto ignorado. Así el
  texto generado es el mismo en todas las repeticiones, y eso importa en cuanto entre la especulativa.

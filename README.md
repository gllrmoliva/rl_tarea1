# GridWorld - Tarea 1

Este proyecto es una modificación del entorno **GridWorld (gym)**, desarrollada para la Tarea 1 de la asignatura Aprendizaje por Refuerzo.
Incluye la implementación de los siguientes métodos de aprendizaje por refuerzo:

* Q-Learning
* Double Q-Learning
* SARSA
* Q(λ)-Learning
* SARSA(λ)

Repositorio original: [https://github.com/opocaj92/GridWorldEnvs](https://github.com/opocaj92/GridWorldEnvs)

## Requisitos de instalación

1. Tener instalada la versión **Python 3.9.23** (se recomienda usar pyenv para la gestión de versiones).
2. Instalar las dependencias del proyecto ejecutando:
   ```
   pip install -r requirements.txt
   ```

## Uso

El archivo principal de ejecución es **`tarea1.py`**.
La sintaxis general es la siguiente:

```
python tarea1.py [args]
```

### Argumentos disponibles

| Argumento      | Descripción                                                                                                                           | Valores posibles                                                                                    | Valor por defecto |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------- |
| `--method`     | Método de aprendizaje a utilizar.                                                                                                     | `0` = Q-Learning<br>`1` = SARSA<br>`2` = Double Q-Learning<br>`3` = Q(λ)-Learning<br>`4` = SARSA(λ) | `0`               |
| `--exp_name`   | Nombre del experimento (para los archivos de salida).                                                                                 | Texto libre                                                                                         | `'default'`       |
| `--plot_title` | Título que se mostrará en los gráficos generados.                                                                                     | Texto libre                                                                                         | `''`              |
| `--render`     | Controla la visualización interactiva del entorno.                                                                                    | `0` = sin renderizado<br>`1` = con renderizado                                                      | `0`               |
| `--map`        | Mapa en el que se realizarán los experimentos.                                                                                        | `'map1'`, `'map2'`                                                                                  | `'map1'`          |
| `--fail_rate`  | Porcentaje de fallo en la acción elegida. Si ocurre un fallo, el agente se mueve con un 50% de probabilidad a la izquierda o derecha. | `[0, 1]`                                                                                            | `0`               |
| `--episodes`   | Número de episodios de entrenamiento.                                                                                                 | Entero positivo                                                                                     | `10000`           |
| `--max_steps`  | Máximo de pasos por episodio.                                                                                                         | Entero positivo                                                                                     | `100`             |
| `--lr`         | Tasa de aprendizaje (learning rate).                                                                                                  | `[0, 1]`                                                                                            | `0.2`             |
| `--gamma`      | Factor de descuento.                                                                                                                  | `[0, 1]`                                                                                            | `0.90`            |
| `--lambda`     | Valor de λ (para métodos con traza de elegibilidad).                                                                                  | `[0, 1]`                                                                                            | `0`               |
| `--epsilon`    | Valor inicial de ε para la política ε-greedy.                                                                                         | `[0, 1]`                                                                                            | `1`               |
| `--debug_step` | Intervalo de pasos para mostrar resultados parciales.                                                                                 | Entero positivo                                                                                     | `100`             |


### Ejemplo de uso

```bash
python tarea1.py \
  --method 0 \
  --exp_name 'example' \
  --plot_title 'Ejemplo con Q-Learning, mapa 1' \
  --render 1 \
  --map map1 \
  --fail_rate 0.0 \
  --episodes 10000 \
  --max_steps 100 \
  --lr 0.20 \
  --gamma 0.90 \
  --lambda 0.0 \
  --epsilon 1.0 \
  --debug_step 100
```


> Para realizar los experimentos realizados en la tarea se debe ejecutar `run_experiments.sh` de la siguiente manera: 
 ```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

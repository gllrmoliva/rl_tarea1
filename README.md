## GridWorld Tarea 1
Este proyecto es una modificación de Gridworld (gym) para responder a la tarea
de la asignatura Aprendizaje por Refuerzo. Está cuenta con la implementación de
los métodos: Q-Learning, Double Q-Learning, SARSA, Lambda Q-Learning y Lambda SARSA

Repositorio original: https://github.com/opocaj92/GridWorldEnvs

## Requisitos instalación
1. Tener instalada la versión 3.9.23 de python, esto se puede hacer mediante pyenv.
2. Instalar dependencias mediante `pip install -r requirements.txt`.

## Uso
El archivo principal es `tarea1.py`, su uso está dado por:

``python tarea.py [args]``

los argumentos disponibles son:
- `--method`: método que se usara, puede tomar el valor de: 0(qlearning), 1(sarsa), 2(d-qlearning), 3(qlearning lambda), 4(sarsa lambda). Su valor por defecto es 0.
- `--exp_name`: nombre que tendran los archivos de cierto experimento. su valor por defecto es 'default'.
- `--plot_title`: nombre de método que se mostrara en el gráfico generado.
- `--render`: si es 0 no se mostrara visualización interactiva, si es 1 si se mostrará. se valor por defecto es 0
- `--map`: para en el que se haran los experimentos, puede ser: 'map1' o 'map2'. Valor por defecto es map1
- `--fail_rate`: Porcentaje de fallos que puede tener una acción. si falla tiene 50% de probabilidades de moverse a la izquieda y 50% a la derecha. Valor por defecto: 0
- `--episodes`: cantidad de episodios de entrenamiento. Valor por defecto: 10,000
- `--max_steps`: cantidad de pasos máximos por episodio, valor por defecto: 100
- `--lr`: learning rate, valor por defecto: 0.2
- `--gamma`: valor gamma, valor por defecto: 0.90
- `--lambda`: valor lambda, valor por defecto: 0
- `--epsilon`: epsilon inicial, valor por defecto: 1
- `--debug_step`: cantidad de pasos en las que se mostrara resultados de un episodio, valor por defecto: 100

En caso de querer ejecutar experimentos predeterminados, ejecutar:

``
chmod +x run_experiments.sh
./run_experiments.sh
``

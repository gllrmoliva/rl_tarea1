import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

def save_data(filename, rewards, epsilons, dones):

    with open(filename, 'w') as csvfile:  

        csvwriter = csv.writer(csvfile)  

        csvwriter.writerow(['episode','reward', 'epsilon', 'done'])  

        rows = [[i, x, y, z] for i, (x, y, z) in enumerate(zip(rewards, epsilons, dones))]
        csvwriter.writerows(rows)
    print(f"data saved in {filename}")

def save_params(filename, PARAMS):

    with open(filename, 'w') as csvfile:  

        csvwriter = csv.writer(csvfile)  

        for k,v in PARAMS.items():
            csvwriter.writerow([k,v])  

    print(f"params saved in {filename}")

# filename: nombre de svg
def plot_csv(csv_path, filename, title, step=50):

    df = pd.read_csv(csv_path)
    df_with_step = df.groupby(df.index // step).mean().reset_index(drop=True)
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    sns.set(style="whitegrid")
    
    plt.figure(figsize=(7, 3))
    sns.lineplot(data=df_with_step, x="episode", y="reward")
    plt.title(f"{title} : Reward por Episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    
    output_path = os.path.join("experiments", filename)
    plt.savefig(output_path, format="svg")
    plt.close()
    print("plot saved!")

def cargar_mapa(ruta):
    """
    Carga el mapa desde un archivo .txt con caracteres:
    0=libre, 1=pared, x=inicio, G=meta, B=bomba

    #######################################################################
    Warning: Esta función fue hecha con ayuda de Google Gemini, solo para apoyo visual
    en informe.
    #######################################################################
    """
    with open(ruta, 'r') as f:
        lineas = [line.strip() for line in f if line.strip()]
    
    # Convertimos cada línea a una lista de caracteres
    mapa = np.array([list(linea) for linea in lineas])

    alto, ancho = mapa.shape  # alto = filas, ancho = columnas
    return mapa, ancho, alto

def mostrar_recorrido(mapa, recorrido, output_path):
    """
    Muestra el mapa con colores y el recorrido del agente.

    #######################################################################
    Warning: Esta función fue hecha con ayuda de Google Gemini, solo para apoyo visual
    en informe.
    #######################################################################
    """
    # Definir colores RGB por tipo de casilla
    colores = {
        '0': [1, 1, 1],      # libre - blanco
        '1': [0, 0, 0],      # pared - negro
        'x': [1, 1, 0],      # inicio - amarillo
        'G': [0, 1, 0],      # meta - verde
        'B': [1, 0, 0],      # bomba - rojo
    }

    # Convertimos a imagen RGB
    img = np.zeros((*mapa.shape, 3))
    for simbolo, color in colores.items():
        img[mapa == simbolo] = color

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, interpolation='none')

    # Dibujar recorrido con flechas
    for i in range(len(recorrido) - 1):
        y1, x1 = recorrido[i]
        y2, x2 = recorrido[i + 1]
        dx, dy = x2 - x1, y2 - y1
        ax.arrow(
            x1, y1,
            dx, dy,
            head_width=0.3, head_length=0.3,
            fc='blue', ec='blue', length_includes_head=True
        )

    # Configurar cuadrícula
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, mapa.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, mapa.shape[0], 1))

    ax.grid(color='black', linewidth=0.8)

    ax.set_xlim(-0.5, mapa.shape[1] - 0.5)
    ax.set_ylim(mapa.shape[0] - 0.5, -0.5)

    #plt.show()
    plt.savefig(output_path, format = "svg")
    print("example saved!")

def parse_args():
    parser = argparse.ArgumentParser(
        description="experimentos de tarea1 rl",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--method', type=int, default=0,
        help="0: qlearning\n1: sarsa\n2: double qlearning\n3: qlearning(lambda)\n4: sarsa(lambda)"
    )
    parser.add_argument(
        '--exp_name', type=str, default="default"
    )
    parser.add_argument(
        '--plot_title', type=str, default="default"
    )
    parser.add_argument(
        '--render', type=int, default=0
    )
    parser.add_argument(
        '--map', type=str, default="map1"
    )
    parser.add_argument(
        '--fail_rate', type=float, default=0.0
    )
    parser.add_argument(
        '--episodes', type=int, default=10_000
    )
    parser.add_argument(
        '--max_steps', type=int, default=100
    )
    parser.add_argument(
        '--lr', type=float, default=0.20
    )
    parser.add_argument(
        '--gamma', type=float, default=0.90
    )
    parser.add_argument(
        '--lambda', type=float, default=0.5, dest='lambda_' # palabra asignada O.O
    )
    parser.add_argument(
        '--epsilon', type=float, default=1.0
    )
    parser.add_argument(
        '--debug_step', type=int, default=100
    )

    args = parser.parse_args()

    params = {
        "METHOD": args.method,
        "EXP_NAME": args.exp_name,
        "PLOT_TITLE": args.plot_title,
        "MAP": args.map,
        "FAIL_RATE": args.fail_rate,
        "DEBUG_STEP": args.debug_step,
        "EPISODES": args.episodes,
        "MAX_STEPS": args.max_steps,
        "LEARNING_RATE": args.lr,
        "GAMMA": args.gamma,
        "LAMBDA": args.lambda_,
        "EPSILON": args.epsilon,
        "RENDER": args.render,
    }
    return params

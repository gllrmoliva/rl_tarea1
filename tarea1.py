import numpy as np
import gym
import gym_gridworld
from utils import save_data, save_params, plot_csv, cargar_mapa, mostrar_recorrido, parse_args

def qlearning(env, PARAMS):

    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.
    epsilon = PARAMS["EPSILON"]

    Q = np.zeros((STATES, ACTIONS)) #Inicializa la Q table con 0s.
    rewards = []
    epsilons = []
    dones = []

    for episode in range(PARAMS["EPISODES"]):
        rewards_epi=0
        state = env.reset() #Reinicia el ambiente
        for actual_step in range(PARAMS["MAX_STEPS"]):
                
            #Escoge un valor al azar entre 0 y 1. Si es menor al valor de 
            #epsilon, escoge una acción al azar.
            if np.random.uniform(0, 1) < epsilon: 
                action = env.action_space.sample() 
            else:
                #De lo contrario, escogerá el estado con el mayor valor.
                action = np.argmax(Q[state, :]) 

            #Ejecuta la acción en el ambiente y guarda los nuevos parámetros
            #(estado siguiente, recompensa, ¿terminó?).
            next_state, reward, done, _ = env.step(action) 
            rewards_epi=rewards_epi+reward

            # Calcula la nueva Q table.
            Q[state, action] = Q[state, action] + PARAMS["LEARNING_RATE"] * (reward + PARAMS["GAMMA"] * np.max(Q[next_state, :]) - Q[state, action]) 

            state = next_state

            if done:
                dones.append(1)
                break  

            if (PARAMS["MAX_STEPS"] - 1) == actual_step:
                dones.append(0)

        #Guardar datos en listas
        rewards.append(rewards_epi) 
        epsilons.append(epsilon)

        if (episode % PARAMS["DEBUG_STEP"] == 0):
            print (f"Episode {episode} rewards: {rewards_epi}")
            print(f"Value of epsilon: {epsilon}")
        
        # Actualizar epsilon, TODO: esto podría ser con función externa
        if epsilon > 0.1: epsilon -= 0.0001

    return Q, rewards, epsilons, dones

def doubleqlearning(env, PARAMS):
    
    epsilon = PARAMS["EPSILON"]
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles.

    Q_1 = np.zeros((STATES, ACTIONS)) #Inicializa la Q table con 0s.
    Q_2 = np.zeros((STATES, ACTIONS)) #Inicializa la Q table con 0s.

    rewards = []
    epsilons = []
    dones = []

    for episode in range(PARAMS["EPISODES"]):

        rewards_epi=0
        state = env.reset() #Reinicia el ambiente

        for actual_step in range(PARAMS["MAX_STEPS"]):

            # Escoge un valor al azar entre 0 y 1. Si es menor al valor de 
            # epsilon, escoge una acción al azar.
            if np.random.uniform(0, 1) < epsilon: 
                action = env.action_space.sample() 
            else:
                # CAMBIO: Ahora se escoge acción con respecto a ambas tablas Q
                action = np.argmax(Q_1[state, :] + Q_2[state, :])


            # Ejecuta la acción en el ambiente y guarda los nuevos parámetros 
            # (estado siguiente, recompensa, ¿terminó?).
            next_state, reward, done, _ = env.step(action) 
            rewards_epi=rewards_epi+reward

            # CAMBIO: Ahora se elige de forma aleatoria una de las tablas y
            # se actualiza con respecto a la estimación de la otra tabla 
            # otra tabla
            if np.random.uniform(0, 1) <= 0.5:
                action_max = np.argmax(Q_1[next_state, :])
                Q_1[state, action] += PARAMS["LEARNING_RATE"] * (reward + PARAMS["GAMMA"] * Q_2[next_state, action_max] - Q_1[state, action])
            else:
                action_max = np.argmax(Q_2[next_state, :])
                Q_2[state, action] += PARAMS["LEARNING_RATE"] * (reward + PARAMS["GAMMA"] * Q_1[next_state, action_max] - Q_2[state, action])

            state = next_state

            if done:
                dones.append(1)
                break  

            if (PARAMS["MAX_STEPS"] - 1) == actual_step:
                dones.append(0)

        #Guardar datos en listas
        rewards.append(rewards_epi) 
        epsilons.append(epsilon)

        if (episode % PARAMS["DEBUG_STEP"] == 0):
            print(f"Episode {episode} rewards: {rewards_epi}")
            print(f"Value of epsilon: {epsilon}")

        if epsilon > 0.1: epsilon -= 0.0001

    # CAMBIO: ahora devolvemos ambas tablas por separado
    return Q_1, Q_2, rewards, epsilons, dones


def sarsa(env, PARAMS):

    epsilon = PARAMS["EPSILON"]
    rewards = []
    epsilons = []
    dones = []
    STATES =  env.n_states #Cantidad de estados en el ambiente.
    ACTIONS = env.n_actions #Cantidad de acciones posibles

    Q = np.zeros((STATES, ACTIONS))
    for episode in range(PARAMS["EPISODES"]):
        rewards_epi=0
        #Reinicia el ambiente
        state = env.reset() 

        #Escoge un valor al azar entre 0 y 1. Si es menor al valor de epsilon, 
        #escoge una acción al azar.
        if np.random.uniform(0, 1) < epsilon: 
            action = env.action_space.sample() 
        else:
            #De lo contrario, escogerá el estado con el mayor valor.
            action = np.argmax(Q[state, :]) 

        for actual_step in range(PARAMS["MAX_STEPS"]):

            next_state, reward, done, _ = env.step(action)

            #Escoge un valor al azar entre 0 y 1. Si es menor al valor de 
            # epsilon, escoge una acción al azar.
            if np.random.uniform(0, 1) < epsilon: 
                action2 = env.action_space.sample() 
            else:
                #De lo contrario, escogerá el estado con el mayor valor.
                action2 = np.argmax(Q[next_state, :]) 

            #Calcula la nueva Q table.
            Q[state, action] = Q[state, action] + PARAMS["LEARNING_RATE"] * (reward + PARAMS["GAMMA"] * Q[next_state, action2] - Q[state, action]) 
            rewards_epi=rewards_epi+reward
            state = next_state
            action = action2
            if done:
                dones.append(1)
                break  

            if (PARAMS["MAX_STEPS"] - 1) == actual_step:
                dones.append(0)

        #Guardar datos en listas
        rewards.append(rewards_epi) 
        epsilons.append(epsilon)

        if (episode % PARAMS["DEBUG_STEP"] == 0):
            print(f"Episode {episode} rewards: {rewards_epi}")
            print(f"Value of epsilon: {epsilon}")

        if epsilon > 0.1: epsilon -= 0.0001

    return Q, rewards, epsilons, dones


def qlearning_lambda(env, PARAMS):

    STATES = env.n_states
    ACTIONS = env.n_actions
    epsilon = PARAMS["EPSILON"]

    Q = np.zeros((STATES, ACTIONS))
    rewards = []
    epsilons = []
    dones = []

    for episode in range(PARAMS["EPISODES"]):

        state = env.reset()
        
        # aquí se guardan los traces
        e = np.zeros((STATES, ACTIONS))
        rewards_epi = 0

        for step in range(PARAMS["MAX_STEPS"]):

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            # definición de delta
            delta = reward + PARAMS["GAMMA"] * np.max(Q[next_state, :]) - Q[state, action]

            # incrementar trace
            e[state, action] += 1

            # actualizar Q 
            Q += PARAMS["LEARNING_RATE"] * delta * e

            # decaer todos los traces por gamma y lambda
            e *= PARAMS["GAMMA"] * PARAMS["LAMBDA"]

            state = next_state

            if done:
                dones.append(1)
                break
            if step == PARAMS["MAX_STEPS"] - 1:
                dones.append(0)

        rewards.append(rewards_epi)
        epsilons.append(epsilon)

        if episode % PARAMS["DEBUG_STEP"] == 0:
            print(f"Episode {episode} rewards: {rewards_epi}, epsilon: {epsilon}")
        
        if epsilon > 0.1:
            epsilon -= 0.0001

    return Q, rewards, epsilons, dones


def sarsa_lambda(env, PARAMS):

    STATES = env.n_states
    ACTIONS = env.n_actions
    epsilon = PARAMS["EPSILON"]

    Q = np.zeros((STATES, ACTIONS))

    rewards = []
    epsilons = []
    dones = []

    for episode in range(PARAMS["EPISODES"]):

        state = env.reset()

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Aquí se van a guardar los traces
        e = np.zeros((STATES, ACTIONS))

        rewards_epi = 0

        for step in range(PARAMS["MAX_STEPS"]):
            next_state, reward, done, _ = env.step(action)
            rewards_epi += reward

            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            # definición de delta
            delta = reward + PARAMS["GAMMA"] * Q[next_state, next_action] - Q[state, action]

            # incrementar trace
            e[state, action] += 1

            # actualizamos q values a partir de traces
            Q += PARAMS["LEARNING_RATE"] * delta * e 

            # decaer traces con respecto a gamma y lambda
            e *= PARAMS["GAMMA"] * PARAMS["LAMBDA"]

            state = next_state
            action = next_action

            if done:
                dones.append(1)
                break
            if step == PARAMS["MAX_STEPS"] - 1:
                dones.append(0)

        rewards.append(rewards_epi)
        epsilons.append(epsilon)

        if episode % PARAMS["DEBUG_STEP"] == 0:
            print(f"Episode {episode} rewards: {rewards_epi}, epsilon: {epsilon}")
        
        if epsilon > 0.1:
            epsilon -= 0.0001

    return Q, rewards, epsilons, dones



# Función que simula un episodio y devuelve los estados en los que estuvo
def playgame(env, Q, max_steps = 100):

    env.reset()
    observation = env.reset()

    steps = []
    steps.append(observation)

    for _ in range(max_steps):

        action = np.argmax(Q[observation, :]) #La acción a realizar esta dada por la política
        observation, reward, done, info = env.step(action)

        steps.append(observation)

        if done:
            break

    env.close()
    return steps


#Función para correr juegos siguiendo una determinada política
def playgames(env, Q, num_games, render = True):
    wins = 0
    env.reset()
    #pause=input()
    env.render() 

    for i_episode in range(num_games): # print(f"Episode {i_episode} started")

        rewards_epi=0
        observation = env.reset()
        t = 0

        while True:

        #for _ in range(300):
            action = np.argmax(Q[observation, :]) #La acción a realizar esta dada por la política
            observation, reward, done, info = env.step(action)
            rewards_epi=rewards_epi+reward
            if render: env.render()

            print(observation)
            pause=input()

            if done:
                if reward >= 0:
                    wins += 1
                print(f"Episode {i_episode} finished after {t+1} timesteps with reward {rewards_epi}")
                break
            t += 1
    pause=input()
    env.close()
    print("Victorias: ", wins)

if __name__ == "__main__":

    folder_path = "experiments/"

    PARAMS = parse_args()

    env = gym.make("GridWorld-v0",
                   file_name    = PARAMS["MAP"] + ".txt",
                   fail_rate    = PARAMS["FAIL_RATE"]
                   )

    env.verbose = True
    _ =env.reset()

    Q = None
    
    if (PARAMS["METHOD"] == 0):
        Q, rewards, epsilons, dones = qlearning(env = env,
                                                PARAMS = PARAMS
                                                )
    elif (PARAMS["METHOD"] == 1):
        Q, rewards, epsilons, dones = sarsa(env = env,
                                            PARAMS = PARAMS
                                            )
    elif (PARAMS["METHOD"] == 2):
        Q_1, Q_2, rewards, epsilons, dones = doubleqlearning(env = env,
                                                             PARAMS = PARAMS
                                                             )
        Q = (Q_1 + Q_2) / 2
    elif (PARAMS["METHOD"] == 3):
        Q, rewards, epsilons, dones = qlearning_lambda(env = env,
                                                       PARAMS = PARAMS
                                                       )
    elif (PARAMS["METHOD"] == 4):
        Q, rewards, epsilons, dones = sarsa_lambda(env = env,
                                                   PARAMS = PARAMS
                                                   )
    else:
        print("Ingrese un método válido")
        exit()
    
    # Guardar datos de experiimento
    file_path = folder_path + PARAMS["EXP_NAME"]
    save_data(file_path + ".csv" , rewards, epsilons, dones)
    save_params(file_path + "_params.csv" , PARAMS)

    movements = playgame(env, Q)

    plot_csv(csv_path   = file_path + ".csv",
             filename   = PARAMS["EXP_NAME"] + ".svg",
             title      = PARAMS["PLOT_TITLE"]
             )

    map_, width, height = cargar_mapa("gym_gridworld/envs/" + PARAMS["MAP"] + ".txt")
    steps = []

    for i in movements:
             steps.append((i//width, i%width))

    mostrar_recorrido(map_, steps, file_path + "_example.svg")

    if (PARAMS["RENDER"] == 1):
        playgames(env, Q, 100, True)

    env.close()

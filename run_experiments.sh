
# EXPERIMENTOS PREGUNTA 1

# MAPA 1

python tarea1.py --method 0 --exp_name 'p1_qlearning_map1_default' --plot_title 'Q-learning, mapa 1' --render 0 --map map1 --fail_rate 0.0 --episodes 10000 --max_steps 100 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p1_sarsa_map1_default' --plot_title 'SARSA, mapa 1' --render 0 --map map1 --fail_rate 0.0 --episodes 10000 --max_steps 100 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

# MAPA 2

# default

python tarea1.py --method 0 --exp_name 'p1_qlearning_map2_default' --plot_title 'Q-learning, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 10000 --max_steps 100 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p1_sarsa_map2_default' --plot_title 'SARSA, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 10000 --max_steps 100 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

# aumento max_steps a 500

python tarea1.py --method 0 --exp_name 'p1_qlearning_map2_maxstep500' --plot_title 'Q-learning, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 10000 --max_steps 500 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p1_sarsa_map2_maxstep500' --plot_title 'SARSA, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 10000 --max_steps 500 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

# aumento episodios a 15000 + max_steps a 500

python tarea1.py --method 0 --exp_name 'p1_qlearning_map2_maxstep500_ep15' --plot_title 'Q-learning, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p1_sarsa_map2_maxstep500_ep15' --plot_title 'SARSA, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

# aumento gamma a 0.95 + ep 15000 + maxsteps 500

python tarea1.py --method 0 --exp_name 'p1_qlearning_map2_gamma95' --plot_title 'Q-learning, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p1_sarsa_map2_ep_gamma95' --plot_title 'SARSA, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100

# aumento gamma a 0.99 + ep 15000 + maxsteps 500

python tarea1.py --method 0 --exp_name 'p1_qlearning_map2_gamma99' --plot_title 'Q-learning, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.99 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p1_sarsa_map2_gamma99' --plot_title 'SARSA, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.99 --lambda 0.0 --epsilon 1.0 --debug_step 100

# disminución lr a 0.10 + ep 15000 + maxsteps 500

python tarea1.py --method 0 --exp_name 'p1_qlearning_map2_lr10' --plot_title 'Q-learning, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.10 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p1_sarsa_map2_lr10' --plot_title 'SARSA, mapa 2' --render 0 --map map2 --fail_rate 0.0 --episodes 15000 --max_steps 500 --lr 0.10 --gamma 0.90 --lambda 0.0 --epsilon 1.0 --debug_step 100

# EXPERIMENTOS PREGUNTA 2

# mapa 1 con gamma 0.95

 python tarea1.py --method 0 --exp_name 'p2_qlearning_stocastic_map1' --plot_title 'Q-learning estocástico, mapa 1' --render 0 --map map1 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100

 python tarea1.py --method 1 --exp_name 'p2_sarsa_stocastic_map1' --plot_title 'SARSA estocástico, mapa 1' --render 0 --map map1 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100

# mapa2 con gamma 0.95

python tarea1.py --method 0 --exp_name 'p2_qlearning_stocastic_g95_map2' --plot_title 'Q-learning estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p2_sarsa_stocastic_g95_map2' --plot_title 'SARSA estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100


## EXPERIMENTOS PREGUNTA 3

python tarea1.py --method 0 --exp_name 'p3_qlearning_stocastic_g95_map2' --plot_title 'Q-learning estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 2 --exp_name 'p3_doubleqlearning_stocastic_g95_map2' --plot_title 'Double Q-learning estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.95 --lambda 0.0 --epsilon 1.0 --debug_step 100


# EXPERIMENTOS PREGUNTA 4

python tarea1.py --method 0 --exp_name 'p4_qlearning_stocastic_map2' --plot_title 'Q-learning estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.99 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 1 --exp_name 'p4_sarsa_stocastic_map2' --plot_title 'SARSA estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.99 --lambda 0.0 --epsilon 1.0 --debug_step 100

python tarea1.py --method 3 --exp_name 'p4_qlearninglambda_stoc_map2' --plot_title 'Q-learning Lambda estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.99 --lambda 0.50 --epsilon 1.0 --debug_step 100

python tarea1.py --method 4 --exp_name 'p4_sarsalambda_stoc_map2' --plot_title 'SARSA Lambda estocástico, mapa 2' --render 0 --map map2 --fail_rate 0.30 --episodes 15000 --max_steps 500 --lr 0.20 --gamma 0.99 --lambda 0.50 --epsilon 1.0 --debug_step 100

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')

#parametros do agente

alpha = 0.1 # É a taxa de aprendizado, define o quanto novas informações sobrepõe as antigas
gamma = 0.995 # O quão importante são as recompensas
epsilon = 1 # Aleatoriedade das ações
epsilon_decay = 0.9991 # Queda do epsilon
min_epsilon = 0.001 #Número mínimo do epsilon, garante sempre uma mínima aleatóriedade
num_episodes = 2000
max_steps = 200 # Quantidade maxima de ações que o agente pode realizar até um episódio ser encerrado

#q-table

q_table = np.zeros((env.observation_space.n, env.action_space.n)) # quantos estados o jogo pode ter e quantas ações posso tomar por estado 



#funcionamento das escolhas das ações

def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])


rewards_per_episode = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    total_reward = 0
    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
    
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
    
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    
        state = next_state
        total_reward += reward
    
        if done or truncated:
            break

    # após o loop interno de steps:
    rewards_per_episode.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 1000 == 0:
        print(f"Episódio {episode}, epsilon={epsilon:.3f}")
#reduz caso não esteja abaixo do mínimo
        
#iniciando o ambiente 

env = gym.make('Taxi-v3', render_mode='human')

for episode in range(5):
    state, _ = env.reset()
    done = False
    
     # garante que o ambiente não está terminado ao iniciar
    while done:
        state, _ = env.reset()

    print('Episode', episode)

    for step in range(max_steps):
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        env.render()

        if done or truncated:
            env.render()
            print('Finished episode', episode, 'with reward', reward)
            break
        
#printando o treinamento

plt.plot(rewards_per_episode)
plt.xlabel('Episódio')
plt.ylabel('Recompensa total')
plt.title('Recompensa por Episódio durante o Treinamento')
plt.grid()
plt.savefig("recompensa_treinamento.png")
print("Gráfico salvo como 'recompensa_treinamento.png'")



env.close()        
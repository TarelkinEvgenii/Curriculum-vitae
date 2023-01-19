import copy
import math
from collections import deque, namedtuple
from itertools import count

import gym
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pyglet.window import key
from torch import nn, optim
from torch.distributions import Categorical


def interactive_cartpole():
    """
    Allows you to control the cart with the arrow keys.
    Press Space to reset the cart-pole and press Escape to close the window.
    """

    env = gym.make('CartPole-v1')
    # Make sure to render once so that the viewer exists
    env.reset()
    env.render()
    # Inject key_handler
    key_handler = key.KeyStateHandler()
    env.viewer.window.push_handlers(key_handler)

    action = 0
    already_done = False
    t = 0
    while True:
        if key_handler[key.ESCAPE]:
            break

        if key_handler[key.SPACE]:
            env.reset()
            action = 0
            t = 0
            already_done = False

        if key_handler[key.LEFT]:
            action = 0
        elif key_handler[key.RIGHT]:
            action = 1

        observation, reward, done, info = env.step(action)
        env.render()

        if done and not already_done:
            print(f'Episode finished after {t + 1} time steps')
            already_done = True

        t += 1

    env.close()


def discretize(state, ANGLE_MINMAX, ANGLE_DOT_MINMAX, ANGLE_STATE_SIZE, ANGLE_DOT_STATE_SIZE):
    # state[2] -> angle (Pole Angle)
    # state[3] -> angle_dot (Pole Angular Velocity)
    discretized = np.array([0, 0])  # Initialised discrete array
    angle_window = (ANGLE_MINMAX - (-ANGLE_MINMAX)) / ANGLE_STATE_SIZE
    discretized[0] = (state[2] - (-ANGLE_MINMAX)) // angle_window
    discretized[0] = min(ANGLE_STATE_SIZE - 1, max(0, discretized[0]))

    angle_dot_window = (ANGLE_DOT_MINMAX - (-ANGLE_DOT_MINMAX)) / ANGLE_DOT_STATE_SIZE
    discretized[1] = (state[3] - (-ANGLE_DOT_MINMAX)) // angle_dot_window
    discretized[1] = min(ANGLE_DOT_STATE_SIZE - 1, max(0, discretized[1]))

    return tuple(discretized.astype(np.int))


def cart_pole_SARSA(env, EPISODES=100, rendering=False,
                    DISCOUNT=0.95, LEARNING_RATE=0.25, EPSILON=0.2,
                    EPISODES_WITH_BEST_MATRIX=100):
    env.reset()
    """
    # env.action_space = 0 - left or 1 - right
    # new_state [cart position , cart velocity , pole angle , pole angular velocity]

    # A reward of +1 is given for every step taken and an episode terminates when
    # 1) Pole angle ≥12 degrees
    # 2) The cart centre reaches display edge
    # 3) Episode length is ≥ 500 (200 for v0)

    # observation_space describe the format of valid observations
    # The Box space represents an n-dimensional box, so valid observations will be an array of 4 numbers.
    # We can also check the Box’s bounds:
    # print(env.observation_space.high)
    # print(env.observation_space.low)
    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Cart Position         | -4.8*                 | 4.8*                |
    | 1   | Cart Velocity         | -Inf                 | Inf                |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°)** | ~ 0.418 rad (24°)** |
    | 3   | Pole Angular Velocity | -Inf                 | Inf                |
    **Note:** above denotes the ranges of possible observations for each element, but in two cases this range exceeds the
    range of possible values in an un-terminated episode:
    - `*`: the cart x-position can be observed between `(-4.8, 4.8)`, but an episode terminates if the cart leaves the
    `(-2.4, 2.4)` range.
    - `**`: Similarly, the pole angle can be observed between  `(-.418, .418)` radians or precisely **±24°**, but an episode is
    terminated if the pole angle is outside the `(-.2095, .2095)` range or precisely **±12°**
    """

    # Initialization
    angle_minmax = env.observation_space.high[2]
    angle_dot_minmax = math.radians(50)
    angle_state_size = 50
    angle_dot_state_size = 50
    matrix_of_Q_values = np.random.randn(angle_state_size, angle_dot_state_size, env.action_space.n)

    previous_mean = 0
    matrix_of_Q_values_max_mean = []

    total_returns = []
    for i in range(EPISODES):
        done = False
        returns = 0
        current_state = env.reset()

        disc_current_state = discretize(state=current_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                        ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)
        # Random action or not
        if np.random.random() > EPSILON:
            action = np.argmax(matrix_of_Q_values[disc_current_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        while not done:
            new_state, reward, done, info = env.step(action)

            disc_new_state = discretize(state=new_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                        ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)
            if np.random.random() > EPSILON:
                new_action = np.argmax(matrix_of_Q_values[disc_new_state])
            else:
                new_action = np.random.randint(0, env.action_space.n)

            if not done:
                matrix_of_Q_values[disc_current_state + (action,)] = \
                    matrix_of_Q_values[disc_current_state + (action,)] + \
                    LEARNING_RATE * (reward + DISCOUNT * matrix_of_Q_values[
                        disc_new_state + (new_action,)] -
                                     matrix_of_Q_values[
                                         disc_current_state + (action,)])

            disc_current_state = disc_new_state
            action = new_action
            returns += reward

            print('\r' + str(i + 1) + ' / ' + str(EPISODES), end='')
            if rendering: env.render()

        total_returns.append(returns)

        ####################################################################################################

        # Getting the best matrix_of_Q_values
        if i == 0:
            current_mean = total_returns[0]
        else:
            current_mean = np.mean(total_returns)

        if previous_mean < current_mean:
            matrix_of_Q_values_max_mean = np.copy(matrix_of_Q_values)
            previous_mean = current_mean

        # you can check changing
        # lp=np.reshape(matrix_of_Q_values_max_mean==matrix_of_Q_values,[-1])
        # minlp=lp[np.argmin(np.reshape(matrix_of_Q_values_max_mean==matrix_of_Q_values,[-1]))]

    # Using matrix_of_Q_values_max_mean for computing 100 in row
    total_returns_with_best_Q_matrix = []
    for i in range(EPISODES_WITH_BEST_MATRIX):
        done = False
        returns = 0
        current_state = env.reset()
        disc_current_state = discretize(state=current_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                        ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)
        while not done:
            action = np.argmax(matrix_of_Q_values_max_mean[disc_current_state])

            new_state, reward, done, info = env.step(action)

            disc_current_state = discretize(state=new_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                            ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)
            returns += reward
        total_returns_with_best_Q_matrix.append(returns)

    env.close()
    return total_returns, total_returns_with_best_Q_matrix


def cart_pole_Q_learning(env, EPISODES=100, rendering=False,
                         DISCOUNT=0.95, LEARNING_RATE=0.25, EPSILON=0.2,
                         EPISODES_WITH_BEST_MATRIX=100):
    env.reset()
    """
    # env.action_space = 0 - left or 1 - right
    # new_state [cart position , cart velocity , pole angle , pole angular velocity]

    # A reward of +1 is given for every step taken and an episode terminates when
    # 1) Pole angle ≥12 degrees
    # 2) The cart centre reaches display edge
    # 3) Episode length is ≥ 500 (200 for v0)

    # observation_space describe the format of valid observations
    # The Box space represents an n-dimensional box, so valid observations will be an array of 4 numbers.
    # We can also check the Box’s bounds:
    # print(env.observation_space.high)
    # print(env.observation_space.low)
    | Num | Observation           | Min                  | Max                |
    |-----|-----------------------|----------------------|--------------------|
    | 0   | Cart Position         | -4.8*                | 4.8*               |
    | 1   | Cart Velocity         | -Inf                 | Inf                |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°)** | ~ 0.418 rad (24°)** |
    | 3   | Pole Angular Velocity | -Inf                 | Inf                |
    **Note:** above denotes the ranges of possible observations for each element, 
    but in two cases this range exceeds the range of possible values in an un-terminated episode:
    - `*`: the cart x-position can be observed between `(-4.8, 4.8)`, but an episode terminates if the cart leaves the
    `(-2.4, 2.4)` range.
    - `**`: Similarly, the pole angle can be observed between  `(-.418, .418)` radians or precisely **±24°**, 
    but an episode is terminated if the pole angle is outside the `(-.2095, .2095)` range or precisely **±12°**
    """

    # Initialization
    angle_minmax = env.observation_space.high[2]
    angle_dot_minmax = math.radians(50)
    angle_state_size = 50
    angle_dot_state_size = 50
    matrix_of_Q_values = np.random.randn(angle_state_size, angle_dot_state_size, env.action_space.n)

    previous_mean = 0
    matrix_of_Q_values_max_mean = []

    total_returns = []
    for i in range(EPISODES):
        done = False
        returns = 0
        current_state = env.reset()

        disc_current_state = discretize(state=current_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                        ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)

        while not done:
            # Random action or not
            if np.random.random() > EPSILON:
                action = np.argmax(matrix_of_Q_values[disc_current_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, info = env.step(action)

            disc_new_state = discretize(state=new_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                        ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)

            if not done:
                matrix_of_Q_values[disc_current_state[0], disc_current_state[1], action] = \
                    matrix_of_Q_values[disc_current_state[0], disc_current_state[1], action] + \
                    LEARNING_RATE * (reward +
                                     DISCOUNT * np.max(matrix_of_Q_values[disc_new_state[0], disc_new_state[1]]) -
                                     matrix_of_Q_values[disc_current_state[0], disc_current_state[1], action])

            disc_current_state = disc_new_state
            returns += reward

            print('\r' + str(i + 1) + ' / ' + str(EPISODES), end='')
            if rendering: env.render()

        total_returns.append(returns)

        ####################################################################################################

        # Getting the best matrix_of_Q_values
        if i == 0:
            current_mean = total_returns[0]
        else:
            current_mean = np.mean(total_returns)

        if previous_mean < current_mean:
            matrix_of_Q_values_max_mean = np.copy(matrix_of_Q_values)
            previous_mean = current_mean

        # you can check changing
        # lp=np.reshape(matrix_of_Q_values_max_mean==matrix_of_Q_values,[-1])
        # minlp=lp[np.argmin(np.reshape(matrix_of_Q_values_max_mean==matrix_of_Q_values,[-1]))]

    # Using matrix_of_Q_values_max_mean for computing 100 in row
    total_returns_with_best_Q_matrix = []
    for i in range(EPISODES_WITH_BEST_MATRIX):
        done = False
        returns = 0
        current_state = env.reset()
        disc_current_state = discretize(state=current_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                        ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)
        while not done:
            action = np.argmax(matrix_of_Q_values_max_mean[disc_current_state])

            new_state, reward, done, info = env.step(action)

            disc_current_state = discretize(state=new_state, ANGLE_MINMAX=angle_minmax, ANGLE_DOT_MINMAX=angle_dot_minmax,
                                            ANGLE_STATE_SIZE=angle_state_size, ANGLE_DOT_STATE_SIZE=angle_state_size)
            returns += reward
        total_returns_with_best_Q_matrix.append(returns)

    env.close()
    return total_returns, total_returns_with_best_Q_matrix


class Cart_pole_FC_NN(nn.Module):
    def __init__(self):
        super(Cart_pole_FC_NN, self).__init__()
        self.fc1 = nn.Linear(4, 100)  # 4 parameters as the observation space and 100 to get more data from 4 parameters
        # self.fc2 = nn.Linear(100, 100)  # 2 hidden layers is too unstable for this reason it is useless
        self.for_action = nn.Linear(100, 2)  # 2 number of actions
        self.for_probabilities = nn.Linear(100, 1)  # but choose 1 action
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.for_probabilities(x)
        probability = F.softmax(self.for_action(x), dim=-1)
        return probability, action


def select_action(state, model, List_of_Saved_Action):
    # for choosing actions after training model if necessary
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(List_of_Saved_Action(m.log_prob(action), state_value))
    return action.item()


def train_NN(env, model, List_of_Saved_Action,
             PRINT_STATISTIC=True,
             PRINT_EVERY_Nth_EPISODE=10,
             N_IN_ROW_DONE=100,
             DONE_CONDITION=300):
    learning_rate = 0.005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    eps = np.finfo(np.float32).eps.item()
    total_returns = []

    for episode in count():  # inf episodes
        state = env.reset()
        ep_return = 0
        for _ in range(501):
            action = select_action(state, model, List_of_Saved_Action)
            state, reward, done, info = env.step(action)
            model.rewards.append(reward)
            ep_return += reward
            if done:
                break

        total_returns.append(ep_return)
        if episode == 0:
            ep_return_mean = ep_return
        else:
            ep_return_mean = np.mean(total_returns)

        # For updating weights
        if True:
            # Calculate losses and perform backpropagation
            reward_r = 0
            saved_actions = model.saved_actions
            policy_losses = []
            value_losses = []
            returns = []

            for itr_r in model.rewards[::-1]:
                reward_r = itr_r + 0.99 * reward_r  # insert discounting
                returns.insert(0, reward_r)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            for (probability, value), reward_r in zip(saved_actions, returns):
                advantage = reward_r - value.item()

                policy_losses.append(-probability * advantage)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([reward_r])))

            optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

            loss.backward()
            optimizer.step()

            del model.rewards[:]
            del model.saved_actions[:]

        if PRINT_STATISTIC:
            if episode % PRINT_EVERY_Nth_EPISODE == 0:  # We will print some things out
                last_n_done_return = total_returns[-PRINT_EVERY_Nth_EPISODE:]
                print('Episode: ' + str(episode) +
                      '  \tReturn: ' + str(round(ep_return)) +
                      '   \tWhole Mean Return: ' + str(round(ep_return_mean * 10) / 10) +
                      '  \tLast ' + str(PRINT_EVERY_Nth_EPISODE) + ' Mean Return: '
                      + str(round(np.mean(last_n_done_return))) +
                      '   \tLast ' + str(PRINT_EVERY_Nth_EPISODE) + ' Min: '
                      + str(round(last_n_done_return[np.argmin(last_n_done_return)])))
        else:
            print('\r' + str(episode), end='')

        # Conditions for completion

        # if mean of ep returns is above env.spec.reward_threshold
        # env.spec.reward_threshold in cart pole v0 is equal 195
        # env.spec.reward_threshold in cart pole v1 is equal 475
        if ep_return_mean > env.spec.reward_threshold:
            return total_returns

        # if last n returns is equal or above env.spec.reward_threshold
        last_n_done_return = total_returns[-N_IN_ROW_DONE:]
        if last_n_done_return[np.argmin(last_n_done_return)] >= DONE_CONDITION:
            return total_returns

    return total_returns


def train_NN_with_experience_replay(env, model, List_of_Saved_Action,
                                    PRINT_STATISTIC=True,
                                    PRINT_EVERY_Nth_EPISODE=10,
                                    N_IN_ROW_DONE=100,
                                    DONE_CONDITION=300,
                                    MEMORY_SIZE=1000000,
                                    AMOUM_OF_MEMORY_PER_EXPERIMENT=100):
    learning_rate = 0.005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    eps = np.finfo(np.float32).eps.item()
    total_returns = []

    replay_memory_saved_actions = deque(maxlen=MEMORY_SIZE)
    replay_memory_returns = deque(maxlen=MEMORY_SIZE)
    replay_memory_state = deque(maxlen=MEMORY_SIZE)
    torch.autograd.set_detect_anomaly(True)

    for episode in count():  # inf episodes
        state = env.reset()
        ep_return = 0
        for _ in range(501):
            # choosing action
            if True:
                state = torch.from_numpy(state).float()
                replay_memory_state.append(copy.copy(state))
                probs, state_value = model(state)
                m = Categorical(probs)
                action = m.sample()
                model.saved_actions.append(List_of_Saved_Action(m.log_prob(action), state_value))
                replay_memory_saved_actions.append(
                    copy.copy(List_of_Saved_Action(m.log_prob(action), state_value)))  # replay memory
                action = action.item()

            state, reward, done, info = env.step(action)
            model.rewards.append(reward)
            ep_return += reward
            if done:
                break

        total_returns.append(ep_return)
        if episode == 0:
            ep_return_mean = ep_return
        else:
            ep_return_mean = np.mean(total_returns)

        # For updating weights
        if True:
            # Calculate losses and perform backpropagation
            reward_r = 0
            saved_actions = model.saved_actions
            policy_losses = []
            value_losses = []
            returns = []

            for itr_r in model.rewards[::-1]:
                reward_r = itr_r + 0.99 * reward_r  # insert discounting
                returns.insert(0, reward_r)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # replay memory start
            list_of_returns = list(returns)
            for i in range(len(list_of_returns)):
                replay_memory_returns.append(list_of_returns)
            # replay memory end

            for (probability, value), reward_r in zip(saved_actions, returns):
                advantage = reward_r - value.item()

                policy_losses.append(-probability * advantage)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([reward_r])))

            optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

            loss.backward(retain_graph=True)

            optimizer.step()

            del model.rewards[:]
            del model.saved_actions[:]

        # Using Experience Replay buffer
        if len(replay_memory_returns) < AMOUM_OF_MEMORY_PER_EXPERIMENT:
            current_amount_of_memory = len(replay_memory_returns)
        else:
            current_amount_of_memory = AMOUM_OF_MEMORY_PER_EXPERIMENT
        samples = np.random.choice(len(replay_memory_returns), current_amount_of_memory, replace=False)
        for i in range(current_amount_of_memory):
            # if replay_memory_saved_actions[samples[i]] not in model.saved_actions:
            model.rewards.append(copy.copy(replay_memory_returns[samples[i]]))
            model.saved_actions.append(copy.copy(replay_memory_saved_actions[samples[i]]))
            probs, state_value = model(copy.copy(replay_memory_state[samples[i]]))

        raise ValueError('\nCUSTOM ERROR: Multiple attempts to make this task 2.C was not crowned with success.\n'
                         '\tCommented code block that issues the errors located below the raise of this error. '
                         '\n\tThese Errors described in the report.\n')

        # For updating weights
        if True:
            # Calculate losses and perform backpropagation
            reward_r = 0
            saved_actions = model.saved_actions
            policy_losses = []
            value_losses = []
            returns = []
            discount = 0.99

            for itr_r in model.rewards[::-1]:
                reward_r = itr_r + discount * reward_r  # insert discounting
                returns.insert(0, reward_r)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            for (probability, value), reward_r in zip(saved_actions, returns):
                advantage = reward_r - value.item()

                policy_losses.append(-probability * advantage)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([reward_r])))

            optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

            loss.backward()

            optimizer.step()

            del model.rewards[:]
            del model.saved_actions[:]

        if PRINT_STATISTIC:
            if episode % PRINT_EVERY_Nth_EPISODE == 0:  # We will print some things out
                last_N_done_return = total_returns[-PRINT_EVERY_Nth_EPISODE:]
                print('Episode: ' + str(episode) +
                      '  \tReturn: ' + str(round(ep_return)) +
                      '   \tWhole Mean Return: ' + str(round(ep_return_mean * 10) / 10) +
                      '  \tLast ' + str(PRINT_EVERY_Nth_EPISODE) + ' Mean Return: '
                      + str(round(np.mean(last_N_done_return))) +
                      '   \tLast ' + str(PRINT_EVERY_Nth_EPISODE) + ' Min: '
                      + str(round(last_N_done_return[np.argmin(last_N_done_return)])))
        else:
            print('\r' + str(episode), end='')

        # Conditions for completion

        # if mean of ep returns is above env.spec.reward_threshold
        # env.spec.reward_threshold in cart pole v0 is equal 195
        # env.spec.reward_threshold in cart pole v1 is equal 475
        if ep_return_mean > env.spec.reward_threshold:
            return total_returns

        # if last n returns is equal or above env.spec.reward_threshold
        last_N_done_return = total_returns[-N_IN_ROW_DONE:]
        if last_N_done_return[np.argmin(last_N_done_return)] >= DONE_CONDITION:
            return total_returns

    return total_returns


if __name__ == '__main__':

    # Initialization

    # Necessary booleans
    TASK2A_SARSA = False
    TASK2A_Q_LEARNING = False
    TASK2B = True
    TASK2C = False

    if TASK2A_SARSA:
        print('Task 2A SARSA')
        env = gym.make('CartPole-v1')

        total_returns, total_returns_with_best_q_matrix = cart_pole_SARSA(env, EPISODES=50000, rendering=False)

        # Plotting n in row
        index_of_min_value = np.argmin(total_returns_with_best_q_matrix)
        min_value = total_returns_with_best_q_matrix[index_of_min_value]

        total_epoch = np.arange(1, len(total_returns_with_best_q_matrix) + 1, 1)
        plt.plot(total_epoch, total_returns_with_best_q_matrix, label='Results with best Q-matrix')

        plt.plot(index_of_min_value + 1, min_value, 'r.',
                 label='Min value:' + str(round(min_value)) + ' idx:' + str(index_of_min_value))

        plt.xlabel('All episodes')
        plt.ylabel('Total return')
        plt.title('Cart pole v1 SARSA\n' + str(len(total_epoch)) + ' in row with best Q-matrix')
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Cart_Pole_SARSA_' + str(len(total_returns)) + '_in_row.png')
        plt.show()

        # Plotting main plot
        total_epoch = np.arange(1, len(total_returns) + 1, 1)
        # Computing avg and median
        mean_sarsa = []
        median_sarsa = []
        print('\nGetting median and average')
        for itr_total_returns in range(len(total_returns)):
            if itr_total_returns == 0:
                mean_sarsa.append(total_returns[itr_total_returns])
                median_sarsa.append(total_returns[itr_total_returns])
            else:
                mean_sarsa.append(np.mean(total_returns[:itr_total_returns + 1]))
                median_sarsa.append(np.median(total_returns[:itr_total_returns + 1]))
            print('\r' + str(itr_total_returns + 1) + ' / ' + str(len(total_returns)), end='')

        index_max_mean = np.argmax(mean_sarsa)
        index_max_median = np.argmax(median_sarsa)
        # Plotting
        plt.plot(total_epoch, total_returns, label='Results')
        plt.plot(total_epoch, mean_sarsa, label='Mean')
        plt.plot(total_epoch, median_sarsa, label='Median')

        plt.plot(index_max_mean + 1, round(mean_sarsa[index_max_mean]),
                 'r.', label='MaxMean:' + str(round(mean_sarsa[index_max_mean])) + ' idx:' + str(index_max_mean))
        plt.plot(index_max_median + 1, round(median_sarsa[index_max_median]),
                 'b.', label='MaxMedian:' + str(round(median_sarsa[index_max_median])) + ' idx:' + str(index_max_median))

        plt.xlabel('All episodes')
        plt.ylabel('Total return')
        plt.title('Cart pole v1 SARSA')
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Cart_Pole_SARSA_' + str(len(total_returns)) + '.png')
        plt.show()

        print()

    if TASK2A_Q_LEARNING:
        print('Task 2A Q-Learning')
        env = gym.make('CartPole-v1')

        total_returns, total_returns_with_best_q_matrix = cart_pole_Q_learning(env, EPISODES=50000, rendering=False,
                                                                               EPISODES_WITH_BEST_MATRIX=100)

        # Plotting n in row
        total_epoch = np.arange(1, len(total_returns_with_best_q_matrix) + 1, 1)
        plt.plot(total_epoch, total_returns_with_best_q_matrix, label='Results with best Q-matrix')
        plt.xlabel('All episodes')
        plt.ylabel('Total return')
        plt.title('Cart pole v1 Q-Learning\n' + str(len(total_epoch)) + ' in row with best Q-matrix')
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Cart_Pole_Q-Learning_' + str(len(total_returns)) + '_in_row.png')
        plt.show()

        # Plotting main plot
        total_epoch = np.arange(1, len(total_returns) + 1, 1)
        # Computing avg and median
        mean_q = []
        median_q = []
        print('\nGetting median and average')
        for itr_total_returns in range(len(total_returns)):
            if itr_total_returns == 0:
                mean_q.append(total_returns[itr_total_returns])
                median_q.append(total_returns[itr_total_returns])
            else:
                mean_q.append(np.mean(total_returns[:itr_total_returns + 1]))
                median_q.append(np.median(total_returns[:itr_total_returns + 1]))
            print('\r' + str(itr_total_returns + 1) + ' / ' + str(len(total_returns)), end='')

        index_max_mean = np.argmax(mean_q)
        index_max_median = np.argmax(median_q)
        # Plotting main plot
        plt.plot(total_epoch, total_returns, label='Results')
        plt.plot(total_epoch, mean_q, label='Mean')
        plt.plot(total_epoch, median_q, label='Median')

        plt.plot(index_max_mean + 1, round(mean_q[index_max_mean]),
                 'r.', label='MaxMean:' + str(round(mean_q[index_max_mean])) + ' idx:' + str(index_max_mean))
        plt.plot(index_max_median + 1, round(median_q[index_max_median]),
                 'b.', label='MaxMedian:' + str(round(median_q[index_max_median])) + ' idx:' + str(index_max_median))

        plt.xlabel('All episodes')
        plt.ylabel('Total return')
        plt.title('Cart pole v1 Q-Learning')
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Cart_Pole_Q-Learning_' + str(len(total_returns)) + '.png')
        plt.show()

        print()

    if TASK2B:
        env = gym.make('CartPole-v1')
        model = Cart_pole_FC_NN()
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        # done_condition=env.spec.reward_threshold
        done_condition = 475
        n_in_row = 100

        total_returns = train_NN(env, model, SavedAction, N_IN_ROW_DONE=n_in_row, DONE_CONDITION=done_condition)

        total_epoch = np.arange(1, len(total_returns) + 1, 1)

        # Computing avg and median
        mean_q = []
        median_q = []
        print('\nGetting median and average')
        for itr_total_returns in range(len(total_returns)):
            if itr_total_returns == 0:
                mean_q.append(total_returns[itr_total_returns])
                median_q.append(total_returns[itr_total_returns])
            else:
                mean_q.append(np.mean(total_returns[:itr_total_returns + 1]))
                median_q.append(np.median(total_returns[:itr_total_returns + 1]))
            print('\r' + str(itr_total_returns + 1) + ' / ' + str(len(total_returns)), end='')
        index_max_mean = np.argmax(mean_q)
        index_max_median = np.argmax(median_q)
        # Plotting main plot
        plt.plot(total_epoch, total_returns, label='Results')
        plt.plot(total_epoch, mean_q, label='Mean')
        plt.plot(total_epoch, median_q, label='Median')

        plt.plot(index_max_mean + 1, round(mean_q[index_max_mean]),
                 'r.', label='MaxMean:' +
                             str(round(mean_q[index_max_mean])) +
                             ' idx:' + str(index_max_mean))
        plt.plot(index_max_median + 1, round(median_q[index_max_median]),
                 'b.', label='MaxMedian:' +
                             str(round(median_q[index_max_median])) +
                             ' idx:' + str(index_max_median))

        plt.xlabel('All episodes')
        plt.ylabel('Total return')
        plt.title('Cart pole v1 with NN\nat least ' +
                  str(done_condition) +
                  ' for ' + str(n_in_row) + ' episodes in a row')
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Cart_Pole_NN_' + str(len(total_returns)) + '.png')
        plt.show()

    if TASK2C:
        env = gym.make('CartPole-v1')
        model = Cart_pole_FC_NN()
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        # or env.spec.reward_threshold
        done_condition = 475
        n_in_row = 100

        total_returns = train_NN_with_experience_replay(env, model,
                                                        SavedAction,
                                                        N_IN_ROW_DONE=n_in_row,
                                                        DONE_CONDITION=done_condition)

        total_epoch = np.arange(1, len(total_returns) + 1, 1)

        # Computing avg and median
        mean_q = []
        median_q = []
        print('\nGetting median and average')
        for itr_total_returns in range(len(total_returns)):
            if itr_total_returns == 0:
                mean_q.append(total_returns[itr_total_returns])
                median_q.append(total_returns[itr_total_returns])
            else:
                mean_q.append(np.mean(total_returns[:itr_total_returns + 1]))
                median_q.append(np.median(total_returns[:itr_total_returns + 1]))
            print('\r' + str(itr_total_returns + 1) + ' / ' + str(len(total_returns)), end='')
        index_max_mean = np.argmax(mean_q)
        index_max_median = np.argmax(median_q)
        # Plotting main plot
        plt.plot(total_epoch, total_returns, label='Results')
        plt.plot(total_epoch, mean_q, label='Mean')
        plt.plot(total_epoch, median_q, label='Median')

        plt.plot(index_max_mean + 1, round(mean_q[index_max_mean]),
                 'r.', label='MaxMean:' +
                             str(round(mean_q[index_max_mean])) +
                             ' idx:' + str(index_max_mean)
                 )
        plt.plot(index_max_median + 1, round(median_q[index_max_median]),
                 'b.', label='MaxMedian:' +
                             str(round(median_q[index_max_median])) +
                             ' idx:' + str(index_max_median)
                 )

        plt.xlabel('All episodes')
        plt.ylabel('Total return')
        plt.title('Cart pole v1 with NN\nat least ' +
                  str(done_condition) + ' for ' +
                  str(n_in_row) + ' episodes in a row\ntask 2C'
                  )
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Cart_Pole_NN_' + str(len(total_returns)) + '_2C.png')
        plt.show()

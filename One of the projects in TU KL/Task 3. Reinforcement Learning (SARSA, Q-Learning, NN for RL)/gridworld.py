import json
import random
import tkinter

from tkinter import Tk, Canvas, BOTH
import traceback
import matplotlib.pyplot as plt
import numpy as np


def apply_action(state, action):
    x, y = state.x, state.y
    if action == 'left':
        x -= 1
    elif action == 'right':
        x += 1
    elif action == 'up':
        y -= 1
    elif action == 'down':
        y += 1

    return x, y


class Cell:
    """
    Cell base class. This corresponds to the behaviour of a blank cell
    """

    def __init__(self, world, x, y):
        self.world = world
        self.x = x
        self.y = y
        self.reachable = True
        self.terminal = False

    def to_dict(self):
        res = self.__dict__.copy()
        res['__cls__'] = type(self).__name__
        del res['world']
        return res

    def from_dict(self, dictionary):
        if '__cls__' in dictionary:
            del dictionary['__cls__']

        self.__dict__.update(dictionary)

    def step(self, action):
        """
        Sample the next state when an agent performs the action while in this state.

        :param action: The action that the agent wants to take
        :return: the resulting state
        """
        if self.terminal:
            return self

        x, y = apply_action(self, action)

        state = self.world.get_state(x, y)

        return state

    def allow_enter(self, old_state, action):
        """
        Specify whether the agent can enter this state. The base implementation simply checks if the old state is next
        to the current state.
        :param old_state: The previous state that the agent is coming from
        :param action: The action that the agent wants to take
        :return: bool that specifies if the agent can enter or not
        """
        if not self.reachable:
            return False

        return (abs(self.x - old_state.x) + abs(self.y - old_state.y)) <= 1

    def get_afterstates(self, action):
        """
        This should return all possible states that can be output by step
        if that method is called with parameter action
        """
        if self.terminal:
            return [self]

        x, y = apply_action(self, action)
        return [self.world.get_state(x, y)]

    def p_step(self, action, new_state):
        """
        Compute the probability that self.step(action) will return new_state
        """
        if self.terminal:
            return int(new_state == self)

        x, y = apply_action(self, action)

        return int(new_state == self.world.get_state(x, y))

    def p_enter(self, old_state, action):
        """
        Compute the probability that self.allow_enter(old_state, action) will return True
        """
        if self.allow_enter(old_state, action):
            return 1

        return 0

    def __eq__(self, other):
        """
        Overwriting __eq__ and __hash__ means that Cells can be used as dictionary keys.
        """
        if not isinstance(other, Cell):
            return False

        return self.x == other.x and self.y == other.y

    def __hash__(self):
        """
        Overwriting __eq__ and __hash__ means that Cells can be used as dictionary keys.
        """
        return hash((self.x, self.y))

    def __str__(self):
        return f'{type(self).__name__} at ({self.x}, {self.y})'

    def __repr__(self):
        return str(self)


class BlankCell(Cell):
    pass


class StartCell(Cell):
    def allow_enter(self, old_state, action):
        return super(StartCell, self).allow_enter(old_state, action) or old_state.terminal


class GoalCell(Cell):
    def __init__(self, world, x, y):
        super(GoalCell, self).__init__(world, x, y)

        self.terminal = True


class WallCell(Cell):
    def __init__(self, world, x, y):
        super(WallCell, self).__init__(world, x, y)

        self.reachable = False


class ArrowCell(Cell):
    def __init__(self, world, x, y, direction='left'):
        super(ArrowCell, self).__init__(world, x, y)

        self.direction = direction

    def step(self, action):
        return Cell.step(self, self.direction)

    def get_afterstates(self, action):
        x, y = apply_action(self, self.direction)
        return [self.world.get_state(x, y)]

    def p_step(self, action, new_state):
        if new_state == self.step(action):
            return 1

        return 0


class SwampCell(Cell):
    def __init__(self, world, x, y, stick_prob=0.5):
        super(SwampCell, self).__init__(world, x, y)

        self.stick_prob = stick_prob

    def step(self, action):
        if self.world.rng.random() < self.stick_prob:
            return self

        return super(SwampCell, self).step(action)

    def get_afterstates(self, action):
        x, y = apply_action(self, action)
        return [self, self.world.get_state(x, y)]

    def p_step(self, action, new_state):
        if new_state == Cell.step(self, action):
            return 1 - self.stick_prob
        elif new_state == self:
            return self.stick_prob

        return 0


class PitCell(GoalCell):
    pass


class Reward_task_1_e_3:
    possible_rewards = [0, -1, -1000]

    @staticmethod
    def reward_f(old_state, action, new_state):
        goal_state = [12, 7]  # goal x and goal y

        # If Goal reached and we want to go out of this Goal cell
        if old_state.terminal:
            return 0
        # For Pit Cell
        if isinstance(new_state, PitCell):
            return -1000

        return - np.absolute(new_state.x - goal_state[0]) - np.absolute(new_state.y - goal_state[1])

    @staticmethod
    def reward_p(reward, new_state, old_state, action):
        """
        Computes p(R_{t+1} | S_{t+1}=new_state, S_t=old_state, A_t=action)
        """
        goal_state = [12, 7]  # goal x and goal y

        # If Goal reached and we want to go out of this Goal cell
        if old_state.terminal:
            true_r = 0
        # For Pit Cell
        elif isinstance(new_state, PitCell):
            true_r = -1000
        else:
            true_r = - np.absolute(new_state.x - goal_state[0]) - np.absolute(new_state.y - goal_state[1])

        return int(true_r == reward)


class Reward_task_1_e_2:
    possible_rewards = [0, 1, -1000]

    @staticmethod
    def reward_f(old_state, action, new_state):
        # If Goal reached and we want to go out of this Goal cell
        if old_state.terminal:
            return 0
        # For Pit Cell
        if isinstance(new_state, PitCell):
            return -1000
        if isinstance(new_state, GoalCell):
            return 1

        return 0

    @staticmethod
    def reward_p(reward, new_state, old_state, action):
        """
        Computes p(R_{t+1} | S_{t+1}=new_state, S_t=old_state, A_t=action)
        """
        # If Goal reached and we want to go out of this Goal cell
        if old_state.terminal:
            true_r = 0

        # For Pit Cell
        elif isinstance(new_state, PitCell):
            true_r = -1000
        elif isinstance(new_state, GoalCell):
            true_r = 1
        else:
            true_r = 0

        return int(true_r == reward)


class DefaultReward:
    possible_rewards = [0, -1, -1000]

    @staticmethod
    def reward_f(old_state, action, new_state):
        # If Goal reached and we want to go out of this Goal cell
        if old_state.terminal:
            return 0
        # For Pit Cell
        if isinstance(new_state, PitCell):
            return -1000

        return -1

    @staticmethod
    def reward_p(reward, new_state, old_state, action):
        """
        Computes p(R_{t+1} | S_{t+1}=new_state, S_t=old_state, A_t=action)
        """
        # If Goal reached and we want to go out of this Goal cell
        if old_state.terminal:
            true_r = 0
        # For Pit Cell
        elif isinstance(new_state, PitCell):
            true_r = -1000
        else:
            true_r = -1

        return int(true_r == reward)


class World:
    def __init__(self, reward_class=DefaultReward):
        self.reward_class = reward_class
        self.rng = random.Random()
        self.grid = None
        self.start_states = None
        self.current_state = None

    def change_reward_1_e_2(self):
        self.reward_class = Reward_task_1_e_2

    def change_reward_1_e_3(self):
        self.reward_class = Reward_task_1_e_3

    def save_to_file(self, path):
        g_list = self.grid.flatten().tolist()
        g_list = [o for o in g_list if not isinstance(o, BlankCell)]

        world_info = dict(size=self.grid.shape, reward_class=self.reward_class.__name__,
                          grid=g_list)

        class CellEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()

                return json.JSONEncoder.default(self, obj)

        with open(path, mode='w') as f:
            json.dump(world_info, f, cls=CellEncoder, ensure_ascii=False, indent=4)

    @classmethod
    def load_from_file(cls, path):
        def hook(dct):
            if '__cls__' not in dct:
                return dct

            klass = globals()[dct['__cls__']]
            obj = klass(None, dct['x'], dct['y'])
            obj.from_dict(dct)
            return obj

        with open(path, mode='r') as f:
            world_info = json.load(f, object_hook=hook)  # Creating dict for world info from .json

        rew_class = globals()[world_info['reward_class']]
        world = World(rew_class)  # Creating object of class World and Initializing him
        world.start_states = []

        # Initializing grid of world_info.size of Blank Cells
        world.grid = np.array([[BlankCell(world, x, y) for x in range(world_info['size'][0])]
                               for y in range(world_info['size'][1])], dtype=np.object)

        # Initializing grid of our world from .json
        for cell in world_info['grid']:
            cell.world = world
            if isinstance(cell, StartCell):
                world.start_states.append(cell)
            world.grid[cell.y, cell.x] = cell

        return world

    def get_state(self, x, y):
        if self.grid is None:
            raise RuntimeError
        if not (0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]):
            raise ValueError

        return self.grid[y, x]

    def reset(self):
        self.current_state = self.rng.choice(
            self.start_states)  # Returns one random element from the non-empty sequence self.start_states

    def step(self, action):
        proposed_state = self.current_state.step(action)
        if proposed_state.allow_enter(self.current_state, action):
            new_state = proposed_state
        else:
            new_state = self.current_state

        reward = self.reward_class.reward_f(self.current_state, action, new_state)
        done = new_state.terminal
        self.current_state = new_state

        return new_state, reward, done

    def p(self, new_state, reward, old_state, action):
        """
        Computes p(S_{t+1}=new_state, R_{t+1}=reward | S_t=old_state, A_t=action)
        """
        reward_p = self.reward_class.reward_p(reward, new_state, old_state, action)
        if reward_p == 0:
            return 0

        step_p = old_state.p_step(action, new_state)

        if new_state != old_state:
            enter_p = new_state.p_enter(old_state, action)
            return reward_p * step_p * enter_p

        if step_p == 1:
            return reward_p

        sum = 0
        count = 0
        for s in old_state.get_afterstates(action):
            if s != old_state:
                sum += s.p_enter(old_state, action)
                count += 1
        return reward_p * (step_p + (1 - step_p) * (1 - sum / count))


def random_walk(steps=100):
    world = World.load_from_file('world.json')
    world.reset()

    print(f'Starting at pos. ({world.current_state.x}, {world.current_state.y}).')

    ep_return = 0
    last_termination = 0
    for s in range(steps):
        print(f'Step {s + 1}/{steps}...')
        action = random.choice(['left', 'right', 'up', 'down'])
        print(f'Going {action}!')
        current_state = [world.current_state.y, world.current_state.x]
        new_state, reward, done = world.step(action)
        ep_return += reward
        print(f'Received a reward of {reward}!')
        print(f'New position is ({new_state.x}, {new_state.y}) on a {type(new_state).__name__}.')
        if done:
            print(f'Episode terminated after {s + 1 - last_termination} steps. Total Return was {ep_return}.')
            ep_return = 0
            last_termination = s + 1
            print(f'Resetting the world...')
            world.reset()


def SARSA(number_of_episodes, EPSILON=0.1, ALPHA=1, GAMMA=1,
          visualize_console=False, print_all_steps=False,
          visualize_window=False, visualize_window_each_done=False, visualize_window_final_done=False,
          change_reward_1e2=False, change_reward_1e3=False):
    world = World.load_from_file('world.json')
    world.reset()

    if change_reward_1e2:
        world.change_reward_1_e_2()
        print("Reward function changed to 1_e_2")

    if change_reward_1e3:
        world.change_reward_1_e_3()
        print("Reward function changed to 1_e_3")

    last_termination = 0
    actions_list = ['left', 'right', 'up', 'down']
    matrix_of_Q_values = np.zeros((4, 16, 16))

    if print_all_steps:
        print(f'Starting at pos. ({world.current_state.x}, {world.current_state.y}).')
    if visualize_console:
        # time.sleep(0.5)  # sleep for 0.5 seconds
        my_visualize_map(world)
    if visualize_window or visualize_window_each_done:
        window = Tk()
        canvas = Canvas(background="white")  # lightyellow
        my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)

    total_returns = []  # for collecting all returns of all episodes
    total_steps = []
    total_stuck_wall = []
    finish = 0
    pit = 0
    for i in range(number_of_episodes):
        done = False
        ep_return = 0
        ep_steps = 0
        stuck_wall = 0
        future_action = 0
        print("\rEpisode " + str(i + 1) + " / " + str(number_of_episodes), end="")
        while done != True:

            current_state = [world.current_state.y, world.current_state.x]  # transfer to column, raw to raw, column

            if (future_action == 0):
                # Random step or not
                random_step = random.random()  # generate a random probability (0<=number<1)

                # if Q values in all directions of current state is empty we need to randomly choose the number
                if (np.sum(matrix_of_Q_values[:, current_state[0], current_state[1]]) == 0):
                    random_step = 0

                # Random case
                if (random_step <= EPSILON):
                    if (print_all_steps):
                        print("Random case")
                    action = random.choice(actions_list)

                # Not Random case
                else:
                    directions_of_Q_values = matrix_of_Q_values[:, current_state[0], current_state[1]]
                    max_of_Q_values = directions_of_Q_values[np.argmax(directions_of_Q_values)]  # searching max value

                    indices_of_max_Q_values = np.where(
                        directions_of_Q_values == max_of_Q_values)  # get the tuple of indices
                    indices_of_max_Q_values = np.array(indices_of_max_Q_values)[0]  # convert tuple to np array

                    # if  there are several maximum directions we randomly between them
                    if (len(indices_of_max_Q_values) != 1):
                        action = actions_list[random.choice(indices_of_max_Q_values)]
                    # if there are one maximal direction we choose this direction
                    else:
                        action = actions_list[indices_of_max_Q_values[0]]
            else:
                action = future_action

            # If we stand to ArrowCell any action that the agent wants to execute is ignored.
            # For this reason not modifying the ArrowCell directions
            current_state_cell_name = type(world.current_state).__name__

            # The action already chosen so execute action
            new_state_full_information, reward, done = world.step(action)
            new_state = [world.current_state.y, world.current_state.x]

            # # # # # # #Generates new action for new state
            # if (True): # only for \t of code to make code more clear to read
            if (True):
                # Random step or not
                random_step = random.random()  # generate a random probability (0<=number<1)

                # if Q values in all directions of new state is empty we need to randomly choose the number
                if np.sum(matrix_of_Q_values[:, new_state[0], new_state[1]]) == 0:
                    random_step = 0

                # Random case
                if random_step <= EPSILON:
                    if (print_all_steps):
                        print("Random case of future action")
                    future_action = random.choice(actions_list)

                # Not Random case
                else:
                    directions_of_Q_values = matrix_of_Q_values[:, new_state[0], new_state[1]]
                    max_of_Q_values = directions_of_Q_values[
                        np.argmax(directions_of_Q_values)]  # searching max value

                    indices_of_max_Q_values = np.where(
                        directions_of_Q_values == max_of_Q_values)  # get the tuple of indices of max elements
                    indices_of_max_Q_values = np.array(indices_of_max_Q_values)[0]  # convert tuple to np array

                    # if there are several maximum directions we randomly between them
                    if (len(indices_of_max_Q_values) != 1):
                        future_action = actions_list[random.choice(indices_of_max_Q_values)]
                    # if there are one maximal direction we choose this direction
                    else:
                        future_action = actions_list[indices_of_max_Q_values[0]]

                if print_all_steps:
                    print("Current action " + action + ". Future step is " + future_action)

            # If we are not going to the wall or we are not stuck in the swamp
            if current_state != new_state and current_state_cell_name != "ArrowCell":

                ep_return += reward
                ep_steps += 1

                # getting a index of current action in currect direction
                index_of_direction = np.array(np.where(np.array(actions_list) == action))
                index_of_direction = index_of_direction[0][0]

                # Q[S';A']
                # getting a index of new action in new direction
                index_of_new_direction = np.array(np.where(np.array(actions_list) == future_action))
                index_of_new_direction = index_of_new_direction[0][0]

                matrix_of_Q_values[index_of_direction, current_state[0], current_state[1]] = matrix_of_Q_values[
                                                                                                 index_of_direction,
                                                                                                 current_state[0],
                                                                                                 current_state[1]
                                                                                             ] + ALPHA * (
                                                                                                     reward + GAMMA *
                                                                                                     matrix_of_Q_values[
                                                                                                         index_of_new_direction,
                                                                                                         new_state[0],
                                                                                                         new_state[1]] -
                                                                                                     matrix_of_Q_values[
                                                                                                         index_of_direction,
                                                                                                         current_state[
                                                                                                             0],
                                                                                                         current_state[
                                                                                                             1]])
                # Section for visualization (start)
                if print_all_steps:
                    print(f'Going {action}!')
                    print(f'Received a reward of {reward}!')
                    print(
                        f'New position is ({new_state_full_information.x}, {new_state_full_information.y}) on a {type(new_state_full_information).__name__}.' + "\n")
                if visualize_console:
                    my_visualize_map(world)
                if visualize_window:
                    # Clear all canvas
                    canvas.delete("all")
                    window.update()
                    # print new states
                    my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
                # Section for visualization (end)
                if done:
                    if print_all_steps:
                        print(
                            f'Episode terminated after {ep_steps} steps. Total Return was {ep_return}.')
                        print(f'Resetting the world...')
                    if (visualize_window_each_done):
                        # Clear all canvas
                        canvas.delete("all")
                        window.update()
                        # print new states
                        my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
                    if (type(new_state_full_information).__name__ == "GoalCell"):
                        finish += 1
                    if (type(new_state_full_information).__name__ == "PitCell"):
                        pit += 1
                    world.reset()
            else:
                if print_all_steps:
                    if (type(new_state_full_information).__name__ == "SwampCell"):
                        print(f'***Stucked is swamp***')
                    elif (type(new_state_full_information).__name__ != "ArrowCell"):
                        print(f'***Stucked near the wall***')
                        stuck_wall += 1
                    elif (type(new_state_full_information).__name__ == "ArrowCell"):
                        print(f'***Moving in the direction of the arrow***')

        total_returns.append(ep_return)
        total_steps.append(ep_steps)
        total_stuck_wall.append(stuck_wall)

    # End of cycle
    if visualize_window_final_done == True:
        if visualize_window == True or visualize_window_each_done == True:
            # Clear all canvas
            canvas.delete("all")
            window.update()
            # print new states
            my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
            window.mainloop()
        else:
            window = Tk()
            canvas = Canvas(background="white")  # lightyellow
            my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
            window.mainloop()

    print("\nDone of learning")
    return finish, total_returns


def Q_learning(number_of_episodes, EPSILON=0.1, ALPHA=1, GAMMA=1,
               visualize_console=False, print_all_steps=False,
               visualize_window=False, visualize_window_each_done=False, visualize_window_final_done=False,
               change_reward_1e2=False, change_reward_1e3=False):
    world = World.load_from_file('world.json')
    world.reset()

    if change_reward_1e2:
        world.change_reward_1_e_2()
        print("Reward function changed to 1_e_2")

    if change_reward_1e3:
        world.change_reward_1_e_3()
        print("Reward function changed to 1_e_3")

    last_termination = 0
    actions_list = ['left', 'right', 'up', 'down']
    matrix_of_Q_values = np.zeros((4, 16, 16))

    if print_all_steps:
        print(f'Starting at pos. ({world.current_state.x}, {world.current_state.y}).')
    if visualize_console:
        # time.sleep(0.5)  # sleep for 0.5 seconds
        my_visualize_map(world)
    if visualize_window or visualize_window_each_done:
        window = Tk()
        canvas = Canvas(background="white")  # lightyellow
        my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)

    total_returns = []  # for collecting all returns of all episodes
    total_steps = []
    total_stuck_wall = []
    finish = 0
    pit = 0
    for i in range(number_of_episodes):
        done = False
        ep_return = 0
        ep_steps = 0
        stuck_wall = 0
        print("\rEpisode " + str(i + 1) + " / " + str(number_of_episodes), end="")
        while done != True:
            # Random step or not
            random_step = random.random()  # generate a random probability (0<=number<1)

            current_state = [world.current_state.y, world.current_state.x]  # transfer to column, raw to raw, column

            # if Q values in all directions of current state is empty we need to randomly choose the number
            if np.sum(matrix_of_Q_values[:, current_state[0], current_state[1]]) == 0:
                random_step = 0

            # Random case
            if random_step <= EPSILON:
                if (print_all_steps):
                    print("Random case")
                action = random.choice(actions_list)

            # Not Random case
            else:
                directions_of_Q_values = matrix_of_Q_values[:, current_state[0], current_state[1]]
                max_of_Q_values = directions_of_Q_values[np.argmax(directions_of_Q_values)]  # searching max value

                indices_of_max_Q_values = np.where(
                    directions_of_Q_values == max_of_Q_values)  # get the tuple of indices
                indices_of_max_Q_values = np.array(indices_of_max_Q_values)[0]  # convert tuple to np array
                try:
                    # if  there are several maximum directions we randomly between them
                    if (len(indices_of_max_Q_values) != 1):
                        action = actions_list[random.choice(indices_of_max_Q_values)]
                    # if there are one maximal direction we choose this direction
                    else:
                        action = actions_list[indices_of_max_Q_values[0]]
                except:
                    print(traceback.format_exc())

            # If we stand to ArrowCell any action that the agent wants to execute is ignored.
            # For this reason not modifying the ArrowCell directions
            current_state_cell_name = type(world.current_state).__name__

            # the action already chosen
            new_state_full_information, reward, done = world.step(action)
            new_state = [world.current_state.y, world.current_state.x]

            # If we are not going to the wall or we are not stuck in the swamp
            if current_state != new_state and current_state_cell_name != "ArrowCell":
                ep_return += reward
                ep_steps += 1

                index_of_direction = np.array(np.where(np.array(actions_list) == action))
                index_of_direction = index_of_direction[0][0]

                new_state_values = matrix_of_Q_values[:, new_state[0], new_state[1]]
                max_in_new_state = new_state_values[np.argmax(new_state_values)]

                matrix_of_Q_values[index_of_direction, current_state[0], current_state[1]] = matrix_of_Q_values[
                                                                                                 index_of_direction,
                                                                                                 current_state[0],
                                                                                                 current_state[1]
                                                                                             ] + ALPHA * (
                                                                                                     reward + GAMMA * max_in_new_state -
                                                                                                     matrix_of_Q_values[
                                                                                                         index_of_direction,
                                                                                                         current_state[
                                                                                                             0],
                                                                                                         current_state[
                                                                                                             1]])
                # Section for visualization (start)
                if print_all_steps:
                    print(f'Going {action}!')
                    print(f'Received a reward of {reward}!')
                    print(
                        f'New position is ({new_state_full_information.x}, {new_state_full_information.y}) on a {type(new_state_full_information).__name__}.' + "\n")
                if visualize_console:
                    my_visualize_map(world)
                if visualize_window:
                    # Clear all canvas
                    canvas.delete("all")
                    window.update()
                    # print new states
                    my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
                # Section for visualization (end)
                if done:
                    if print_all_steps:
                        print(
                            f'Episode terminated after {ep_steps} steps. Total Return was {ep_return}.')
                        print(f'Resetting the world...')
                    if visualize_window_each_done:
                        # Clear all canvas
                        canvas.delete("all")
                        window.update()
                        # print new states
                        my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
                    if type(new_state_full_information).__name__ == "GoalCell":
                        finish += 1
                    if type(new_state_full_information).__name__ == "PitCell":
                        pit += 1
                    world.reset()
            else:
                if print_all_steps:
                    if type(new_state_full_information).__name__ == "SwampCell":
                        print(f'***Stucked is swamp***')
                    elif type(new_state_full_information).__name__ != "ArrowCell":
                        print(f'***Stucked near the wall***')
                        stuck_wall += 1
                    elif type(new_state_full_information).__name__ == "ArrowCell":
                        print(f'***Moving in the direction of the arrow***')

        total_returns.append(ep_return)
        total_steps.append(ep_steps)
        total_stuck_wall.append(stuck_wall)

    # End of learning reached
    if visualize_window_final_done == True:
        if visualize_window == True or visualize_window_each_done == True:
            # Clear all canvas
            canvas.delete("all")
            window.update()
            # print new states
            my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
            window.mainloop()
        else:
            window = Tk()
            canvas = Canvas(background="white")  # lightyellow
            my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list)
            window.mainloop()
    print("\nDone of learning")
    return finish, total_returns


def my_visualize_map(world):
    grid = np.array(world.grid)
    grid_visual = np.zeros_like(grid)
    # Filling map
    for i in range(len(grid)):
        for j in range(len(grid)):
            if isinstance(grid[i, j], WallCell):
                grid_visual[i, j] = "W  "
            elif isinstance(grid[i, j], PitCell):
                grid_visual[i, j] = "P  "
            elif isinstance(grid[i, j], BlankCell):
                grid_visual[i, j] = "B  "
            elif isinstance(grid[i, j], ArrowCell):
                grid_visual[i, j] = "Ar "
            elif isinstance(grid[i, j], GoalCell):
                grid_visual[i, j] = "Go "
            elif isinstance(grid[i, j], SwampCell):
                grid_visual[i, j] = "Sw "
            elif isinstance(grid[i, j], StartCell):
                grid_visual[i, j] = "St "

    # Current state
    grid_visual[world.current_state.y, world.current_state.x] = "@> "

    # Printing
    for i in range(len(grid)):
        for j in range(len(grid)):
            print(grid_visual[i, j], end="")
        print()

    pass


def my_visual_window(window, canvas, world, matrix_of_Q_values, actions_list):
    window.config(width=600, height=600, background="black")
    window.title("Visualize learning")
    number_of_cells = 16

    cell_size = 58
    my_geometry = str(number_of_cells * cell_size) + "x" + str(number_of_cells * cell_size) + "+0+0"
    window.geometry(my_geometry)  # 16*32
    window.update()

    canvas.pack(fill=BOTH, expand=True)
    window.update()

    # Skeletal cells
    lines = []
    for i in range(number_of_cells + 1):
        lines.append(i * cell_size)
    # Horizontal lines
    for i in range(number_of_cells + 1):
        canvas.create_line(0, lines[i], number_of_cells * cell_size, lines[i], width=2)
    # Vertical lines
    for i in range(number_of_cells + 1):
        canvas.create_line(lines[i], 0, lines[i], number_of_cells * cell_size, width=2)

    # Slash lines (/)
    for i in range(number_of_cells + 1):
        canvas.create_line(0, lines[i], lines[i], 0)
    for i in range(number_of_cells + 1):
        canvas.create_line(number_of_cells * cell_size, lines[i], lines[i], number_of_cells * cell_size)

    # Backslash lines(\)
    for i in range(number_of_cells + 1):
        canvas.create_line(lines[i], 0, lines[len(lines) - 1], lines[len(lines) - 1 - i])
    for i in range(number_of_cells + 1):
        canvas.create_line(0, lines[i], lines[len(lines) - 1 - i], lines[len(lines) - 1])
    window.update()

    # visualization of world

    grid = np.array(world.grid)
    grid_visual = np.zeros_like(grid)
    # Filling map
    for i in range(len(grid)):
        for j in range(len(grid)):
            if isinstance(grid[i, j], WallCell):
                grid_visual[i, j] = "W"
            elif isinstance(grid[i, j], PitCell):
                grid_visual[i, j] = "P"
            elif isinstance(grid[i, j], BlankCell):
                grid_visual[i, j] = "B"
            elif isinstance(grid[i, j], ArrowCell):
                grid_visual[i, j] = "Ar"
            elif isinstance(grid[i, j], GoalCell):
                grid_visual[i, j] = "Go"
            elif isinstance(grid[i, j], SwampCell):
                grid_visual[i, j] = "Sw"
            elif isinstance(grid[i, j], StartCell):
                grid_visual[i, j] = "St"

    # Current state
    grid_visual[world.current_state.y, world.current_state.x] = "@>"

    # Computing central points and filling the map
    points = []
    for i in range(number_of_cells):
        points.append(cell_size / 2 + cell_size * i)

    for i in range(number_of_cells):
        for j in range(number_of_cells):
            if grid_visual[i, j] == "W":
                canvas.create_text(
                    points[j], points[i], justify="center", font="Verdana 7",
                    text=grid_visual[i, j], fill="black")
            if grid_visual[i, j] == "P":
                canvas.create_text(
                    points[j], points[i], justify="center", font="Verdana 7",
                    text=grid_visual[i, j], fill="blue")
            if grid_visual[i, j] == "Sw":
                canvas.create_text(
                    points[j], points[i], justify="center", font="Verdana 7",
                    text=grid_visual[i, j], fill="green")
            if grid_visual[i, j] == "St" or grid_visual[i, j] == "Go":
                canvas.create_text(
                    points[j], points[i], justify="center", font="Verdana 7",
                    text=grid_visual[i, j], fill="red")
            if grid_visual[i, j] == "Ar":
                canvas.create_text(
                    points[j], points[i], justify="center", font="Verdana 7",
                    text=grid_visual[i, j], fill="purple")
            if grid_visual[i, j] == "@>":
                canvas.create_text(
                    points[j], points[i], justify="center", font="Verdana 7",
                    text=grid_visual[i, j], fill="red")

    # Visualization of learning parameters
    # actions_list = ['left', 'right', 'up', 'down']
    # matrix_of_Q_values,actions_list

    # Code below for testing visualization
    # for i in range(len(matrix_of_Q_values[0])):
    #     for j in range(len(matrix_of_Q_values[0])):
    #         matrix_of_Q_values[0, i, j] = i * j * 2 + j + 100
    #         matrix_of_Q_values[1, i, j] = i * j * 2 + j + 200
    #         matrix_of_Q_values[2, i, j] = i * j * 2 + j + 3000
    #         matrix_of_Q_values[3, i, j] = i * j * 2 + j + 4000

    quarter = cell_size / 4
    for i in range(len(actions_list)):
        if actions_list[i] == "left":
            for j in range(len(points)):
                for k in range(len(points)):
                    # points[k] - quarter, points[j]
                    canvas.create_text(
                        lines[k], points[j], justify="center", font="Verdana 5",
                        text=" " + str(round(matrix_of_Q_values[i, j, k])), fill="black", anchor=tkinter.W)

        elif actions_list[i] == "right":
            for j in range(len(points)):
                for k in range(len(points)):
                    canvas.create_text(
                        points[k] + quarter, points[j], justify="center", font="Verdana 5",
                        text=" " + str(round(matrix_of_Q_values[i, j, k])), fill="black")


        elif actions_list[i] == "up":
            for j in range(len(points)):
                for k in range(len(points)):
                    canvas.create_text(
                        points[k], points[j] - quarter, justify="center", font="Verdana 5",
                        text=str(round(matrix_of_Q_values[i, j, k])), fill="black")


        elif actions_list[i] == "down":
            for j in range(len(points)):
                for k in range(len(points)):
                    canvas.create_text(
                        points[k], points[j] + quarter, justify="center", font="Verdana 5",
                        text=str(round(matrix_of_Q_values[i, j, k])), fill="black")

    window.update()


if __name__ == '__main__':

    # random_walk(1000)

    # Initialization

    # Not necessary booleans
    TRYING_TO_FIND_ALPHA = True  # GROUP 9 attempts
    TASK_B_C_D = True  # Visualization windows form after calculation best paths

    # Necessary booleans
    TASK_B_C_D_VISUALIZATION = True  # Calculation of SARSA and Q-learning with different probabilities of choosing a random action
    TASK_1_E = True
    TASK_1_F = True

    # Trying to find alpha with the initial reward function
    if TRYING_TO_FIND_ALPHA:
        alpha_for_tests = np.arange(0.5, 1.1, 0.1)
        for i in range(len(alpha_for_tests)):
            finish, total_returns = Q_learning(50000, ALPHA=alpha_for_tests[i], visualize_window_each_done=False,
                                               visualize_window_final_done=False,
                                               visualize_window=False, print_all_steps=False)

            print("Q-learning alpha = " + str(alpha_for_tests[i]) + "\tnumber of finish = " + str(finish))

            finish, total_returns = SARSA(50000, ALPHA=alpha_for_tests[i], visualize_window_each_done=False,
                                          visualize_window_final_done=False,
                                          visualize_window=False, print_all_steps=False)

            print("SARSA alpha = " + str(alpha_for_tests[i]) + "\tnumber of finish = " + str(finish) + "\n")

        print("End of finding alpha")

    if TASK_B_C_D:
        finish, total_returns = SARSA(50000, visualize_window_final_done=True)
        finish, total_returns = Q_learning(50000, visualize_window_final_done=True)

    if TASK_B_C_D_VISUALIZATION:
        # in this program we don't show the case where epsilon is equal 0, cause for both SARSA and Q-learning it stack algorithm
        # mentioned above case we described in report
        epsilon_array = np.arange(0.1, 1, 0.1)

        for i in range(len(epsilon_array)):
            finish, total_returns = SARSA(10000, EPSILON=epsilon_array[i])

            total_epoch = np.arange(1, len(total_returns) + 1, 1)
            plt.plot(total_epoch, total_returns, label="Epsilon=" + str(round(epsilon_array[i] * 10) / 10))
        plt.xlabel("All episodes")
        plt.ylabel("Total return")
        plt.title("SARSA with different Epsilon parameters")
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('SARSA_with_different_epsilon.png')
        plt.show()

        for i in range(len(epsilon_array)):
            finish, total_returns = Q_learning(10000, EPSILON=epsilon_array[i])

            total_epoch = np.arange(1, len(total_returns) + 1, 1)
            plt.plot(total_epoch, total_returns, label="Epsilon=" + str(round(epsilon_array[i] * 10) / 10))
        plt.xlabel("All episodes")
        plt.ylabel("Total return")
        plt.title("Q-learning with different Epsilon parameters")
        plt.legend(loc=0, fontsize=7, framealpha=0.1)
        plt.savefig('Q-learning_with_different_epsilon.png')
        plt.show()

    if TASK_1_E:
        print("Task 1 e")
        print("Changing policy to 1.e.2")

        print("SARSA")
        finish, total_returns = Q_learning(10000, EPSILON=0, GAMMA=0.9, ALPHA=0.9, visualize_window_each_done=False,
                                           change_reward_1e2=True,
                                           visualize_window_final_done=True)
        print("Reached Finish " + str(finish))

        print("Q-learning")
        finish, total_returns = Q_learning(10000, EPSILON=0, GAMMA=0.9, ALPHA=0.9, visualize_window_each_done=False,
                                           change_reward_1e2=True,
                                           visualize_window_final_done=True)
        print("Reached Finish " + str(finish))

        print("\nChanging policy to 1.e.3")
        print("SARSA")
        finish, total_returns = Q_learning(10000, EPSILON=0.1, GAMMA=0.1, ALPHA=0.9, visualize_window_each_done=False,
                                           change_reward_1e3=True,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

        print("Q-learning")
        finish, total_returns = Q_learning(10000, EPSILON=0.1, GAMMA=0.1, ALPHA=0.9, visualize_window_each_done=False,
                                           change_reward_1e3=True,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

    if TASK_1_F:
        TASK_1_F_GAMMA = np.arange(0, 1.1, 0.1)
        SHORTEST_WAY = 24
        gamma_for_policies = []

        # For task 1_e_1
        matrix = np.full_like(TASK_1_F_GAMMA, -1)
        for i in range(SHORTEST_WAY - 1):
            matrix = -1 + TASK_1_F_GAMMA * matrix

        index = np.argmax(matrix)
        print("For task 1e1 with optimal policy the return is equal to " + str(matrix[index]) +
              " with gamma equal to " + str(TASK_1_F_GAMMA[index]))
        gamma_for_policies.append(TASK_1_F_GAMMA[index])

        # For task 1_e_2
        matrix = np.full_like(TASK_1_F_GAMMA, 1)
        for i in range(SHORTEST_WAY - 1):
            matrix = TASK_1_F_GAMMA * matrix
        index = np.argmax(matrix)
        print("For task 1e2 with optimal policy the return is equal to " + str(matrix[index]) +
              " with gamma equal to " + str(TASK_1_F_GAMMA[index]))
        gamma_for_policies.append(TASK_1_F_GAMMA[index])

        # For task 1_e_3
        # format x, y
        FINISH = [12, 7]
        PATH = [[12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14],
                [11, 14], [10, 14],
                [10, 13], [10, 12], [10, 11], [10, 10], [10, 9], [10, 8],
                [9, 8], [8, 8], [7, 8], [6, 8], [5, 8], [4, 8], [3, 8], [2, 8]]
        rewards = []
        for i in range(len(PATH)):
            rewards.append(-np.absolute(FINISH[0] - PATH[i][0]) - np.absolute(FINISH[1] - PATH[i][1]))

        matrix = np.full_like(TASK_1_F_GAMMA, 0)
        for i in range(SHORTEST_WAY - 1):
            matrix = rewards[i] + TASK_1_F_GAMMA * matrix
        index = np.argmax(matrix)
        print("For task 1e3 with optimal policy the return is equal to " + str(matrix[index]) +
              " with gamma equal to " + str(TASK_1_F_GAMMA[index]))
        gamma_for_policies.append(TASK_1_F_GAMMA[index])

        print("Testing optimal policies\n")

        print("\nStandart policy")
        print("SARSA")
        finish, total_returns = Q_learning(50000, EPSILON=0.1, GAMMA=gamma_for_policies[0], ALPHA=1,
                                           visualize_window_each_done=False,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

        print("Q-learning")
        finish, total_returns = Q_learning(50000, EPSILON=0.1, GAMMA=gamma_for_policies[0], ALPHA=1,
                                           visualize_window_each_done=False,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

        print("\nChanging policy to 1.e.2")
        print("SARSA")
        finish, total_returns = Q_learning(10000, EPSILON=0.1, GAMMA=gamma_for_policies[1], ALPHA=1,
                                           visualize_window_each_done=False,
                                           change_reward_1e2=True,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

        print("Q-learning")
        finish, total_returns = Q_learning(10000, EPSILON=0.1, GAMMA=gamma_for_policies[1], ALPHA=1,
                                           visualize_window_each_done=False,
                                           change_reward_1e2=True,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

        print("\nChanging policy to 1.e.3")
        print("SARSA")
        finish, total_returns = Q_learning(10000, EPSILON=0.1, GAMMA=gamma_for_policies[2], ALPHA=1,
                                           visualize_window_each_done=False,
                                           change_reward_1e3=True,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

        print("Q-learning")
        finish, total_returns = Q_learning(10000, EPSILON=0.1, GAMMA=gamma_for_policies[2], ALPHA=1,
                                           visualize_window_each_done=False,
                                           change_reward_1e3=True,
                                           visualize_window_final_done=False)
        print("Reached Finish " + str(finish))

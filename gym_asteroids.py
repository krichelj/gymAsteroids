import os
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import time
import pickle
import math
import itertools
import gym
import gym.envs.classic_control.rendering as gym_rendering
import tqdm
from typing import Tuple, List, Dict

RENDER = True


class AgentState2D:
    def __init__(self,
                 position: np.array,
                 screen_ratios: np.array = None):
        self.position = position

        if RENDER:
            assert screen_ratios is not None

            self._transform = gym_rendering.Transform()
            self.screen_ratios = screen_ratios

    def reposition(self,
                   position: np.array):
        self.__set_position(position)

    def step(self,
             step_vector: np.array):
        self.__set_position(self.position + step_vector)

    def __set_position(self,
                       position: np.array):
        self.position = position

        if RENDER:
            translation = self.position * self.screen_ratios
            self._transform.set_translation(*tuple(translation))

    def __str__(self) -> str:
        return str(tuple(self.position))


class CircleAgentState2D(AgentState2D):
    def __init__(self,
                 position: np.array,
                 radius: float = None,
                 screen_ratios: np.array = None,
                 color: Tuple = None):
        super(CircleAgentState2D, self).__init__(position, screen_ratios)

        if RENDER:
            assert radius is not None
            assert screen_ratios is not None

            self.__radius = radius
            self.circle = gym_rendering.make_circle(radius)
            self.circle.add_attr(self._transform)

            if color:
                self.circle.set_color(*color)


class SquareAgentState2D(AgentState2D):
    def __init__(self,
                 position: np.array,
                 side_length: int = None,
                 screen_ratios: np.array = None,
                 color: Tuple = None):
        super(SquareAgentState2D, self).__init__(position, screen_ratios)

        if RENDER:
            assert side_length is not None
            assert screen_ratios is not None

            self.__side_length = side_length
            position_x, position_y = position
            half_side_length = side_length / 2
            left = position_x - half_side_length
            right = position_x + half_side_length
            top = position_y + half_side_length
            bottom = position_y - half_side_length

            square_points_positions = [(left, bottom), (left, top), (right, top), (right, bottom)]
            self.square = gym_rendering.make_polygon(square_points_positions, filled=True)
            self.square.add_attr(self._transform)

            if color:
                self.square.set_color(*color)


class AsteroidsGameState:
    """
    Asteroids game environment state base class

    Args:

      quad: The quadrotor agent object
      asteroids: A dictionary of the asteroid agents numbers and objects
      observed_moves: All possible observed elements coordinates with respect to the quadrotor
      grid_side_length: The length of the grid size, which is assumed to be a square
    """

    def __init__(self, quad: CircleAgentState2D, asteroids: Dict[int, CircleAgentState2D],
                 observed_moves: List[np.array], grid_side_length: int):
        self.__quad = quad
        self.__asteroids = asteroids
        self.__grid_side_length = grid_side_length

        rounded_grid_side_length = round(self.__grid_side_length)
        self.__xy_grid = np.zeros([rounded_grid_side_length, rounded_grid_side_length])

        for asteroid in self.__asteroids.values():
            asteroid_x, asteroid_y = asteroid.position
            if self.__grid_side_length > asteroid_y > 0:
                self.__xy_grid[asteroid_x, asteroid_y] = 1

        quad_position = self.__quad.position
        self.__k_positions = {}

        for i, observed_move in enumerate(observed_moves):
            observed_position_x, observed_position_y = quad_position + observed_move
            if self.__grid_side_length > observed_position_x > 0 and self.__grid_side_length > observed_position_y > 0:
                self.__k_positions[i] = self.__xy_grid[observed_position_x, observed_position_y]

    def id(self) -> Tuple:
        return tuple((i for i in self.__k_positions.values()))

    def num_of_observed_asteroids(self) -> int:
        return len([i for i in self.__k_positions.values() if i == 1])

    def has_collided(self) -> int:
        collided = any([np.array_equal(asteroid.position, self.__quad.position)
                        for asteroid in self.__asteroids.values()])
        collided_boolean_int = int(collided)

        return collided_boolean_int

    def is_terminal(self) -> bool:
        quad_y = self.__quad.position[1]
        return all([asteroid.position[1] <= quad_y for asteroid in self.__asteroids.values()])

    def __str__(self) -> str:
        return f"Quad: {self.__quad}\nClose positions:\n" + \
            '\n'.join([f'{position}: {is_dangerous}' for position, is_dangerous in self.__k_positions.items()])


class QLearner:
    """
    Q-Learning class

    Attributes
    ----------
    Q_table : np.array
        The Q-table used to perform Q-learning, assuming each observed spot either has an asteroid or not
    """

    def __init__(self,
                 action_space_n: int,
                 observed_moves_n: int,
                 alpha: float,
                 gamma: float,
                 epsilon: float):
        """
        :param action_space_n: The number of available actions to perform at each step
        :param observed_moves_n: The number of observed elements the agent has at each state,
                assumed to be a perfect square
        :param alpha: The learning rate - used to decide how much we accept the new value vs the old value
        :param gamma: The discount factor - used to balance immediate and future reward
        :param epsilon: The threshold for the randomized epsilon-greedy algorithm to generate next action
        """

        self.__action_space_n = action_space_n
        self.__observed_moves_n = observed_moves_n
        self.__n = int(math.sqrt(self.__observed_moves_n))
        self.Q_table = np.zeros([2] * observed_moves_n + [action_space_n])
        self.__alpha = alpha
        self.__gamma = gamma
        self.__epsilon = epsilon
        self.learning_episodes = 0

    def get_learning_next_action_id(self,
                                    old_state_id: Tuple) -> int:
        """
        Epsilon-greedy algorithm for choosing the next learning action:
            1. First draw a float random_draw in [0, 1] and compare it to the value of self.__epsilon
            2. If random_draw < self.__epsilon return a random action
            3. Else take the action that has the maximal future reward value

        :param old_state_id: The id of the old state
        :return Next action id in the range [0, 1, ..., self.__action_space_n-1]

        Notes
        ----------
            - The case where len(old_state_id) < self.__observed_moves_n is where the quadrotor reached a corner
        """

        random_draw = np.random.uniform(0, 1)

        if random_draw < self.__epsilon:
            next_action_id = np.random.randint(self.__action_space_n)
        elif len(old_state_id) == self.__observed_moves_n:
            next_action_id = self.get_learning_next_action_id(old_state_id)
        else:
            next_action_id = self.__action_space_n

        return next_action_id

    def update_q_matrix(self,
                        old_state_id: Tuple,
                        new_state_id: Tuple,
                        reward: int,
                        next_action_id: int):
        """
        Q-Learning update rule implementation
        Updates the Q-table inplace with respect to the learning parameters

        :param old_state_id: The id of the old state
        :param new_state_id: The id of the new state
        :param reward: The reward for performing the next action
        :param next_action_id: The id of the next action to perform
        """

        if next_action_id in range(self.__action_space_n) and \
                len(old_state_id) == len(new_state_id) == self.__observed_moves_n:
            old_state_id = [int(k) for k in old_state_id]
            new_state_id = [int(k) for k in new_state_id]
            old_q_value = self.Q_table[old_state_id]
            max_value = np.max(self.Q_table[new_state_id, :])

            self.Q_table[old_state_id] = old_q_value + self.__alpha * (reward + self.__gamma * max_value - old_q_value)

    def get_maximal_expected_value_action(self,
                                          old_state_id: Tuple) -> int:
        """
        Retrieve the action that maximizes the expectation value

        :param old_state_id: The id of the old state
        :return Next action id in the range [0, 1, ..., self.__action_space_n-1]
        """

        old_state_id = [int(k) for k in old_state_id]
        q_table_slice_to_consider = self.Q_table[old_state_id]
        maximum_value_actions = np.where(q_table_slice_to_consider == q_table_slice_to_consider.max())[0]
        next_action_id = np.random.choice(maximum_value_actions)

        return next_action_id


class AsteroidsGameEnv(gym.Env):
    """
    Asteroid game environment base class

    Attributes
    ----------
    __observed_moves : List
        The 2-D coordinates of all the observed locations - cartesian product of the 1-D coordinates from observed_moves
    __action_space: Discrete
        A discrete gym space to model the possible moves of the quadrotor agent
    """

    def __init__(self,
                 grid_side_length: int,
                 q_p_0: np.array,
                 num_of_asteroids: int,
                 actions: Dict[str, Tuple[float, float]],
                 observed_moves: Dict[str, List[int]], alpha: float,
                 gamma: float,
                 epsilon: float,
                 quad_radius: float = None,
                 asteroid_radius: float = None,
                 screen_width: int = None,
                 screen_height: int = None
                 ):
        super(AsteroidsGameEnv, self).__init__()
        self.__grid_side_length = grid_side_length

        self.__observed_moves = list(itertools.product(observed_moves['x'], observed_moves['y']))
        observed_moves_n = len(self.__observed_moves)
        self.__actions = {i: action for i, action in enumerate(actions.values())}
        self.__action_space_n = len(actions)
        self.__action_space = gym.spaces.discrete.Discrete(n=self.__action_space_n)
        self.q_learner = QLearner(action_space_n=self.__action_space_n,
                                  observed_moves_n=observed_moves_n,
                                  alpha=alpha,
                                  gamma=gamma,
                                  epsilon=epsilon)
        self.seed()

        if RENDER:
            assert screen_width is not None
            assert screen_height is not None

            self.__screen_shape = np.array([screen_width, screen_height])
            screen_ratios = self.__screen_shape / self.__grid_side_length
            observed_locations_squares_side_length = 15
            square_color = (255, 255, 0)
        else:
            screen_ratios, observed_locations_squares_side_length, square_color = [None] * 3

        self.__q_p_0 = q_p_0
        q_p_null = np.array([0, 0])
        quad_color = (1, 0, 0)
        self.__quad = CircleAgentState2D(position=q_p_null,
                                         radius=quad_radius,
                                         screen_ratios=screen_ratios,
                                         color=quad_color)

        self.__observed_elements = {element_id: SquareAgentState2D(position=q_p_null,
                                                                   side_length=observed_locations_squares_side_length,
                                                                   screen_ratios=screen_ratios,
                                                                   color=square_color)
                                    for element_id, observed_move in enumerate(self.__observed_moves)}

        self.__asteroids = {asteroid_id: CircleAgentState2D(position=self.__randomize_asteroid_position(),
                                                            radius=asteroid_radius,
                                                            screen_ratios=screen_ratios)
                            for asteroid_id in range(num_of_asteroids)}

        self.__np_random, self.__viewer, self.__t, self.__done, self.__learning = [None] * 5
        self.__state = self.reset()

        self.__frames = []

    def seed(self, seed=None) -> List[int]:
        self.__np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def __randomize_asteroid_position(self) -> np.array:
        pos = np.random.randint(low=0,
                                high=self.__grid_side_length,
                                size=(2,))
        return pos

    def __reposition_observed_elements(self, quad_position: np.array):
        if RENDER and not self.__learning:
            for element_id, observed_element in self.__observed_elements.items():
                current_observed_move = self.__observed_moves[element_id]
                new_element_position = quad_position + current_observed_move
                observed_element.reposition(new_element_position)

    def reset(self) -> AsteroidsGameState:
        self.__quad.reposition(self.__q_p_0)
        self.__reposition_observed_elements(self.__q_p_0)

        for asteroid in self.__asteroids.values():
            new_initial_position = self.__randomize_asteroid_position()
            asteroid.reposition(new_initial_position)

        new_state = AsteroidsGameState(quad=self.__quad,
                                       asteroids=self.__asteroids,
                                       observed_moves=self.__observed_moves,
                                       grid_side_length=self.__grid_side_length)
        self.__t = 0
        self.__done = False

        return new_state

    def __agents_step(self, quad_vector: np.array, reposition_quad: bool = False) \
            -> AsteroidsGameState:
        if reposition_quad:
            self.__quad.reposition(quad_vector)
            self.__reposition_observed_elements(quad_vector)
        else:
            self.__quad.step(quad_vector)

            if RENDER and not self.__learning:
                for element_id, observed_element in self.__observed_elements.items():
                    observed_element.step(quad_vector)

        asteroids_unit_vector = np.array([0, -1])
        for asteroid in self.__asteroids.values():
            asteroid.step(asteroids_unit_vector)

        new_state = AsteroidsGameState(quad=self.__quad,
                                       asteroids=self.__asteroids,
                                       observed_moves=self.__observed_moves,
                                       grid_side_length=self.__grid_side_length)
        self.__t += 1

        return new_state

    def step(self,
             action_id: int) -> Tuple[AsteroidsGameState, int, bool]:
        new_state = None
        done = False
        reward = 0

        if action_id in self.__actions:  # check what happens here
            quad_step_vector = self.__actions[action_id]
            step_vector_norm = np.linalg.norm(quad_step_vector)
            new_state = self.__agents_step(quad_step_vector)
            num_of_observed_asteroids = new_state.num_of_observed_asteroids()
            has_collided = new_state.has_collided()

            reward = - 50 * step_vector_norm - 10 * num_of_observed_asteroids - 100 * has_collided
            if new_state.is_terminal():
                done = True
        else:
            if action_id == self.__action_space_n:
                quad_reposition_vector = self.__q_p_0
                new_state = self.__agents_step(quad_reposition_vector, reposition_quad=True)
            done = True

        return new_state, reward, done

    def render(self,
               mode: str = 'human'):
        if self.__viewer is None:
            self.__viewer = gym_rendering.Viewer(*self.__screen_shape)

            for observed_element in self.__observed_elements.values():
                observed_element_square = observed_element.square
                self.__viewer.add_geom(observed_element_square)

            self.__viewer.add_geom(self.__quad.circle)

            for asteroid in self.__asteroids.values():
                asteroid_circle = asteroid.circle
                self.__viewer.add_geom(asteroid_circle)

            screen_width, screen_height = self.__screen_shape
            self.__viewer.set_bounds(0, screen_width, 0, screen_height)

        if not self.__state:
            return None

        render_result = self.__viewer.render(return_rgb_array=mode == 'rgb_array')
        self.__frames += [render_result]

        return render_result

    def learn(self,
              episodes_num: int,
              render: bool = True,
              step_delay: float = None):
        print('Starting to learn...')
        self.__learning = True

        for _ in tqdm.tqdm(range(episodes_num)):
            self.__state = self.reset()

            while not self.__done:
                if RENDER and render:
                    self.render()

                if step_delay:
                    time.sleep(step_delay)

                old_state_index = self.__state.id()
                next_action_id = self.q_learner.get_learning_next_action_id(old_state_index)
                self.__state, reward, self.__done = self.step(next_action_id)
                new_state_id = self.__state.id()
                self.q_learner.update_q_matrix(old_state_index, new_state_id, reward, next_action_id)

            self.q_learner.learning_episodes += 1

        print('Learning done')
        self.__learning = False

        self.close()
        # self.__save_frames_as_gif()

    def act_out_optimally(self,
                          step_delay: float = 0.1):
        self.__state = self.reset()
        num_of_collisions = 0

        while not self.__done:
            if RENDER:
                self.render()

            if step_delay:
                time.sleep(step_delay)

            old_state_index = self.__state.id()
            next_action_id = self.q_learner.get_maximal_expected_value_action(old_state_index)
            quad_step_vector = self.__actions[next_action_id % self.__action_space_n]
            self.__state = self.__agents_step(quad_step_vector)

            has_collided = self.__state.has_collided()

            if has_collided:
                num_of_collisions += 1
                print(f'OH NO! Collision!!!')

            if self.__state.is_terminal():
                self.__done = True

        print(f'Test done. ' + (f'Collided a total of {num_of_collisions} times' if num_of_collisions else
                                'NO COLLISIONS! HOORAY!'))

    def save_Q_learner(self,
                       saved_Q_table_filename: str):
        with open(saved_Q_table_filename, 'wb') as f:
            pickle.dump(self.q_learner, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'\nQ-learner saved to {saved_Q_table_filename}. '
              f'Total learning episodes: {self.q_learner.learning_episodes}\n\n\n')

    def load_Q_learner(self,
                       saved_Q_table_filename: str):
        if os.path.exists(saved_Q_table_filename):
            with open(saved_Q_table_filename, 'rb') as f:
                self.q_learner = pickle.load(f)

            print(f'Q-learner loaded from {saved_Q_table_filename}. '
                  f'Total learning episodes: {self.q_learner.learning_episodes}')

    def close(self):
        if RENDER and self.__viewer:
            self.__viewer.close()
            self.__viewer = None

    def __save_frames_as_gif(self,
                             path: str = './',
                             filename: str = 'gym_animation.gif'):

        # Mess with this to change frame size
        plt.figure(figsize=(self.__frames[0].shape[1] / 72.0, self.__frames[0].shape[0] / 72.0),
                   dpi=72)

        patch = plt.imshow(self.__frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(self.__frames[i])

        anim = animation.FuncAnimation(fig=plt.gcf(),
                                       func=animate,
                                       frames=len(self.__frames),
                                       interval=50)
        anim.save(filename=path + filename,
                  writer='imagemagick',
                  fps=60)


if __name__ == '__main__':
    grid_side_length = 100
    quad_radius = 3
    q_p_0 = np.array([int(grid_side_length / 2), int(grid_side_length / 3)])
    asteroids_radius = 5
    num_of_asteroids = 400
    screen_width = 1400
    screen_height = int(screen_width / 1.5)
    episodes_num = 1000
    step_delay = 1
    observed_moves = {'x': [-1, 0, 1],
                      'y': [0, 1, 2]}
    actions = {
        'right': (1, 0),
        'two right': (2, 0),
        'left': (-1, 0),
        'two left': (-2, 0),
        'stay put': (0, 0),
    }
    alpha = 0.628
    gamma = 0.9
    epsilon = 0.2
    saved_Q_learner_filename = 'Q_learner.pickle'

    env = AsteroidsGameEnv(grid_side_length=grid_side_length,
                           q_p_0=q_p_0,
                           num_of_asteroids=num_of_asteroids,
                           actions=actions,
                           observed_moves=observed_moves,
                           alpha=alpha,
                           gamma=gamma,
                           epsilon=epsilon,
                           quad_radius=quad_radius,
                           asteroid_radius=asteroids_radius,
                           screen_width=screen_width,
                           screen_height=screen_height)

    # env.load_Q_learner(saved_Q_learner_filename)
    # env.learn(episodes_num, render=False)
    # env.save_Q_learner(saved_Q_learner_filename)
    env.act_out_optimally()

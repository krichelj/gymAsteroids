from gym import Env, spaces
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.envs.classic_control.rendering import Viewer, FilledPolygon, Transform, make_circle
from gym.spaces.space import Space
import numpy as np
from typing import Tuple, List, Dict
from math import sqrt


class AgentState2D:
    def __init__(self, p_0: np.array, radius: float, screen_shape: np.array, screen_ratios: np.array):
        self._pos = p_0
        self._radius = radius
        self._circle = make_circle(radius)
        self._transform = Transform()
        self._circle.add_attr(self._transform)
        self._screen_shape = screen_shape
        self._screen_ratios = screen_ratios

    def translate(self, unit_vector: np.array):
        self._pos += unit_vector
        new_translation = self._pos * self._screen_ratios + self._screen_shape / 2
        self._transform.set_translation(*new_translation)

    def __str__(self) -> str:
        return f'{tuple(self._pos)}'

    @property
    def circle(self) -> FilledPolygon:
        return self._circle

    @property
    def transform(self) -> Transform:
        return self._transform


class Observation:
    def __init__(self, grid_side_length: float, q_radius: float, a_radius: float, num_of_asteroids: int,
                 screen_shape: np.array, screen_ratios: np.array):
        half_length = grid_side_length / 2
        q_p_0 = np.array([0, - half_length + q_radius])
        quad_0 = AgentState2D(q_p_0, q_radius, screen_shape, screen_ratios)

        asteroids_0 = {}

        for asteroid_num in range(num_of_asteroids):
            a_p_0 = np.random.uniform(-1, 1, (2,)) * half_length
            asteroids_0[asteroid_num] = AgentState2D(a_p_0, a_radius, screen_shape, screen_ratios)

        self._quad = quad_0
        self._asteroids = asteroids_0

    def translate_quad(self, unit_vector: np.array):
        self._quad.translate(unit_vector)

    def translate_asteroids(self, unit_vectors: Dict[int, np.array]):
        for asteroid_num, xy in unit_vectors.items():
            asteroid = self._asteroids[asteroid_num]
            asteroid.translate(xy)

    @property
    def quad(self):
        return self._quad

    @property
    def asteroids(self):
        return self._asteroids

    def __str__(self):
        return f"Quad: {self._quad}\nAsteroids:\n" + \
               '\n'.join([f'{asteroid_num}: {asteroid_pos}' for asteroid_num, asteroid_pos in self._asteroids.items()])


class Space8Directions(Space):
    def __init__(self):
        self.base_space = Discrete(8)
        super(Space8Directions, self).__init__()

    def sample(self) -> str:
        sample: int = self.base_space.sample()
        samples_vs_directions = {0: 'up',
                                 1: 'up right',
                                 2: 'right',
                                 3: 'down right',
                                 4: 'down',
                                 5: 'left down',
                                 6: 'left',
                                 7: 'up left'}
        direction = samples_vs_directions[sample]

        return direction

    def contains(self, x):
        return x in range(8)


class AsteroidsGameEnv(Env):
    def __init__(self, grid_side_length: float, q_radius: float, a_radius: float, num_of_asteroids: int,
                 screen_width: int, screen_height: int):
        super(AsteroidsGameEnv, self).__init__()
        self._grid_side_length = grid_side_length
        self._q_radius = q_radius
        self._a_radius = a_radius
        self._num_of_asteroids = num_of_asteroids

        self.action_space = Space8Directions()
        self.observation_space = spaces.Box(
            low=-self._grid_side_length,
            high=self._grid_side_length,
            shape=(self._num_of_asteroids,),
            dtype=np.float32
        )

        self._np_random = None
        self.seed()

        self._screen_shape = np.array([screen_width, screen_height])
        self._screen_ratios = self._screen_shape / self._grid_side_length
        self._observation = Observation(self._grid_side_length, self._q_radius, self._a_radius, self._num_of_asteroids,
                                        self._screen_shape, self._screen_ratios)
        self._quad = self._observation.quad
        self._asteroids = self._observation.asteroids
        self._t = 0
        self._viewer = None

    def seed(self, seed=None) -> List[int]:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Observation:
        self._observation = Observation(self._grid_side_length, self._q_radius, self._a_radius, self._num_of_asteroids,
                                        self._screen_shape, self._screen_ratios)
        self._t = 0
        return self._observation

    def step(self, action: str) -> Tuple[Observation, float, bool, dict]:
        sqrt2 = sqrt(2)

        actions_vs_tuples = {'up': (0, 1),
                             'up right': (sqrt2, sqrt2),
                             'right': (1, 0),
                             'down right': (sqrt2, -sqrt2),
                             'down': (0, -1),
                             'left down': (-sqrt2, -sqrt2),
                             'left': (-1, 0),
                             'up left': (-sqrt2, sqrt2)}

        quad_unit_vector = np.array(actions_vs_tuples[action])
        self._observation.translate_quad(quad_unit_vector)

        asteroids_unit_vectors = {asteroid_num: np.array(actions_vs_tuples['down'])
                                  for asteroid_num in self._observation.asteroids.keys()}
        self._observation.translate_asteroids(asteroids_unit_vectors)
        self._t += 1

        return self._observation, 0, True, {}

    def render(self, mode='human'):
        print('#' * 50 + f' t={self._t} ' + '#' * 50 + f'\n{self._observation}\n')

        if self._viewer is None:
            self._viewer = Viewer(*self._screen_shape)

            self._viewer.add_geom(self._quad.circle)

            for asteroid in self._asteroids.values():
                self._viewer.add_geom(asteroid.circle)

        if not self._observation:
            return None

        return self._viewer.render(return_rgb_array=mode == 'rgb_array')

    def play(self, T: int):
        for t in range(T):
            self.render()
            action = self.action_space.sample()
            observation, reward, done, info = self.step(action)
            # if done:
            #     print("Episode finished after {} timesteps".format(t + 1))
            #     break
        self.close()


if __name__ == '__main__':
    grid_side_length = 1000
    q_radius = 3
    a_radius = 5
    num_of_asteroids = 100
    screen_width = 600
    screen_height = 400
    T = 1000

    env = AsteroidsGameEnv(grid_side_length, q_radius, a_radius, num_of_asteroids, screen_width, screen_height)
    env.play(T)

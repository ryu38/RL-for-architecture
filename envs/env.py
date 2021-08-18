import numpy as np
import gym
from gym.spaces import Discrete, Box
from modules.constValid_module import const_valid

class MyEnv(gym.Env):

    def __init__(self, config, refer):
        self.action_space = Discrete(9)
        self.observation_space = Box(-1., 1., shape=(19, ))

        self.params = config['params']
        self.nelz = self.params[2]
        self.const_valid = refer['const_valid']
        self.size_df = refer['size_df']

        x_spans = self.params[0]
        y_spans = self.params[1]
        self.elem_num = {
            'beam': (4 + 3 * (x_spans - 1)) * y_spans - x_spans * (y_spans - 1),
            'l_pillar': (2 + (x_spans - 1)) * 2,
            'r_pillar': 2 + (x_spans - 1),
        }

        elem_init = config['elem_init']
        elem_min = [400, 200, 9, 12, 350, 12, 350, 12]

        self.states_max = np.array((self.nelz, *elem_init, 0, self.nelz, *elem_init), dtype=float)
        self.states_least = np.array((0, *elem_min, 0, 0, *elem_min), dtype=float)
        self.states_max[9] = self._cal_vol(self.states_max[1:9])
        self.states_least[9] = self._cal_vol(self.states_least[1:9])
        diffmax = self.states_max - self.states_least
        self.states_diffmax = np.where(diffmax == 0, 1, diffmax)
        self.states_init = np.copy(self.states_max)
        self.states_init[11:19] = self.states_least[1:9]

        self.reset()

    def reset(self):
        self.states = np.copy(self.states_init)
        self.structure = np.repeat(self.states[None, 1:9], self.nelz, axis=0)

        self.vol = []
        for z_i in range(int(self.nelz)):
            vol = self._cal_vol(self.structure[z_i], z_i)
            self.vol.append(vol)
        self.vol = np.array(self.vol)

        self.result = None

        return self.states_scaled()

    def _cal_vol(self, array, z_i=None):
        def cal_vol(data, gf=False):
            h, b, tw, tf, l_d, l_t, r_d, r_t = data
            beam = h * b - (h - 2*tf) * (b - tw)
            beam *= self.params[4] * self.elem_num['beam']
            l_pillar = l_d**2 - (l_d - 2*l_t)**2
            r_pillar = r_d**2 - (r_d - 2*r_t)**2
            if gf:
                pillar_length = self.params[5] + 1400
            else:
                pillar_length = self.params[5]
            l_pillar *= pillar_length * self.elem_num['l_pillar']
            r_pillar *= pillar_length * self.elem_num['r_pillar']
            return np.array((beam, l_pillar, r_pillar)).sum()

        if z_i is None:
            vol = cal_vol(array) * (self.nelz - 1) + cal_vol(array, gf=True)
            return vol

        if z_i == 0:
            return cal_vol(array, gf=True)

        return cal_vol(array)

    def input_struc(self, input_struc):
        self.reset()
        self.structure = np.copy(input_struc)
        z_i = int(self.states[0] - 1)
        self.states[1:9] = self.structure[z_i]
        if z_i + 1 != self.nelz:
            self.states[11:11+8] = self.structure[z_i+1]
        else:
            self.states[11:11+8] = self.states_least[1:9]

        self.vol = []
        for z_i in range(int(self.nelz)):
            vol = self._cal_vol(self.structure[z_i], z_i)
            self.vol.append(vol)
        self.vol = np.array(self.vol)
        self.states[9] = self.vol.sum()
        
        if self._check_constraints(init=True):
            done = False
        else:
            done = True
        return done

    def step(self, action_i):
        if not action_i in set(range(9)):
            raise ValueError

        if action_i == 0:
            reward, done = self._move()
        else:
            reward, done = self._reduce(action_i)

        z_i = int(self.states[0] - 1)
        if z_i + 1 != self.nelz:
            self.states[11:11+8] = self.structure[z_i+1]
        else:
            self.states[11:11+8] = self.states_least[1:9]

        return self.states_scaled(), float(reward), done, {}
    
    def _move(self):
        next = self.states[0] - 1
        if next > 0.5:
            self.states[0] = next
        else:
            self.states[0] = self.nelz

        z_i = int(self.states[0] - 1)
        self.states[1:9] = self.structure[z_i]
        self.states[10] -= 1
        if self._check_moved_count():
            reward = 0
            done = False
        else:
            self.result = self.structure
            reward = -1
            done = True
        return reward, done

    def _reduce(self, i):
        size_list = self.size_df.iloc[i-1].dropna().values
        size_i = np.where(size_list == self.states[i])[0]
        if size_i != 0:
            self.states[i] = size_list[size_i - 1]
        else:
            self.states[i] = self.states_least[i] - self.states_diffmax[i]

        z_i = int(self.states[0] - 1)
        struc_old = np.copy(self.structure)
        vol_old = self.states[9]
        self.structure[z_i] = self.states[1:9]
        self.vol[z_i] = self._cal_vol(self.structure[z_i], z_i)
        self.states[9] = self.vol.sum()
        self.states[10] = self.nelz
        if self._check_constraints():
            reward = self._cal_reward(vol_old)
            done = False
        else:
            self.result = struc_old
            reward = -1
            done = True
        return reward, done

    def states_scaled(self):
        states_scaled = np.zeros(19)
        states_scaled = (self.states - self.states_least) / self.states_diffmax
        return states_scaled

    def _cal_reward(self, vol_old):
        point = (vol_old - self.states[9]) / self.states_diffmax[9]
        return point

    def _check_constraints(self, init=False):
        if not init:
            d = self.states_diffmax[1:9]
            check = self.states[1:9]
            if (check - self.states_least[1:9] < - (d / 2)).any():
                return False

        self._update_labels()
        constraints = self.const_valid(self.params, self.labels)
        for r in constraints:
            r = np.array(r)
            if r.min() < -1e-10:
                return False
        return True

    def _update_labels(self):
        beams = np.arange(1, self.nelz+1)[:, None]
        l_pillars = beams
        r_pillars = beams + self.nelz
        beams = np.concatenate((beams, self.structure[:, :4]), axis=1).flatten()
        l_pillars = np.concatenate((l_pillars, self.structure[:, 4:6]), axis=1).flatten()
        r_pillars = np.concatenate((r_pillars, self.structure[:, 6:8]), axis=1).flatten()
        self.labels = np.concatenate((beams, l_pillars, r_pillars)).tolist()

    def _check_moved_count(self):
        if self.states[10] == 0:
            return False
        return True
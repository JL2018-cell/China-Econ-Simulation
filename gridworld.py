"""
Grid-World Markov Decision Processes (MDPs).

The MDPs in this module are actually not complete MDPs, but rather the
sub-part of an MDP containing states, actions, and transitions (including
their probabilistic character). Reward-function and terminal-states are
supplied separately.

Some general remarks:
    - Edges act as barriers, i.e. if an agent takes an action that would cross
    an edge, the state will not change.

    - Actions are not restricted to specific states. Any action can be taken
    in any state and have a unique inteded outcome. The result of an action
    can be stochastic, but there is always exactly one that can be described
    as the intended result of the action.
"""

import numpy as np
import math
from itertools import product, permutations
from sklearn.preprocessing import StandardScaler

class GridWorld:
    """
    Basic deterministic grid world MDP.

    The attribute size specifies both widht and height of the world, so a
    world will have size**2 states.

    Args:
        size: The width and height of the world as integer.

    Attributes:
        n_states: The number of states of this MDP.
        n_actions: The number of actions of this MDP.
        p_transition: The transition probabilities as table. The entry
            `p_transition[from, to, a]` contains the probability of
            transitioning from state `from` to state `to` via action `a`.
        size: The width and height of the world.
        actions: The actions of this world as paris, indicating the
            direction in terms of coordinates.
    """
    class prob_transition:
        def __init__(self, n_states, n_actions, buildUpLimit, resource_points):
            self.buildUpLimit = buildUpLimit
            self.n_actions = n_actions
            self.resource_points = resource_points
            self.shape = (n_states, n_states, n_actions)
        def shape(self):
            return self.shape
        def state_to_int(self, state):
            base = 10
            result = 0
            for c, i in enumerate(state):
                result += i * 10**(len(state) - c)
            return result
        def histogram(self, n_states, trajectories):
            p = np.zeros(n_states)
        
            for t in trajectories:                  # for each trajectory
                p[self.state_to_int(t.transitions()[0][0])] += 1.0     # increment starting state
        
            return p / len(trajectories)            # normalize
        def transition_func(self, frm, to, act):
            step = to - frm
            if step[0] > self.buildUpLimit['Agriculture'] \
              or step[1] > self.buildUpLimit['Energy'] \
              or sum(step[2:]) > self.resource_points:
                return 0
            else:
                return 1 / self.n_actions

    def __init__(self, max_scale):
        # Industries & Upper limit
        self.industries = {'Agriculture': max_scale, 'Energy': max_scale, 'Finance': max_scale, \
                           'IT': max_scale, 'Minerals': max_scale, 'Tourism': max_scale}
        self.buildUpLimit = {'Agriculture': 10, 'Energy': 10}
        self.resourcePt_contrib = {"Agriculture": 10, "Energy": 10}
        self.resource_points = 2

        # Build/break industies
        self.actions = np.concatenate((np.eye(len(self.industries.keys())), \
                                       -np.eye(len(self.industries.keys()))), \
                                      axis = 0)
        """
        self.actions = [(1, 0, 0, 0, 0, 0),
                        (-1, 0, 0, 0, 0, 0),
                        (0, 1, 0, 0, 0, 0),
                        (0, -1, 0, 0, 0, 0),
                        (0, 0, 1, 0, 0, 0),
                        (0, 0, -1, 0, 0, 0),
                        (0, 0, 0, 1, 0, 0),
                        (0, 0, 0, -1, 0, 0),
                        (0, 0, 0, 0, 1, 0),
                        (0, 0, 0, 0, -1, 0),
                        (0, 0, 0, 0, 0, 1),
                        (0, 0, 0, 0, 0, -1)]
        """

        # Additional 1 state for each industry is industry = 0.
        self.n_states = math.prod([v + 1 for v in self.industries.values()])
        self.n_actions = self.get_num_actions()

        self.p_transition = self.prob_transition(self.n_states, self.n_actions, \
                                                 self.buildUpLimit, self.resource_points)

    def get_num_actions(self):
        self.n_actions = self.buildUpLimit['Agriculture'] * self.buildUpLimit['Energy'] * (1 + self.resource_points)
        return self.n_actions

    def state_to_int(self, ls, n = 0):
        if len(ls) == 0:
            return 0
        elif len(ls) == 1:
            return ls[0] * 10**n
        else:
            return ls[-1] * 10**n + self.state_to_int(ls[0:-1], n + 1)

    def int_to_state(self, ls, n = 0):
        if num < 10:
            return [num]
        else:
            return self.int_to_state(num//10) + [num % 10]

    def state_index_transition(self, s, a):
        """
        Perform action `a` at state `s` and return the intended next state.

        Does not take into account the transition probabilities. Instead it
        just returns the intended outcome of the given action taken at the
        given state, i.e. the outcome in case the action succeeds.

        Args:
            s: The state at which the action should be taken.
            a: The action that should be taken.

        Returns:
            The next state as implied by the given action and state.
        """
        return s + a

    # Return all possible actions
    def all_actions(self):
        many_actions = self.state_features(self.industries)
        return np.array(list(filter(lambda x: x[0] <= self.buildUpLimit['Agriculture'] \
                                              and x[1] <= self.buildUpLimit['Energy'] \
                                              and x[2:].sum() <= self.resource_points, many_actions)))
    # Return all possible states
    def state_features(self, d):
        x = d
        state = np.zeros(len(x.values()))
        all_states = state.reshape(1, state.shape[0])
        for i, limit in enumerate(x.values()):
            for j in range(limit):
                state[i] += 1
                for pemu in set(permutations(state)):
                    all_states = np.concatenate((all_states, np.array(pemu).reshape(1, len(pemu))), axis = 0)
        return all_states
    
    

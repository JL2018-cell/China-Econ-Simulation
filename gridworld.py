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
from itertools import product


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

    def __init__(self):
        # Industries & Upper limit
        self.industries = {'Agriculture': 2000, 'Energy': 2000, 'Finance': 2000, \
                           'IT': 2000, 'Minerals': 2000, 'Tourism': 2000}
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
        self.n_actions = len(self.actions)

        # Agent can only build/break 1 unit in each step.
        # Pr[state_from, state_to, action]
        self.p_transition = lambda frm, to, act: 0 if np.linalg.norm(frm - to) > 1 else 1

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



    

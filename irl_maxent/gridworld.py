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
import threading

class p_transition:
    def __init__(self, shape, table):
        self.shape = shape
        # Probability Function
        self.table = table 
    def p_trans(self, s1, s2, a):
        return self.table(s1, s2, a)

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

    def __init__(self, dim, size):
        self.size = size
        self.dimension = dim

        # self.actions = [tuple(int(math.copysign(1, i)) if i == j else 0 for i in range(-dim, dim)) for j in range(-dim, dim)]
        self.actions = [tuple(1 if i == j else 0 for i in range(dim)) for j in range(dim)] + [tuple(-1 if i == j else 0 for i in range(dim)) for j in range(dim)] + [tuple(0 for _ in range(dim))]

        # Assume dimension number = dim, each dimension has grid size = size.
        self.n_states = size**dim
        self.n_actions = len(self.actions)

        self.p_transition = p_transition(shape=(self.n_states, self.n_states, self.n_actions), table = self._transition_prob)
        #self._transition_prob_table()

    def state_index_to_point(self, state):
        """
        Convert a state index to the coordinate representing it.

        Args:
            state: Integer representing the state.

        Returns:
            The coordinate as tuple of integers representing the same state
            as the index.
        """
        point = []
        for d in range(self.dimension):
            point.append(state % (self.size**d))
        #while state >= self.size:
        #    point.append(state % self.size)
        #    state = state // self.size
        point.reverse()
        return tuple(point)

    def state_point_to_index(self, state):
        """
        Convert a state coordinate to the index representing it.

        Note:
            Does not check if coordinates lie outside of the world.

        Args:
            state: Tuple of integers representing the state.

        Returns:
            The index as integer representing the same state as the given
            coordinate.
        """
        state = list(state)
        state.reverse()
        index = 0
        for i, x in enumerate(state):
            index += x * self.size**i
        return index

    def state_point_to_index_clipped(self, state):
        """
        Convert a state coordinate to the index representing it, while also
        handling coordinates that would lie outside of this world.

        Coordinates that are outside of the world will be clipped to the
        world, i.e. projected onto to the nearest coordinate that lies
        inside this world.

        Useful for handling transitions that could go over an edge.

        Args:
            state: The tuple of integers representing the state.

        Returns:
            The index as integer representing the same state as the given
            coordinate if the coordinate lies inside this world, or the
            index to the closest state that lies inside the world.
        """
        s = (max(0, min(self.size - 1, state[0])), max(0, min(self.size - 1, state[1])))
        return self.state_point_to_index(s)

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
        s = self.state_index_to_point(s)
        s = s[0] + self.actions[a][0], s[1] + self.actions[a][1]
        return self.state_point_to_index_clipped(s)

    def _transition_prob_table(self):
        """
        Builds the internal probability transition table.

        Returns:
            The probability transition table of the form

                [state_from, state_to, action]

            containing all transition probabilities. The individual
            transition probabilities are defined by `self._transition_prob'.
        """
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        return table

    def _transition_prob(self, s_from, s_to, a):
        """
        Compute the transition probability for a single transition.

        Args:
            s_from: The state in which the transition originates.
            s_to: The target-state of the transition.
            a: The action via which the target state should be reached.

        Returns:
            The transition probability from `s_from` to `s_to` when taking
            action `a`.
        """
        fx, fy = self.state_index_to_point(s_from)
        tx, ty = self.state_index_to_point(s_to)
        ax, ay = self.actions[a]

        # deterministic transition defined by action
        if fx + ax == tx and fy + ay == ty:
            return 1.0

        # we can stay at the same state if we would move over an edge
        if fx == tx and fy == ty:
            if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
                return 1.0

        # otherwise this transition is impossible
        return 0.0

    def __repr__(self):
        return "GridWorld(size={})".format(self.size)


class IcyGridWorld(GridWorld):
    """
    Grid world MDP similar to Frozen Lake, just without the holes in the ice.

    In this worlds, agents will slip with a specified probability, causing
    the agent to end up in a random neighboring state instead of the one
    implied by the chosen action.

    Args:
        size: The width and height of the world as integer.
        p_slip: The probability of a slip.

    Attributes:
        p_slip: The probability of a slip.

    See `class GridWorld` for more information.
    """

    def __init__(self, industry_dstr, buildUpLimit, industries_limit, resourcePt_contribution, dim, size, p_slip=0.2):
        self.p_slip = p_slip
        self.industry_dstr = industry_dstr
        self.buildUpLimit = buildUpLimit
        self.industries_limit = industries_limit
        self.resource_points = 10
        self.resourcePt_contribution = resourcePt_contribution

        super().__init__(dim, size)

    def _transition_prob(self, s_from, s_to, a):
        """
        Compute the transition probability for a single transition.

        Args:
            s_from: The state in which the transition originates.
            s_to: The target-state of the transition.
            a: The action via which the target state should be reached.

        Returns:
            The transition probability from `s_from` to `s_to` when taking
            action `a`.
        """
        frm = self.state_index_to_point(s_from)
        to = self.state_index_to_point(s_to)
        action = self.actions[a]

        step = np.array(to) - np.array(frm)
        #print("_transition_prob", a)
        if np.array_equal(step, np.array(action)):
            if step[0] <= self.buildUpLimit['Agriculture'] \
              and step[1] <= self.buildUpLimit['Energy'] \
              and step[2:].sum() <= self.resource_points:
                return 1 / self.n_actions
        else:
            return 0.

    def __repr__(self):
        return "IcyGridWorld(size={}, p_slip={})".format(self.size, self.p_slip)


def state_features(world):
    """
    Return the feature matrix assigning each state with an individual
    feature (i.e. an identity matrix of size n_states * n_states).

    Rows represent individual states, columns the feature entries.

    Args:
        world: A GridWorld instance for which the feature-matrix should be
            computed.

    Returns:
        The coordinate-feature-matrix for the specified world.
    """
    return np.identity(world.n_states)


def coordinate_features(world):
    """
    Symmetric features assigning each state a vector where the respective
    coordinate indices are nonzero (i.e. a matrix of size n_states *
    world_size).

    Rows represent individual states, columns the feature entries.

    Args:
        world: A GridWorld instance for which the feature-matrix should be
            computed.

    Returns:
        The coordinate-feature-matrix for the specified world.
    """
    features = np.zeros((world.n_states, world.size))

    for s in range(world.n_states):
        x, y = world.state_index_to_point(s)
        features[s, x] += 1
        features[s, y] += 1

    return features

# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from foundation.base.base_agent import BaseAgent, agent_registry
import copy

@agent_registry.add
class localGov(BaseAgent):
    """
    A basic mobile agent represents an individual actor in the economic simulation.
    """

    name = "localGov"
    #preference = {} 
    #{inv: 0.5 for inv in self._registered_inventory}
    def __init__(self, loc, buildUpLimit, idx=None, multi_action_mode=None):
        BaseAgent.__init__(self, idx=idx, multi_action_mode=multi_action_mode)
        self.state["loc"] = loc
        self.resource_points = 0
        self.buildUpLimit = buildUpLimit
        self.buildUpIncrm = copy.copy(buildUpLimit)
        self.industries = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism']
        self.preference = {inv: 0.5 for inv in self.industries}
        #self.preference = {inv: 0.5 for inv in self._registered_inventory}
        #(Pdb) p self.state
        #{'loc': [0, 0], 'inventory': {}, 'escrow': {}, 'endogenous': {}}

    """
    #See if agent usea more than buildUpLimit and resource_points than it has.
    #type(actions): []
    #If actions do not comply with constraint and reset flag = True, then resample actions until it complies with constraint.
    #Otherwise, return how it violate the constraint.
    def check_actions(self, actions, reset = False):
        #See if agent uses all of its buildUpLimit and resource_points.
        resourceful = True

        #build up Agriculture & Energy industry < buildUpLimit
        if actions['Construct.build_Agriculture'] > self.buildUpLimit['Agriculture']:
            resourceful = False
        if actions['Construct.build_Energy'] > self.buildUpLimit['Energy']:
            resourceful = False

        #build up other industries < resource point
        builds = {k: v for k, v in actions.items() if "Construct.build_" in k}
        builds.pop('Construct.build_Agriculture')
        builds.pop('Construct.build_Energy')
        #The agent use too much resource points
        if sum([sum(v) for v in builds.values()]) > self.resource_points:
            resourceful = False

        return actions

    #Update state of agent after taking actions
    #No need to worry this. Each component will take care of this in def component_step(.):
    def parse_actions(self, actions):
        print(self.idx, "parse action.")
        for action, magnitude in actions.items():
            if "Construct" in action:
                if "break" in action:
                    self.state['inventory'][action.split("_")[-1]] -= magnitude
                    self.resource_points += magnitude
                elif "build" in action:
                    self.state['inventory'][action.split("_")[-1]] += magnitude
                    self.resource_points -= magnitude
            elif "ContinuousDoubleAuction" in action:
                pass
                #if "Buy" in action:
                #elif "Sell" in action:
    """

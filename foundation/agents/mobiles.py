# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from foundation.base.base_agent import BaseAgent, agent_registry

@agent_registry.add
class localGov(BaseAgent):
    """
    A basic mobile agent represents an individual actor in the economic simulation.
    """

    name = "localGov"
    #preference = {} 
    #{inv: 0.5 for inv in self._registered_inventory}
    def __init__(self, idx=None, multi_action_mode=None):
        BaseAgent.__init__(self, idx=None, multi_action_mode=None)
        self.preference = {inv: 0.5 for inv in self._registered_inventory}
        #(Pdb) p self.state
        #{'loc': [0, 0], 'inventory': {}, 'escrow': {}, 'endogenous': {}}

#@agent_registry.add
#class BasicMobileAgent(BaseAgent):
#    """
#    A basic mobile agent represents an individual actor in the economic simulation.
#
#    "Mobile" refers to agents of this type being able to move around in the 2D world.
#    """
#
#    name = "BasicMobileAgent"

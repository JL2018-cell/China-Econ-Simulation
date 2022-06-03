from foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

from foundation.scenarios.MacroEcon.layout import MacroEconLayout #Get data of GDP, CO2 contribution.
import numpy as np
import random

#Build an industry
@component_registry.add
class Construct(BaseComponent):

    name = "Construct"
    component_type = "Construct"
    required_entities = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism']
    agent_subclasses = ["localGov"]


    def __init__(self, *base_component_args, payment=10, **base_component_kwargs):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.payment = payment

    def component_step(self):
        world = self.world
        build = []

        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():
            for action, magnitude in agent.action.items():
                print("Agent", agent.idx, ":", action, "-", magnitude)
                if "Construct" in action:
                    if "break_" in action:
                        target_industry = action.split("_")[-1]
                        # After breaking industry, resultant resource point allocated on it should not < 0.
                        if agent.state['inventory'][target_industry] - magnitude > 0:
                            # target_industry is Agriculture or Energy Industry.
                            if target_industry in self.required_entities[0:2]:
                                    agent.state['inventory'][target_industry] -= magnitude
                                    agent.buildUpLimit[target_industry] += magnitude
                            else:
                                    agent.state['inventory'][target_industry] -= magnitude
                                    agent.resource_points += magnitude
                        else:
                            #Incorporate punishment of agent acts outside limit.
                            pass
                    elif "build_" in action:
                        target_industry = action.split("_")[-1]
                        # After building industry, resultant resource point allocated on it should not > preference i.e. upper limit.
                        if agent.state['inventory'][target_industry] + magnitude <= agent.preference[target_industry]:
                            # target_industry is Agriculture or Energy Industry.
                            if target_industry in self.required_entities[0:2]:
                                    agent.state['inventory'][target_industry] += magnitude
                                    agent.buildUpLimit[target_industry] -= magnitude
                            else:
                                    agent.state['inventory'][target_industry] += magnitude
                                    agent.resource_points -= magnitude
                        else:
                            #Incorporate punishment of agent acts outside limit.
                            pass
            # In the next timestep, agent gets resources to build Agriculture and Energy industry.
            for industry in agent.buildUpLimit.keys():
                agent.buildUpLimit[industry] += agent.buildUpIncrm[industry]

            #Each agent has a prefernce list to construct or vreak industry.
        for agent in self.world.agents:
            agent.state["resource_points"] = agent.resource_points
            agent.state["buildUpLimit"] = agent.buildUpLimit
            """
            action list:
            ['Construct.build_Agriculture', 'Construct.break_Agriculture', 
             'Construct.build_Energy', 'Construct.break_Energy', 
             'Construct.build_Finance', 'Construct.break_Finance', 
             'Construct.build_IT', 'Construct.break_IT', 
             'Construct.build_Minerals', 'Construct.break_Minerals', 
             'Construct.build_Tourism', 'Construct.break_Tourism']
            """

    def generate_masks(self, completions = 0):
        #Refer to ai_ChinaEcon\foundation\components\continuous_double_auction.py:580
        masks = {}
        for agent in self.world.agents:
            masks[agent.idx] = {}
            #localGov is free to build or break its industries.
            for entity in self.required_entities:
                #masks[agent.idx][entity] = np.array([True, True]) #np.array([build, break])
                masks[agent.idx]["build_" + entity] = np.array([True])
                masks[agent.idx]["break_" + entity] = np.array([True])
        masks[self.world.planner.idx] = {}
        for entity in self.required_entities:
            masks[self.world.planner.idx]["charge_" + entity] = np.array([True])
            masks[self.world.planner.idx]["release_" + entity] = np.array([True])
        return masks


    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe: 
        How much local government spend on each industry. 
        How much is spent in transporting resources from other provinces.

        The central government (planner) does not observe anything from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                #How much is paid to build industries.
                #"build_payment": agent.state["build_payment"] / self.payment,
                #How much resource is generated.
                #"build_resources": self.sampled_skills[agent.idx],
                "loc": agent.state["loc"],
                "Agriculture": np.array(agent.state["inventory"]["Agriculture"]),
                "Energy": np.array(agent.state["inventory"]["Energy"])
            }

        return obs_dict

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        Update CO2, GDP, Labour variables of localGov.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"build_payment": float(self.payment), "build_resources": 1}
        if agent_cls_name == "localGov":
          #Keep track on how many points does localGov spend on constructing industries and transporting resources form other provinces.
          return {"construct_payment": 0.0, "transport_resources": 0.0}
        raise NotImplementedError

    #Define actions available for agent
    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (build) for mobile agents.
        """
        #Local government can choose to:
        #1. Increase industry point by 1.
        #1. Decrease industry point by 1.
        if agent_cls_name == "localGov":
            #Inappropriate. This is telling agent can move in 2 direction in world map.
            #return [(entity, 2) for entity in self.required_entities]
            actions = []
            for c in self.required_entities:
                actions.append(("build_{}".format(c), 20))
                actions.append(("break_{}".format(c), 20))
            return actions
        if agent_cls_name == "centralGov":
            charges = []
            for c in self.required_entities:
                charges.append(("charge_{}".format(c), 20))
                charges.append(("release_{}".format(c), 20))
            return charges
        return None




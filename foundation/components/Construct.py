
#problem: industry_choice = np.random.choice(industries, size = 1, p = industry_preference)
#Where does industries, industry_preference come from?

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

            action = agent.get_component_action(self.name)
            #Each agent has a prefernce list to construct or vreak industry.
            """
            action list:
            ['Construct.build_Agriculture', 'Construct.break_Agriculture', 
             'Construct.build_Energy', 'Construct.break_Energy', 
             'Construct.build_Finance', 'Construct.break_Finance', 
             'Construct.build_IT', 'Construct.break_IT', 
             'Construct.build_Minerals', 'Construct.break_Minerals', 
             'Construct.build_Tourism', 'Construct.break_Tourism']
            """
            for agent in self.world.agents:
                if action[0] > 0:
                    if random.random() < agent.preference['Agriculture']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Agriculture'] += 1
                if action[1] > 0: #Destroy Agriculture industry
                    if random.random() > agent.preference['Agriculture']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Agriculture'] -= 1
                if action[2] > 0:
                    if random.random() < agent.preference['Energy']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Energy'] += 1
                if action[3] > 0:
                    if random.random() > agent.preference['Energy']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Energy'] -= 1
                if action[4] > 0:
                    if random.random() < agent.preference['Finance']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Finance'] += 1
                if action[5] > 0:
                    if random.random() > agent.preference['Finance']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Finance'] -= 1
                if action[6] > 0:
                    if random.random() < agent.preference['IT']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['IT'] += 1
                if action[7] > 0:
                    if random.random() > agent.preference['IT']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['IT'] -= 1
                if action[8] > 0:
                    if random.random() < agent.preference['Minerals']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Minerals'] += 1
                if action[9] > 0:
                    if random.random() > agent.preference['Minerals']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Minerals'] -= 1
                if action[10] > 0:
                    if random.random() < agent.preference['Tourism']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Tourism'] += 1
                if action[11] > 0:
                    if random.random() > agent.preference['Tourism']: #Construct Agriulture industry.
                        self.world.agents[0].inventory['Tourism'] -= 1
            #self.world.agents[0].preference

            """
            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Build! (If you can.)
            elif action == 1:
                if self.agent_can_build(agent):
                    # Remove the resources that the local government has.
                    for resource, cost in self.resource_cost.items():
                        agent.state["inventory"][resource] -= cost

                    # Create an industry in the location of local government. 
                    loc_r, loc_c = agent.loc
                    industry_choice = np.random.choice(industries, size = 1, p = industry_preference)
                    world.create_landmark(industry_choice, loc_r, loc_c, agent.idx)

                    # Receive feedback from the industry i.e. GDP growth, carbon dioxide emission
                    agent.state["performance"]["GDP"] += 100
                    agent.state["performance"]["CO2"] += 100

                    # Incur the labor cost for building
                    agent.state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": agent.idx,
                            "loc": np.array(agent.loc),
                            "income": float(agent.state["build_payment"]),
                        }
                    )

            else:
                raise ValueError
            """

        #self.builds.append(build)

        return []

    def generate_masks(self, completions = 0):
        #Refer to ai_ChinaEcon\foundation\components\continuous_double_auction.py:580
        masks = {}
        for agent in self.world.agents:
            masks[agent.idx] = {}
            if isinstance(agent.idx, int):
                #localGov is free to build or break its industries.
                for entity in self.required_entities:
                    #masks[agent.idx][entity] = np.array([True, True]) #np.array([build, break])
                    masks[agent.idx]["build_" + entity] = np.array([True])
                    masks[agent.idx]["break_" + entity] = np.array([True])
            else:
                    masks[agent.idx]["store_" + entity] = np.array([True])
                    masks[agent.idx]["release_" + entity] = np.array([True])

        agent = self.world.planner
        masks[agent.idx] = {}
        for entity in self.required_entities:
            masks[agent.idx]["store_{}".format(entity)] = np.array([True])
            masks[agent.idx]["release_{}".format(entity)] = np.array([True])

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
                "Agriculture": agent.state["inventory"]["Agriculture"],
                "Energy": agent.state["inventory"]["Energy"]
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
                actions.append(("build_{}".format(c), 1))
                actions.append(("break_{}".format(c), 1))
            return actions

        if agent_cls_name == "centralGov":
            actions = []
            for c in self.required_entities:
                actions.append(("store_{}".format(c), 1))
                actions.append(("release_{}".format(c), 1))
            return actions

        return None




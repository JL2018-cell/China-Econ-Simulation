
#problem: industry_choice = np.random.choice(industries, size = 1, p = industry_preference)
#Where does industries, industry_preference come from?

from foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

from foundation.scenarios.MacroEcon.layout import MacroEconLayout #Get data of GDP, CO2 contribution.
import numpy as np

#Build an industry
@component_registry.add
class Construct(BaseComponent):

    name = "Construct"
    component_type = "Construct"
    required_entities = ["Agriculture", "Minerals"]
    agent_subclasses = ["BasicMobileAgent"]


    def __init__(self, *base_component_args, payment=10, **base_component_kwargs):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.payment = payment

    def component_step(self):
        world = self.world
        build = []

        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

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

        self.builds.append(build)

        return []

    def generate_masks(self):
        masks = {}
        # Mobile agents' build action is masked if they cannot build with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([self.agent_can_build(agent)])

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
                "build_payment": agent.state["build_payment"] / self.payment,
                #How much resource is generated.
                "build_resources": self.sampled_skills[agent.idx],
            }

        return obs_dict

    def get_additional_state_fields(self):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"build_payment": float(self.payment), "build_resources": 1}
        raise NotImplementedError

    def get_n_actions(self):
        """
        See base_component.py for detailed description.

        Add a single action (build) for mobile agents.
        """
        # This component adds 1 action that mobile agents can take: build a house
        if agent_cls_name == "BasicMobileAgent":
            return 1

        return None




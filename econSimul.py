from foundation.entities.resources import Resource, resource_registry
from foundation.base.base_component import BaseComponent, component_registry
from foundation.base.base_env import BaseEnvironment, scenario_registry
from foundation.base.base_agent import BaseAgent, agent_registry
from foundation.scenarios.MacroEcon.layout import MacroEconLayout
from foundation.components import component_registry
from foundation.components.Construct import Construct
from foundation.components.Transport import Transport
import foundation
import numpy as np

@resource_registry.add
class Widget(Resource):
    name = "Widget"
    color = [1, 1, 1]
    collectible = False # <--- Goes in agent inventory, but not in the world

@component_registry.add
class BuyWidgetFromVirtualStore(BaseComponent):
    name = "BuyWidgetFromVirtualStore"
    required_entities = ["Coin", "Widget"]  # <--- We can now look up "Widget" in the resource registry
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        widget_refresh_rate=0.1,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.widget_refresh_rate = widget_refresh_rate
        self.available_widget_units = 0
        self.widget_price = 5

def get_n_actions(self, agent_cls_name):
    # This component adds 1 binary action that mobile agents can take: buy widget (or not).
    if agent_cls_name == "BasicMobileAgent":
        return 1  # Buy or not.

    return None

def generate_masks(self, completions=0):
    masks = {}
    # Mobile agents' buy action is masked if they cannot build with their
    # current coin or if no widgets are available.
    for agent in self.world.agents:
        masks[agent.idx] = np.array([
            agent.state["inventory"]["Coin"] >= self.widget_price and self.available_widget_units > 0
        ])

    return masks

@agent_registry.add
class province(BaseAgent):
    name = "GuangDong"

#industries
new_rsc_cls = resource_registry.get("Widget")
print(new_rsc_cls)


new_cmp_cls = component_registry.get("BuyWidgetFromVirtualStore")
print(new_cmp_cls)

#Province
new_agn_cls = agent_registry.get("GuangDong")
print(new_agn_cls)
print(dir(new_cmp_cls))



env_config = {
    'scenario_name': 'layout/MacroEcon',
    #to be contnued after layout construction on foundation/scenarios/MacroEcon.
    'world_size': [100, 100],
    'n_agents': 3,
    'agent_names': ["GuangDong", "HeBei", "XinJiang"],

    'components': [
        #Build industries
        {"Construct": {}},
        #Import resources from other provinces
        {"Transport": {}},
        #Exchange resources, industry points by auction.
        {'ContinuousDoubleAuction': {'max_num_orders': 5}},
    ],
    'industries': ['Agriculture', 'Energy', 'Finance', \
                   'IT', 'Minerals', 'Tourism'], #Help to define actions of localGov


    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    'multi_action_mode_agents': True,
    'multi_action_mode_planner': True,


    # (optional) kwargs of the chosen scenario class
    'starting_agent_resources': {"Food": 10., "Energy": 10.} #food, energy
}

print("MacroEconLayout:", MacroEconLayout)

test_env_cls = scenario_registry.get("layout/MacroEcon")
print("scenario_registry.get:", test_env_cls)

env = foundation.make_env_instance(**env_config)

print("get agent")
print(env.get_agent(0))

#call ai_ChinaEcon\foundation\base\base_env.py:852
obs = env.reset()

def sample_random_action(agent, mask):
    """Sample random UNMASKED action(s) for agent."""
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        return [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask/mask.sum())


def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""
        
    actions = {
        #a_obs['action_mask'] defined in ai_ChinaEcon\foundation\base\base_env.py:702
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions


actions = sample_random_actions(env, obs)
#call step to advance the state and advance time by one tick.
obs, rew, done, info = env.step(actions)

print("Computation done.")
print("obs.keys:\n", obs.keys())
print("Investigate items.")
for key, val in obs['0'].items(): 
    print("{:50} {}".format(key, type(val)))
print("Reward of agents.")
for agent_idx, reward in rew.items(): 
    print("{:2} {:.3f}".format(agent_idx, reward))
print("Done")
print(done)

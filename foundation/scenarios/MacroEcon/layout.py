import numpy as np
from scipy import signal

from foundation.base.base_env import BaseEnvironment, scenario_registry
from foundation.scenarios.utils import rewards, social_metrics
from foundation.base.base_env import BaseEnvironment, scenario_registry

@scenario_registry.add
class agriculture_industry(BaseEnvironment):
  def __init__(self):
    self.resource = 0

@scenario_registry.add
class energy_industry(BaseEnvironment):
  def __init__(self):
    self.resource = 0
    
@scenario_registry.add
class MacroEconLayout(BaseEnvironment):
  """
  Map of China.
  """
  name = "layout/MacroEcon"
  
  """
  Industries
  """
  
  agent_subclasses = ["LocalGov", "CentralGov"]
  required_industries = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism']
  required_entities = required_industries

  def __init__(self, starting_agent_resources, industries, **kwargs):
    self.world_size = [100, 100]
    self.energy_cost = 100
    self.expn_per_day = 100
    self.pt_per_day = 100
    self.toCarbonEffcy = 0.5
    self.toGDPEffcy = 0.5
    self.GDP_contrib = {"Minerals": 100, "Agriculture": 50}
    self.CO2_contrib = {"Minerals": 110, "Agriculture": 100} 
    self.agric = starting_agent_resources["Food"]
    self.energy = starting_agent_resources["Energy"]
    self.resource_points = 100.
    self.industries = industries
    #assert self.starting_agent_coin >= 0.0
    super().__init__(**kwargs)


  """ p dir(self)
  ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', 
  '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', 
  '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', 
  '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', 
  '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', 
  '_agent_lookup', '_allow_observation_scaling', '_build_packager', '_completions', 
  '_components', '_components_dict', '_create_dense_log_every', '_dense_log', 
  '_dense_log_this_episode', '_entities', '_episode_length', '_finalize_logs', 
  '_flatten_masks', '_flatten_observations', '_generate_masks', '_generate_observations', 
  '_generate_rewards', '_last_ep_dense_log', '_last_ep_metrics', '_last_ep_replay_log', 
  '_package', '_packagers', '_register_entities', '_replay_log', '_shorthand_lookup', 
  '_world_dense_log_frequency', 'additional_reset_steps', 'agent_subclasses', 'agric', 
  'all_agents', 'collate_agent_info', 'collate_agent_obs', 'collate_agent_rew', 
  'collate_agent_step_and_reset_data', 'components', 'compute_reward', 'dense_log', 
  'endogenous', 'energy', 'energy_cost', 'episode_length', 'expn_per_day', 'generate_observations', 
  'generate_rewards', 'get_agent', 'get_component', 'inv_scale', 'landmarks', 'metrics', 
  'multi_action_mode_agents', 'multi_action_mode_planner', 'n_agents', 'name', 'num_agents', 
  'parse_actions', 'previous_episode_dense_log', 'previous_episode_metrics', 'previous_episode_replay_log', 
  'pt_per_day', 'replay_log', 'required_entities', 'required_industries', 'reset', 'reset_agent_states', 
  'reset_starting_layout', 'resource_points', 'resources', 'scenario_metrics', 'scenario_step', 'seed', 
  'set_agent_component_action', 'step', 'toCarbonEffcy', 'toGDPEffcy', 'world', 'world_size']
  """

  def compute_reward(self):
      #weights = {k:1 for k in agent.state['inventory'].keys()}
      #for endogn in agent.state['endogenous']:
      #    weights[endogn] = agent.state['endogenous'][endogn]
      #weights = np.array(list(agent.state['inventory'].keys()) \
      #                   + list(agent.state['endogenous'].keys()))

      #Reward = linear combination of ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism', 'CO2', 'GDP', 'Labor']
      #Assume all weights = 1
      rewards = {}
      for agent in self.world.agents:
        #rewards[agent.idx] = sum(agent.state['inventory'].values()) + sum(agent.state['endogenous'].values())
        weights = np.array([1 for i in list(agent.state['inventory'].keys()) + list(agent.state['endogenous'].keys())])
        rewards[agent.idx] = np.dot(np.array(list(agent.state['inventory'].values()) + list(agent.state['endogenous'].values())), weights)
      print("In layout, rewards:", rewards)
      return rewards

  def generate_observations(self):
      #Include ALL agents and object in this world. Refer to ai_ChinaEcon\foundation\base\base_env.py:648
      obs = {}
      obs[self.world.planner.idx] = {
          "industry-" + k: v * self.inv_scale
          for k, v in self.world.planner.inventory.items()
      }
      #Show location of modelled provinces/lcal government
      #obs['map'] = self.world.maps.state

      #for agent in self.world.agents:
      #for planner in self.world.planners:
      """
      #refer to ai_ChinaEcon_v2\foundation\base\world.py
      #Local government i: {"Industry-Agriculture": points, "Resources": points}
      """
      return obs

  def reset_agent_states(self):
      self.agric = 100.
      self.energy = 100.

  def reset_starting_layout(self):
      self.agric = 100.
      self.energy = 100.

  def scenario_step(self):
      self.agric += 1.
      self.energy += 1.


  """
  @property
  def observe(self):
    return []
  
  def reset_scenario(self):
    self.agric = 100.
    self.energy = 100.
  
  def next_step(self):
    self.agric += 1
    self.energy += 1
    
  def get_rewards(self):
    return 0
  """

import numpy as np

from foundation.base.base_env import BaseEnvironment, scenario_registry
from foundation.scenarios.utils import rewards, social_metrics

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
    #Total CO2 emission of all localGov.
    self.total_CO2 = 0.
    self.world_size = [100, 100]
    self.energy_cost = 100
    self.expn_per_day = 100
    self.pt_per_day = 100
    self.toCarbonEffcy = 0.5
    self.toGDPEffcy = 0.5
    self.resourcePt_contrib = {"Agriculture": 10, "Energy": 10}
    self.GDP_contrib = {"Agriculture": 100, "Energy": 50}
    self.CO2_contrib = {"Agriculture": 110, "Energy": 100} 
    self.agric = starting_agent_resources["Food"]
    self.energy = starting_agent_resources["Energy"]
    self.resource_points = 100.
    self.industries = industries
    #assert self.starting_agent_coin >= 0.0
    super().__init__(**kwargs)


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
      #Observe agents
      for agent in self.world.agents:
          obs[str(agent.idx)] = {}
          obs[str(agent.idx)]['actions'] = [act for act, b in agent.action.items() if b > 0]
          obs[str(agent.idx)]['industries'] = agent.state['inventory']
          obs[str(agent.idx)]['endogenous'] = agent.state['endogenous']
      #Observe planner
      obs[self.world.planner.idx] = {}
      obs[self.world.planner.idx]['actions'] = self.world.planner.action
      obs[self.world.planner.idx]['storage'] = self.world.planner.inventory
      return obs

  def reset_agent_states(self):
      self.agric = 100.
      self.energy = 100.

  def reset_starting_layout(self):
      self.agric = 100.
      self.energy = 100.

  def scenario_step(self):
      for agent in self.world.agents:
          agent.resource_points += self.resourcePt_contrib["Agriculture"] + self.resourcePt_contrib["Energy"]
          for k, v in agent.action.items():
              if v > 0:
                  try:
                      agent.state['endogenous']['CO2'] += self.CO2_contrib[k]
                  except KeyError:
                      pass
      self.total_CO2 = sum([agent.state['endogenous']['CO2'] for agent in self.world.agents])
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

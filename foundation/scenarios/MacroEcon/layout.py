import numpy as np
import copy
from foundation.base.base_env import BaseEnvironment, scenario_registry
from foundation.scenarios.utils import rewards, social_metrics
#from irl_maxent.main import irl_maxent
import irl_maxent.main
import os
import pickle
import pandas as pd

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
  # required_industries = ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism']
  # required_entities = required_industries

  def __init__(self, starting_agent_resources, contribution, industries, industries_chin, industry_depreciation, irl, irl_data_path, **kwargs):
      self.required_entities = list(industries.keys())
      # Depreciation of industry in each time step.
      # Format: {"agent 1": {industry 1: int}, "agent 2": ...}
      self.industry_depreciation = industry_depreciation
      # Total CO2 emission of all localGov.
      self.total_CO2 = 0.
      self.irl = irl
      self.irl_data_path = irl_data_path
      self.world_size = [100, 100]
      self.energy_cost = 100
      self.expn_per_day = 100
      self.pt_per_day = 100
      self.toCarbonEffcy = 0.5
      self.toGDPEffcy = 0.5
      self.resourcePt_contrib = contribution["resource_points"]
      self.GDP_contrib = contribution["GDP"]
      self.CO2_contrib = contribution["CO2"]
      self.agric = starting_agent_resources["Food"]
      self.energy = starting_agent_resources["Energy"]
      self.resource_points = 100.
      #assert self.starting_agent_coin >= 0.0
      kwargs = {**kwargs, **{"industries": self.required_entities}}
      super().__init__(**kwargs)
      # Use inverse reinforcement learning to define reward function of agents.
      if self.irl:
          root, directory, files = list(os.walk("./irl_maxent"))[0]
          if "rewards.pkl" in files:
              print("Read reward.pkl from file.")
              self.rewards = pd.read_pickle("./irl_maxent/rewards.pkl")
          else:
              print("Learn rewards of agents from data.")
              self.rewards = irl_maxent.main.irl_maxent(self.irl_data_path, list(industry_depreciation.keys()), list(industries.keys()), industries_chin, kwargs["industry_init_dstr"], contribution, kwargs["buildUpLimit"], industries)
              with open('./irl_maxent/rewards.pkl', 'wb') as f:
                  pickle.dump(self.rewards, f)
      for industry, upperLimits in industries.items():
          if isinstance(upperLimits, list): #Each region has its own upper limit of building industries.
              for i, agent in enumerate(self.world.agents):
                  agent.preference.update({industry: upperLimits[i]})
          elif isinstance(upperLimits, (int, float)): #All regions have same upper limit of building industries.
              for agent in self.world.agents:
                  agent.preference.update({industry: upperLimits})
          else:
              raise ValueError("Upper limit of building industries shoud have type List or int.")

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
          index += x * 2**i
      return index

  def compute_reward(self):
      if self.irl:
          rewards = {}
          for agent in self.world.agents:
              rewards[agent.idx] = 0
              max_val = max(agent.state['inventory'].values())
              tmp = max_val
              enum = 0
              while(tmp > 1):
                  idx = self.state_point_to_index(np.array(list(agent.state['inventory'].values())) > tmp)
                  rewards[agent.idx] += self.rewards[agent.state['name']][idx, enum]
                  enum += 1
                  if (enum == self.rewards[agent.state['name']].shape[1]):
                      break
                  tmp = tmp // 2
          rewards[self.world.planner.idx] = self.total_GDP - self.total_CO2
          return rewards
      else:
          rewards = {}
          for agent in self.world.agents:
              weights = np.array(list(agent.industry_weights.values()) + [1. for i in agent.state['endogenous'].keys()])
              rewards[agent.idx] = np.dot(np.array(list(agent.state['inventory'].values()) + list(agent.state['endogenous'].values())), weights)
          rewards[self.world.planner.idx] = self.total_GDP - self.total_CO2
          print("In layout, rewards:", rewards)
          return rewards

  def generate_observations(self):
      #Include ALL agents and object in this world. Refer to ai_ChinaEcon\foundation\base\base_env.py:648
      obs = {}
      #Observe agents
      for agent in self.world.agents:
          obs[str(agent.idx)] = {}
          obs[str(agent.idx)]['actions'] = {k: np.array(int(v)) for k, v in agent.action.items()}
          obs[str(agent.idx)]['industries'] = {k: np.array(int(v)) for k, v in agent.state['inventory'].items()}
          obs[str(agent.idx)]['endogenous'] = {k: np.array(int(v)) for k, v in agent.state['endogenous'].items()}
      #Observe planner
      obs[self.world.planner.idx] = {}
      obs[self.world.planner.idx]['actions'] = {k: np.array(int(v)) for k, v in self.world.planner.action.items()}
      obs[self.world.planner.idx]['storage'] = {k: np.array([int(v)]) for k, v in self.world.planner.inventory.items()}
      return obs

  def reset_agent_states(self):
      self.agric = 100.
      self.energy = 100.

  def reset_starting_layout(self):
      self.agric = 100.
      self.energy = 100.

  def scenario_step(self):
      for agent in self.world.agents:
          agent_name = agent.state["name"]
          idx = agent.idx
          # Agriculture and energy industry produce resource points to build other industries
          agent.resource_points += sum(self.resourcePt_contrib[agent_name].values())
          # Calculate cumulative CO2, GDP produced by each industry in each agent.
          for k, v in agent.action.items():
              if v > 0:
                  industry = k.split("_")[-1]
                  try:
                      self.world.agents[idx].state['endogenous']['CO2'] += self.CO2_contrib[agent_name][industry]
                  except KeyError:
                      pass
                  try:
                      self.world.agents[idx].state['endogenous']['GDP'] += self.GDP_contrib[agent_name][industry]
                  except KeyError:
                      pass
          # Industry depreciate over time.
          for industry in agent.state['inventory'].keys():
              if agent.state["inventory"][industry] - self.industry_depreciation[agent.state["name"]][industry] > 0:
                  self.world.agents[idx].state['inventory'][industry] -= self.industry_depreciation[agent.state["name"]][industry]
              else:
                  self.world.agents[idx].state['inventory'][industry] = 0

          #agent.state['inventory'] = {k: agent.state["inventory"][k] - v 
          #                            for k, v in self.industry_depreciation[agent.state["name"]].items()
          #                            if agent.state["inventory"][k] - v > 0 else k: 0}
      self.total_CO2 = sum([agent.state['endogenous']['CO2'] for agent in self.world.agents])
      self.total_GDP = sum([agent.state['endogenous']['GDP'] for agent in self.world.agents])
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

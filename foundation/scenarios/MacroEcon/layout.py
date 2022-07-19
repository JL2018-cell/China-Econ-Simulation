#Do not handle overflow in adding GDP, CO2 to agent state.

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

  def __init__(self, starting_agent_resources, contribution, industries, industries_chin, industry_depreciation, irl, irl_data_path, contribution_chg_rate, **kwargs):
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
      self.contribution_chg_rate = contribution_chg_rate
      self.contribution = contribution
      self.buildUpLimit = kwargs['buildUpLimit']
      #self.resourcePt_contrib = contribution["resource_points"]
      #self.GDP_contrib = contribution["GDP"]
      #self.CO2_contrib = contribution["CO2"]
      self.agric = starting_agent_resources["Food"]
      self.energy = starting_agent_resources["Energy"]
      self.resource_points = 100.
      kwargs = {**kwargs, **{"industries": self.required_entities}}
      super().__init__(**kwargs)
      self.backup = {"contribution_chg_rate": contribution_chg_rate, "contribution": contribution, "buildUpLimit": kwargs['buildUpLimit'], "industry_init_dstr": kwargs["industry_init_dstr"]}
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
      # Upper limit of industry that each agent can build.
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
      # Use result of IRL.
      if self.irl:
          rewards = {}
          for agent in self.world.agents:
              rewards[agent.idx] = 0
              max_val = max(agent.state['inventory'].values())
              tmp = max_val
              enum = 0 # Count iteration
              while(tmp > 1):
                  idx = self.state_point_to_index(np.array(list(agent.state['inventory'].values())) > tmp)
                  rewards[agent.idx] += self.rewards[agent.state['name']][idx, enum]
                  enum += 1
                  if (enum == self.rewards[agent.state['name']].shape[1]):
                      break
                  tmp = tmp // 2
          rewards[self.world.planner.idx] = self.total_GDP - self.total_CO2
          return rewards
      # Use self-defined reward function.
      else:
          rewards = {}
          for agent in self.world.agents:
              #rewards[agent.idx] = sum(agent.industry_weights.values()) * agent.state['endogenous']['GDP'] - agent.state['endogenous']['CO2']
              rewards[agent.idx] = agent.state['endogenous']['GDP'] - agent.state['endogenous']['CO2']
          rewards[self.world.planner.idx] = self.total_GDP - self.total_CO2
          #print("In layout, rewards:", rewards)
          return rewards

  def generate_observations(self):
      #Include ALL agents and object in this world. Refer to ai_ChinaEcon\foundation\base\base_env.py:648
      obs = {}
      #Observe agents
      for agent in self.world.agents:
          if agent.resource_points < 0:
              print("Agent", agent.idx, "resource_points < 0")
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
      for agent in self.world.agents:
          self.world.agents[agent.idx].buildUpLimit = copy.copy(self.backup["buildUpLimit"])
          for k in self.world.agents[agent.idx].endogenous.keys():
              self.world.agents[agent.idx].endogenous[k] = 0
          for k in self.world.agents[agent.idx].escrow.keys():
              self.world.agents[agent.idx].escrow[k] = 0
          for k in self.world.agents[agent.idx].industry_weights.keys():
              self.world.agents[agent.idx].industry_weights[k] = 1
          for k in self.world.agents[agent.idx].inventory.keys():
              self.world.agents[agent.idx].inventory[k] = self.backup["industry_init_dstr"][agent.state["name"]][k]
          for k in self.world.agents[agent.idx].state.keys():
              if k == 'escrow':
                  for ky in self.world.agents[agent.idx].state[k].keys():
                      self.world.agents[agent.idx].state[k][ky] = 0
              if k == 'endogenous':
                  for ky in self.world.agents[agent.idx].state[k].keys():
                      self.world.agents[agent.idx].state[k][ky] = 0
              if k == 'buildUpLimit':
                  for ky in self.world.agents[agent.idx].state[k].keys():
                      self.world.agents[agent.idx].state[k][ky] = copy.copy(self.backup["buildUpLimit"][ky])
              if k == 'resource_points':
                  self.world.agents[agent.idx].state[k] = 100
          self.world.agents[agent.idx].state["inventory"] = copy.copy(self.backup["industry_init_dstr"][agent.state["name"]])
      self.agric = 100.
      self.energy = 100.

  def reset_starting_layout(self):
      self.agric = 100.
      self.energy = 100.
      self.total_CO2 = 0.
      self.total_GDP = 0.
      self.contribution_chg_rate = copy.copy(self.backup["contribution_chg_rate"])
      self.contribution = copy.copy(self.backup["contribution"])
      self.buildUpLimit = copy.copy(self.backup["buildUpLimit"])

  def linear_exp(self, x):
      if x > 0:
          return x
      else:
          return np.exp(x)

  def scenario_step(self):
      for agent in self.world.agents:
          agent_name = agent.state["name"]
          idx = agent.idx
          # Agriculture and energy industry produce resource points to build other industries
          cntrb = copy.copy(self.contribution["resource_points"][agent_name])
          cntrb.pop("bias")
          self.world.agents[idx].resource_points += 10 * self.linear_exp(np.dot(np.array(list(agent.state['inventory'].values())), np.array(list(cntrb.values()))))
          """
          if np.dot(np.array(list(agent.state['inventory'].values())), np.array(list(cntrb.values()))) < 601:
              self.world.agents[idx].resource_points += self.linear_exp(np.dot(np.array(list(agent.state['inventory'].values())), np.array(list(cntrb.values()))))
          else:
              self.world.agents[idx].resource_points += np.exp(np.dot(np.array(list(agent.state['inventory'].values())), np.array(list(cntrb.values()))))
          """

          #np.dot(np.array(agent.state['inventory'].values()), np.array(self.contribution["resource_points"][agent_name].values()))
          # Calculate cumulative CO2, GDP produced by each industry in each agent.
          for k, v in agent.action.items():
              this_CO2 = 0
              this_GDP = 0
              if v > 0:
                  industry = k.split("_")[-1]
                  try:
                      this_CO2 += self.contribution["CO2"][agent_name][industry] * agent.state['inventory'][industry]
                  except KeyError:
                      pass
                  try:
                      this_GDP += self.contribution["GDP"][agent_name][industry] * agent.state['inventory'][industry]
                  except KeyError:
                      pass
          self.world.agents[idx].state['endogenous']['CO2'] += self.linear_exp(this_CO2 + self.contribution["CO2"][agent_name]["bias"])
          self.world.agents[idx].state['endogenous']['GDP'] += self.linear_exp(this_GDP + self.contribution["GDP"][agent_name]["bias"])
          """
          if (this_CO2 + self.contribution["CO2"][agent_name]["bias"]) > 601:
              self.world.agents[idx].state['endogenous']['CO2'] += self.linear_exp(this_CO2 + self.contribution["CO2"][agent_name]["bias"])
          else:
              self.world.agents[idx].state['endogenous']['CO2'] += np.exp(this_CO2 + self.contribution["CO2"][agent_name]["bias"])
          if (this_GDP + self.contribution["GDP"][agent_name]["bias"]) > 601:
              self.world.agents[idx].state['endogenous']['GDP'] += self.linear_exp(this_GDP + self.contribution["GDP"][agent_name]["bias"])
          else:
              self.world.agents[idx].state['endogenous']['GDP'] += np.exp(this_GDP + self.contribution["GDP"][agent_name]["bias"])
          """

          # Industry depreciate over time.
          for industry in agent.state['inventory'].keys():
              if agent.state["inventory"][industry] - self.industry_depreciation[agent.state["name"]][industry] > 0:
                  self.world.agents[idx].state['inventory'][industry] -= self.industry_depreciation[agent.state["name"]][industry]
              else:
                  self.world.agents[idx].state['inventory'][industry] = 0

      self.total_CO2 += sum([agent.state['endogenous']['CO2'] for agent in self.world.agents])
      self.total_GDP += sum([agent.state['endogenous']['GDP'] for agent in self.world.agents])
      self.agric += 1.
      self.energy += 1.

      # Update contribution of industries to CO2, GDP, resource points over time.
      for metrics, provinces in self.contribution.items():
          for province, attrs in provinces.items():
              for k, v in attrs.items():
                  growth = self.contribution_chg_rate[metrics][province][k][0]
                  sd = self.contribution_chg_rate[metrics][province][k][1]
                  self.contribution[metrics][province][k] *= np.random.normal(growth, sd)


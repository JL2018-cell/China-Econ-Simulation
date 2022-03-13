from copy import deepcopy

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
  
  agent_subclasses = ["LocalGov", "CentralGov"] #地方政府、中央政府。
  required_industries = ["Agriculture", "Minerals"] #产业
  
  def __init__(self, **kwargs): #放进不同产业的生产能力、碳排放、GDP 贡献。
    self.energy_cost = 100
    self.expn_per_day = 100
    self.pt_per_day = 100
    self.toCarbonEffcy = 0.5
    self.toGDPEffcy = 0.5
    
    self.agric = 100.
    self.energy = 100.
    assert self.starting_agent_coin >= 0.0
    super().__init__(**kwargs)


  def compute_reward(self):
      return {0:0}
  def generate_observations(self):
      return {0:0}

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

  #参考 foundation\scenarios\simple_wood_and_stone\layout_from_file.py
  

from copy import deepcopy

import numpy as np
from scipy import signal

from foundation.base.base_env import BaseEnvironment, scenario_registry
from foundation.scenarios.utils import rewards, social_metrics
from foundation.base.base_env import BaseEnvironment, scenario_registry

class agriculture_industry(BaseEnvironment):
  def __init__(self):

class energy_industry(BaseEnvironment):
  def __init__(self):
    
@scenario_registry.add
class LayoutFromFile(BaseEnvironment):
  """
  Map of China.
  """
  name = "layout_from_file/MacroEcon"
  
  """
  Industries
  """
  
  agent_subclasses = ["LocalGov", "CentralGov"] #地方政府、中央政府。
  required_industries = ["Agriculture", "Minerals"] #产业
  
  def __init__(self): #放进不同产业的生产能力、碳排放、GDP 贡献。
    self.energy_cost = 100
    self.expn_per_day = 100
    self.pt_per_day = 100
    self.toCarbonEffcy = 0.5
    self.toGDPEffcy = 0.5
  
  #参考 foundation\scenarios\simple_wood_and_stone\layout_from_file.py
  

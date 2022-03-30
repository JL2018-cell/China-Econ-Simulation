# China-Econ-Simulation

<p> Files editing: </p>
<li> China-Econ-Simulation/econSimul.py </li>
<p> Main program </p>
<li> China-Econ-Simulation/foundation/scenarios/MacroEcon/Layout.py </li>
<p> layout of China Economy </p>
<li> China-Econ-Simulation/foundation/agents/planners.py </li>
<p> Agent: Central government </p>
<li> China-Econ-Simulation/foundation/agents/mobilers.py </li>
<p> Agent: Local governments. </p>
<li> China-Econ-Simulation/foundation/entities/resources.py </li>
<p> industries </p>

<h1> Technical Details </h1>
<li> agents should not be 'BasicMobileAgent' but 'localGov' ai_ChinaEcon\foundation\agents\mobiles.py </li>
```
See agent assignment at ai_ChinaEcon_v2\foundation\base\base_env.py:348, 532
        mobile_class = agent_registry.get("localGov")
        planner_class = agent_registry.get("centralGov")
```
<li> should not include agent 'map' in generate_observations. Should include agents and planner only. </li>
```«ai_ChinaEcon\foundation\components\Construct.py» 134 lines, 4526 characters```
<li> Register endogenous variables and other industries </li>
```
{'resources': ['Coin'], 'landmarks': [], 'endogenous': ['Labor']} «ai_ChinaEcon\foundation\base\base_env.py» [Lf] line 369 of 1151. 

Solution: Register in ai_ChinaEcon\foundation\entities\endogenous.py, ai_ChinaEcon\foundation\entities\landmarks.py, ai_ChinaEcon\foundation\entities\resources.py

Update corresponding endogeneous variables in affected codes as well e.g. ContinuousDoubleAuction
> ai_ChinaEcon\foundation\base\base_env.py(313)__init__()
-> self._register_entities(component_cls.required_entities)
(Pdb) p component_cls
<class 'foundation.components.continuous_double_auction.ContinuousDoubleAuction'>
(Pdb) c
Traceback (most recent call last):
  File "C:\python\lib\pdb.py", line 1704, in main
    pdb._runscript(mainpyfile)
  File "C:\python\lib\pdb.py", line 1573, in _runscript
    self.run(statement)
  File "C:\python\lib\bdb.py", line 580, in run
    exec(cmd, globals, locals)
  File "<string>", line 1, in <module>
  File "ai_ChinaEcon\tt.py", line 104, in <module>
    env = foundation.make_env_instance(**env_config)
  File "ai_ChinaEcon\foundation\__init__.py", line 18, in make_env_instance
    return scenario_class(**kwargs)
  File "ai_ChinaEcon\foundation\scenarios\MacroEcon\layout.py", line 48, in __init__
    super().__init__(**kwargs)
  File "ai_ChinaEcon\foundation\base\base_env.py", line 313, in __init__
    self._register_entities(component_cls.required_entities)
  File "ai_ChinaEcon\foundation\base\base_env.py", line 380, in _register_entities
    raise KeyError("Unknown entity: {}".format(entity))
KeyError: 'Unknown entity: Coin'
Uncaught exception. Entering post mortem debugging
Running 'cont' or 'step' will restart the program
> ai_ChinaEcon\foundation\base\base_env.py(380)_register_entities()
-> raise KeyError("Unknown entity: {}".format(entity))
```
<li> Include industry names into states of agents ai_ChinaEcon\foundation\entities\resources.py.  To record how agent develop the industry. </li>
<li> Change location of localGov in world map. Different provinces have different locations. Their distances between one another affects transport cost. </li>
```
ai_ChinaEcon\foundation\components\Construct.py
(Pdb) p self.world.agents[2].state
{'loc': [0, 0], 'inventory': {'Agriculture': 0, 'Coin': 0, 'Energy': 0, 'Minerals': 0}, 'escrow': {'Agriculture': 0, 'Coin': 0, 'Energy': 0, 'Minerals': 0}, 'endogenous': {'CO2': 0, 'GDP': 0, 'Labor': 0}}
May come from «ai_ChinaEcon\foundation\base\base_agent.py» [Lf] line 85 of 491
> ai_ChinaEcon\foundation\base\base_env.py(313)__init__()
-> self._register_entities(component_cls.required_entities)
(Pdb) p component_cls
<class 'foundation.components.continuous_double_auction.ContinuousDoubleAuction'>
(Pdb) c
Traceback (most recent call last):
  File "C:\python\lib\pdb.py", line 1704, in main
    pdb._runscript(mainpyfile)
  File "C:\python\lib\pdb.py", line 1573, in _runscript
    self.run(statement)
  File "C:\python\lib\bdb.py", line 580, in run
    exec(cmd, globals, locals)
  File "<string>", line 1, in <module>
  File "ai_ChinaEcon\tt.py", line 104, in <module>
    env = foundation.make_env_instance(**env_config)
  File "ai_ChinaEcon\foundation\__init__.py", line 18, in make_env_instance
    return scenario_class(**kwargs)
  File "ai_ChinaEcon\foundation\scenarios\MacroEcon\layout.py", line 48, in __init__
    super().__init__(**kwargs)
  File "ai_ChinaEcon\foundation\base\base_env.py", line 313, in __init__
    self._register_entities(component_cls.required_entities)
  File "ai_ChinaEcon\foundation\base\base_env.py", line 380, in _register_entities
    raise KeyError("Unknown entity: {}".format(entity))
KeyError: 'Unknown entity: Coin'
Uncaught exception. Entering post mortem debugging
Running 'cont' or 'step' will restart the program
> ai_ChinaEcon\foundation\base\base_env.py(380)_register_entities()
-> raise KeyError("Unknown entity: {}".format(entity))
```
<li> What actions can be taken by localGov, centralGov? Where to define actions? </li>
```
Answer: Action 1a: increase industry 1 by 1 point. Action 1b: decrease industry 1 by 1 point. Action 1c: Transport dustry point out by 1 point. Action 1d: Transport industry point in by 1 point, etc.
Refer to  line 16, «ai_ChinaEcon\foundation\base\base_agent.py» [Lf] line 116 of 490
```

<h2> Useful references </h2>
<li> attributes of agent </li>
```
(Pdb) p BaseAgent
<class 'foundation.base.base_agent.BaseAgent'>
(Pdb) p dir(BaseAgent)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_incorporate_component', 'action_spaces', 'endogenous', 'escrow', 'escrow_to_inventory', 'flatten_masks', 'get_component_action', 'get_random_action', 'has_component', 'idx', 'inventory', 'inventory_to_escrow', 'loc', 'name', 'parse_actions', 'populate_random_actions', 'register_components', 'register_endogenous', 'register_inventory', 'reset_actions', 'set_component_action', 'total_endowment']
(Pdb) p agent
<foundation.agents.mobiles.localGov object at 0x000001D7C56EB6D0>
(Pdb) agent.state
{'loc': [0, 0], 'inventory': {'Agriculture': 0, 'Coin': 0, 'Minerals': 0}, 'escrow': {'Agriculture': 0, 'Coin': 0, 'Minerals': 0}, 'endogenous': {'Labor': 0}}
```

<li> attributes of planner </li>
```
(Pdb) p dir(self.world.planner)
['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_action_names', '_idx', '_incorporate_component', '_multi_action_dict', '_noop_action_dict', '_one_component_single_action', '_passive_multi_action_agent', '_premask', '_registered_components', '_registered_endogenous', '_registered_inventory', '_total_actions', '_unique_actions', 'action', 'action_dim', 'action_spaces', 'endogenous', 'escrow', 'escrow_to_inventory', 'flatten_masks', 'get_component_action', 'get_random_action', 'has_component', 'idx', 'inventory', 'inventory_to_escrow', 'loc', 'multi_action_mode', 'name', 'parse_actions', 'populate_random_actions', 'register_components', 'register_endogenous', 'register_inventory', 'reset_actions', 'set_component_action', 'single_action_map', 'state', 'total_endowment']
```

<li> Attributes of Auction </li>
```
(Pdb) p self.world.agents[0].action_dim
{'Build': 1, 'ContinuousDoubleAuction.Buy_Agriculture': 11, 'ContinuousDoubleAuction.Sell_Agriculture': 11, 'ContinuousDoubleAuction.Buy_Minerals': 11, 'ContinuousDoubleAuction.Sell_Minerals': 11, 'ContinuousDoubleAuction.Buy_Stone': 11, 'ContinuousDoubleAuction.Sell_Stone': 11, 'ContinuousDoubleAuction.Buy_Wood': 11, 'ContinuousDoubleAuction.Sell_Wood': 11}

(Pdb) p self.world.agents[0].action
{'Build': 0, 'ContinuousDoubleAuction.Buy_Agriculture': 0, 'ContinuousDoubleAuction.Sell_Agriculture': 0, 'ContinuousDoubleAuction.Buy_Minerals': 0, 'ContinuousDoubleAuction.Sell_Minerals': 0, 'ContinuousDoubleAuction.Buy_Stone': 0, 'ContinuousDoubleAuction.Sell_Stone': 0, 'ContinuousDoubleAuction.Buy_Wood': 0, 'ContinuousDoubleAuction.Sell_Wood': 0}
```



# Current task

<li> How to insert final data and intermediate data into model? </li>

<li> How to declare localGov and centralGov? </li>
<p> Used in returning observation (China-Econ-Simulation/foundation/scenarios/MacroEcon/Layout.py)
Solution: Defined in 'n_agents': 10 in dictionary env_config at the main program. i.e. econSimul.py
-> 10 local gov & 1 central gov. </p>

<li> How to return observations of agents? </li>
<p> In China-Econ-Simulation/foundation/scenarios/MacroEcon/Layout.py
self.world.agents - properties of agents
self.world.maps - resources available e.g. agriculture, minerals. 

(Pdb) p world_obs.keys()
dict_keys(['p', 'map']) #Include too littel objects in generate_observation in layout.py
(Pdb) p agent_wise_planner_obs
{'p0': {}, 'p1': {}, 'p2': {}} #3 local governments </p>

<li> How to represent Map? </li>
<p> Square grids. Some grids are empty. Some are non-empty (represent geographical centre of a province).
Distance between non-empty grids = distance between centre of provinces. </p>

<li> How to Modify actions of agents? </li>
<p> Specified in main program: env_config = {'components': [ ...] ...
  
```
{'n_agents': 4, 'world_size': [15, 15], 'episode_length': 1000, 'multi_action_mode_agents': False, 'multi_action_mode_planner': True, 'flatten_observations': False, 'flatten_masks': True, 'components': [{'Build': {}}, {'ContinuousDoubleAuction': {'max_num_orders': 5}}, {'Gather': {}}]}
```
  
ai_chinaEcon\foundation\scenarios\simple_wood_and_stone\dynamic_layout.py:107
base_env_kwargs

d:\tools\ai_chinaecon_v2\foundation\base\base_env.py(344)__init__()
agent.register_components(self._components)
(Pdb) p self._components
[<foundation.components.build.Build object at 0x000001E2883D4490>, <foundation.components.continuous_double_auction.ContinuousDoubleAuction object at 0x000001E28F1F7D60>, <foundation.components.move.Gather object at 0x000001E28F1F7EB0>]

ai_ChinaEcon\foundation\scenarios\MacroEcon\layout.py: self.world.agents, self.world.planners </p>



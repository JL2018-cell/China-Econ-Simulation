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



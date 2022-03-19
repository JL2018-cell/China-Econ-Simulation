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
```
How to declare localGov and centralGov?
Used in returning observation (China-Econ-Simulation/foundation/scenarios/MacroEcon/Layout.py)
Solution: see how template works.
```

in China-Econ-Simulation/foundation/scenarios/MacroEcon/Layout.py
self.world.agents - properties of agents
self.world.maps - resources available e.g. agriculture, minerals.

Map representation:
Square grids. Some grids are empty. Some are non-empty (represent geographical centre of a province).
Distance between non-empty grids = distance between centre of provinces.

How to insert final data and intermediate data into model?

20220319
Current problem:
(Pdb) p world_obs.keys()
dict_keys(['p', 'map']) #Include too littel objects in generate_observation in layout.py
(Pdb) p agent_wise_planner_obs
{'p0': {}, 'p1': {}, 'p2': {}} #3 local governments



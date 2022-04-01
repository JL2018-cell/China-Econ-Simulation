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
<li> (Done) agents should not be 'BasicMobileAgent' but 'localGov'. </li>
<li> (Done) should not include agent 'map' in generate_observations. Should include agents and planner only. </li>
<li> Register endogenous variables e.g. GDP, CO2 and other industries e.g. Finance, Tourism. </li>
<li> Include industry names into states of agents. To record how agent develop the industry. </li>
<li> Change location of localGov in world map. Different provinces have different locations. Their distances between one another affects transport cost. </li>
<li> What actions can be taken by localGov, centralGov? Where to define actions?  </li>
<p> Answer: Action 1a: increase industry 1 by 1 point. Action 1b: decrease industry 1 by 1 point. Action 1c: Transport dustry point out by 1 point. Action 1d: Transport industry point in by 1 point, etc.
Role of Auction: Province can use agriculture/energy to exchange energy/agriculture points, industry points. </p>
Defined in line 191 of ai_ChinaEcon\foundation\base\base_component.py
<li> How to gernerate action masks? </li>
<p> Refer to line 180 of ai_ChinaEcon\foundation\components\build.py </p>
<li> Consider enabling "multi_action_mode" because government can carry out >1 policy simultaneously. </li>
<li> (Done) Make action mask and action consistent. </li>
<li> Complete: evolvement of agents defined in component_step in ai_ChinaEcon\foundation\components\Construct.py </li>
<p> This relates to actions of agents. </p>

# Done task
<li> How to declare localGov and centralGov? </li>
<p> Used in returning observation (China-Econ-Simulation/foundation/scenarios/MacroEcon/Layout.py)
Solution: Defined in 'n_agents': 10 in dictionary env_config at the main program. i.e. econSimul.py
-> 10 local gov & 1 central gov. </p>

<li> How to return observations of agents? </li>
<p> In China-Econ-Simulation/foundation/scenarios/MacroEcon/Layout.py
self.world.agents - properties of agents
self.world.maps - resources available e.g. agriculture, minerals. 

<li> How to represent Map? </li>
<p> Square grids. Some grids are empty. Some are non-empty (represent geographical centre of a province).
Distance between non-empty grids = distance between centre of provinces. </p>


# Coding setting
<li> Change industries available in macroEcon: </li>
industries in env_config (a dictionary) in tt.py
required_entities in ai_ChinaEcon\foundation\components\continuous_double_auction.py
classes in ai_ChinaEcon\foundation\entities\resources.py
required_entities in ai_ChinaEcon\foundation\components\Construct.py



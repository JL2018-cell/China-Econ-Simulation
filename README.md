# China-Econ-Simulation

<h1> Current Status </h1>
<li> Can run normally without bugs. </li>
<li> Needs more variables to track status of agents. </li>

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
<li> Local Governments acts after there is instructions. Doubtful if this is realistic.</li>
<p> See component_step() in ai_ChinaEcon\foundation\components\Construct.py </p>

# Current task
<li> Let agents (localGov) act intelligently, instead of randomly. </li>
<li> Impose limit of Energy, Agriculture industry growth rate on local Gov. </li>
<p> Impose upper bound of building agriculture, energy industry per time step. Defined in env_config. </p>
<li> How to insert final data and intermediate data into model? </li>
<li> Building industries involves no cost so far. agent.state["performance"]["GDP"], agent.state["endogenous"]["Labor"], agent.state["performance"]["CO2"], effects to map are skipped temporarily. See component_step() in ai_ChinaEcon\foundation\components\Construct.py</li>
<li> Add resource points. Produced from Energy & agriculture industry,used to build other industries. </li>
<p> self.resource_points = 0 </p>
<li> self.resourcePt_contrib = {"Agriculture": 10, "Energy": 10} can read from env_config </li>
<li> Where is "Labor" in endogenous variable be computed? </li>
<p> foundation\components\continuous_double_auction.py:196:        agent.state[endogenous][Labor] += self.order_labor
foundation\components\continuous_double_auction.py:227:        agent.state[endogenous][Labor] += self.order_labor
The computation not agree with MacroEcon setting. Should change it. </p>
<li> Remove excessive items in obs of "obs, rew, done, info = env.step(actions)". </li>
<li> Planner: Observe planner (generated every time step) actions taken, industry points stored. </li>
<li> CO2 should be aggregated and observed by all agents. </li>
<p> ai_ChinaEcon\foundation\scenarios\MacroEcon\layout.py </p>
<li> Update states of agents </li>
<p> for component in self._components: in ai_ChinaEcon_v3\foundation\base\base_env.py </p>
<li> replace def sample_random_actions(env, obs): in main program with intelligent actions. </li>
<li> Pay attention to timescale of model </li>
<p> ai_ChinaEcon\foundation\base\base_env.py: 653 </p>
<li> Setting of payment: choose resource (industry) to pay randomly. This is not good. </li>
<p> ai_ChinaEcon\foundation\components\continuous_double_auction.py: 193 </>
<li> Generate action mask for building industries over resource points. </li>
<li> RL: Given state -> nextstate, compute action required using e.g. Q-learning. </li>
<li> Curriculum learning </li>
<p> BuildUpLimit ~ reward. > BuildUpLimit -> punishment. Otherwise, reward. This has small magnitude at beginning but larger later. </p>
<li> Know action dimension & action magnitude in def generate_n_actions() of each component. </li>
<p> <ul>
       <li> agent.reset_actions() in ai_ChinaEcon\foundation\base\base_env.py line 1034 of 1166 </li>
       <li> agent.parse_actions() in mobiles.py </li>
    </ul>
</p>



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

<li> agents should not be 'BasicMobileAgent' but 'localGov'. </li>
<li> should not include agent 'map' in generate_observations. Should include agents and planner only. </li>

<li> Register endogenous variables e.g. GDP, CO2 and other industries e.g. Finance, Tourism. </li>
<p> See ai_ChinaEcon\foundation\base\base_env.py:294 </p>

<li> How to gernerate action masks? </li>
<p> Refer to line 180 of ai_ChinaEcon\foundation\components\build.py </p>

<li> What actions can be taken by localGov, centralGov? Where to define actions?  </li>
<p> Answer: Action 1a: increase industry 1 by 1 point. Action 1b: decrease industry 1 by 1 point. Action 1c: Transport dustry point out by 1 point. Action 1d: Transport industry point in by 1 point, etc.
Role of Auction: Province can use agriculture/energy to exchange energy/agriculture points, industry points. </p>
Defined in line 191 of ai_ChinaEcon\foundation\base\base_component.py

<li> Complete: evolvement of agents defined in component_step in ai_ChinaEcon\foundation\components\Construct.py </li>
<p> This relates to actions of agents. </p>

<li> agent.action_spaces and obs['action_mask'] does not match. </li>
<p> problem: ai_ChinaEcon\foundation\base\base_env.py: _generate_observations(self, flatten_observations=False, flatten_masks=False), _generate_masks(self, flatten_masks=True)
ai_ChinaEcon\foundation\scenarios\MacroEcon\layout.py: generate_observations(self) </p>

<li> (Done) Make action mask and action consistent. </li>

<li> Compute rewards for each agent. </li>
<p> See ai_ChinaEcon\foundation\scenarios\MacroEcon\layout.py line 74 of 123 </p>
<p> For example, ['Agriculture', 'Energy', 'Finance', 'IT', 'Minerals', 'Tourism']
10 * 'Agriculture'
+ 10 * 'Energy'
+ 1 * 'Finance'
+ 2 * 'IT'
+ 10 * 'Minerals'
+ 5 * 'Tourism' </p>

<li> Define position and name of each agent. </li>
<p> e.g. (0,0), "GuangDong" </p>

<li> Include industry names into states of agents. To record how agent develop the industry. </li>

<li> Consider enabling "multi_action_mode" because government can carry out >1 policy simultaneously. </li>

<li> Change location of localGov in world map. Different provinces have different locations. Their distances between one another affects transport cost. </li>

<li> BuildUpLimit for industries </li>
<p> ai_ChinaEcon\foundation\base\base_env.py, ai_ChinaEcon_v3\foundation\base\world.py </p>

<li> Enable multi_action_mode </li>
<p> ai_ChinaEcon\foundation\base\base_agent.py: 183 </p>

<li> Reward function of agent </li>
<p> ai_ChinaEcon\foundation\scenarios\MacroEcon\layout.py: 73 </p>

<li> location, name of localGov </li>
<p> ai_ChinaEcon\foundation\base\world.py </p>



# Further improvement
<li> Actions of agents are not defined in compact way. </li>
<p> e.g. build_Finance, destroy_Finance -> construct.Finance, action dimension = 2. </p>
<li> Parameters to change </li>
<p> <ul> 
       <li> Maximum, minimum actions agent can take inn ai_ChinaEcon\econSimul.py </li>
       <li> Change action dimension in def fet_n_actions construct.py </li>
     </ul>
 </p>



# Coding setting
<li> Change industries available in macroEcon: </li>
industries in env_config (a dictionary) in tt.py
required_entities in ai_ChinaEcon\foundation\components\continuous_double_auction.py
classes in ai_ChinaEcon\foundation\entities\resources.py
required_entities in ai_ChinaEcon\foundation\components\Construct.py




# China-Econ-Simulation
<ul>
  <li> <h2> Can Run in GPU Farm </h2> </li>
  <li> <h2> Trained in Rllib framework. </h2> </li>
  <li> <h2> Useful reference: Training APIs </h2>
    <p> https://docs.ray.io/en/latest/rllib/rllib-training.html </p>
    <p> https://docs.ray.io/en/latest/rllib/index.html </p> </li>
  <li> <h2> Policies in rllib </h2> </li>
    <p> https://docs.ray.io/en/latest/rllib/rllib-concepts.html </p>
    <p> https://docs.ray.io/en/latest/rllib/package_ref/policy.html </p>
</ul>

<ul> 
  <li> <h2> Functionalities Records </h2> </li>
    <ul>
      <li> ai-economist-master </li>
      <p> Original copy from official Github "AI-economist" </p>
      <li> ai_ChinaEcon_train_v2 </li>
      <p> untidied fit-to-rllib
      Impose no limit to building, breaking industries.
      Impose no limit to DoubleContinuousAuction. </p>
      <li> China-Econ-Simulation-simple </li>
      <p> Same as in Github branch "train".
      Tidied version of "ai_ChinaEcon_train_v2"
      Impose no limit to building, breaking industries.
      Impose no limit to DoubleContinuousAuction. </p>
      <li> China-Econ-Simulation-simple_v7 </li>
      <p> Utilize GPU to accelerate training. </p>
      <li> irl_maxent </li>
      <p> Original copy of reverse RL. </p>
      <li> irl_maxent_v2 </li>
      <p> Modified to fit into ChinaEcon Simulation context. </p>
</ul>
  
<h2> Task </h2>
<ul>
  <li> Understand visualization of training result shown in tensorboard. </li>
     <p> tensorboard --logdir=~/ray_results
     What tensorflow model does rllib uses?
       Refer to https://docs.ray.io/en/latest/rllib/user-guides.html </p>
  <li> How to incorporate data into my model with rllib framework (deep RL)? </li>
    <p> solution: Watch videos
        Ray RLlib: How to Use Deep RL Algorithms to Solve Reinforcement Learning Problems
        https://www.youtube.com/watch?v=HteW2lfwLXM
        Build a trading bot with Deep Reinforcement Learning using Ray and RLlib
        https://www.youtube.com/watch?v=sVqbbl6U8OY
        Ray RLlib: How to Visualize Results Using Tensorboard
        https://www.youtube.com/watch?v=eLY8YAVnx_w </p>
  <li> Accelerate training by GPU </li>
  <li> Generate and Visualize the Environment's Dense Logs. </li>
  <p> Refer to D:\Tools\ai_ChinaEcon_train_v2\multi_agent_training_with_rllib.py </p>
  <li> Find relation between deep learning & RL in this context. </li>
  <p> Collect data from RL environment, then train deep neural network using data from RL.
    Set up deep neural network between action and state. </p>
  <li> How to accelerate deep neural network training & rllib RL training simultaneously? </li>
  <p> RL environment
      RL Algorithm: PPO
      Configuration
      Experiment runner: ray tune
      Average total reward per episode. </p>
  <li> Plotting </li>
      <ul>
        <li> Industry distribution </li>
        <li> BuildUpLimit </li>
        <li> Resource_points </li>
        <li> CO2 </li>
        <li> GDP </li>
        <li> Actions taken </li>
        <p> Construct, ContinuousDoubleAuction: market rate, price history... </p>
      </ul>
      <ul>
        <li> Parameters </li>
    <ul>
      <li> buildUpLimit </li>
      <li> Depreciation of industry </li>
      <li> Fineness of timestep/episode length </li>
      <li> starting_agent_resources </li>
      <li>initial industry distribution </li>
      </ul>
</ul>

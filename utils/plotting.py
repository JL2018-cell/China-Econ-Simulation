

def plotting(dense_logs):

    #Plot local gov states
    n_agents = len(dense_logs[0]['states'][1].keys()) - 1
    n_episodes = len(dense_logs[0]['states'])
    fig, axs = plt.subplots(n_episodes, n_agents)
    for n_ep in range(n_episodes):
        for n_agn in range(n_agents):
            labels = dense_logs[0]['states'][n_ep][str(n_agn)]['inventory'].keys()
            sizes = dense_logs[0]['states'][n_ep][str(n_agn)]['inventory'].values()
            axs[n_ep][n_agn].pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            axs[n_ep][n_agn].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('Industry_distribution.png', dpi = 250)

    #Plot GPD of local gov
    base_year = 2021
    years = len(dense_logs[0]['states'])
    labels = [str(base_year + i) for i in range(years)]
    n_agents = len(dense_logs[0]['states'][1].keys()) - 1
    fig, ax = plt.subplots()
    width = 0.35
    for n_agn in range(n_agents):
        GDPs = [dense_logs[0]['states'][year][str(n_agn)]['endogenous']['GDP'] for year in range(years)]
        ax.bar(labels, GDPs, width, label=str(dense_logs[0]['states'][0][str(n_agn)]['loc']))
    ax.set_ylabel('GDP')
    ax.set_title('GDP by Provinces')
    ax.legend()
    plt.savefig('GDP.png', dpi = 250)

    #Plot CO2 of local gov
    base_year = 2021
    years = len(dense_logs[0]['states'])
    labels = [str(base_year + i) for i in range(years)]
    n_agents = len(dense_logs[0]['states'][1].keys()) - 1
    fig, ax = plt.subplots()
    width = 0.35
    for n_agn in range(n_agents):
        CO2s = [dense_logs[0]['states'][year][str(n_agn)]['endogenous']['CO2'] for year in range(years)]
        ax.bar(labels, CO2s, width, label=str(dense_logs[0]['states'][0][str(n_agn)]['loc']))
    ax.set_ylabel('CO2')
    ax.set_title('CO2 by Provinces')
    ax.legend()
    plt.savefig('CO2.png', dpi = 250)
    
    #Plot reward of each agent, include planner.
    n_agents = len(dense_logs[0]['rewards'][0].keys())
    log2[0]['rewards'][0]
    n_episodes = len(dense_logs[0]['rewards'])
    fig, axs = plt.subplots(n_agents, 1)
        index = list(range(n_episodes))
    bar_width = 0.3
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    for n_agn in range(n_agents):
        rewards = [dense_logs[0]['rewards'][n_ep][str(n_agn)] for n_ep in range(n_episodes)]
        axs[n_agn].bar(index, rewards, bar_width, alpha = opacity, error_kw = error_config)
        axs[n_agn].set_ylabel('Rewards')
        axs[n_agn].set_title('Reward by Provinces')
        axs[n_agn].legend()
    plt.savefig('rewards.png', dpi = 250)

    #Plot buildUpLimit, resource points of local gov
    n_agents = len(dense_logs[0]['states'][0]) - 1
    fig, axs = plt.subplots(3, 1)
    base_year = 2021
    years = len(dense_logs[0]['states'])
    labels = [str(base_year + i) for i in range(years)]
    bldUpLmt_Eng = []
    bldUpLmt_Agr = []
    rsc_pts = []
    for n_agn in range(n_agents):
        bldUpLmt_Agr.append([dense_logs[0]['states'][year][str(n_agn)]['buildUpLimit']['Agriculture'] for year in range(years)])
        bldUpLmt_Eng.append([dense_logs[0]['states'][year][str(n_agn)]['buildUpLimit']['Energy'] for year in range(years)])
        rsc_pts.append([dense_logs[0]['states'][year][str(n_agn)]['resource_points'] for year in range(years)])
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    for n_agn in range(n_agents):
        rects1 = axs[n_agn].bar(x - width/2, bldUpLmt_Agr[n_agn], width, label='buildUpLimit - Agriculture')
        rects2 = axs[n_agn].bar(x, bldUpLmt_Eng[n_agn], width, label='buildUpLimit - Energy')
        rects3 = axs[n_agn].bar(x + width/2, rsc_pts[n_agn], width, label='Resource points')
        axs[n_agn].set_ylabel('Points')
        axs[n_agn].set_title('Building resourced by Provinces')
        axs[n_agn].set_xticks(x, labels)
        axs[n_agn].legend()
        #axs[n_agn].bar_label(rects1, padding=3)
        #axs[n_agn].bar_label(rects2, padding=3)
        #axs[n_agn].bar_label(rects3, padding=3)
    plt.savefig('resources.png', dpi = 250)


    #Plot actions of local gov
    n_agents = len(dense_logs[0]['actions'][0].keys()) - 1
    years = len(dense_logs[0]['actions'])
    for n_agn in range(n_agents):
        if n_agn == 0:
            actions = set(dense_logs[0]['actions'][n_agn].keys())
        else:
            actions = actions | set(dense_logs[0]['actions'][n_agn].keys())
    actions = list(actions)
    
    theta = np.linspace(0.0, 2 * np.pi, len(actions), endpoint=False)
    radii = np.zeros(n_agents, len(actions))
    
    for n_agn in range(n_agents):
        for year in range(years):
            for i, action in enumerate(actions):
                try:
                    radii[n_agn, i] += dense_logs[0]['actions'][year][str(n_agn)][action]
                except KeyError:
                    pass
                    #radii[n_agn, i] = 0
    
    colors = plt.cm.viridis(radii / 10.)
    
    ax = plt.subplot(projection='polar')
    ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5, tick_label = actions)
    
    plt.savefig("polar_bar.png", dpi = 250)

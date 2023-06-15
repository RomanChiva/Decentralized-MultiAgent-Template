import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
#plt.style.use('dark_background')




class Agent:

    def __init__(self) -> None:
    
        self.state = 0

    def move(self, neighborhood):



        if neighborhood.shape[0] <= 0:
            v = np.random.randint(-2,3,(1,2),dtype=int)
        else:
            v =  np.zeros((1,2), dtype=int)

        return v



class Env:

    def __init__(self, env_size, n_agents, neighborhood_radius) -> None:

        self.env_size = env_size
        self.n_radius = neighborhood_radius

        self.agents = [{'agent':Agent(), 
                        'pos':np.random.randint(-env_size,env_size,(1,2), dtype=int)} 
                        for agent in range(n_agents)]


    def timestep(self):

        # Perform all actions simultaneously (although they are computed separately)
        # ASSUMPTION: Agents operate synchronously
        

        # Gather Actions

        tstep_actions = []

        for agent in self.agents:

            n = self.get_neighborhood(agent['pos'])
            v = agent['agent'].move(n)
            tstep_actions.append(v)
        

        # Perform actions
        for i, action in enumerate(tstep_actions):
            self.agents[i]['pos'] += action

        
            


    def get_neighborhood(self, pos):

        # Remove self from list
        positions = self.make_positions_list()
        positions = positions[np.all(positions!=pos, axis=1)]
        
        # Compute norm on relative position vectors
        relative_positions = positions - pos
        norm = np.linalg.norm(relative_positions, axis=1)

        # Neighborhood
        neighborhood = relative_positions[norm <= self.n_radius]
        return neighborhood



    def make_positions_list(self):

        # Make a list containing the positions of all the agents

        positions = [agent['pos'] for agent in self.agents]
        
        return np.squeeze(np.array(positions))
    
    



            
def run_sim(env):

    # Run the simulation and show an animation

    # Create figure
    fig = plt.figure()
    fig.suptitle('Decentralized Multi-Agent Template')
    # Create plot inside figure
    ax1 = fig.add_subplot(1,1,1)
    

    def update(i):

        env.timestep()

        positions = env.make_positions_list()

        ax1.clear()

        # Set Axis Limits
        ax1.set_xlim(-env.env_size, env.env_size)
        ax1.set_ylim(-env.env_size, env.env_size)

        # Set Ticks and Make Grid

        # Major ticks every 10, minor ticks every 5
        major_ticks = np.arange(-env.env_size, env.env_size, 20)
        minor_ticks = np.arange(-env.env_size, env.env_size, 10)

        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.set_yticks(major_ticks)
        ax1.set_yticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax1.grid(which='both')

        # Plot Scatter
        ax1.scatter(positions.T[0], positions.T[1])

    ani = animation.FuncAnimation(fig, update, interval = 100)
    plt.show()



if __name__ == '__main__':

    env = Env(100,10,15)
    run_sim(env)

    

    
    






        
    


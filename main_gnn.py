# Deep Learning Pong Project
import DL_gnn as DL
import numpy as np
from PIL import Image, ImageOps
import time
import pickle
from os.path import isfile
import matplotlib.pyplot as plt
import gym
import random

'''
# Some notes about the general process
# Observation refers to the 210x160x3 array for the image
# Reward refers to whether the AI has gained or loss a point per frame
# The game is first to 21.
# Convert observation to grayscale
# May need Velocity information which would require a subtraction from the previous frame
# Can use multiple environments to speed up run time.

Thoughts
-Make a stopping condition (stop training once the enemy can't score against AI)
-Implement pickle to save models
-Try out different mutation probabilities and intensities
-Better to have a large pop. size with small game steps per network
or small pop. size with large game steps?
-Pop. size needs to be even number or else an error happens
newPongAIList.append( pongNetworkList[i].crossover( pongNetworkList[i+1] ) )
IndexError: list index out of range
'''

def runPong():
    # pip install gym[atari]
    # pip install gym[accept-rom-license]

    env = gym.make('ALE/Pong-v5', render_mode='human')
    observation, info = env.reset(seed=0, return_info=True)

    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            observation, info = env.reset(return_info=True)
        
    env.close()

def trainGNN(populationSize, generations, game_steps, should_load=True):
    print("%d population, %d generations, %d game steps" % (populationSize, generations, game_steps))
    
    time_start = time.time()
    pongNetworkList = []
    pongEnv = []
    envDone = []
    networkAction = []
    timeTillScore = []
    previousFrame = []
    
    generationGraphList = []
    maxFitnessGraphList = []

    totalRunning = populationSize
    img_w = 60
    img_h = 44
    img_size = img_w * img_h

    load_model = isfile('model.pickle') and should_load
    if load_model:
        pongNetworkList = pickle.load(open('model.pickle', "rb"))['networks']

    for i in range(populationSize):
        if not load_model:
            pongAI = DL.GNN()
            pongAI.addLayer( DL.FullyConnectedLayer(img_size, 3) )
            pongAI.addLayer( DL.SoftmaxLayer() )

            #pongAI.setMutationIntensity(0.2)
            #pongAI.setMutationProbability(0.3)
        
            pongNetworkList.append(pongAI)

        env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
        env.reset(seed = random.randint(0, 1000))

        pongEnv.append( env )
        envDone.append( False )
        networkAction.append( 0 )
        timeTillScore.append( 0 )
        previousFrame.append( np.zeros( (img_h, img_w)) )

    for gen in range(generations):
        gen_start = time.time()

        pickle.dump({'networks': pongNetworkList}, open('model.pickle', "wb"))
        
        for i in range(populationSize):
            fitness_adj = 0
            
            for step in range(game_steps):
                if not envDone[i]:
                    
                    action = networkAction[i]
                    observation, reward, done, info = pongEnv[i].step(action)
                    if action != 0:
                        fitness_adj += 0.01
                    
                    timeTillScore[i] += 1
                    if reward != 0:
                        finalFitness = pongNetworkList[i].getFitnessValue() + reward 
                        pongNetworkList[i].setFitnessValue( finalFitness )
                        timeTillScore[i] = 0
                        fitness_adj = 0
                    
                    # Prepare the image
                    img = Image.fromarray(observation)
                    img = ImageOps.grayscale(img)
                    img = img.resize( size=(84, 64) )
                    
                    # Get only the ball (crop out scoreboard and paddles)
                    area = (12,12,84-12,64-8) # (top, left, bottom, right)
                    crop_img = img.crop(area)
                    
                    imgArray = np.asarray(crop_img) / 255.0 #Convert to [0 - 1]
                    buffer = imgArray - previousFrame[i] #Get difference between last frame
                    previousFrame[i] = imgArray #Set previous frame as current frame for next time
                    
                    finalImgMatrix = buffer.flatten()

                    # run network on observation to generate next action
                    output = pongNetworkList[i].forward(finalImgMatrix)
                    
                    indexOfMax = np.argmax(output)

                    if(indexOfMax == 0):
                        networkAction[i] = 0 #No action
                    if(indexOfMax == 1):
                        networkAction[i] = 2 #Move up
                    if(indexOfMax == 2):
                        networkAction[i] = 3 #Move down
                    
                    if done:
                        pongEnv[i].reset(seed = 0)
                        envDone[i] = True
                        totalRunning -= 1
                        break

            if(totalRunning <= 0):
                break
        
        #sort by fitness values
        pongNetworkList.sort(key=lambda x: x.getFitnessValue(), reverse=True)

        maxFitnessValue = pongNetworkList[0].getFitnessValue()
        minFitnessValue = pongNetworkList[-1].getFitnessValue()
        elapsed = time.time() - gen_start
        print("Generation %d\tMax Fitness: %.2f\tMin Fitness: %.2f\tElapsed: %.2f s" % (gen, maxFitnessValue, minFitnessValue, elapsed))

        #Stuff for the graph of fitness versus generation
        generationGraphList.append(gen)
        maxFitnessGraphList.append(maxFitnessValue)

        if(gen != generations-1):
            #Crossover neural networks. The mutation happens in the crossover.
            #Get rid of the bottom half
            halfListSize = int(len(pongNetworkList)/2)
            newPongAIList = pongNetworkList[:halfListSize]

            for i in range(0, populationSize, 2):
                newPongAIList.append( pongNetworkList[i].crossover( pongNetworkList[i+1] ) )
            
            pongNetworkList = newPongAIList
            
            #repeat everything above until some sort of convergence.
            totalRunning = populationSize
            for i in range(populationSize):
                pongNetworkList[i].setFitnessValue(0)
                timeTillScore[i] = 0
                previousFrame[i] = np.zeros( (img_h, img_w))
                envDone[i] = False
                networkAction[i] = 0
                pongEnv[i].reset(seed = random.randint(0, 1000))

    for i in range(0, populationSize):
        pongEnv[i].close()
    
    #Display graph
    plt.figure()
    # plot Log Loss line
    plt.plot(generationGraphList, maxFitnessGraphList)
    plt.title("Genetic Algorithm Pong")
    plt.xlabel("Generation")
    plt.ylabel("MaxFitness")

    plt.show()

    # Test best performing one
    env = gym.make('ALE/Pong-v5', render_mode='human')
    env.reset(seed = random.randint(0, 1000))

    for _ in range(1000):
        action = networkAction[0]
        observation, reward, done, info = env.step(action)

        #run network on observation to generate next action
        img = Image.fromarray(observation)
        img = ImageOps.grayscale(img)
        img = img.resize( size=(84, 64) )

        area = (12,12,84-12,64-8)
        img = img.crop(area)
        
        imgArray = np.asarray(img) / 255.0
        buffer = imgArray - previousFrame[0]
        previousFrame[0] = imgArray
        
        finalImgMatrix = buffer.flatten()
        
        output = pongNetworkList[0].forward(finalImgMatrix)

        indexOfMax = np.argmax(output)

        if(indexOfMax == 0):
            networkAction[0] = 0 #No action
        if(indexOfMax == 1):
            networkAction[0] = 2 # Up
        if(indexOfMax == 2):
            networkAction[0] = 3 # Down
        
        if done:
            env.reset()
            break
        
    env.close()
    
def main():
    #runPong()

    trainGNN(20, 40, 1000, should_load=False)

if __name__ == "__main__":
    main()

# Deep Learning Pong Project
from math import gamma
import DL
import numpy as np
from PIL import Image, ImageOps
import time
import pickle
import Pong
import random

import gym

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
        print(i)
        time.sleep(0.3)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            observation, info = env.reset(return_info=True)
        
    env.close()

def trainGNN(populationSize, generations, game_steps):
    print("%d population, %d generations, %d game steps" % (populationSize, generations, game_steps))
    
    time_start = time.time()
    pongNetworkList = []
    pongEnv = []
    envDone = []
    networkAction = []
    timeTillScore = []
    previousFrame = []
    
    totalRunning = populationSize

    for i in range(populationSize):
        imageSize = 60*44 #Cropped image when scaled down will be 84x44
        pongAI = DL.GNN()
        pongAI.addLayer( DL.FullyConnectedLayer(imageSize, 3) )
        pongAI.addLayer( DL.SigmoidLayer() )
        # pongAI.addLayer( DL.FullyConnectedLayer(8, 3) )
        # pongAI.addLayer( DL.SoftmaxLayer() )

        #pongAI.setMutationIntensity(0.2)
        #pongAI.setMutationProbability(0.3)
        
        pongNetworkList.append(pongAI)

        env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
        env.reset(seed = 0)

        pongEnv.append( env )
        envDone.append( False )
        networkAction.append( 0 )
        timeTillScore.append( 0 )
        previousFrame.append( np.zeros( (44, 60)) )

    for gen in range(generations):
        gen_start = time.time()
        
        for i in range(populationSize):
            for step in range(game_steps):
                if(envDone[i] == False):
                    '''
                    At start, Pong takes a few frames to load the enemy and the ball.
                    These frames don't affect the game, so they can be skipped.
                    '''
                    if step < 15:
                        pongEnv[i].step(0)
                        continue
                    
                    action = networkAction[i]
                    observation, reward, done, info = pongEnv[i].step(action)
                    
                    timeTillScore[i] += 1
                    if reward != 0:
                        finalFitness = pongNetworkList[i].getFitnessValue() + reward
                        pongNetworkList[i].setFitnessValue( finalFitness )
                        timeTillScore[i] = 0
                    
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
                previousFrame[i] = np.zeros( (44, 60))
                envDone[i] = False
                networkAction[i] = 0
                pongEnv[i].reset(seed = 0)

    for i in range(0, populationSize):
        pongEnv[i].close()
    
    # Test best performing one
    env = gym.make('ALE/Pong-v5', render_mode='human')
    env.reset(seed = 0)

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

def trainDRL():
    time_start = time.time()
    batch_size = 32
    learning_rate = 1e-3
    reward_discount = 0.99

    imageSize = 80*80

    # Create the network
    pongAI = DL.DRL()
    #FC 1
    pongAI.addLayer( DL.FullyConnectedLayer(imageSize, 256) )
    #relu
    pongAI.addLayer( DL.ReluLayer() )
    #FC 2
    pongAI.addLayer( DL.FullyConnectedLayer(256, 1) )
    #sigmoid
    pongAI.addLayer( DL.SigmoidLayer() )
    #Log loss
    pongAI.addLayer( DL.LogLoss())


def testSmallPong():
    game = Pong.Game(True)

    while(game.getRunning()):
        game.step(random.choice([0, 1, 2]))
        processImage = game.getImage() #Image is 210 x 160 grayscale numpy array. No text is rendered
        if(game.getFramesPassed() > 1000):
            break
        time.sleep(0.0333)

def main():
    # runPong()

    # trainGNN(20, 20, 1000)
    testSmallPong()

if __name__ == "__main__":
    main()

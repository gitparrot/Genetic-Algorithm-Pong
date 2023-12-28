import math
import pygame
import random

import numpy as np

class Paddle():
    def __init__(self, playerNum, moveSpeed):
        self.__x = 0
        self.__y = 80
        self.__sizeX = 7.5
        self.__sizeY = 20
        self.__moveSpeed = moveSpeed
        
        self.__playerNum = playerNum
        if(self.__playerNum == 0):
            self.__x = 10
        else:
            self.__x = 210 - 10

    def moveUp(self):
        self.__y -= self.__moveSpeed
        if(self.__y < 0):
           self.__y = 0
    
    def moveDown(self):
        self.__y += self.__moveSpeed
        if(self.__y > 160):
            self.__y = 160
        
    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), (self.__x, self.__y, self.__sizeX, self.__sizeY) )
        
    def drawInternal(self, imageArray):
        #manually fill imageArray
        x1 = int(max(0, self.__x))
        y1 = int(max(0, self.__y))
        x2 = int(min(209, self.__x+self.__sizeX))
        y2 = int(min(159, self.__y+self.__sizeY))

        for x in range(x1, x2):
            for y in range(y1, y2):
                imageArray[x, y] = 255   #Red converted to black and white

    def reset(self):
        self.__y = 80
        if(self.__playerNum == 0):
            self.__x = 10
        else:
            self.__x = 210 - 10
    
    def getX(self):
        return self.__x
    
    def getY(self):
        return self.__y

    def getXSize(self):
        return self.__sizeX
    
    def getYSize(self):
        return self.__sizeY
    
class Ball():
    def __init__(self, playerServe):
        self.__x = 105
        self.__y = 80

        self.__sizeX = 7.5
        self.__sizeY = 7.5

        #Potentially add angles 15 30 45 and others
        self.__dir = 0
        
        if(playerServe == 0):
            self.__dir = random.choice( [45, 315] ) #Serve with respect to left player
        else:
            self.__dir = random.choice( [135, 225] ) #Serve with respect to right player

        self.__velocity = 5

    def move(self):
        moveX = self.__velocity * math.cos( math.radians(self.__dir) )
        moveY = self.__velocity * -math.sin( math.radians(self.__dir) )

        self.__x += moveX
        self.__y += moveY

        #Hard code cause simple enough
        if(moveX > 0):
            if(self.__y <= 0):
                self.__dir = 315
            elif(self.__y >= 160):
                self.__dir = 45
        else:
            if(self.__y <= 0):
                self.__dir = 225
            elif(self.__y >= 160):
                self.__dir = 135

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), (self.__x, self.__y, self.__sizeX, self.__sizeY) )
    
    def drawInternal(self, imageArray):
        #manually fill imageArray
        x1 = int(max(0, self.__x))
        y1 = int(max(0, self.__y))
        x2 = int(min(209, self.__x+self.__sizeX))
        y2 = int(min(159, self.__y+self.__sizeY))

        for x in range(x1, x2):
            for y in range(y1, y2):
                imageArray[x, y] = 81   #Red converted to black and white
    
    def getX(self):
        return self.__x
    
    def getY(self):
        return self.__y

    def getXSize(self):
        return self.__sizeX
    
    def getYSize(self):
        return self.__sizeY
    
    def reverseDirection(self):
        self.__dir += 180
        self.move() #Prevents the ball from getting stuck when bouncing near paddle and edge
        
    
class Game():
    def __init__(self, type):
        self.__renderType = type
        self.__P1Score = 0
        self.__P2Score = 0
        self.__Frame = 0
        self.__Player1 = Paddle(0, 5)
        self.__Player2 = Paddle(1, 2.5)
        self.__Ball = Ball(0) #serve side could be random too
        self.__running = True
        self.__imageArray = np.zeros((210, 160))

        self.__screen = None
        self.__myfont = None
        if(type):
            pygame.init()
            self.__screen = pygame.display.set_mode([210, 160])
            self.__myfont = pygame.font.SysFont('Arial', 24)
    
    def step(self, action):
        #action is a number corresponding to input
        #0 - noop
        #1 - move up
        #2 - move down
        if(self.__running == False):
            return -9999
        
        self.__Frame += 1

        self.__Ball.move()

        if(action == 1):
            self.__Player1.moveUp()
        elif(action == 2):
            self.__Player1.moveDown()
        
        #AI controlled player 2
        if(self.__Player2.getY() < self.__Ball.getY()):
            self.__Player2.moveDown()
        else:
            self.__Player2.moveUp()

        self.getCollision()

        if(self.__renderType):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__running = False
            self.draw()
            pygame.display.flip()
        
        self.drawInternal() #draws to an array for processing

        if(self.__Ball.getX() >= 210):
            self.__P1Score += 1
            self.__Ball = Ball(1)
            return 1

        if(self.__Ball.getX() <= 0):
            self.__P2Score += 1
            self.__Ball = Ball(0)
            return -1

        return 0

    def getCollision(self):
        #Axis Aligned Bounding Box collision
        #Modified slightly so that the front part of the paddle is the only part counted
        #check for collision between Ball and Player1
        xCol = False
        yCol = False
        if(self.__Ball.getX() + self.__Ball.getXSize() > self.__Player1.getX()
        and self.__Ball.getX() - self.__Ball.getXSize() < self.__Player1.getX() + self.__Player1.getXSize() ):
            xCol = True
        
        if(self.__Ball.getY() + self.__Ball.getYSize() > self.__Player1.getY() - self.__Player1.getYSize() 
        and self.__Ball.getY() - self.__Ball.getYSize() < self.__Player1.getY() + self.__Player1.getYSize() ):
            yCol = True  
        
        if(xCol and yCol):
            #potentially speed the ball up
            self.__Ball.reverseDirection()
            return

        #check for collision between Ball and Player2
        xCol = False
        yCol = False
        if(self.__Ball.getX() + self.__Ball.getXSize() > self.__Player2.getX() - self.__Player2.getXSize()
        and self.__Ball.getX() - self.__Ball.getXSize() < self.__Player2.getX() ):
            xCol = True
        
        if(self.__Ball.getY() + self.__Ball.getYSize() > self.__Player2.getY() - self.__Player2.getYSize() 
        and self.__Ball.getY() - self.__Ball.getYSize() < self.__Player2.getY() + self.__Player2.getYSize() ):
            yCol = True  
        
        if(xCol and yCol):
            #potentially speed the ball up
            self.__Ball.reverseDirection()
            return
    
    def draw(self):

        self.__screen.fill((0,0,0))

        self.__Player1.draw(self.__screen)
        self.__Player2.draw(self.__screen)
        self.__Ball.draw(self.__screen)

        textSurface = self.__myfont.render(str(self.__P1Score) + " : " + str(self.__P2Score), False, (255, 255, 255))

        self.__screen.blit(textSurface, (80, 0))
    
    def drawInternal(self):
        self.__imageArray = np.zeros( (210, 160) )
        self.__Player1.drawInternal(self.__imageArray)
        self.__Player2.drawInternal(self.__imageArray)
        self.__Ball.drawInternal(self.__imageArray)

    def reset(self):
        self.__P1Score = 0
        self.__P2Score = 0
        self.__Frame = 0
        self.__Player1 = Paddle(0, 5)
        self.__Player2 = Paddle(1, 2.5)
        self.__Ball = Ball(0)
        self.__running = True
    
    def getRunning(self):
        return self.__running

    def getImage(self):
        #Returns current frame
        return self.__imageArray

    def getFramesPassed(self):
        return self.__Frame
    
    def getPlayer1Score(self):
        return self.__P1Score
    
    def getPlayer2Score(self):
        return self.__P2Score
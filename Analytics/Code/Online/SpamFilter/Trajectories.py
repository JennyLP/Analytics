import numpy as np
import pandas as pd
from collections import namedtuple
import pylab
import time
import random
import matplotlib.pyplot as plt


TimePoint = namedtuple('TimePoint', 'x y t')

def Update( myDict ):

    myMin = min( myDict.values() )
    return myMin, [ key for key, val in myDict.iteritems() if myMin == val ]


class Trajectory(object):

    def __init__(self, myFilename):

        myTrajectory = pd.read_csv( myFilename, quotechar="'" )
        pd.to_datetime( myTrajectory['t'] )
        
        myHeaders = myTrajectory.columns.values.tolist()
        myUserID_index = myHeaders.index('userID')

        self.trajectories = {}

        myRows = myTrajectory.values.tolist()

        for row in myRows:
            myUser = row[myUserID_index]
            if myUser not in self.trajectories:
                self.trajectories[ myUser ] = []
            row.pop(myUserID_index)
            self.trajectories[ myUser ].append( TimePoint(*row) )


    def Draw(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = {}

        for userID in self.trajectories.keys():
            
            # sort by timestamp
            self.trajectories[userID] = sorted(self.trajectories[userID], key = lambda elem: elem[2])
            print self.trajectories[userID]
  
            # generate a separate color for each user 
            colors[userID] = (random.random(), random.random(), random.random() )
        
        print colors

        myPointers = { userID : 0 for userID in self.trajectories.keys() }
        currTime = min( [ elem[0][2] for elem in self.trajectories.values() ] )
        lastTime = max( [ elem[0][2] for elem in self.trajectories.values() ] )

        print "Pointers"
        print myPointers
        myTimes = { userID : trajectory[0][2] for userID, trajectory in self.trajectories.iteritems() }
        print currTime

        numSeconds = 0

        while myPointers != []:
            
            currTime, UpdateList = Update( myTimes )
            print currTime, UpdateList

            for userID in UpdateList:

                myPoints = self.trajectories[userID][myPointers[userID]][0:2]
                plt.plot( [myPoints[0]], [myPoints[1]], marker = 'o', color = colors[userID] )
                myPointers[userID] += 1
                if myPointers[userID] == len(self.trajectories[userID]):
                    myPointers.pop(userID)
                plt.ion()
                plt.show()
                plt.pause(1)


myTrajectories = Trajectory( "./Data/Trajectories.csv" )
myTrajectories.Draw()

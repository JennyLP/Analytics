#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, getopt
import copy
import operator
reload(sys)
sys.setdefaultencoding('utf-8')

import random
import itertools
import re, math
import networkx as nx
import numpy as np

from prettytable import PrettyTable
from collections import Counter
from collections import OrderedDict
from collections import namedtuple
from itertools import chain

from Utils import color

from Users import UsersInfo

WORD = re.compile(r'\w+')

from TopicsHierarchy import buildNetworkFromList
from TopicsHierarchy import plotFancyGraph
from TopicsHierarchy import fancyTopicClassification
from TopicsHierarchy import computeFromAdjacency


####### USEFUL FUNCTIONS ########

def FindTheLoner(M, F, indexM, indexF):

            for ind in [0,1]:
                if F[indexF][ind] in M[indexM]:
                    return F[indexF][1-ind]


def SwitchGenders(M, F):

    M, F = F, M


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def affinityScore( userID1, userID2, userDict ):

    list_user1 = userDict[ userID1 ]
    list_user2 = userDict[ userID2 ]

    user1_aff_for_user2 = float( len ( set(list_user1).intersection( list_user2 ) ) ) / len( list_user1 )
    user2_aff_for_user1 = float( len ( set(list_user2).intersection( list_user1 ) ) ) / len( list_user2 )

    return user1_aff_for_user2, user2_aff_for_user1


def cosineAffinityScore( userID1, userID2, userDict ):

    list_user1 = userDict[ userID1 ]
    list_user2 = userDict[ userID2 ]

    intersection = set(list_user1.keys()) & set(list_user2.keys())
    numerator = sum( [list_user1[x] * list_user2[x] for x in intersection] )

    sum1 = sum( [list_user1[x]**2 for x in list_user1.keys()] )
    sum2 = sum( [list_user2[x]**2 for x in list_user2.keys()] )
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def textToVector(text):

     words = WORD.findall(text)
     return Counter(words)


##################################

def sampling(myList, k):

    # fill the reservoir to start                                                                                                                                                                         
    result = myList[0:k]

    n = k
    for item in myList[k:]:
        n += 1
        s = random.randint(0, n)
        if s < k:
            result[s] = item

    return result


def searchPairIndex(M, userID):

    flatM = list(chain.from_iterable(M))
    
    try:
        myIndex = flatM.index(userID) / 2
    except:
        # userID is not in the list
        return None

    return myIndex


NGEN = 10
MUTPB = 0.3 # mutation probability
MUTB = 0.3

# individual
class Suggestion(list):

    def __init__(self, myPopulation):     # population is in a usersInfo format

        self.population = myPopulation.users
        self.matchingMatrix = myPopulation.matchingMatrix
                                                                                                               
        self.initialize()

        # needs to be a shallow copy
        
        # should be weighted average with powerScore for powerUsers
        self.meetingScores = [ 0.5 * self.matchingMatrix[elem1][elem2] + 0.5 * self.matchingMatrix[elem2][elem1] for elem1, elem2 in self ]
        self.fitness = self.evaluate()


    def initialize( self ):
    
        myListOfUsers = self.population.keys()
        random.shuffle( myListOfUsers )
    
        myResults = [ (x, y) for x, y in itertools.izip( myListOfUsers[0::2], myListOfUsers[1::2] ) ]
        if len(myListOfUsers) % 2 == 1:
            myResults.append( (myListOfUsers[-1],) )
    
        list.__init__(self, myResults)


    def __str__(self):

        myString = color.BOLD + "------------------------------------- \n" + color.END

        for myPair in self:
            myString += (color.BLUE + " {0} : " + color.END +  "{1} \n").format( myPair, self.meetingScores[ self.index(myPair) ] )
        
        myString +=  color.BOLD + "-------------------------------------- \n" + color.END
       
        return myString

    __repr__ = __str__


    def evaluate( self ):

        myMin = min(self.meetingScores)
        myAvg = np.mean(self.meetingScores)

        #return myMin
        return (myMin, myAvg, )


    def mutate( self ):

        #myList = copy.deepcopy(self)
        myLoner = None

        if myLoner == None and self[-1][1] == None:
            myLoner = self[-1][0]
            self.pop[-1]

        lowest_fitness = 1.0
        index_lowest = 0
        bestUserA, bestUserB = self[0]

        index = 0

        for index in range(len(self)):
            currFitness = self.meetingScores[index]
            if self.meetingScores[index] < lowest_fitness:
                lowest_fitness = currFitness
                bestUserA, bestUserB = self[index]
                lowest_index = index

        otherUserA, otherUserB = bestUserA, bestUserB
   
        while (otherUserA, otherUserB) == (bestUserA, bestUserB):
            myPair_index = random.randint(0, len(self) - 1)
            otherUserA, otherUserB = self[myPair_index]

        myReservoir = [ bestUserA, bestUserB, otherUserA, otherUserB ]
        if myLoner:
            myReservoir.append( myLoner )
        random.shuffle( myReservoir )

        myNewPair_1 = sampling( myReservoir, 2 )
        #print myReservoir
        #print myNewPair_1

        myReservoir.remove( myNewPair_1[0] )
        myReservoir.remove( myNewPair_1[1]) 
        if myLoner == None:
            myNewPair_2 = myReservoir
        else:
            myNewPair_2 = sampling( MyReservoir, 2 )
        #print myNewPair_2

        self.__setitem__(lowest_index, tuple(myNewPair_1))
        self.__setitem__(myPair_index, tuple(myNewPair_2))
        if myLoner:
            self.append((myLoner,))

        self.meetingScores = [ 0.5 * self.matchingMatrix[elem1][elem2] + 0.5 * self.matchingMatrix[elem2][elem1] for elem1, elem2 in self ]
        self.fitness = self.evaluate()


    def mate(self, other):

        M = copy.deepcopy(self)
        F = copy.deepcopy(other)
        
        '''
        print "Starting Mating..."

        print "MALE"
        print M
        print M.meetingScores

        print "FEMALE"
        print F
        print F.meetingScores

        assert(0)
        '''

        child = []
        myPool = []


        while len(M)!=0 and len(F)!=0:
            
            # pick a random user from father, pair and then index
            indexM    = random.randint(0, len(M)-1)
            subindexM = random.randint(0, 1)

            # find which pair the user is in @ mother
            indexF = searchPairIndex(F, M[indexM][subindexM])

            # make the father the 'preferred' individual
            if M.meetingScores[indexM] < F.meetingScores[indexF]:
                SwitchGenders(M, F)
                
            child.extend( M[indexM] )
            myLoner = FindTheLoner(M, F, indexM, indexF)

            M.pop(indexM)
            M.meetingScores.pop(indexM)
            F.pop(indexF)
            F.meetingScores.pop(indexF)

            indexMb = searchPairIndex(M, myLoner)
                
            if indexMb != None:
                child.extend( M[indexMb] )
                M.pop(indexMb)
                M.meetingScores.pop(indexMb)
            # else, does it mean we looped the loop?

        myChild = copy.deepcopy(self)
        #print type(myChild)
        myChild[:] = list( zip( child[0::2], child[1::2] ) )
        myChild.fitness = myChild.evaluate()

        return myChild


class Suggestions(list):

    def __init__(self, myPopulation, n = 200):

        matchingMatrix = Suggestion(myPopulation)

        list.__init__(self, [ Suggestion(myPopulation) for _ in range(n) ])
        self.size = n
        self.genNum = 0
        self.bestIndividual = np.nan
        self.bestFitness = np.nan

        for individual in self:
            print individual


    def evolve(self, numGenerations = 10):

        self.genNum += numGenerations

        for gen in range(numGenerations):

            for index in range( len(self) ):
                if random.random() < MUTPB:
                    self[index].mutate()
            
            myParents = [ individual for individual in self ]
            random.shuffle( myParents )

            myParents = [ (x,y) for x, y in itertools.izip( myParents[0::2], myParents[1::2] ) ]
            
            myNewPopulation = []

            for myFather, myMother in myParents:
                
                myChild = myFather.mate(myMother)
                
                if myFather.fitness[1] <= min(myMother.fitness[1], myChild.fitness[1]):
                    myNewPopulation.extend( (myMother, myChild) )
                elif myMother.fitness[1] <= min(myFather.fitness[1], myChild.fitness[1]):
                    myNewPopulation.extend( (myFather, myChild) )
                else:
                    myNewPopulation.extend( (myMother, myFather) )
              
            # keep two best out of father, mother and child
            self.__setslice__(0, len(self), myNewPopulation[0:] )


    def select(self, verbose = False):

        #print type(self)
        
        list.sort(self, key = lambda x: x.fitness, reverse = True)
        
        if verbose:
            for suggestion in self:
                print suggestion.fitness
                    
        self.bestIndividual = self[0]
        self.bestScores  = self[0].meetingScores
        self.bestFitness = self[0].fitness


    def writeToFile(self, myOutputName = "./myOutput/Test"):
            
        myOutputName += "_" + str(self.size) + "ind_" + str(self.genNum) + ".txt"
        myOutputFile = open( myOutputName, 'wb' )

        print>>myOutputFile, self.bestFitness
        for meeting in self.bestIndividual:
            print>>myOutputFile, meeting, self.bestScores[ self.bestIndividual.index(meeting) ]


def main( argv ):

    #myInputFile = '/Users/jennifer/Brainstorms/Data/Sample.csv'
    myInputFile = '/Users/jennifer/Brainstorms/Data/UsersInfo.csv'
    myOutputDir = './myOutput/'
    myAirport = "SJC"
    genNum = 50
   
    def printHelp():
        print 'suggestion.py -u <users> -s <size> -g <generations> -a <airport> -o <output>'

    try:
      opts, args = getopt.getopt(argv, "hu:s:g:a:o:", ["users=", "populationSize=", "generations=", "airport=", "outputDir="])
   
    except getopt.GetoptError:
      printHelp()
      sys.exit(2)
   
    for opt, arg in opts:
    
        if opt == '-h':
            printHelp()
            sys.exit()
        elif opt in ("-u", "--users"):
            myInputfile = arg
        elif opt in ("-g", "--generations"):
            genNum = arg
        elif opt in ("-a", "--airport"):
            myAirport = arg
        elif opt in ("-o", "--outputDir"):
            myOutputfile = arg
   
    print 'Users file is "', myInputFile
    print 'Output directory is "', myOutputDir

    # Get List Of Users for Area
    myUsersInfo = UsersInfo( myInputFile )
    #print myUsersInfo

    oneSuggestion = Suggestion( myUsersInfo )

    print oneSuggestion
    oneSuggestion.mutate()
    print oneSuggestion

    mySuggestions = Suggestions( myUsersInfo, n = 10 )
    
    for mySuggestion in mySuggestions:
        print mySuggestion.meetingScores

    mySuggestions.evolve( 5 )

    for mySuggestion in mySuggestions:
        print mySuggestion.meetingScores

    mySuggestions.select()
    mySuggestions.writeToFile()


if __name__ == "__main__":

   main(sys.argv[1:])

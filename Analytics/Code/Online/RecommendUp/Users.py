from prettytable import PrettyTable

import numpy as np
import pandas as pd
import math
import operator
from Utils import color
from Utils import airportFeatures
from Utils import geoParams


# Default values for user members
DEFAULT_WALKINGSPEED = 300
DEFAULT_MINMEETINGLENGTH = 20

################################################################################

def LevenshteinDistance(s1,s2):

    if len(s1) > len(s2):
        s1,s2 = s2,s1

    distances = range(len(s1) + 1)

    for index2,char2 in enumerate(s2):

        newDistances = [index2+1]

        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    
    return distances[-1]


def LongestCommonString(s1, s2):

    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]

    longest, x_longest = 0, 0

    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):

            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1

                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0

    return s1[x_longest - longest: x_longest]


#s = 'Computer Vision'
#t = 'Comput. Vis.'

#print LevenshteinDistance(s, t)
#print LongestCommonString(s, t)
#assert(0)

################################################################################
# make that class virtual

class User(object):

    def __init__(self, myUserId, myDict):

        self.userId = myUserId
        
        # everything that we read from the profile
        for myKey, myVal in myDict.iteritems():
            if not 'List' in myKey:
                setattr(self, myKey, myVal)
            else:
                #print myKey, myVal
                # try and except might be faster at runtime...
                try:
                    setattr(self, myKey, myVal.split(';'))
                except:
                    setattr(self, myKey, [])

        self.X, self.Y    = self.computeXY()    # GPS Coordinates
        self.destinationX = self.X    # for now the user is considered static
        self.destinationY = self.Y

        #print self.headline
        #assert(0)

        if type(self.headline) == float:
            self.headline = ''

        self.walkingSpeed     = DEFAULT_WALKINGSPEED      # feet / min
        self.minMeetingLength = DEFAULT_MINMEETINGLENGTH  # in minutes
        self.placesList       = []    # favorite restaurants, etc.

        if self.nPastMeetings == '':
            self.nPastMeetings = 0
        if self.avgRating == '':
            self.avgRating = 4
        if self.acceptRate == '':
            self.acceptRate = 0.5

        self.flightNum  = None
        if self.flightNum == None:
            self.flightNum = self.guessFlight()
        
        self.freeTime   = 60          # in minutes

        #self.status     = Type.FREE

        self.vipScore   = 0
        

    def computeXY(self):

        myX = ( (self.lon - airportFeatures.CENTER_LON) * geoParams.EQUATOR_CIRCUMFERENCE) / 180.0
        myY = ( (self.lat - airportFeatures.CENTER_LAT) * geoParams.EQUATOR_CIRCUMFERENCE) * np.pi / 360.0

        return myX, myY


    def computeMaxDist(self):

        return 0.5 * ( self.freeTime - self.minMeetingLength - self.epsilon ) * self.walkingSpeed


    def computeDist(self, otherUser):

        # myDist
        return myDist


    def computeGate(self):

        self.gate = np.nan


    def guessFlight(self):
        pass


    def guessGate(self):
        # guess the gate from patterns, when not provided by the traveler
        pass


    def computeAffinity(self, user2):

        # skills_user1 = [ 'Computer Vision', 'Optimization', 'Genetic Algorithms', 'Machine Learning' ]
        # skills_user2 = [ 'Computer Science', 'Machine Learning' ]
        
        skills_user1 = self.skillsList
        skills_user2 = user2.skillsList

        # will use more complex data structure later to include weights
        intersection = set(skills_user1).intersection(set(skills_user2))
        numerator    = len(intersection)
        denominator = len( set.union( *( set(skills_user1), set(skills_user2) ) ) ) 

        score = 0.0

        if denominator > 0:
            score += 0.5 * float(numerator) / denominator

        # headline_user1 = VP of Engineering @ OpunUp
        # headline_user2 = VP of Product @ LendUp

        myMax = max( len(self.headline), len(user2.headline) )
        if myMax > 0:
            score += 0.5 * len( LongestCommonString(self.headline, user2.headline) ) / myMax
        else:
            score += 0.25

        return score


    def __str__(self): 

        myString = ""
        
        myString += color.BOLD + "--------------------------- \n"
        myString += "User Details for user %s \n" % self.userId 
        myString +=  "--------------------------- \n" + color.END
        for myFeature in self.__dict__:
            if myFeature == 'userId':
                pass
            myString += (color.PURPLE + " {0} : " + color.END +  "{1} \n").format( myFeature, getattr(self, myFeature) )
        myString += "\n"

        return myString


    __repr__ = __str__


##################################################################################################

# helpful function

def cosineAffinityScore( user1, user2 ):

    list_user1 = user1.skillsList
    list_user2 = user2.skillsList

    intersection = set(list_user1.keys()) & set(list_user2.keys())
    numerator = sum( [list_user1[x] * list_user2[x] for x in intersection] )

    sum1 = sum( [list_user1[x]**2 for x in list_user1.keys()] )
    sum2 = sum( [list_user2[x]**2 for x in list_user2.keys()] )
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


class UsersInfo(object):

    def __init__(self, myFile = './Data/UsersInfo.csv'):

        myData = pd.read_csv( myFile, quotechar = "'" )
        myHeaders = myData.columns.values.tolist()

        myIndex_userID = myHeaders.index( 'userId' )
        myHeaders.pop(myIndex_userID)

        self.users = {}
        for myUserRow in myData.values.tolist():
            myUserID = myUserRow[myIndex_userID]
            myUserRow.pop(myIndex_userID)
            self.users[ myUserID ] = User( myUserID, dict( zip( myHeaders, myUserRow ) ) )

        self.size = len(self.users)
        self.matchingMatrix = self.fillMatchingMatrix()


    def fillMatchingMatrix(self, algo = "circle", verbose = False ):    # using the circle algorithm

        if algo not in [ "circle", "ellipse" ]:
            assert(0)

        matchingMatrix = {}

        if self.size > 0:

            matchingMatrix = { userA : { userB : 0.0 for userB in self.users.keys() } for userA in self.users.keys() }

            for userA_id, userA in self.users.iteritems():
                for userB_id, userB in self.users.iteritems():

                    #if userA.computeDist(userB) <= (maxDist_userA + maxDist_userB):
                    matchingMatrix[ userA_id ][ userB_id ] = userA.computeAffinity( userB )

            # The lines below are just for pretty printing functionalities...

            myTableHeaders = [ ' ' ]
            myTableHeaders.extend( sorted(matchingMatrix.keys()) )
            myTable = PrettyTable( myTableHeaders )

            if verbose:

                for myKey, myVal in sorted(matchingMatrix.iteritems()):
                    myRow = [ myKey ]
                    myRow.extend( [ round(val,2) for key, val in sorted(myVal.iteritems(), key = operator.itemgetter(0)) ] )
                    myTable.add_row( myRow )

                print myTable

        return matchingMatrix


    def __str__( self ):

        myString = ""

        for myUserID, myUserObject in self.users.iteritems():
            myString += myUserObject.__str__()

        return myString


    __repr__ = __str__

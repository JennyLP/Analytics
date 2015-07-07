import operator
import random
from itertools import chain

def mutate(M):

    return M


fitnessMatrix = [ [1.00, 0.36, 0.79, 0.16, 0.97, 0.00, 0.15, 0.62, 0.33, 0.35],
                  [0.22, 1.00, 0.80, 0.18, 0.32, 0.07, 0.15, 0.67, 0.45, 0.85],
                  [0.77, 0.40, 1.00, 0.25, 0.18, 0.80, 0.15, 0.67, 0.19, 0.35],
                  [0.20, 0.58, 0.81, 1.00, 0.49, 0.15, 0.35, 0.80, 0.62, 0.38],
                  [0.20, 0.40, 0.76, 0.10, 1.00, 0.11, 0.11, 0.08, 0.66, 0.12],
                  [0.88, 0.40, 0.80, 0.71, 0.12, 1.00, 0.15, 0.22, 0.92, 0.18],
                  [0.20, 0.55, 0.24, 0.69, 0.18, 0.28, 1.00, 0.63, 0.40, 0.35],
                  [0.20, 0.40, 0.80, 0.04, 0.42, 0.30, 0.15, 1.00, 0.61, 0.05],
                  [0.19, 0.46, 0.23, 0.15, 0.26, 0.39, 0.48, 0.44, 1.00, 0.34],
                  [0.03, 0.40, 0.17, 0.09, 0.31, 0.75, 0.59, 0.67, 0.37, 1.00],
                ]


class Meetings(list):

    def __init__(self, *args):

        list.__init__(self, *args)
        self.fitnesses = [ fitnessMatrix[userA-1][userB-1] for (userA, userB) in self ]
        #print self.fitnesses


def searchPairIndex(M, userID):
    
    flatM = list(chain.from_iterable(M))
    myIndex = flatM.index(userID) / 2
    return myIndex


def mate(M, F):

    MUTB = 0.3
    
    for index in range(len(M)):
        M[index] = sorted(M[index])
        F[index] = sorted(F[index])

    M = sorted(M, key = operator.itemgetter(0))
    F = sorted(F, key = operator.itemgetter(0))

    M = Meetings(M)
    F = Meetings(F)

    print M
    print M.fitnesses
    print F
    print F.fitnesses

    if random.random() < MUTB:
        mutate(M)
            
    if random.random() < MUTB:
        mutate(F)

    child1 = []
    child2 = []

    indexM = (random.randint(0, len(M)-1), random.randint(0,1))
    indexF = searchPairIndex(F, M[indexM[0]][indexM[1]])

    if M.fitnesses[ indexM[0] ] > F.fitnesses[ indexM[0] ]:
        child1.append( M[indexM[0]] )
        print child1

M = [ (1,5), (8,3), (2,6), (10,9), (4,7) ]
F = [ (2,10), (6,7), (8,1), (3,4), (9,5) ]

mate(M, F)

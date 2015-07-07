class Individual(list):

    def __init__(self, *args):

        list.__init__(self, *args)
        self.fitness = 1.0

    def __str__(self):
        
        print self


myIndividual = Individual( [1,4,6] )
print myIndividual.fitness
myIndividual.__str__

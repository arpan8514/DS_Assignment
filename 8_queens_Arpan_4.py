#------------------------------------------------------------------------------#
import random
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
class eight_queens_Arpan(object):

    def __init__(self, board):
        self.board = board

    ##----------------------------------------------------------------##
    def fitness_measure(self, chromosome, maxFitness):
        horizontal_collisions = \
            sum([chromosome.count(queen)-1 for queen in chromosome])/2
        diagonal_collisions = 0

        n = len(chromosome)
        left_diagonal = [0] * 2*n
        right_diagonal = [0] * 2*n
        for i in range(n):
            left_diagonal[i + chromosome[i] - 1] += 1
            right_diagonal[len(chromosome) - i + chromosome[i] - 2] += 1

        diagonal_collisions = 0
        for i in range(2*n-1):
            counter = 0
            if left_diagonal[i] > 1:
                counter += left_diagonal[i]-1
            if right_diagonal[i] > 1:
                counter += right_diagonal[i]-1
            diagonal_collisions += counter / (n-abs(i-n+1))
    
        return int(maxFitness - (horizontal_collisions + diagonal_collisions))
    ##----------------------------------------------------------------##

    ##----------------------------------------------------------------##
    def chomosome_display(self, chromosome):
        print("Chromosome = {},  Fitness = {}"
              .format(str(chromosome), self.fitness_measure(chromosome)))
    ##----------------------------------------------------------------##

    ##----------------------------------------------------------------##
    def probability_measure(self, chromosome, fitness):
        return fitness(chromosome) / maxFitness
    ##----------------------------------------------------------------##

    ##----------------------------------------------------------------##
    def random_pick(self, population, probabilities):
        populationWithProbabilty = zip(population, probabilities)
        total = sum(w for c, w in populationWithProbabilty)
        r = random.uniform(0, total)
        limit_val = 0
        for c, w in zip(population, probabilities):
            if limit_val + w >= r:
                return c
            limit_val = limit_val + w
        assert False
    ##----------------------------------------------------------------##

    ##----------------------------------------------------------------##
    def genetic_reproduce(self, node_1, node_2):
        n = len(node_1)
        c = random.randint(0, n - 1)
        return node_1[0:c] + node_2[c:n]
    ##----------------------------------------------------------------##

    ##----------------------------------------------------------------##
    def genetic_mutate(self, node):
        n = len(node)
        cross = random.randint(0, n - 1)
        mutation = random.randint(1, n)
        node[cross] = mutation
        return node
    ##----------------------------------------------------------------##

    ##----------------------------------------------------------------##
    def genetic_queen(self, population, fitness):
        mutation_probability = 0.03
        new_population = []
        probabilities = [self.probability_measure(n, fitness) for n in population]
        for i in range(len(population)):
            x = self.random_pick(population, probabilities)
            y = self.random_pick(population, probabilities)
            child = self.genetic_reproduce(x, y)
            if (random.random() < mutation_probability):
                child = self.genetic_mutate(child)
            self.chomosome_display(child)
            new_population.append(child)
            if (fitness(child) == maxFitness): 
                break
        return new_population
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
def eight_queens_puzzle(nq):

    board = []
    for x in range(nq):
        board.append(["x"] * nq)

    maxFitness = (nq*(nq-1))/2  # 8*7/2 = 28
    random_chromosome = [random.randint(1, nq) for _ in range(nq)]
    population = [random_chromosome for _ in range(100)]
    #print (population)

    generation = 1
    eight_queens_obj = eight_queens_Arpan(board)

    while not maxFitness in \
        [eight_queens_obj.fitness_measure(chrom, maxFitness) for chrom in population]:
        print("=== Generation {} ===".format(generation))
        population = eight_queens_obj.genetic_queen(population, 
                     eight_queens_obj.fitness_measure)
        print("")
        print("Maximum Fitness = {}"
            .format(max([eight_queens_obj.fitness_measure(n, maxFitness) for n in population])))
        generation += 1

    chrom_out = []
    print("Solved in Generation {}!".format(generation-1))
    for chrom in population:
        if (eight_queens_obj.fitness_measure(chrom, maxFitness) == maxFitness):
            print("");
            print("One of the solutions: ")
            chrom_out = chrom
            eight_queens_obj.chomosome_display(chrom)
            
    for i in range(nq):
        board[nq-chrom_out[i]][i]="Q"
            
    return board
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
nq = 8
#maxFitness = (nq*(nq-1))/2  # 8*7/2 = 28

result = eight_queens_puzzle(nq)
for row in result:
    print (" ".join(row))
#------------------------------------------------------------------------------#


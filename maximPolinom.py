import math
from typing import List, Tuple
import random
import bisect
import copy

fout = open('Evolutie.txt', 'w')

POPULATION_SIZE = 20
LEFT_BOUND, RIGHT_BOUND = -1, 2
COEF2, COEF1, COEF0 = -1, 1, 2
PRECISION = 6
CROSSOVER_PROB = 25 / 100
MUTATION_PROB = 1 / 100
STEPS = 100


class Individual:
    def __init__(self, genome: List[int], fitness: float = 0) -> None:
        self.genome: List[int] = genome
        self.fitness: float = fitness


class GenomeGenerator:
    def __init__(
        self,
        left: float,
        right: float,
        precision: int,
        coeffs: Tuple[float, float, float],
        populationSize: int,
        steps: int,
        crossOverProb: float,
        mutationProb: float,
    ) -> None:
        self.left = left
        self.right = right
        self.genomeSize = math.ceil(math.log((right - left) * 10**precision, 2))
        self.discreteStep = (right - left) / 2**self.genomeSize
        self.fitnessFunction = lambda x: coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
        self.population: List[Individual] = [
            self.generateRandomIndividual() for _ in range(populationSize)
        ]
        self.steps = steps
        self.crossOverProb = crossOverProb
        self.mutationProb = mutationProb
        self.elite = None
        
        fout.write("Populatia initiala\n")
        for (i, x) in enumerate(self.population):
            res = str(i).zfill(len(str(len(self.population)))) + ": "
            res += "".join([str(y) for y in x.genome]) + " "
            
            floatVal = self.fromGenome(x.genome)
            res += "x= "
            if floatVal > 0:
                res += " "
                
            res += '%.5f' % floatVal + " "
            res += "f= " + str(x.fitness)
            
            fout.write(res + "\n")

        fout.write('\n')
        
    def generateRandomIndividual(self):
        x = random.uniform(self.left, self.right)
        g = self.toGenome(x)
        f = self.fitnessFunction(x)

        return Individual(g, f)

    def toGenome(self, x: float) -> str:
        normalized = math.floor((x - self.left) / self.discreteStep)
        return [int(x) for x in bin(normalized)[2:].zfill(self.genomeSize)]

    def fromGenome(self, x: str) -> float:
        decimal = int("".join([str(y) for y in x]), 2)
        return self.left + decimal * self.discreteStep

    def selection(self, isFirst = True) -> List[Individual]:
        
        newPopulation = []
    
        n = len(self.population)
        intervals = [0] * (n + 1)
        suma = sum([x.fitness for x in self.population])

        for i in range(1, n):
            intervals[i] = intervals[i - 1] + self.population[i - 1].fitness

        intervals[n] = suma

        for i in range(1, n + 1):
            intervals[i] /= suma
            
        self.elite = 0
        maxFitness = 0
        for i in range(len(self.population)):
            if self.population[i].fitness > maxFitness:
                maxFitness = self.population[i].fitness
                self.elite = i
            
        # always keep elite
        newPopulation.append(copy.deepcopy(self.population[self.elite]))
        
        # if isFirst:
        #     fout.write("Intervale probabilitati selectie\n")
            
        #     for i in range(len(probs)):
        #         res = str(probs[i][0]) + " "
        #         if i > 0 and i % 4 == 0:
        #             res += '\n'
        #         fout.write(res)
        #     fout.write('\n')
        
        for _ in range(1, len(self.population)):
            u = random.uniform(0, 1)
            idx = bisect.bisect_right(intervals, u) - 1
            newPopulation.append(copy.deepcopy(self.population[idx]))

        self.population = newPopulation
        return newPopulation

    def crossOver(self, isFirst = True) -> List[Individual]:
        parents = []
        for i in range(len(self.population)):
            u = random.uniform(0, 1)

            if u < self.crossOverProb:
                parents.append(i)

        l, r = 0, len(parents) - 1

        while l < r:
            p1 = self.population[l]
            p2 = self.population[r]

            crossPoint = random.randint(0, self.genomeSize - 1)
            g1 = p1.genome[:crossPoint] + p2.genome[crossPoint:]
            g2 = p2.genome[:crossPoint] + p1.genome[crossPoint:]

            p1.genome = copy.deepcopy(g1)
            p2.genome = copy.deepcopy(g2)

            l, r = l + 1, r - 1

        return self.population

    def mutation(self, isFirst = True) -> List[Individual]:
        
        for (i, x) in enumerate(self.population):
            u = random.uniform(0, 1)

            if u < self.mutationProb:
                mutationPoint = random.randint(0, self.genomeSize - 1)
                x.genome[mutationPoint] = 1 - x.genome[mutationPoint]

        return self.population

    def maximizeFunction(self):
        
        self.selection()
        self.crossOver()
        self.mutation()
        
        for step in range(1, self.steps):
            self.elite = None
            for idx, individual in enumerate(self.population):
                individual.fitness = self.fitnessFunction(
                    self.fromGenome(individual.genome)
                )

            self.selection(False)
            self.crossOver(False)
            self.mutation(False)

            print("Max= ", max([x.fitness for x in self.population]), end="   ")
            print(
                "Mean= ",
                sum([x.fitness for x in self.population]) / len(self.population),
            )


genomeGenerator = GenomeGenerator(
    LEFT_BOUND,
    RIGHT_BOUND,
    PRECISION,
    (COEF2, COEF1, COEF0),
    POPULATION_SIZE,
    STEPS,
    CROSSOVER_PROB,
    MUTATION_PROB,
)

genomeGenerator.maximizeFunction()

import math
from typing import List, Tuple
import random
import bisect

fout = open('Evolutie.txt', 'w')

POPULATION_SIZE = 20
LEFT_BOUND, RIGHT_BOUND = -1, 2
COEF2, COEF1, COEF0 = -1, 1, 2
PRECISION = 6
CROSSOVER_PROB = 25 / 100
MUTATION_PROB = 1 / 100
STEPS = 50


class Individual:
    def __init__(self, genome: str, fitness: float = 0) -> None:
        self.genome = genome
        self.fitness = fitness


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
        
        fout.write("Populatia initiala\n")
        for (i, x) in enumerate(self.population):
            res = str(i).zfill(len(str(len(self.population)))) + ": "
            res += x.genome + " "
            
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
        return bin(normalized)[2:].zfill(self.genomeSize)

    def fromGenome(self, x: str) -> float:
        decimal = int(x, 2)
        return self.left + decimal * self.discreteStep

    def selection(self, isFirst = True) -> List[Individual]:
        
        newPopulation = []

        fitnessValues = [x.fitness for x in self.population]
        sumFitness = sum(fitnessValues)

        # pi = f(Xi) / sum(f(Xj))
        probs: List[Tuple[int, float]] = [
            (x.fitness / sumFitness, i) for (i, x) in enumerate(self.population)
        ]
        
        if isFirst:
            fout.write("Probabilitati selectie\n")
            
            for (val, i) in probs:
                res = "cromozom   "
                res += str(i).zfill(len(str(len(self.population)))) + " "
                res += "probabilitate "
                res += str(val)
                
                fout.write(res + "\n")
            fout.write("\n")
            
        probs.sort()
        

        elite = self.population[probs[-1][1]]
        # always keep elite
        newPopulation.append(elite)

        probs.insert(0, (0, -1))

        # sum of probabilities qi
        for i in range(1, len(probs)):
            probs[i] = (probs[i - 1][0] + probs[i][0], probs[i][1])

        probs.append((1, -1))
        
        if isFirst:
            fout.write("Intervale probabilitati selectie\n")
            
            for i in range(len(probs)):
                res = str(probs[i][0]) + " "
                if i > 0 and i % 4 == 0:
                    res += '\n'
                fout.write(res)
            fout.write('\n')
        
        for _ in range(1, len(self.population)):
            u = random.uniform(0, 1)
            idx = bisect.bisect_right(probs, u, key=lambda x: x[0]) + 1
            newPopulation.append(self.population[probs[idx][1]])

        self.population = newPopulation
        return newPopulation

    def crossOver(self, isFirst = True) -> List[Individual]:
        parents = []
        for i in range(len(self.population)):
            u = random.uniform(0, 1)

            if u < self.crossOverProb:
                parents.append(i)

        maxF = max([x.fitness for x in self.population])
        l, r = 0, len(parents) - 1

        while l < r:
            p1 = self.population[l]
            p2 = self.population[r]

            crossPoint = random.randint(0, self.genomeSize - 1)
            g1 = p1.genome[:crossPoint] + p2.genome[crossPoint:]
            g2 = p2.genome[:crossPoint] + p1.genome[crossPoint:]

            # always keep elite
            if p1.fitness != maxF:
                p1.genome = g1
            if p2.fitness != maxF:
                p2.genome = g2

            l, r = l + 1, r - 1

        return self.population

    def mutation(self, isFirst = True) -> List[Individual]:
        
        maxF = max([x.fitness for x in self.population])
        for x in self.population:
            u = random.uniform(0, 1)

            # always keep elite
            if u < self.mutationProb and x.fitness != maxF:
                mutationPoint = random.randint(0, self.genomeSize - 1)
                flipped = "1" if x.genome[mutationPoint] == "0" else "0"
                x.genome = (
                    x.genome[:mutationPoint] + flipped + x.genome[mutationPoint + 1 :]
                )

        return self.population

    def maximizeFunction(self):
        
        self.selection()
        self.crossOver()
        self.mutation()
        
        for step in range(1, self.steps):
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

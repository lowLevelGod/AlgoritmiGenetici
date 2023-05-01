import math
from typing import List, Tuple
import random
import bisect
import copy
import matplotlib.pyplot as plt

fout = open("Evolutie.txt", "w")

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
        for i, x in enumerate(self.population):
            res = str(i + 1).zfill(len(str(len(self.population)))) + ": "
            res += "".join([str(y) for y in x.genome]) + " "

            floatVal = self.fromGenome(x.genome)
            res += "x= "
            if floatVal > 0:
                res += " "

            res += "%.5f" % floatVal + " "
            res += "f= " + str(x.fitness)

            fout.write(res + "\n")

        fout.write("\n")

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

    def selection(self, isFirst=True) -> List[Individual]:
        newPopulation = []

        n = len(self.population)
        intervals = [0] * (n + 1)
        suma = sum([x.fitness for x in self.population])

        if isFirst:
            fout.write("Probabilitati selectie\n")
            for i, x in enumerate(self.population):
                col = "cromozom   "
                col += str(i + 1).zfill(len(str(len(self.population))))
                col += " probabilitate "
                col += str(x.fitness / suma)
                fout.write(col + "\n")

        for i in range(1, n):
            intervals[i] = intervals[i - 1] + self.population[i - 1].fitness

        intervals[n] = suma

        for i in range(1, n + 1):
            intervals[i] /= suma

        if isFirst:
            fout.write("Intervale probabilitati selectie\n")
            for i in intervals:
                fout.write(str(i) + "\n")

        self.elite = 0
        maxFitness = 0
        for i in range(len(self.population)):
            if self.population[i].fitness > maxFitness:
                maxFitness = self.population[i].fitness
                self.elite = i

        # always keep elite
        newPopulation.append(copy.deepcopy(self.population[self.elite]))

        for _ in range(1, len(self.population)):
            u = random.uniform(0, 1)
            idx = bisect.bisect_right(intervals, u) - 1
            newPopulation.append(copy.deepcopy(self.population[idx]))

            if isFirst:
                col = "u=" + str(u) + "   "
                col += "selectam cromozomul "
                col += str(idx + 1)
                fout.write(col + "\n")

        self.population = newPopulation

        if isFirst:
            fout.write("Dupa selectie:\n")
            for i, x in enumerate(self.population):
                res = str(i + 1).zfill(len(str(len(self.population)))) + ": "
                res += "".join([str(y) for y in x.genome]) + " "

                floatVal = self.fromGenome(x.genome)
                res += "x= "
                if floatVal > 0:
                    res += " "

                res += "%.5f" % floatVal + " "
                res += "f= " + str(x.fitness)

                fout.write(res + "\n")
            fout.write("\n")

        return newPopulation

    def crossOver(self, isFirst=True) -> List[Individual]:
        if isFirst:
            fout.write(
                "Probabilitatea de incrucisare " + str(self.crossOverProb) + "\n"
            )

        parents = []
        for i in range(len(self.population)):
            u = random.uniform(0, 1)

            if u < self.crossOverProb:
                parents.append(i)

            if isFirst:
                res = str(i + 1).zfill(len(str(len(self.population)))) + ": "
                res += "".join([str(y) for y in self.population[i].genome]) + " "
                res += "u=" + str(u)
                if u < self.crossOverProb:
                    res += "<" + str(self.crossOverProb) + " participa"
                fout.write(res + "\n")

        if isFirst:
            fout.write("\n")

        l, r = 0, len(parents) - 1

        while l < r:
            p1 = self.population[l]
            p2 = self.population[r]

            crossPoint = random.randint(0, self.genomeSize - 1)
            g1 = p1.genome[:crossPoint] + p2.genome[crossPoint:]
            g2 = p2.genome[:crossPoint] + p1.genome[crossPoint:]

            if isFirst:
                fout.write(
                    "Recombinare dintre cromozomul "
                    + str(l)
                    + " "
                    + "cu cromozomul "
                    + str(r)
                    + "\n"
                )
                res = "".join([str(y) for y in self.population[l].genome]) + " "
                res += "".join([str(y) for y in self.population[r].genome]) + " "
                res += "punct " + str(crossPoint) + "\n"

                res += "Rezultat   "
                res += "".join([str(y) for y in g1]) + " "
                res += "".join([str(y) for y in g2]) + "\n"
                fout.write(res)

            p1.genome = copy.deepcopy(g1)
            p2.genome = copy.deepcopy(g2)

            l, r = l + 1, r - 1

        if isFirst:
            fout.write("Dupa recombinare:\n")
            for i, x in enumerate(self.population):
                res = str(i + 1).zfill(len(str(len(self.population)))) + ": "
                res += "".join([str(y) for y in x.genome]) + " "

                floatVal = self.fromGenome(x.genome)
                res += "x= "
                if floatVal > 0:
                    res += " "

                res += "%.5f" % floatVal + " "
                res += "f= " + str(x.fitness)

                fout.write(res + "\n")
            fout.write("\n")

        return self.population

    def mutation(self, isFirst=True) -> List[Individual]:
        if isFirst:
            fout.write(
                "Probabilitate de mutatie pentru fiecare gena "
                + str(self.mutationProb)
                + "\n"
            )
            fout.write("Au fost modificati cromozomii:\n")

        for i, x in enumerate(self.population):
            u = random.uniform(0, 1)

            if u < self.mutationProb:
                if isFirst:
                    fout.write(str(i + 1) + "\n")
                mutationPoint = random.randint(0, self.genomeSize - 1)
                x.genome[mutationPoint] = 1 - x.genome[mutationPoint]
        if isFirst:
            fout.write("Dupa mutatie:\n")
            for i, x in enumerate(self.population):
                res = str(i + 1).zfill(len(str(len(self.population)))) + ": "
                res += "".join([str(y) for y in x.genome]) + " "

                floatVal = self.fromGenome(x.genome)
                res += "x= "
                if floatVal > 0:
                    res += " "

                res += "%.5f" % floatVal + " "
                res += "f= " + str(x.fitness)

                fout.write(res + "\n")
            fout.write("\n")

        return self.population

    def maximizeFunction(self):
        
        xs = [i + 1 for i in range(self.steps)]
        ys = []
        
        self.selection()
        self.crossOver()
        self.mutation()

        ys.append(max([x.fitness for x in self.population]))

        fout.write("Evolutia maximului\n")

        for step in range(1, self.steps):
            self.elite = None
            for idx, individual in enumerate(self.population):
                individual.fitness = self.fitnessFunction(
                    self.fromGenome(individual.genome)
                )

            self.selection(False)
            self.crossOver(False)
            self.mutation(False)

            mx = max([x.fitness for x in self.population])
            mn = sum([x.fitness for x in self.population]) / len(self.population)
            
            ys.append(mx)
    
            fout.write("Max= " + str(mx) + "   " + "Mean= " + str(mn) + "\n")

            print("Max= ", mx, end="   ")
            print("Mean= ", mn)
        
        for (x, y) in zip(xs, ys):
            plt.plot(x, y)
        
        plt.xlabel("Generatii")
        plt.ylabel("Max fitness")
        
        plt.show()


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

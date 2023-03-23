##########################################################################################
# Copyright (c) 2022, Nassim Mokhtari, Alexis Nédélec, Marlène Gilles and Pierre De Loor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################################
import random
from IFA.Utils import GlobalInformation


class Population:
    """
    Implementation of the population of solutions
    """
    def __init__(self, pop_size, solution):
        """
        :param pop_size: population size
        :param solution: solution's class to use
        """
        self.pop_size = pop_size
        self.population = []
        self.Solution = solution
        self.sum_fitness = 0

    def create_new_sol(self, matrix=None, encoding=None):
        """
        Crates a new solution from the matrix and the encoding
        @param matrix
        @param encoding in the case of nas_bench_101
        @return: solution created from the matrix and the encoding
        """
        return self.Solution(matrix=matrix, encoding=encoding)

    def init_pop(self):
        """
        Init. the population
        @return: Population of solution
        """
        self.population = []
        for i in range(self.pop_size):
            sol = self.create_new_sol()
            if sol:
                sol.evaluate()
                print("new sol added with fitness", sol.fitness)
                self.population.append(sol)
            else:
                i -= 1

    @staticmethod
    def to_tuples(solution):
        """
        Converts solution's information to a tuple
        @param solution
        @return: tuple containing solution's information
        """
        if GlobalInformation.Params.is_nas_bench_101:
            return (solution.matrix, solution.encoding), solution.fitness
        else:
            return solution.matrix, solution.fitness

    def get_from_pop(self, solution_information):
        """
        @param solution_information: Information concerning a solution
        @return: solution corresponding to the solution_information
        """
        if GlobalInformation.Params.is_nas_bench_101:
            matrix, encoding = solution_information
        else:
            matrix = solution_information
            encoding = None
        return self.create_new_sol(matrix=matrix, encoding=encoding)

    def load_population(self, solutions_information):
        """
        Loads a population from an array of solution_information
        @param solutions_information: array of solution_information
        """
        del self.population
        self.population = []
        for x in solutions_information:
            couches, fitness = x
            s = self.get_from_pop(couches)
            s.set_fitness(fitness)
            self.population.append(s)

    def get_best(self):
        """
        returns the best solution in the population
        @return: solution
        """
        best = self.population[0]
        for x in self.population:
            if x.is_better(best):
                best = x
        return best.clone()

    def sort_population(self):
        """
        In-place ascending sorting of the population
        """
        self.population.sort(key=lambda x: x.fitness, reverse=False)
        self.sum_fitness = sum([x.fitness for x in self.population])

    def get_population(self):
        """
        @return: population
        """
        return self.population

    def get(self, index):
        """
        returns the solution at the specified index of the population
        @param index: index of the solution to return
        @return: solution
        """
        return self.population[index]

    def length(self):
        """
        @return: size of the population
        """
        return len(self.population)

    def set_population(self, new_pop):
        """
        Set the population
        @param new_pop: list of solution
        @return:
        """
        del self.population
        self.population = new_pop

    def select_random(self):
        """
        Select a solution from the population randomly
        @return: selected solution
        """
        return self.population[random.randint(0, len(self.population) - 1)]

    def select_fit_based(self):
        """
        Select a solution from the population using roulette wheel selection based on fitness
        @return: selected solution
        """
        q = random.random()*self.sum_fitness
        i = 0
        while i < len(self.population):
            q -= self.population[i].fitness
            if q <= 0:
                break
            i += 1

        return self.population[i]

    def select_rank_based(self):
        """
        Select a solution from the population using roulette wheel selection based on rank
        @return: selected solution
        """
        nb_sols = len(self.population)
        sum_ranks = int((nb_sols - 1) * nb_sols / 2)
        q = random.randint(1, sum_ranks)
        i = 0
        while i < len(self.population):
            q -= i+1
            if q <= 0:
                break
            i += 1
        return self.population[i]

    def select(self, strategy='random'):
        """
        Select a solution from the population using the specified strategy
        @param strategy :
        @return: selected solution
        """
        if strategy == 'random':
            return self.select_random()
        elif strategy == 'fit':
            return self.select_fit_based()
        elif strategy == 'rank':
            return self.select_rank_based()
        else:
            raise ValueError('Error in solution selection ! Wrong type passed as param : "'+strategy
                             + '" unknown ! Please use "fit", "rank" or "random" ')

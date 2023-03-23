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

import pickle
import time
import random
from IFA.Population import Population


class Search_FA:
    """
    Implementation of the Firefly Algorithm
    """

    def __init__(self, pop_size, alpha=0.5, beta=0.95, gamma=0.15):
        from IFA.Solution import SolutionFA as Solution
        self.population = Population(pop_size, Solution)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def go_search(self, path, path_best, end, search_information=None):
        """
        :param path: path where to save the information of the search (last population, best solution, ...)
        :param path_best: path where to save the best solution
        :param end: stopping criterion (max epochs)
        :param search_information: python dictionary containing the the information of the search, None if new search
        :return: best solution
        """
        if search_information:
            self.population.load_population(search_information['pop'])
            g_best = search_information['best']
            history = search_information['hist']
            start = len(history)
        else:
            start = 0
            search_information = {}
            self.population.init_pop()
            g_best = self.population.get_best()
            history = []
        print("### Pop init Done ###")
        for i in range(start, end):
            deb = time.time()
            print("Generation : ", i)
            pop = []
            local_best = None
            self.population.sort_population()
            for x in range(self.population.length() - 1):
                for y in range(x + 1, self.population.length()):
                    if self.population.get(y).is_better(self.population.get(x)):
                        self.population.get(x).move_to(self.population.get(y), self.alpha, self.beta, self.gamma)

            for x in self.population.get_population():
                x.evaluate()
                pop.append(self.population.to_tuples(x))
                if x.is_better(local_best):
                    local_best = x

            if local_best.is_better(g_best):
                print(local_best.fitness, "is better then", g_best.fitness)
                g_best = local_best.clone()
            search_information['pop'] = pop
            search_information['best'] = g_best
            t = time.time() - deb
            history.append({'best': g_best.fitness, 'local': local_best.fitness, 'time': t})
            search_information['hist'] = history
            print(local_best.get_final_accuracy())
            print(g_best.get_final_accuracy())
            print("Gen ", i, "done , local best :", local_best.fitness, 'global best', g_best.fitness, 'took ',
                  str(t) + 's')
            pickle.dump(search_information, open(path, 'wb'))
            pickle.dump(g_best, open(path_best, 'wb'))
        return g_best

    def run(self, num_gen, path, path_best, force_restart=False):
        """
        :param num_gen: Number of generations
        :param path: path where to save the information of the search (last population, best solution, ...)
        :param path_best: path where to save the best solution
        :param force_restart: restart the search scratch if True, else continue
        :return: the best solution found
        """
        print("Start search : ", "Forcing restart", force_restart)
        import os
        if force_restart or not os.path.exists(path):
            return self.go_search(path, path_best, num_gen)
        else:
            search_information = pickle.load(open(path, 'rb'))
            return self.go_search(path, path_best, num_gen, search_information)


class Search_IFA:
    """
    Implementation of the Improved Firefly Algorithm
    """

    def __init__(self, pop_size, alpha=0.5, beta=0.95, gamma=0.15, prob_mut=1, chances=5, selection='random'):
        from IFA.Solution import SolutionIFA as Solution
        self.population = Population(pop_size, Solution)
        self.Solution = Solution
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.chances = chances
        self.prob_mut = prob_mut
        self.selection = selection

    def cross(self, solution1, solution2):
        """
        :param solution1: parent solution 1
        :param solution2: parent solution 2
        :return: offspring  resulting from the crossover between parent solutions
        """
        return self.Solution.cross(solution1, solution2)

    def mutation(self, off):
        """
        :return: mutant solution (according to the standard mutation or the DE mutation)
        """
        if off and random.random() < self.prob_mut:
            off.mutate()
        return off

    def cross_population(self):
        """
        @return: The new population after crossover
        """
        print('_____ crossover _____')
        new_pop = []
        pop = []
        size = 0
        while size < self.population.length():
            parent_1 = self.population.select(self.selection)
            parent_2 = self.population.select(self.selection)
            off = self.cross(parent_1, parent_2)
            off = self.mutation(off)
            if off:
                off.evaluate()
                new_pop.append(off)
                pop.append(self.population.to_tuples(off))
                size += 1
        self.population.set_population(new_pop)
        return pop

    def go_search(self, path, path_best, end, search_information=None):
        """
        :param path: path where to save the information of the search (last population, best solution, ...)
        :param path_best: path where to save the best solution
        :param end: stopping criterion (max epochs)
        :param search_information: python dictionary containing the the information of the search, None if new search
        :return: best solution
        """
        if search_information:
            self.population.load_population(search_information['pop'])
            bests = search_information['bests']
            best_fit = max([x.fitness for x in bests])
            g_best = [x for x in bests if x.fitness == best_fit][0]
            history = search_information['hist']
            counter = search_information['counter']
            chances = search_information['chances']
            bests = search_information['bests']
            start = len(history)
        else:
            start = 0
            search_information = {}
            self.population.init_pop()
            g_best = self.population.get_best()
            history = []
            counter = self.chances
            chances = self.chances
            bests = []
        print("### Pop init Done ###")
        for i in range(start, end):
            deb = time.time()
            print("Generation : ", i)
            pop = []
            local_best = None
            self.population.sort_population()
            for x in range(self.population.length() - 1):
                for y in range(x + 1, self.population.length()):
                    if self.population.get(y).is_better(self.population.get(x)):
                        self.population.get(x).move_to(self.population.get(y), self.alpha, self.beta, self.gamma)

            for x in self.population.get_population():
                x.evaluate()
                pop.append(self.population.to_tuples(x))
                if x.is_better(local_best):
                    local_best = x.clone()
            if local_best.is_better(g_best):
                if g_best:
                    print(local_best.fitness, "is better then", g_best.fitness)
                g_best = local_best.clone()
                counter = chances
            else:
                counter -= 1
            if counter == 0:
                pop = self.cross_population()
                local_best = self.population.get_best()
                counter = chances
                bests.append(g_best)
                g_best = None

            search_information['pop'] = pop
            search_information['bests'] = bests
            search_information['chances'] = chances
            search_information['counter'] = counter
            t = time.time() - deb
            history.append({'local': local_best.fitness, 'time': t})
            search_information['hist'] = history
            print("Gen ", i, "done , local best :", local_best.fitness, 'took ', str(t) + 's')
            pickle.dump(search_information, open(path, 'wb'))
            print('local', local_best.fitness, local_best.get_final_accuracy(), 'chances', counter)
            pickle.dump(search_information, open(path, 'wb'))
        bests.append(g_best)
        best_fitness = max([x.fitness for x in bests])
        best = [x for x in bests if x.fitness == best_fitness][0]
        pickle.dump(best, open(path_best, 'wb'))
        return best

    def run(self, num_gen, path, path_best, force_restart=False):
        """
          :param num_gen: Number of generations
          :param path: path where to save the information of the search (last population, best solution, ...)
          :param path_best: path where to save the best solution
          :param force_restart: restart the search scratch if True, else continue
          :return: the best solution found
        """
        print("Start search : ", "Forcing restart ", force_restart)
        import os
        if force_restart or not os.path.exists(path):
            return self.go_search(path, path_best, num_gen)
        else:
            search_information = pickle.load(open(path, 'rb'))
            return self.go_search(path, path_best, num_gen, search_information)

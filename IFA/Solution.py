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


import numpy as np
import math
import random
from IFA.Utils import GlobalInformation
from IFA.Utils.Evaluation import get_ICD_score_torch_gpu
from IFA.NASBENCH101.nas101_model import Network
from nasbench.lib.model_spec import ModelSpec
from xautodl.models import get_cell_based_tiny_net


class SolutionBaseNASBENCH101:
    """Implementation of the base solution for NAS-BENCH-101"""

    def __init__(self, matrix=None, encoding=None):
        available = ['maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu']
        if matrix:
            self.matrix = matrix
            self.encoding = encoding
            self.generate_labels()
        else:
            architecture_hash = random.sample(GlobalInformation.Params.nas_bench_101_iterator, 1)[0]
            a, b = GlobalInformation.Params.nasbench.get_metrics_from_hash(architecture_hash)
            self.matrix = a['module_adjacency'].tolist()
            self.labels = a['module_operations']
            self.encoding = [available.index(x) / len(available) for x in self.labels[1:-1]]
        self.model = None

    def generate_labels(self):
        """
        @return: operator's label
        """
        labels = ['input']
        for x in self.encoding:
            if x < 0.33:
                labels.append('maxpool3x3')
            elif x < 0.66:
                labels.append('conv1x1-bn-relu')
            else:
                labels.append('conv3x3-bn-relu')
        labels.append('output')
        self.labels = labels

    def print_model(self):
        """
        print the current solution
        """
        print(np.asarray(self.matrix))
        print(self.labels)
        print(self.encoding)

    def get_final_accuracy(self):
        """
        @return: get final test accuracy (after training) from the NAS-BENCH-101
        """
        spec = ModelSpec(self.matrix, self.labels, data_format=GlobalInformation.Params.config['data_format'])
        _, b = GlobalInformation.Params.nasbench.get_metrics_from_spec(spec)
        scores = [x['final_test_accuracy'] for x in b[108]]
        return max(scores)

    def compute_incoming_edges(self):
        """
        @return: incoming edges from this solution's matrix
        """
        incoming_edges = dict()
        for i in range(len(self.matrix)):
            incoming_edges[i] = []
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[i][j] == 1:
                    incoming_edges[j].append(i)
        return incoming_edges

    @staticmethod
    def explore(node, incoming_edges):
        """
        Explores the graph and returns the path from the current node to the input @param node: current node @param
        incoming_edges: incoming edges for solution's matrix @return: path from the node to the input, -1 if node is
        the input, 1 if the node is directly connected to the input
        """
        from copy import deepcopy
        index = node[-1]
        open_nodes = incoming_edges[index]
        if len(open_nodes) == 0:
            return -1
        if 0 in open_nodes:
            node.append(0)
            return 1
        else:
            new_ = []
            for x in open_nodes:
                base = deepcopy(node)
                base.append(x)
                new_.append(base)
            return new_

    @staticmethod
    def to_taboo_list(path):
        """
        Converts a path stored in python list of Integers to a path stored in a python list of pairs of Integers
        @param path: path as list of Integers
        @return: list of pairs of Integers
        """
        taboo = []
        for i in range(len(path) - 1, 0, -1):
            taboo.append((path[i], path[i - 1]))
        return taboo

    def taboo(self):
        """
        Creates the taboo list paths
        @return: taboo list of paths
        """
        incoming_edges = self.compute_incoming_edges()
        end = len(self.matrix) - 1
        paths = []
        open_edges = [[end, x] for x in incoming_edges[end]]
        while len(open_edges) > 0:
            current = open_edges.pop(0)
            if 0 in current:
                paths.append(self.to_taboo_list(current))
            new_ = self.explore(current, incoming_edges)
            if new_ == -1:
                continue
            elif new_ == 1:
                paths.append(self.to_taboo_list(current))
            else:
                for x in new_:
                    open_edges.insert(0, x)

        if len(paths) == 0:
            return None
        max_len = max([len(x) for x in paths])
        taboo = [x for x in paths if len(x) == max_len][0]
        return taboo

    def purge(self):
        """
        Purge a solution from extra edges that are not parts of a path from the input to the output
        @return:
        """
        num_edges = sum([sum(x) for x in self.matrix])
        taboo = self.taboo()
        if taboo is None:
            return False
        if num_edges > 9:
            delta = num_edges - 9
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix)):
                    if (i, j) in taboo:
                        continue
                    else:
                        if self.matrix[i][j] == 1:
                            delta -= 1
                            self.matrix[i][j] = 0.0
                            if delta == 0:
                                break
        return True

    def validate(self):
        """
        Check if a solution is valid for the NAS-BENCH-101
        @return: True if valid, else False
        """
        return self.purge()

    def compile_model(self):
        """
        Compiles the current solution into a pytorch model
        """
        self.matrix = np.asarray(self.matrix, dtype=int).tolist()
        spec = ModelSpec(self.matrix, self.labels, data_format=GlobalInformation.Params.config['data_format'])
        self.model = Network(spec)

    def to_keras(self):
        """
        Converts the current solution into a Keras model
        @return: Keras model of the solution
        """
        import tensorflow as tf
        from nasbench_keras import ModelSpec, build_keras_model
        spec = ModelSpec(self.matrix, self.labels, data_format=GlobalInformation.Params.config['data_format'])
        features = tf.keras.Input(GlobalInformation.Params.input_shape,3)
        net_outputs = build_keras_model(spec, features, self.labels, GlobalInformation.Params.config)
        return tf.keras.Model(inputs=features, outputs=net_outputs)


class SolutionBaseNASBENCH201:
    def __init__(self, matrix=None):
        if matrix:
            self.matrix = matrix
        else:
            index = GlobalInformation.Params.api.random()
            string = GlobalInformation.Params.api[index]
            self.matrix = GlobalInformation.Params.api.str2matrix(string).tolist()
        self.model = None

    def print_model(self):
        """
        print the current solution
        """
        print(np.asarray(self.matrix))

    def get_string_arch(self):
        """
        Generates the arch string of the current solution
        @return: Arch string of the current solution
        """
        arch = ''
        for i in range(1, len(self.matrix)):
            arch += '|'
            for j in range(0, i):
                op = self.matrix[i][j]
                target = str(j)
                node = None
                if op == 0:
                    node = 'none~'
                elif op == 1:
                    node = 'skip_connect~'
                elif op == 2:
                    node = 'nor_conv_1x1~'
                elif op == 3:
                    node = 'nor_conv_3x3~'
                elif op == 4:
                    node = 'avg_pool_3x3~'
                arch += node + target + '|'
            arch += '+'

        return arch[:-1]

    def get_final_accuracy(self):
        """
        @return: get final test accuracy (after training) from the NAS-BENCH-101
        """
        string_arch = self.get_string_arch()
        index = GlobalInformation.Params.api.query_index_by_arch(string_arch)
        return GlobalInformation.Params.api.get_more_info(index, GlobalInformation.Params.dataset, None, hp='200',
                                                          is_random=False)[
            'test-accuracy']

    def validate(self):
        """
        Check if  a solution is valid for the NAS-BENCH-201
        @return: True if valid, else False
       """
        string_arch = self.get_string_arch()
        index = GlobalInformation.Params.api.query_index_by_arch(string_arch)
        return not index == -1

    def compile_model(self):
        """
        Compiles the current solution into a pytorch model
        """
        string_arch = self.get_string_arch()
        index = GlobalInformation.Params.api.query_index_by_arch(string_arch)
        config = GlobalInformation.Params.api.get_net_config(index, GlobalInformation.Params.dataset)
        self.model = get_cell_based_tiny_net(config)


class SolutionBase(SolutionBaseNASBENCH101 if GlobalInformation.Params.is_nas_bench_101 else SolutionBaseNASBENCH201):
    """Implementation of the base solution"""

    def __init__(self, **kwargs):
        super(SolutionBase, self).__init__(**kwargs)
        self.evaluated = False
        self.is_nas_bench_101 = GlobalInformation.Params.is_nas_bench_101
        self.fitness = None

    def clone(self):
        """
        @return: a clone of the current solution
        """
        import copy
        s = copy.deepcopy(self)
        return s

    def evaluate(self):
        """
        Evaluates the current solution
        """
        if not self.evaluated:
            self.compile_model()
            self.fitness = get_ICD_score_torch_gpu(self.model, GlobalInformation.Params.eval_data)
            self.fitness = int(self.fitness)
            self.evaluated = True

    def set_fitness(self, fit):
        """
        Sets the fitness of the solution
        @param fit: fitness of the solution
        """
        self.fitness = fit

    def is_better(self, solution):
        """
        Compares the current solution with another one
        :param solution: Solution to compare with
        :return: True if better, false otherwise
        """
        if solution:
            if max:
                return self.fitness > solution.fitness
            else:
                return self.fitness < solution.fitness
        else:
            return True


class SolutionFA(SolutionBase):
    def __init__(self, **kwargs):
        super(SolutionFA, self).__init__(**kwargs)

    def distance(self, solution_2):
        """
        Computes the distance from the current solution to another one
        @param solution_2: solution
        @return: distance
        """
        d = 0
        if self.is_nas_bench_101:
            N = len(self.encoding)
            N2 = len(solution_2.encoding)
            for i in range(N):
                if i >= N2:
                    d += math.pow(self.encoding[i], 2)
                else:
                    d += math.pow(self.encoding[i] - solution_2.encoding[i], 2)

        N = len(self.matrix)
        N2 = len(solution_2.matrix)
        for i in range(N):
            for j in range(N):
                if i >= N2 or j >= N2:
                    d += math.pow(self.matrix[i][j], 2)
                else:
                    d += math.pow(self.matrix[i][j] - solution_2.matrix[i][j], 2)
        return math.sqrt(d)

    def get_attractivity(self, solution_2, beta, gamma):
        """
        Computes the attractivity between the current solution and another one
        @param solution_2: solution
        @param beta: attractivity beta parameter
        @param gamma: attractivity gamma parameter
        @return: Attractivity value
        """
        dist = self.distance(solution_2)
        return beta * math.exp(-gamma * math.pow(dist, 2))

    def move_encode(self, index, solution_2, att, alpha):
        """
        Moves the an operator of the current solution to an operator of another solution
        @param index: index of the operator
        @param solution_2: solution
        @param att: attractivity
        @param alpha: moving alpha parameter
        @return: operator after moving
        """
        if index >= len(solution_2.encoding):
            return self.encoding[index]
        e = self.encoding[index] + (solution_2.encoding[index] - self.encoding[index]) * att + alpha * random.random()
        if e < 0:
            return 0
        elif e > 1:
            return 1
        else:
            return e

    @staticmethod
    def move_matrix_element(index, matrix, matrix2, att, alpha):
        """
        Moves a element of matrix1 to the element of matrix 2
        @param index: index of the element to move
        @param matrix: matrix
        @param matrix2: matrix
        @param att: attractivity
        @param alpha: alpha: moving alpha parameter
        @return: element of matrix1 after moving
        """
        if index >= len(matrix2):
            return matrix[index]
        r = round(matrix[index] + (matrix2[index] - matrix[index]) * att + alpha * random.random())
        if r < 0:
            return 0
        elif r > 1:
            return 1
        else:
            return r

    def move_matrix(self, index, solution_2, att, alpha):
        """
        Moves a line the current solution matrix to another solution's matrix line
        @param index: index of the line to move
        @param solution_2: solution
        @param att: attractivity
        @param alpha:  alpha: moving alpha parameter
        @return: line of the current solution's matrix after moving
        """
        return [self.move_matrix_element(i, self.matrix[index], solution_2.matrix[index], att, alpha) for i in
                range(len(self.matrix[index]))]

    def move_to(self, solution_2, alpha, beta, gamma):
        """
        Moves the current solution to another solution
        @param solution_2: solution
        @param alpha: moving alpha parameter
        @param beta: attractivity beta parameter
        @param gamma: attractivity gamma parameter
        @return: current solution after moving to solution_2
        """
        att = self.get_attractivity(solution_2, beta, gamma)

        if self.is_nas_bench_101:
            new_encoding = []
            for i in range(len(self.encoding)):
                new_encoding.append(self.move_encode(i, solution_2, att, alpha))
            self.encoding = new_encoding
            self.generate_labels()

        new_matrix = []
        for i in range(len(self.matrix)):
            if i >= len(solution_2.matrix):
                new_matrix.append(self.matrix[i])
            else:
                new_matrix.append(self.move_matrix(i, solution_2, att, alpha))
        self.matrix = new_matrix
        for i in range(len(new_matrix)):
            for j in range(i + 1):
                new_matrix[i][j] = 0

        old_matrix = self.matrix
        self.matrix = new_matrix
        if not self.validate():
            self.matrix = old_matrix


class SolutionIFA(SolutionFA):
    def __init__(self, **kwargs):
        super(SolutionIFA, self).__init__(**kwargs)

    @classmethod
    def cross(cls, solution_1, solution_2):
        """
        Computes the crossover between two solutions
        @param solution_1: parent solution 1
        @param solution_2: parent solution 2
        @return: offspring solution
        """
        import copy
        len_parent_1 = len(solution_1.matrix)
        len_parent_2 = len(solution_2.matrix)
        max_, min_ = max([len_parent_1, len_parent_2]), min([len_parent_1, len_parent_2])
        i = int(len_parent_1 / 2)
        new_matrix = copy.deepcopy(solution_1.matrix[:i])
        if len_parent_1 < max_:
            delta = max_ - len_parent_1
            for j in range(i):
                for k in range(delta):
                    new_matrix[j].insert(0, 0)
        new_matrix.extend(copy.deepcopy(solution_2.matrix[-(max_ - i):]))
        if len_parent_2 < max_:
            delta = max_ - len_parent_2
            for j in range(-(max_ - i), 0):
                for k in range(delta):
                    new_matrix[j].insert(0, 0)
        off = SolutionIFA(matrix=new_matrix, encoding=solution_1.encoding)
        if off.validate():
            new_encoding = [x for x in solution_1.encoding[:i]]
            new_encoding.extend([x for x in solution_2.encoding[-(max_ - i - 2):]])
        else:
            new_matrix = copy.deepcopy(solution_1.matrix)
            new_encoding = [x for x in solution_1.encoding[:i]]
            new_encoding.extend([x for x in solution_2.encoding[-(len_parent_1 - i - 2):]])
        off = SolutionIFA(matrix=new_matrix, encoding=new_encoding)
        off.generate_labels()
        return off

    def mutate(self):
        """
        Mutates the current solution, by randomly changing the label of an operator
        """
        i = random.randint(0, len(self.encoding) - 1)
        self.encoding[i] = random.random()
        self.generate_labels()

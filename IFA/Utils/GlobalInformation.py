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

"""
This file is used to store global parameters
"""


class Information:
    def __init__(self, input_shape, nb_classes, eval_data, search_space='nas_bench_101'):
        """
        @param input_shape: data input shape
        @param nb_classes: number of classes
        @param eval_data: Dataset (X,Y)
        @param search_space: 'nas_bench_101' or 'nas_bench_201'
        """
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.eval_data = eval_data
        if search_space == 'nas_bench_101':
            self.nasbench, self.nas_bench_101_iterator = self.load_nas_bench_101()
            self.config = None
            self.is_nas_bench_101 = True
        else:
            self.api = self.load_nas_bench_201()
            self.dataset = 'cifar10'
            self.is_nas_bench_101 = False

    def set_config(self, config):
        """
        Sets the configuration for the NAS-BENCH-101
        @param config: configuration for NAS-BENCH-101
        """
        self.config = config

    @staticmethod
    def load_nas_bench_101():
        """
        Loads the NAS-BENCH-101 Api
        @return:
        """
        from nasbench import api
        print('==========================')
        print('LOADING NAS')
        nasbench = api.NASBench('api/nasbench_only108.tfrecord')
        nas_bench_101_iterator = list(nasbench.hash_iterator())
        print('DONE')
        print('==========================')
        return nasbench, nas_bench_101_iterator

    @staticmethod
    def load_nas_bench_201():
        """
        Loads the NAS-BENCH-201 Api
        @return:
        """
        print('==========================')
        print('LOADING NAS')
        from nas_201_api.api_201 import NASBench201API as API
        api = API('api/NAS-Bench-201-v1_1-096897.pth', verbose=False)
        print('DONE')
        print('==========================')
        return api


Params = None

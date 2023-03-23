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

from IFA.Utils.GlobalInformation import Information
from IFA.Utils import GlobalInformation
from IFA.Search import Search_FA, Search_IFA
import torch
from datetime import date
import os
import argparse


def load_data(path='data/samples.pt'):
    return torch.load(path)


parser = argparse.ArgumentParser(description='Improved FireFly Algorithm')
parser.add_argument('--data_path', default='data/samples.pt', type=str, help='dataset file, default is data/samples.pt')
parser.add_argument('--input_shape', default=(3, 32, 32),  type=int,nargs='+', help='input shape, default is (3,32,32)')
parser.add_argument('--num_labels', default=10, type=int, help='number of labels, default is 10')
parser.add_argument('--result_path', default='results', type=str, help='output folder, default is ./results')
parser.add_argument('--search_technique', default='IFA', type=str,
                    help='search technique to use (IFA or FA), default is IFA')
parser.add_argument('--search_name', default=None, type=str,
                    help='search name, default is the current date (YYYY-MM-DD)')
parser.add_argument('--to_keras', default=False, type=str,
                    help='saves the result as a Keras model (for NAS-BENCH-101 only), default is False')
parser.add_argument('--pop_size', default=20, type=int, help='population size, default is 20')
parser.add_argument('--num_gen', default=100, type=int,
                    help='number of generations (search iterations), default is 100')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='value of the alpha parameter used by the FireFly Algorithm, default is 0.5')
parser.add_argument('--beta', default=0.95, type=float,
                    help='value of the beta parameter used by the FireFly Algorithm, default is 0.95')
parser.add_argument('--gamma', default=0.15, type=float,
                    help='value of the gamma parameter used by the FireFly Algorithm, default is 0.15')
parser.add_argument('--prob_mut', default=1.0, type=float,
                    help='mutation probability used by the Genetic Algorithm, default is 1.0')
parser.add_argument('--chances', default=5, type=float,
                    help='mutation probability used by the Genetic Algorithm, default is 5')
parser.add_argument('--selection', default='random', type=str,
                    help='parent solution selection mode (random, fit or rank) for the Genetic Algorithm, default is '
                         'random')
parser.add_argument('--force_restart', default='True', type=str,
                    help='True to force a new search, False to continue if possible, default is True')
parser.add_argument('--stem_filter_size', default=16, type=int, help='Stem filter size, default is 16')
parser.add_argument('--data_format', default='channels_first', type=str,
                    help='Data format (channels_first or channels_last), default is channels_first')
parser.add_argument('--num_stacks', default=3, type=int, help='Number of stacks in the architectures, default is 3')
parser.add_argument('--num_modules_per_stack', default=3, type=int,
                    help='Number of modules per stacks in the architectures, default is 3')
parser.add_argument('--search_space', default='nas_bench_101', type=str,
                    help='NAS Search space (nas_bench_101 or nas_bench_201), default is nas_bench_101')


def get_search_config(arguments):
    return {
        'pop_size': arguments.pop_size,
        'alpha': arguments.alpha,
        'beta': arguments.beta,
        'gamma': arguments.gamma,
        'prob_mut': arguments.prob_mut,
        'chances': arguments.chances,
        'selection': arguments.selection,
        'num_gen': arguments.num_gen,
        'force_restart': arguments.force_restart.lower() == 'true',
        'stem_filter_size': arguments.stem_filter_size,
        'data_format': arguments.data_format,
        'num_stacks': arguments.num_stacks,
        'num_modules_per_stack': arguments.num_modules_per_stack,
        'num_labels': arguments.num_labels,
        'search_space': arguments.search_space
    }


def main(data_path, input_shape, nb_classes, result_path, search_technique, execution_name, to_keras, search_config):
    GlobalInformation.Params = Information(input_shape=input_shape, nb_classes=nb_classes,
                                           eval_data=load_data(data_path), search_space=search_config['search_space'])

    if search_config['search_space'] == 'nas_bench_101':
        config = {'available_ops': ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
                  'stem_filter_size': search_config['stem_filter_size'],
                  'data_format': search_config['data_format'],
                  'num_stacks': search_config['num_stacks'],
                  'num_modules_per_stack': search_config['num_modules_per_stack'],
                  'num_labels': nb_classes}
        GlobalInformation.Params.set_config(config)

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if execution_name is None:
        execution_name = '{}_{}_{}'.format(search_technique, search_config['search_space'], date.today())
    print(execution_name)

    if search_technique == 'FA':
        search = Search_FA(pop_size=search_config['pop_size'], alpha=search_config['alpha'], beta=search_config['beta'],
                           gamma=search_config['gamma'])
    else:
        search = Search_IFA(pop_size=search_config['pop_size'], alpha=search_config['alpha'],
                            beta=search_config['beta'],
                            gamma=search_config['gamma'], prob_mut=search_config['prob_mut'],
                            chances=search_config['chances'], selection=search_config['selection'])
    best_solution = search.run(num_gen=search_config['num_gen'], path=result_path + '/save_' + execution_name,
                               path_best=result_path + '/best_' + execution_name,
                               force_restart=search_config['force_restart'])
    print(best_solution.get_final_accuracy())

    if to_keras and search_config['search_space'] == 'nas_bench_101':
        model = best_solution.to_keras()
        from keras.models import save_model
        save_model(model, '{}/best_{}_keras.h5'.format(result_path, execution_name))


if __name__ == "__main__":
    args = parser.parse_args()
    search_configuration = get_search_config(args)
    main(data_path=args.data_path, input_shape=tuple(args.input_shape), nb_classes=args.num_labels,
         result_path=args.result_path, search_technique=args.search_technique, execution_name=args.search_name,
         to_keras=args.to_keras, search_config=search_configuration)

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


def get_ICD_score_torch_gpu(network, sub_data):
    """
    Computes the ICD score of a neural network
    @param network: neural network to be scored
    @param sub_data: subset of data
    @return: ICD score
    """
    import torch
    sums = []

    def counting_forward_hook(module, inp, out):
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = inp.clone().detach()
            x = torch.where(x > 0, 1.0, 0.0)
            center = x.mean(0)
            center = center.repeat(100, 1)
            diff = torch.pow(x - center, 2)
            sum_ = torch.sum(diff, 1).cpu().numpy()
            sums.append(sum_)
        except:
            pass

    hooks = []
    for name, module in network.named_modules():
        hooks.append(module.register_forward_hook(counting_forward_hook))
    network(sub_data)
    mat = np.asarray(sums)
    mat = mat.sum(0)
    mat = np.sqrt(mat)
    score = np.mean(mat)
    for hook in hooks:
        hook.remove()
    return score

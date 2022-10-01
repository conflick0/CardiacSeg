#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import numpy as np


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class CoTrLoss(MultipleOutputLoss2):
    '''
    the loss for CoTr deep supervision
    source code: CoTr_package/CoTr/training/network_training/nnUNetTrainerV2_ResTrans.py#72
    '''
    def __init__(self, loss):
      # we need to know the number of outputs of the network
      net_numpool = 4

      # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
      # this gives higher resolution outputs more weight in the loss
      weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

      # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
      mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
      weights[~mask] = 0
      weights = weights / weights.sum()
      ds_loss_weights = weights
      # now wrap the loss
      super().__init__(loss, ds_loss_weights)



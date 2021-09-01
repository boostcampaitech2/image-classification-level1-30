# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn

from sklearn.metrics import f1_score

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    
    # Metric
    def f1_loss(self, y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2
        

        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(dim=1)
                
        f1 = f1_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')

        return f1

    def loss_fn(self, target, x):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        train_metric = self.f1_loss(target, x)
        return loss.mean(), train_metric
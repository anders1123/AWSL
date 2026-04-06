from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch
import torch.nn.functional as F

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg
from collections import OrderedDict


@register.ood_alg_register
class AWSL(BaseOODAlg):


    def __init__(self, config: Union[CommonArgs, Munch]):
        super(AWSL, self).__init__(config)
        self.targets = None
        self.node_repr = None
        self.att = None
        self.hard_node_mask = None
        self.soft_node_mask = None
        self.hard_edge_mask = None
        self.soft_edge_mask = None
        self.node_logit_rationale = None

        self.coef_task = config.ood.extra_param[0]
        self.coef_conn = config.ood.extra_param[1]
        # self.coef_size = config.ood.extra_param[2]
        self.coef_purturb = config.ood.extra_param[4]


    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        self.soft_edge_mask, self.hard_edge_mask, self.logit_pre, self.logit_pre_dropped, self.logit_pre_added = model_output
        return self.logit_pre


    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        # dealing with imbalanced labels
        if config.dataset.dataset_name in ['GOODHIV']:
            weight = 25.0
        elif config.dataset.dataset_name in ['GOODEC50', 'GOODIC50']:
            weight = 1/18.0
        else:
            weight = None

        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss based on GSAT algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on DIR algorithm

        """

        self.spec_loss = OrderedDict()

        self.spec_loss['CONN'] = self.coef_conn * get_conn_loss(self.soft_edge_mask)
        # self.spec_loss['SIZE'] = self.coef_size * get_size_loss(self.soft_edge_mask)

        prob_pre = torch.sigmoid(self.logit_pre)
        prob_pre_added = torch.sigmoid(self.logit_pre_added)
        prob_pre_dropped = torch.sigmoid(self.logit_pre_dropped)
        
        dist_pre = torch.cat([prob_pre, 1 - prob_pre], dim=-1)
        dist_pre_added = torch.cat([prob_pre_added, 1 - prob_pre_added], dim=-1)
        dist_pre_dropped = torch.cat([prob_pre_dropped, 1 - prob_pre_dropped], dim=-1)

        self.spec_loss['ADD'] = self.coef_purturb * dis_add
        self.spec_loss['DEL'] = 0.0 - self.coef_purturb * dis_drop

        self.mean_loss = loss.sum()/mask.sum()
        loss = self.mean_loss + sum(self.spec_loss.values())
        return loss
    
def get_size_loss(edge_mask):
    """
    Compute the size loss in PGExp.
    """
    return edge_mask.sum()/edge_mask.size(0)

def get_conn_loss(edge_mask):
    """
    Compute the connectivity loss in PGExp.
    """
    edge_mask = edge_mask * 0.99 + 0.005
    mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
    
    return torch.mean(mask_ent)





import munch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected, subgraph
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from .Pooling import GlobalMeanPool
from munch import munchify
from .MolEncoders import AtomEncoder, BondEncoder
from GOOD.utils.fast_pytorch_kmeans import KMeans


@register.model_register
class AWSLGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(AWSLGIN, self).__init__(config)

        # --- if environment inference ---
        config.environment_inference = False
        if config.environment_inference:
            self.env_infer_warning = f'#W#Expermental mode: environment inference phase.'
            config.dataset.num_envs = 3
        # --- Test environment inference ---

        self.config = config

        self.learn_edge_att = True
        self.top_ratio = config.ood.extra_param[2]
        self.without_embed = config.ood.extra_param[3]


        fe_kwargs = {'without_embed': True if self.without_embed else False}

        # --- Build networks ---
        self.feature_mlp = EFMLP(config, bn=True)
        self.generator = GINFeatExtractor(config, **fe_kwargs)
        #self.mlp_gen=nn.Linear(config.model.dim_hidden,2)
        #self.mlp_gen=MLP([config.model.dim_hidden,2], dropout=config.model.dropout_rate,
        #                                 config=config, bn=True)
        self.mlp_gen = ExtractorMLP(config)
        self.predictor = GINFeatExtractor(config, **fe_kwargs)
        #**kwargs – without_readout will output node features instead of graph features.

        self.pool = GlobalMeanPool()
        self.classifier = Classifier(config)
        self.edge_mask = None



    def forward(self, *args, **kwargs):
        r"""
        The LECIGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        if self.without_embed:
            data.x = self.feature_mlp(data.x, data.batch)
            kwargs['data'] = data
        node_repr_gen = self.generator.get_node_repr(*args, **kwargs)

        att = self.mlp_gen(node_repr_gen, data.edge_index, data.batch)
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
        soft_edge_att = torch.sigmoid(edge_att)

        # convert soft edge mask to hard
        hard_edge_att = F.gumbel_softmax(soft_edge_att, tau=1, hard=True)

        # control sparsity
        cau_edge_att = control_sparsity(soft_edge_att, top_t = self.top_ratio)
        cau_edge_att_dropped = control_sparsity(soft_edge_att, top_t = self.top_ratio - data.num_graphs/data.num_edges)
        cau_edge_att_added = control_sparsity(soft_edge_att, top_t = self.top_ratio + data.num_graphs/data.num_edges)

        set_masks(cau_edge_att, self.predictor)
        repr_pre = self.predictor(*args, **kwargs)
        clear_masks(self)

        set_masks(cau_edge_att_dropped, self.predictor)
        repr_pre_dropped = self.predictor(*args, **kwargs)
        clear_masks(self)

        set_masks(cau_edge_att_added, self.predictor)
        repr_pre_added = self.predictor(*args, **kwargs)
        clear_masks(self)

        logit_rationale = self.classifier(repr_pre)
        logit_rationale_dropped = self.classifier(repr_pre_dropped)
        logit_rationale_added = self.classifier(repr_pre_added)

        if torch.isnan(logit_rationale).any() or torch.isnan(logit_rationale_dropped).any() or torch.isnan(logit_rationale_added).any():
            raise ValueError('NaN detected!')

        return cau_edge_att, hard_edge_att, logit_rationale, logit_rationale_dropped, logit_rationale_added

    
    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


@register.model_register
class AWSLvGIN(AWSLGIN):
    r"""
    The GIN virtual node version of LECI.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(AWSLvGIN, self).__init__(config)
        fe_kwargs = {'without_embed': True if self.without_embed else False}
        self.generator = vGINFeatExtractor(config, **fe_kwargs)
        self.predictor = vGINFeatExtractor(config, **fe_kwargs)


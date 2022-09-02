import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgePredictor

from model.rgcn.layers import LinearAffine

class MLP_edge_predictor(nn.Module):
    def __init__(self, in_feats, h_feats, layer_num = 2):
        super(MLP_edge_predictor,self).__init__()
        self.activate = F.relu
        self.in_linear = EdgePredictor('cat', in_feats, h_feats)
        self.hidden_linears = nn.ModuleList()
        for i in range(layer_num-2):
            self.hidden_linears = self.hidden_linears.append(nn.Linear(h_feats,h_feats))
        self.out_linear = nn.Linear(h_feats,1)
    def forward(self,src_feats,dst_feats):
        # t = self.activate(self.in_linear(src_feats,dst_feats))
        t = self.in_linear(src_feats,dst_feats)
        for layer in self.hidden_linears:
            t = self.activate(layer(t))
        return self.out_linear(t)
class SingleRelationDismult(nn.Module):
    def __init__(self,feat_num = 500):
        super(SingleRelationDismult,self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, feat_num))
        nn.init.normal_(self.weights,0,1)
    def forward(self,s,o):
        return torch.sum(s * self.weights * o, dim=1)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feat = 500, layer_num = 1):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList()# gcn layers
        for i in range(layer_num-2):
            self.layers = self.layers.append(GraphConv(h_feats, out_feat,activation = F.relu))
        if layer_num >=2: # add the outlayer
            self.layers = self.layers.append(GraphConv(h_feats, out_feat,activation = None))
        if layer_num > 1:
            self.in_layer = GraphConv(in_feats, h_feats, activation = F.relu)
        else:
            self.in_layer = GraphConv(in_feats, out_feat, activation = None)

    def forward(self, g, in_feat,use_weighted_edge):
        g = dgl.add_self_loop(g)
        #deal with the weights
        edge_weight = None
        if use_weighted_edge:
            edge_weight = g.edata['weight']
        h = self.in_layer(g,in_feat,edge_weight = edge_weight)
        for layer in self.layers:
            h = layer(g, h,edge_weight=edge_weight)
        return h

# the final model to get any pair of link weight value from a message graph
# initialize: the features of the whole graph's nodes; and other model parameters
# train input: the sub message graph for this batch train, the corresponding nids, the query triplets
# train output: the predicted weight values of the query triplets; the embeddings for all nodes in the graph
class Encoder(nn.Module):
    def __init__(self, features, GCN_in_dim = 500, gcn_dim=500, h_mlp_dim = 500, use_pre_linearAffine =True, use_MLP_edgepredictor = True):
        super(Encoder, self).__init__()
        self.use_pre_linearAffine = use_pre_linearAffine
        self.linearAffine = LinearAffine(features.shape[1],GCN_in_dim)# to replace cnn
        self.gcn = GCN(GCN_in_dim,gcn_dim , gcn_dim)
        if not use_MLP_edgepredictor:
            self.predictor = EdgePredictor('cat', gcn_dim, out_feats=1)
        else:
            self.predictor = MLP_edge_predictor(gcn_dim,h_mlp_dim)
        # self.predictor = SingleRelationDismult()

    def forward(self, g,triplets,use_weighted_edge):
        if self.use_pre_linearAffine:
            input_features = self.linearAffine(g.ndata['feats'].float())
        else:
            input_features = g.ndata['feats']
        embedding = self.gcn(g,input_features,use_weighted_edge)
        # edge predict
        s = embedding[list(triplets[:,0])]
        o = embedding[list(triplets[:,2])]
        # weights = torch.sum(s * self.w_relation * o, dim=1)
        weights = self.predictor(s,o)
        return weights,embedding

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))+ torch.mean(self.predictor.weights.pow(2))

    def get_loss(self, weights, labels,embedding):
        predict_loss = F.mse_loss(weights,labels)
        # reg_loss = self.regularization_loss(embedding)
        return predict_loss #+ 0.01*reg_loss

    def output_weights(self,writter,epoch):
        pass
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rgcn.layers import LinearAffine,My_RelGraphConv


# an RGCN model with 2 RGCN layers (the encoding part without the initial embedding)
# initialize: the features of the whole graph's nodes, the h_dim of the node features, the out_dim of the node features, and the number of relations
# train input: the sub message graph for this batch train, the corresponding nids(if none, then all in)
# train output: the node embeddings of the sub_message graph's nodes
class RGCN(nn.Module):
    def __init__(self, rgcn_in_dim, h_dim, out_dim, num_rels,
                 regularizer=None, num_bases=-1, dropout=0.):
        super(RGCN, self).__init__()

        if num_bases == -1:
            num_bases = num_rels
        self.conv1 = My_RelGraphConv(rgcn_in_dim, h_dim, num_rels, regularizer,
                                  num_bases,self_loop_dropout=dropout)

    def forward(self, features, g,use_dis_weight = False,drop_out = False):
        x = features
        h = self.conv1(g, x, g.edata['my_type'], use_dis_weight,drop_out)
        return h

# the final model to get any pair of link weight value from a message graph
# initialize: the features of the whole graph's nodes; and other model parameters
# train input: the sub message graph for this batch train, the corresponding nids, the query triplets
# train output: the predicted weight values of the query triplets; the embeddings for all nodes in the graph
class LinkPredict(nn.Module):
    def __init__(self, features,  num_rels,rgcn_in_dim=500, h_dim=500, num_bases=100, dropout=0.2, reg_param=0.01):
        super(LinkPredict, self).__init__()
        self.features = features
        self.linearAffine = LinearAffine(features.shape[1],rgcn_in_dim)
        self.rgcn = RGCN(rgcn_in_dim, h_dim, h_dim, num_rels, regularizer=None, dropout=dropout)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))#the weight for the decoder
        nn.init.normal_(self.w_relation,0,1)

    def forward(self, g,triplets,use_dis_weight = False,drop_out = False,features = None):
        if g is not None:
            input_features = self.linearAffine(g.ndata['feats'].float())
        else:
            input_features = self.linearAffine(features.float())
        embedding = input_features
        if g is not None:
            embedding = self.rgcn(input_features,g,use_dis_weight,drop_out)
        # DistMult
        s = embedding[list(triplets[:,0])]
        r = self.w_relation[list(triplets[:,1])]
        o = embedding[list(triplets[:,2])]
        weights = torch.sum(s * r * o, dim=1)
        return weights,embedding


    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))+ torch.mean(self.w_relation.pow(2))

    def get_loss(self, weights, labels,embedding):
        predict_loss = F.mse_loss(weights,labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss

    def output_weights(self,writter,epoch):
        writter.add_histogram('forward_weight/weights',self.rgcn.conv1.linear_r_forward.W,epoch+1)
        writter.add_histogram('backward_weight/weights',self.rgcn.conv1.linear_r_backward.W,epoch+1)
        writter.add_histogram('self_weight/weights',self.rgcn.conv1.loop_weight,epoch+1)
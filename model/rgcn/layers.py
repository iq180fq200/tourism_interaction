import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import TypedLinear
import torch as th
import dgl.function as fn
class LinearAffine(nn.Module):
    def __init__(self,indim,outdim):
        super(LinearAffine,self).__init__()
        self.linear = nn.Linear(indim,outdim)
    def forward(self,x):
        return F.relu(self.linear(x))


class My_RelGraphConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer=None,
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop_dropout=0.0):
        super().__init__()
        self.linear_r_forward = TypedLinear(in_feat, out_feat, num_rels, regularizer, num_bases)#forward weight
        self.linear_r_backward = TypedLinear(in_feat,out_feat,num_rels,regularizer,num_bases)
        self.bias = bias
        self.activation = activation

        # bias
        if self.bias:
            self.forward_bias = nn.Parameter(th.Tensor(out_feat))
            self.backward_bias = nn.Parameter(th.Tensor(out_feat))
            self.self_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.forward_bias)
            nn.init.zeros_(self.backward_bias)
            nn.init.zeros_(self.self_bias)

        # weight for self loop
        self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
        nn.init.normal_(self.loop_weight,0,1)

        self.dropout = nn.Dropout(self_loop_dropout)

    def forward_message(self, edges):
        """Message function."""
        m = self.linear_r_forward(edges.src['h'], edges.data['etype'])
        if self.bias:
            m+=self.forward_bias
        if 'norm' in edges.data:
            m = m * (edges.data['norm'].reshape(-1,1))
        return {'m' : m}

    def backward_message(self, edges):
        m = self.linear_r_backward(edges.src['h'], edges.data['etype'])
        if self.bias:
            m+=self.backward_bias
        if 'norm' in edges.data:
            m = m * (edges.data['norm'].reshape(-1,1))
        return {'m' : m}

    def forward(self, g, feat, etypes, use_dis_weight,drop_out):
        with g.local_scope():
            g.ndata['h'] = feat
            #forward graph
            if not use_dis_weight:
                g.edata['norm'] = dgl.norm_by_dst(g).unsqueeze(-1)# only suitable when there is only one relation
            else:
                #define a useless message function to tailor the grammer of dgl library
                def message_func(edges):
                    return None
                reduce_func = dgl.function.sum('weight', 'sum_weight')
                g.update_all(message_func,reduce_func)
                g.apply_edges(lambda edges: {'norm' : edges.data['weight']/edges.dst['sum_weight']})
            # g.edata['etype'] = torch.tensor([0]*g.num_edges(),device='cuda')
            g.edata['etype'] = etypes
            # message passing
            g.update_all(self.forward_message, fn.sum('m', 'm_forward'))

            #construct backward graph
            g_back = dgl.reverse(g,copy_edata=True)
            if not use_dis_weight:
                g_back.edata['norm'] = dgl.norm_by_dst(g_back).unsqueeze(-1)# only suitable when there is only one relation
            else:
                #define a useless message function to tailor the grammer of dgl library
                def message_func(edges):
                    return None
                reduce_func = dgl.function.sum('weight', 'sum_weight')
                g_back.update_all(message_func,reduce_func)
                g_back.apply_edges(lambda edges: {'norm' : edges.data['weight']/edges.dst['sum_weight']})

            g_back.edata['etype'] = etypes
            g_back.update_all(self.backward_message, fn.sum('m', 'm_backward'))

            #self loop
            if drop_out:
                self_loop_message = self.dropout(feat @ self.loop_weight) #no bias or activation here
            else:
                self_loop_message = feat @ self.loop_weight

            if self.bias:
                self_loop_message +=self_loop_message+self.self_bias


            return self_loop_message + g.ndata['m_forward'] + g_back.ndata['m_backward']
            # return g.ndata['m_forward']

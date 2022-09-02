import dgl
import numpy as np
import math
import random
import torch.nn.functional as F
import torch
import numpy
import sklearn
class NegativeSampler:
    def __init__(self, num_nodes,all_triplets, negative_sample_rate=0.25):
        self.negative_sample_rate = negative_sample_rate
        self.n_entities = num_nodes
        self.objs={}
        self.subs={}
        for triplet in all_triplets:
            if triplet[0] not in self.objs:
                self.objs[triplet[0]] = []
            self.objs[triplet[0]].append((triplet[1], triplet[2]))
            if triplet[2] not in self.subs:
                self.subs[triplet[2]] = []
            self.subs[triplet[2]].append((triplet[1], triplet[0]))


    def sample(self, triplets, labels):
        size_of_batch = len(triplets)
        number_to_generate = int(size_of_batch * self.negative_sample_rate)

        new_labels = np.zeros(number_to_generate+size_of_batch).astype(np.uint16)
        new_labels[:size_of_batch] = labels

        #copy the triplets to the right number first to generate the new triplets
        xs, zs = math.modf(self.negative_sample_rate)
        rand_num = int(size_of_batch * xs)
        idx = random.sample(range(0, size_of_batch), rand_num) # random choice, can be sequential choice (to be test)
        if int(zs) >= 1:
            new_indexes = triplets.copy()
            for i in range(int(zs)):
                new_indexes = np.concatenate((new_indexes, triplets), axis=0)
            new_indexes = np.concatenate((new_indexes, triplets[idx]), axis=0).astype(np.uint16)
        else:
            new_indexes = np.concatenate((triplets, triplets[idx]), axis=0).astype(np.uint16)

        choices = np.random.binomial(1, 0.5, number_to_generate)
        # get the negative samples
        for i in range(number_to_generate):
            index = i + size_of_batch
            if choices[i]:
                new_indexes[index, 2] = random.randint(0, self.n_entities-1)
                while (new_indexes[index][1], new_indexes[index][2]) in self.objs[new_indexes[index][0]]:
                    new_indexes[index, 2] = random.randint(0, self.n_entities-1)
            else:
                new_indexes[index, 0] = random.randint(0, self.n_entities-1)
                while (new_indexes[index][1], new_indexes[index][0]) in self.subs[new_indexes[index][2]]:
                    new_indexes[index, 0] = random.randint(0, self.n_entities-1)

        return new_indexes[:, :3], new_labels


# return an iterator to :
#1. get the partial message graph after edge dropout
#2. get the query with negative sampling
def get_epoch_data(all_query,all_labels,m_graph,nega_sampler,use_dist_weight = False,drop_out = False):
    # edges,labels = nega_sampler.sample(all_query,all_labels)
    edges = all_query
    labels = all_labels
    if m_graph is None:
        sub_g = None
    else:
    #drop half of the edges as dropout
    # Use only half of the positive edges
        if drop_out:
            total_num_edges = m_graph.num_edges()
            num_nodes = m_graph.num_nodes()
            chosen_ids = np.random.choice(np.arange(total_num_edges),
                                          size=int(total_num_edges / 2),
                                          replace=False)
            src,dst = m_graph.find_edges(chosen_ids)
            sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
            if use_dist_weight:
                sub_g.edata['weight'] = m_graph.edata['weight'][chosen_ids]
        else:
            sub_g = m_graph

        #cppy the datas
        sub_g.ndata['feats'] = m_graph.ndata['feats'] #copy the node features
        sub_g.edata['my_type'] = torch.tensor(np.array([0]*sub_g.num_edges()),dtype=torch.long)
    return sub_g, edges.astype(np.long), labels.astype(np.long) #-1 means all the nodes in the original graph


def calc_metrics(predict_flow,real_flow):
    #get mse
    mse = F.mse_loss(predict_flow,real_flow)

    #get mape
    pre_flow = predict_flow.cpu().detach().numpy()
    real_flow = real_flow.cpu().detach().numpy()
    error_percent = np.abs(pre_flow - real_flow)/real_flow
    mape = np.average(error_percent)
    return mse,mape

def calc_metrics_numpy(predict_flow,real_flow):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(real_flow,predict_flow)
    error_percent = np.abs(predict_flow - real_flow)/real_flow
    mape = np.average(error_percent)
    return mse,mape
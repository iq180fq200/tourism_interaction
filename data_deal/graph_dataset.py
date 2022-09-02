import scipy.sparse as sp
import dgl
import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#add node features
#['s_ranking','s_lon','s_lat','s_adname_id','s_cost','s_type_id','s_comment','s_area','s_level','s_mean_time','e_ranking','e_lon','e_lat','e_adname_id','e_cost','e_type_id','e_comment','e_area','e_level','e_mean_time','distance']
def normalize_features(raw_features,feat_names):
    #devide feature names
    std_fnames = []
    minmax_fnames = []
    for item in feat_names:
        if item in ['s_lon','s_lat','s_adname_id','s_type_id','s_ranking','s_level','distance','e_lon','e_lat','e_adname_id','e_type_id','e_ranking','e_level']:
            minmax_fnames.append(item)
        else:
            std_fnames.append(item)
    try:
        std_feats =raw_features[std_fnames].to_numpy()
        std_feats = StandardScaler().fit_transform(std_feats)
    except:
        std_feats = None

    try:
        minmax_feats =raw_features[minmax_fnames].to_numpy()
        minmax_feats = MinMaxScaler().fit_transform(minmax_feats)
    except:
        minmax_feats = None

    if std_feats is None:
        return minmax_feats
    elif minmax_feats is None:
        return std_feats
    else:
        return np.hstack([minmax_feats,std_feats])

def add_node_features(graph,nodes_data,feat_names):
    # graph.ndata['feats'] = torch.tensor(normalize_features(nodes_data,feat_names),dtype = torch.float)
    # return graph



    feats = nodes_data[feat_names].to_numpy()
    #normalize
    std = MinMaxScaler()
    feature = std.fit_transform(feats)
    graph.ndata['feats'] = torch.tensor(feature,dtype=torch.float)
    return graph



def create_dataset(message_graph_type,threshord,use_weighted_edge,feat_names,dataset_path='./data'):
    #get message graph
    nodes_data = pd.read_csv(os.path.join(dataset_path, 'node_feature.csv'),delimiter = '\t')
    weight_data = pd.read_csv(os.path.join(dataset_path, 'inter_edge.csv'),delimiter = '\t',header = None)
    idx_train = np.genfromtxt(os.path.join(dataset_path, 'idx_train.csv')).astype(int)
    idx_valid = np.genfromtxt(os.path.join(dataset_path, 'idx_valid.csv')).astype(int)
    idx_test = np.genfromtxt(os.path.join(dataset_path, 'idx_test.csv')).astype(int)
    num_nodes = nodes_data.shape[0]
    if message_graph_type=='interaction':
        edge_data = weight_data
        src = edge_data.iloc[idx_train,0].to_numpy()
        dst = edge_data.iloc[idx_train,1].to_numpy()
        if threshord != -1:
            connected_edge_ids=[]
            edge_flow = edge_data.iloc[idx_train,2].to_numpy()
            for idx,item in enumerate(edge_flow):
                if item >threshord:
                    connected_edge_ids.append(idx)
            src = src[connected_edge_ids]
            dst = dst[connected_edge_ids]
        mgraph = dgl.graph((src,dst),num_nodes = num_nodes)
        #add the features to the message graph
        mgraph = add_node_features(mgraph,nodes_data,feat_names)
        mgraph.edata['my_type'] = torch.tensor(np.array([0]*mgraph.num_edges()),dtype=torch.long)#all the edges are of the same type
        if use_weighted_edge == True:
            if threshord != -1:
                mgraph.edata['weight'] = torch.tensor(np.array(edge_data.iloc[idx_train,2])[connected_edge_ids],dtype = torch.float32)
            else:
                mgraph.edata['weight'] = torch.tensor(np.array(edge_data.iloc[idx_train,2]),dtype = torch.float32)




    elif message_graph_type == 'distance':
        edge_data = pd.read_csv('./data/dist_edge.csv',delimiter = '\t',header=None)
        distance = edge_data.iloc[:,2].to_numpy()
        src = edge_data.iloc[:,0].to_numpy()
        dst = edge_data.iloc[:,1].to_numpy()
        if threshord != -1:
            connected_edge_ids=[]
            for i,dis in enumerate(distance):
                if dis < threshord and dis > 0:
                    connected_edge_ids.append(i)
            src = src[connected_edge_ids]
            dst = dst[connected_edge_ids]
        else:
            connected_edge_ids=[]
            for i,dis in enumerate(distance):
                if dis > 0:
                    connected_edge_ids.append(i)
            src = src[connected_edge_ids]
            dst = dst[connected_edge_ids]

        mgraph = dgl.graph((src,dst),num_nodes = num_nodes)
        mgraph = add_node_features(mgraph,nodes_data,feat_names)
        # std = StandardScaler()
        mgraph.edata['my_type'] = torch.tensor(np.array([0]*mgraph.num_edges()),dtype=torch.long)#all the edges are of the same
        if use_weighted_edge == True:
            mgraph.edata['weight'] = torch.tensor((1/distance)[connected_edge_ids],dtype=torch.float32)


    elif message_graph_type == 'mixture':
        pass
    else:
        mgraph = None

    ## get all the features
    if mgraph is not None:
        all_features = mgraph.ndata['feats']
        mgraph = dgl.add_reverse_edges(mgraph,copy_edata=True,copy_ndata=True)# get the reverse graph

    else:
        feats = nodes_data[feat_names].to_numpy()
        std = MinMaxScaler()
        feature = std.fit_transform(feats)
        all_features = torch.tensor(feature,dtype=torch.float)

    ##get all the triplets
    weight_data = np.insert(weight_data.to_numpy(),1,values=[0]*weight_data.shape[0],axis = 1)# this insert is for RGCN's relation
    train_triplets,train_labels = weight_data[idx_train,:3],weight_data[idx_train,3]
    valid_triplets,valid_labels = weight_data[idx_valid,:3],weight_data[idx_valid,3]
    test_triplets,test_labels = weight_data[idx_test,:3],weight_data[idx_test,3]

    triplets = [train_triplets,valid_triplets,test_triplets]
    labels = [train_labels,valid_labels,test_labels]


    return mgraph,all_features,triplets,labels






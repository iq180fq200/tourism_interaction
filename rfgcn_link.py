import argparse
import os
import pickle

import torch
import torch as th
import torch.nn as nn
from dglgo.utils.early_stop import EarlyStopping
import numpy as np
from common.output_results import output_flow, output_model_information
from common.link_utils import NegativeSampler, get_epoch_data, calc_metrics, calc_metrics_numpy
from data_deal.other_dataset import get_distances_all as get_distances
from model.common_components.rf import train_rf_regressor
from model.rfgcn import models
from data_deal.graph_dataset import create_dataset
from torch.utils.tensorboard import SummaryWriter

def main(args):
    # message graphs(without trop out), features, all query_triplets(train,valid,test), corresponding labels
    m_graph,all_features,query_triplets,labels = create_dataset(args.massage_graph_type,args.threshold,args.use_weighted_edge,feat_names=str(args.feature_names).split('@'))
    if m_graph is not None:
        num_nodes = m_graph.num_nodes()
    else:
        num_nodes = len(all_features)
    train_query = query_triplets[0]
    valid_query = query_triplets[1]
    test_query = query_triplets[2]
    train_label,valid_label,test_label = labels[0],labels[1],labels[2]

    train_distance,valid_distance,test_distance = get_distances()


    #all the node features are stored in linkpredict
    encoder = models.Encoder(all_features)
    optimizer = th.optim.Adam(encoder.parameters(), lr=1e-2)

    model_state_file = str(args.model_file).split('@')
    if not os.path.exists('rfgcn_models/'):
        os.makedirs('rfgcn_models/')
    encoder_state_file = 'rfgcn_models/'+model_state_file[0]
    decoder_state_file = 'rfgcn_models/'+model_state_file[1]

    if args.gpu >= 0 and th.cuda.is_available():
        device = th.device(args.gpu)
    else:
        device = th.device('cpu')
    encoder = encoder.to(device)


    if args.mode=='train':
        best_mse = 0
        nega_sampler = NegativeSampler(num_nodes,train_query,args.neg_sample_rate)
        writer = SummaryWriter()
        # train the encoder
        early_stopping = EarlyStopping(patience=500,checkpoint_path=encoder_state_file)
        for epoch in range(50000):
            encoder.train()
            # get the message graph and training samples(including negative samples) for this round
            g, triplets, labels = get_epoch_data(train_query,train_label,m_graph,nega_sampler,args.use_weighted_edge,args.dropout)
            g = g.to(device)
            predict_flow,embed = encoder(g,triplets,args.use_weighted_edge)
            labels = torch.tensor(labels,dtype=torch.float).to(device)
            loss = encoder.get_loss(predict_flow, labels,embed)
            writer.add_scalar('Loss/train', loss.item(), epoch+1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0) # clip gradients
            optimizer.step()
            mse,mape = calc_metrics(predict_flow,labels)
            print("Epoch {:04d} | Loss {:.4f} | Best MSE {:.4f} | MSE {:.4f}".format(epoch, loss.item(), best_mse,mse))

            encoder.eval()
            print("start eval")
            g = m_graph.to(device)
            predict_flow,embed = encoder(g,valid_query,args.use_weighted_edge)# 预测有没有流量，所以test的图还是train的图，test的点是所有点
            labels = torch.tensor(valid_label,dtype=torch.float).to(device)
            mse,mape = calc_metrics(predict_flow, labels)
            #print result and add to tensorboard
            writer.add_scalar('MSE/valid', mse, epoch+1)
            #try early stopping
            if early_stopping.step(-mse,encoder):
                print(f'early stop at epoch [{epoch}]')
                break

            if epoch%50 == 0:
                encoder.output_weights(writer,epoch)
        # train the random forest encoder according to the best embedding model
        # 1. load the best model to generate all the embeddings
        checkpoint = th.load(encoder_state_file)
        encoder.eval()
        encoder.load_state_dict(checkpoint)
        g = m_graph.to(device)
        # 2. get all the training embeddings for the graph
        _,embedding = encoder(g,train_query,args.use_weighted_edge)
        embedding = embedding.cpu().detach().numpy()

        train_src_embeddings = embedding[train_query[:,0]]
        train_dst_embeddings = embedding[train_query[:,2]]
        x_train = np.concatenate((train_src_embeddings,train_distance,train_dst_embeddings),axis = 1)
        y_train = train_label

        valid_src_embeddings = embedding[valid_query[:,0]]
        valid_dst_embeddings = embedding[valid_query[:,2]]
        x_valid = np.concatenate((valid_src_embeddings,valid_distance,valid_dst_embeddings),axis = 1)
        y_valid = valid_label

        train_rf_regressor(x_train,y_train,x_valid,y_valid,decoder_state_file)

    #***********************valid and test
    checkpoint_encoder = th.load(encoder_state_file)
    encoder.eval()
    encoder.load_state_dict(checkpoint_encoder)
    encoder = encoder.to(device)
    g = m_graph.to(device)
    print("start final validating:")
    encoder_predict_flow,embed = encoder(g,valid_query,args.use_weighted_edge)
    labels = torch.tensor(valid_label,dtype=torch.float).to(device)
    if not args.test_use_decoder:
        valid_mse,valid_mape = calc_metrics(encoder_predict_flow,labels)
    else:
        decoder = pickle.load(open(decoder_state_file, 'rb'))
        valid_src_embeddings = embedding[valid_query[:,0]]
        valid_dst_embeddings = embedding[valid_query[:,2]]
        x_valid = np.concatenate((valid_src_embeddings,valid_distance,valid_dst_embeddings),axis = 1)
        y_valid = valid_label
        y_pred_valid = decoder.predict(x_valid)
        valid_mse,valid_mape = calc_metrics_numpy(y_pred_valid,y_valid)
    print("final validation MSE {:.4f}, MAPE {:.4f}".format(valid_mse,valid_mape))

    print("start testing:")
    encoder_predict_flow,embed = encoder(g,test_query,args.use_weighted_edge)
    labels = torch.tensor(test_label,dtype=torch.float).to(device)
    if not args.test_use_decoder:
        test_mse,test_mape = calc_metrics(encoder_predict_flow,labels)
        predict_flow = encoder_predict_flow
    else:
        test_src_embeddings = embedding[test_query[:,0]]
        test_dst_embeddings = embedding[test_query[:,2]]
        x_test = np.concatenate((test_src_embeddings,test_distance,test_dst_embeddings),axis = 1)
        y_test = test_label
        y_pred_test = decoder.predict(x_test)
        predict_flow = y_pred_test
        test_mse,test_mape = calc_metrics_numpy(y_pred_test,y_test)
    print("Test MSE {:.4f}, MAPE {:.4f}".format(test_mse,test_mape))
    output_flow(predict_flow,labels,test_query[:,0],test_query[:,2],'outflow/'+args.outflow_file)


    if args.mode == 'train':
        #write information
        output_model_information(args.model_file,args.feature_names,epoch,[test_mse,test_mape],[valid_mse,valid_mape],args.outflow_file,'rfgcn_models/'+args.out_meta_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='random_forest-GCN for link prediction')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--massage-graph-type", type=str, default='interaction',
                        choices=['interaction', 'distance','mix'],
                        help="Type of message graph")
    parser.add_argument("--use-weighted-edge", type=bool, default=False,
                        help="Whether or not to use distance weight to transform messages")
    parser.add_argument("--threshold", type=int, default=200,
                        help="Whether or not to use distance weight to transform messages")
    parser.add_argument("--feature_names",type = str,default = 'lon@lat@cost@type_id@ranking@comment@level@area@mean_time')
    parser.add_argument("--mode",type =str,default = 'train',choices=['train','test'],
                        help='to train a new model or only use a trained model to test' )
    parser.add_argument("--model_file",type =str,default = 'encoder_mlp_ep.pth@decoder_mlp_ep.sav',
                        help='if is test mode,then enter the directory of the trained model; if is train model, the model will be stored here' )
    parser.add_argument("--outflow_file",type = str, default = 'rfgcn_inter_thre_200_mlp_ep.csv')
    parser.add_argument("--out_meta_file",type = str, default = 'rfgcn.txt')
    parser.add_argument("--dropout",type = bool, default = False)
    parser.add_argument("--neg_sample_rate",type = float,default=0.0)
    parser.add_argument("--test_use_decoder",type = bool,default=True,
                        help='if false, then only use GCN for test')

    args = parser.parse_args()
    print(args)
    main(args)
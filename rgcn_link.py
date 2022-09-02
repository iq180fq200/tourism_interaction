import argparse
import pickle

import numpy as np
import torch
import torch as th
import torch.nn as nn
from dglgo.utils.early_stop import EarlyStopping

from common.output_results import output_flow, output_model_information
from common.link_utils import NegativeSampler, get_epoch_data, calc_metrics, calc_metrics_numpy
from data_deal.other_dataset import get_distances_all
from model.common_components.rf import train_rf_regressor
from model.rgcn import models
from data_deal.graph_dataset import create_dataset
from torch.utils.tensorboard import SummaryWriter

def main(args):
    #********************************prepare all the data
    m_graph,all_features,query_triplets,labels = create_dataset(args.massage_graph_type,args.threshold,args.use_weighted_edge,feat_names=str(args.feature_names).split('@'))
    if m_graph is not None:
        num_nodes = m_graph.num_nodes()
    else:
        num_nodes = len(all_features)
    train_query = query_triplets[0]
    valid_query = query_triplets[1]
    test_query = query_triplets[2]
    train_label,valid_label,test_label = labels[0],labels[1],labels[2]
    train_distance,valid_distance,test_distance = get_distances_all()

    #***************************initialize the model
    model = models.LinkPredict(all_features, 1)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-2)
    if args.gpu >= 0 and th.cuda.is_available():
        device = th.device(args.gpu)
    else:
        device = th.device('cpu')
    model = model.to(device)

    #***************************train the model and early stopping using valid data
    if args.mode=='train':
        # train the r_gcn model (encoder)
        nega_sampler = NegativeSampler(num_nodes,train_query,args.neg_sample_rate)#negative sample for training
        writer = SummaryWriter()#write out the train process to tensorboard
        early_stopping = EarlyStopping(patience=500,checkpoint_path='rgcn_models/'+str(args.model_files).split('@')[0])
        for epoch in range(50000):
            model.train()
            # get the message graph and training samples(including negative samples) for this round
            g, triplets, labels = get_epoch_data(train_query,train_label,m_graph,nega_sampler,args.use_weighted_edge,args.dropout)
            if g is not None:
                g = g.to(device)
            else:
                all_features=all_features.to(device)
            predict_flow,embed = model(g,triplets,args.use_weighted_edge,args.dropout,all_features)
            labels = torch.tensor(labels,dtype=torch.float).to(device)
            loss = model.get_loss(predict_flow, labels, embed)
            writer.add_scalar('Loss/train', loss.item(), epoch+1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients
            optimizer.step()
            mse,mape = calc_metrics(predict_flow,labels)
            print("Train Epoch {:04d} | Loss {:.4f} | MSE {:.4f}".format(epoch, loss.item(),mse))

            model.eval()
            if m_graph is not None:
                g = m_graph.to(device)
            else:
                all_features=all_features.to(device)
            predict_flow,embed = model(g,valid_query,args.use_weighted_edge,features = all_features)# 预测有没有流量，所以test的图还是train的图，test的点是所有点
            labels = torch.tensor(valid_label,dtype=torch.float).to(device)
            mse,mape = calc_metrics(predict_flow, labels)
            #print result and add to tensorboard
            writer.add_scalar('MSE/valid', mse, epoch+1)
            # save best model
            if early_stopping.step(-mse,model):
                print(f'early stop at epoch [{epoch}]')
                break
            #show the parameters
            if epoch%50 == 0:
                model.output_weights(writer,epoch)

        #train the random forest model (decoder)
        if args.use_rf == True:
            #----------get the encoder embeddings----------------------------
            checkpoint = th.load('rgcn_models/'+str(args.model_files).split('@')[0])
            model.eval()
            model.load_state_dict(checkpoint)
            model = model.to(device)
            if m_graph is not None:
                g = m_graph.to(device)
            else:
                all_features = all_features.to(device)
            _,embedding = model(g,valid_query,args.use_weighted_edge,features = all_features)
            #----------------------------------------------------

            # --------------------get data for the random forest decoder and train it----------------------------
            train_src_embeddings = embedding[train_query[:,0]].cpu().detach().numpy()
            train_dst_embeddings = embedding[train_query[:,2]].cpu().detach().numpy()
            x_train = np.concatenate((train_src_embeddings,train_distance,train_dst_embeddings),axis = 1)
            y_train = train_label

            valid_src_embeddings = embedding[valid_query[:,0]].cpu().detach().numpy()
            valid_dst_embeddings = embedding[valid_query[:,2]].cpu().detach().numpy()
            x_valid = np.concatenate((valid_src_embeddings,valid_distance,valid_dst_embeddings),axis = 1)
            y_valid = valid_label

            train_rf_regressor(x_train,y_train,x_valid,y_valid,'rgcn_models/'+str(args.model_files).split('@')[1])
            #--------------------------------------------------------------------------------


    #***************************validate and test the model(pure rgcn)
    checkpoint = th.load('rgcn_models/'+str(args.model_files).split('@')[0])
    model.eval()
    model.load_state_dict(checkpoint)
    model = model.to(device)
    if m_graph is not None:
        g = m_graph.to(device)
    else:
        all_features = all_features.to(device)
    _,embedding = model(g,valid_query,args.use_weighted_edge,features = all_features)
    print("Start final validating for pure rgcn:")
    predict_flow,embed = model(g,valid_query,args.use_weighted_edge,features = all_features)
    labels = torch.tensor(valid_label,dtype=torch.float).to(device)
    valid_mse,valid_mape = calc_metrics(predict_flow,labels)
    #print results
    print("final validation pure rgcn MSE {:.4f}, MAPE {:.4f}".format(valid_mse,valid_mape))


    print("start final testing for pure rgcn:")
    predict_flow,embed = model(g,test_query,args.use_weighted_edge,features = all_features)
    labels = torch.tensor(test_label,dtype=torch.float).to(device)
    test_mse,test_mape = calc_metrics(predict_flow,labels)
    output_flow(predict_flow,labels,test_query[:,0],test_query[:,2],'outflow/'+str(args.outflow_files).split('@')[0])
    #print results
    print("final test pure rgcn MSE {:.4f}, MAPE {:.4f}".format(mse,mape))

    if args.use_rf == True:
        #start testing with rgcn + random forest
        decoder = pickle.load(open('rgcn_models/'+str(args.model_files).split('@')[1], 'rb'))

        valid_src_embeddings = embed[valid_query[:,0]].cpu().detach().numpy()
        valid_dst_embeddings = embed[valid_query[:,2]].cpu().detach().numpy()
        x_valid = np.concatenate((valid_src_embeddings,valid_distance,valid_dst_embeddings),axis = 1)
        y_valid = valid_label
        y_pred_valid = decoder.predict(x_valid)
        valid_mse,valid_mape = calc_metrics_numpy(y_pred_valid,y_valid)

        test_src_embeddings = embed[test_query[:,0]].cpu().detach().numpy()
        test_dst_embeddings = embed[test_query[:,2]].cpu().detach().numpy()
        x_test = np.concatenate((test_src_embeddings,test_distance,test_dst_embeddings),axis = 1)
        y_test = test_label
        y_pred_test = decoder.predict(x_test)
        test_mse,test_mape = calc_metrics_numpy(y_pred_test,y_test)
        print("final test RGCN with RF results: MSE {:.4f}, MAPE {:.4f}".format(test_mse,test_mape))
        output_flow(y_pred_test,labels,test_query[:,0],test_query[:,2],'outflow/'+str(args.outflow_files).split('@')[1])

    if args.mode == 'train':
        #write information
        output_model_information(args.model_files,args.feature_names,epoch,[test_mse,test_mape],[valid_mse,valid_mape],args.outflow_files,'rgcn_models/'+args.out_meta_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for link prediction')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--massage_graph_type", type=str, default='interaction',
                        choices=['interaction', 'distance','none'],
                        help="Type of message graph")
    parser.add_argument("--use-weighted-edge", type=bool, default=False,
                        help="Whether or not to use distance weight to transform messages")
    parser.add_argument("--threshold", type=int, default=200,
                        help="Whether or not to use distance weight to transform messages")
    # parser.add_argument("--feature_names",type = str,default = 'lon@lat@adname_id@cost@type_id@ranking@comment@area@total_trip@level@mean_time')
    parser.add_argument("--feature_names",type = str,default = 'lon@lat@cost@type_id@ranking@comment@level@area@mean_time')
    parser.add_argument("--mode",type =str,default = 'train',choices=['train','test'],
                        help='to train a new model or only use a trained model to test' )
    parser.add_argument("--dropout",type = bool, default = False)
    parser.add_argument("--neg_sample_rate",type = float,default=0.0)
    parser.add_argument("--use_rf",type = bool, default = True,
                        help='if true, add a distance feature for the decoder')
    
    parser.add_argument("--model_files",type =str,default = 'inter_thre_rgcnrf_encoder_1.pth@inter_thre_rgcnrf_decoder_1.pth',
                        help='if is test mode,then enter the directory of the trained model; if is train model, the model will be stored here' )
    parser.add_argument("--outflow_files",type = str, default = 'inter_thre_200_rgcn_1.csv@inter_thre_200_rgcn_rf_1.csv')
    parser.add_argument("--out_meta_file",type = str, default = 'inter_thre_200_rgcn.txt')
    
    args = parser.parse_args()
    print(args)
    main(args)
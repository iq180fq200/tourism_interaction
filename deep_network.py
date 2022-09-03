import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from dglgo.utils.early_stop import EarlyStopping
import torch as th
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common.link_utils import calc_metrics_numpy, calc_metrics
from common.output_results import output_model_information, output_flow
from data_deal.graph_dataset import normalize_features
from data_deal.other_dataset import MyDataset
from model.common_components.rf import train_rf_regressor
from model.deep_network.models import MLP


def main(args):
    connection_data_train = pd.read_csv('data/connection_table_train.csv')
    attributes = ['s_ranking','s_cost','s_type_id','s_comment','s_area','s_level','s_mean_time','e_ranking','e_cost','e_type_id','e_comment','e_area','e_level','e_mean_time','distance']
    x_train = connection_data_train[attributes]
    y_train = connection_data_train['flow']
    connection_data_valid = pd.read_csv('data/connection_table_valid.csv')
    x_valid = connection_data_valid[attributes]
    y_valid = connection_data_valid['flow']
    connection_data_test = pd.read_csv('data/connection_table_test.csv')
    x_test = connection_data_test[attributes]
    y_test = connection_data_test['flow']

    mlp = MLP(x_train.shape[1],256)
    if args.gpu >= 0 and th.cuda.is_available():
        device = th.device(args.gpu)
    else:
        device = th.device('cpu')
    mlp = mlp.to(device)

    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_valid = std.fit_transform(x_valid)
    x_test = std.fit_transform(x_test)
    # feature_names = ['s_ranking','s_lon','s_lat','s_adname_id','s_cost','s_type_id','s_comment','s_area','s_level','s_mean_time','e_ranking','e_lon','e_lat','e_adname_id','e_cost','e_type_id','e_comment','e_area','e_level','e_mean_time','distance']
    # x_train = normalize_features(x_train,feature_names)
    # x_valid = normalize_features(x_valid,feature_names)
    # x_test = normalize_features(x_test,feature_names)
    x_train_tensor = th.tensor(np.array(x_train),device = device, dtype=th.float)
    x_valid_tensor = th.tensor(np.array(x_valid),device = device, dtype=th.float)
    x_test_tensor = th.tensor(np.array(x_test),device = device, dtype=th.float)


    if not os.path.exists('deep_network_models/'):
        os.makedirs('deep_network_models/')
    mlp_flie = 'deep_network_models/'+ str(args.state_file).split('@')[0]
    # rf_flie = 'deep_network_models/'+ str(args.state_file[1]).split('@')[1]

    #USE BATCHES
    # train_dataset = MyDataset(x_train_tensor,y_train)
    # train_dataloader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    if args.mode == 'train':
        writer = SummaryWriter()
        optimizer = optim.RMSprop(mlp.parameters(), lr=5e-4, momentum=0.9)
        early_stopping = EarlyStopping(patience = 500,checkpoint_path=mlp_flie)
        for epoch in range(100000):
            mlp.train()

            predict_flow = mlp(x_train_tensor)
            labels = th.tensor(y_train,dtype=th.float).to(device)
            loss = mlp.get_loss(predict_flow,labels)

            writer.add_scalar('Loss/train', loss.item(), epoch+1)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0) # clip gradients
            optimizer.step()
            mse,mape = calc_metrics(predict_flow,labels)
            # mse=mape=loss=0
            # iteration = 0
            # for i,data in enumerate(train_dataloader):
            #     iteration+=1
            #     pred = mlp(data[0])
            #     labels = th.tensor(data[1],dtype=th.float).to(device)
            #     t_loss = mlp.get_loss(pred,labels)
            #     optimizer.zero_grad()
            #     t_loss.backward()
            #     optimizer.step()
            #     loss+=t_loss
            #     t_mse,t_mape = calc_metrics(pred,labels)
            #     mse += t_mse
            #     mape += t_mape
            # mse/=iteration
            # mape/=iteration
            # loss/=iteration
            print("Epoch {:04d} | Loss {:.4f} | MSE {:.4f}".format(epoch, loss.item(),mse))
            # print("Epoch {:04d} | Loss {:.4f} | MSE {:.4f}".format(epoch, loss,mse))
            # writer.add_scalar('Loss/train', loss, epoch+1)

            mlp.eval()
            print("start eval")
            predict_flow = mlp(x_valid_tensor)
            labels = th.tensor(y_valid,dtype=th.float).to(device)

            mse,mape = calc_metrics(predict_flow, labels)
            #print result and add to tensorboard
            writer.add_scalar('MSE/valid', mse, epoch+1)
            #try early stopping
            if early_stopping.step(-mse,mlp):
                print(f'early stop at epoch [{epoch}]')
                break
            if epoch%50 == 0:
                mlp.output_weights(writer,epoch)
    checkpoint = th.load(mlp_flie)
    mlp.eval()
    mlp.load_state_dict(checkpoint)
    #valid
    with torch.no_grad():
        y_pred_valid = mlp(x_valid_tensor)
    valid_mse,valid_mape = calc_metrics(y_pred_valid,th.tensor(y_valid,dtype=th.float).to(device))


    #test
    y_pred_test = mlp(x_test_tensor)
    test_mse,test_mape = calc_metrics(y_pred_test,th.tensor(y_test,dtype=th.float).to(device))
    print("final testing pure RF results: MSE {:.4f}, MAPE {:.4f}".format(test_mse,test_mape))
    if args.mode == 'train':
        output_model_information(args.state_file,'all_features',epoch,[test_mse,test_mape],[valid_mse,valid_mape],args.outflow_file,'deep_network_models/read_me.txt')
    output_flow(y_pred_test,y_test,list(connection_data_test['s_id']),list(connection_data_test['e_id']),'outflow/'+args.outflow_file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nn for link prediction')
    parser.add_argument("--state_file",type = str, default = 'nn.pth')
    parser.add_argument("--outflow_file",type = str, default = 'nn.csv')
    parser.add_argument("--use_rf",type = bool, default = False)
    parser.add_argument("--mode",type =str,default = 'train',choices=['train','test'],
                        help='to train a new model or only use a trained model to test' )
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    args = parser.parse_args()
    main(args)
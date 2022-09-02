import argparse
import pickle
import pandas as pd
from common.link_utils import calc_metrics_numpy
from common.output_results import output_model_information, output_flow
from model.common_components.rf import train_rf_regressor

def main(args):
    connection_data_train = pd.read_csv('data/connection_table_train.csv')
    x_train = connection_data_train[['s_ranking','s_lon','s_lat','s_adname_id','s_cost','s_type_id','s_comment','s_area','s_level','s_mean_time','e_ranking','e_lon','e_lat','e_adname_id','e_cost','e_type_id','e_comment','e_area','e_level','e_mean_time','distance']]
    y_train = connection_data_train['flow']
    connection_data_valid = pd.read_csv('data/connection_table_valid.csv')
    x_valid = connection_data_valid[['s_ranking','s_lon','s_lat','s_adname_id','s_cost','s_type_id','s_comment','s_area','s_level','s_mean_time','e_ranking','e_lon','e_lat','e_adname_id','e_cost','e_type_id','e_comment','e_area','e_level','e_mean_time','distance']]
    y_valid = connection_data_valid['flow']
    connection_data_test = pd.read_csv('data/connection_table_test.csv')
    x_test = connection_data_test[['s_ranking','s_lon','s_lat','s_adname_id','s_cost','s_type_id','s_comment','s_area','s_level','s_mean_time','e_ranking','e_lon','e_lat','e_adname_id','e_cost','e_type_id','e_comment','e_area','e_level','e_mean_time','distance']]
    y_test = connection_data_test['flow']

    if args.mode == 'train':
        best_regressor = train_rf_regressor(x_train,y_train,x_valid,y_valid,'rf_models/'+str(args.state_file))
    else:
        best_regressor = pickle.load(open('rf_models/'+str(args.state_file), 'rb'))

    #valid
    y_pred_valid = best_regressor.predict(x_valid)
    valid_mse,valid_mape = calc_metrics_numpy(y_pred_valid,y_valid)


    #test
    y_pred_test = best_regressor.predict(x_test)
    test_mse,test_mape = calc_metrics_numpy(y_pred_test,y_test)
    print("final testing pure RF results: MSE {:.4f}, MAPE {:.4f}".format(test_mse,test_mape))
    output_model_information(args.state_file,'all_features',-1,[test_mse,test_mape],[valid_mse,valid_mape],args.outflow_file,'rf_models/read_me.txt')
    output_flow(y_pred_test,y_test,list(connection_data_test['s_id']),list(connection_data_test['e_id']),'outflow/'+args.outflow_file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='random forest for link prediction')
    parser.add_argument("--state_file",type = str, default = 'pure_rf.pth')
    parser.add_argument("--outflow_file",type = str, default = 'pure_rf_result.pth')
    parser.add_argument("--mode",type =str,default = 'train',choices=['train','test'],
                        help='to train a new model or only use a trained model to test' )
    args = parser.parse_args()
    main(args)
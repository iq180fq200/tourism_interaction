import pandas as pd
import numpy as np
def output_flow(predicted_flow,labels,src_ids,dst_ids,file_name):
    try:
        pre_flow = np.squeeze(predicted_flow.cpu().detach().numpy())
    except:
        pre_flow = predicted_flow
    try:
        true_flow = np.squeeze(labels.cpu().detach().numpy())
    except:
        true_flow = labels
    table_data = {
        'pre_flow' : pre_flow,
        'true_flow' : true_flow,
        'src_ids':src_ids,
        'dst_ids':dst_ids
    }
    flow_table = pd.DataFrame(table_data)
    flow_table.to_csv(file_name,index=False)

def output_model_information(name,str_features,epoch,testing_results,valid_results,outflow_file,information_file = 'trained_models/read_me.txt'):
    file = open(information_file,'a')
    file.write('\n')
    file.write('[model name]'+name+'\n')
    file.write('[features]'+str_features+'\n')
    file.write('[early stop at]'+str(epoch)+'\n')
    file.write('[testing results]'+'MSE '+str(testing_results[0].item())+', '+'MAPE '+str(testing_results[1])+'\n')
    file.write('[valid results]'+'MSE '+str(valid_results[0].item())+', '+'MAPE '+str(valid_results[1])+'\n')
    file.write('[outflow file]'+outflow_file+'\n')
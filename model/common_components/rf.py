import pickle

from sklearn.ensemble import RandomForestRegressor

from common.link_utils import calc_metrics_numpy


def train_rf_regressor(x_train,y_train,x_valid,y_valid,state_file):
    # get the best hyperparameter
    best_n_estimator = -1
    best_max_depth = -1
    best_mse = -1
    best_regressor = None
    for max_depth in [20,25,30,35]:
        for n_estimator in [10,30,50]:
            regressor = RandomForestRegressor(n_estimators=n_estimator, random_state=0,max_depth = max_depth)
            regressor.fit(x_train, y_train)
            y_pred_valid = regressor.predict(x_valid)
            mse,mape = calc_metrics_numpy(y_pred_valid,y_valid)
            if best_regressor == None or mse < best_mse:
                best_regressor,best_mse,best_max_depth ,best_n_estimator= regressor,mse,max_depth,n_estimator
    #write out the best decoder
    pickle.dump(best_regressor, open(state_file, 'wb'))
    print(f'best_n_estimator = {best_n_estimator}, best_max_depth = {best_max_depth}, best_mse = {best_mse}')
    return best_regressor
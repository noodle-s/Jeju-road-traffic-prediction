import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import exchange_calendars as ecals
import os
from functools import partial
from eli5.lightgbm import *
from eli5.sklearn import *
import eli5
from catboost import CatBoostRegressor
from imblearn.over_sampling import ADASYN
from bayes_opt import BayesianOptimization
from sklearn.inspection import permutation_importance
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from pyproj import Geod


data_path = '../../data/'
train_file = 'train.csv'
test_file = 'test.csv'
submission_file = 'sample_submission.csv'

def bayes_opt(model, random_state=1):
    pbounds = {'learning_rate': (0.01,0.1), 'min_child_samples':(50,500),'max_depth':(1,16)}

    optimizer = BayesianOptimization(
        f=model,
        pbounds=pbounds,
        verbose=1,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=random_state,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=50
    )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

def train_preprocessing():
    data = pd.read_csv(data_path + train_file, encoding='euckr')
    data = diff_long_lat_add(data)
    data = holiday_add(data)
    # data = road_count_add(data)
    data = month_year_add(data)
    data = time_quarter_add(data)
    data = meter_sec_add(data)
    data = distance_add(data)
    # data = month_visitor_add(data)

    num_col = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude',
               'diff_longtitude', 'diff_latitude', 'maximum_meter_second', 'maximum_speed_limit']
    cat_col = ['road_name', 'start_node_name', 'end_node_name', 'day_of_week']
    scalers = {}
    encoders = {}

    for col in num_col:
        scaler = MinMaxScaler()
        data[col] = scaler.fit_transform(np.array(data[col]).reshape(1,-1).transpose())
        scalers[col] = scaler

    for col in cat_col:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(np.array(data[col]).reshape(1,-1).transpose())
        encoders[col] = encoder

    target = data['target']
    data = data.drop(['target','id', 'multi_linked', 'connect_code', 'vehicle_restricted', 'height_restricted'], axis=1)
    exist_dict = {'없음': 0, '있음': 1}
    data = data.replace({"start_turn_restricted": exist_dict, 'end_turn_restricted': exist_dict})
    group = ['road_name', 'lane_count', 'start_turn_restricted', 'road_type',
             'end_turn_restricted', 'weight_restricted', 'maximum_speed_limit','road_rating','day_of_week']
    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=0, stratify=data[group])
    split_data = {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val}

    return split_data, encoders, scalers

def diff_long_lat_add(data):
    data['diff_longtitude'] = data['end_longitude'] - data['start_longitude']
    data['diff_latitude'] = data['end_latitude'] - data['start_latitude']

    return data

def holiday_add(data):
    krx = ecals.get_calendar("XKRX")
    krx_holiday = pd.DataFrame(krx.schedule.loc["2021-01-01":"2022-12-31"])

    open_date = pd.to_datetime(krx_holiday['open'])
    open_date = open_date.dt.strftime('%Y%m%d')
    open_date = open_date.astype(int)
    open_date = list(open_date)

    data['holiday'] = data['base_date'].isin(open_date)
    holiday_exist = {True: 0, False: 1}
    data['holiday'] = data['holiday'].map(holiday_exist)

    return data

def month_year_add(data):
    data['month'] = data['base_date'].apply(lambda e: str(e)[4:6])
    data['month'] = data['month'].astype(int)
    data['year'] = data['base_date'].apply(lambda e: str(e)[0:4])
    data['year'] = data['year'].astype(int)
    del data['base_date']

    return data

def time_quarter(data):
    quarter = 3

    if 6 > data >= 0:
        quarter = 0

    elif 12 > data >= 6:
        quarter = 1

    elif 18 > data >= 12:
        quarter = 2

    return quarter

def time_quarter_add(data):
    data['time_quarter'] = data['base_hour'].apply(lambda e: time_quarter(e))

    return data

def get_distance(data):
    g = Geod(ellps='WGS84')
    # 2D distance in meters with longitude, latitude of the points
    azimuth1, azimuth2, distance_2d = g.inv(data['start_longitude'], data['start_latitude'], data['end_longitude'],
                                            data['end_latitude'])

    return distance_2d

def distance_add(data):
    data['distance'] = data.apply(get_distance,axis=1)

    return data

def meter_sec_add(data):
    data['maximum_meter_second'] = data['maximum_speed_limit'] / 3.6

    return data

def road_count_add(data):
    # data['base_hour'] = data['base_hour'].astype(str)
    # data['base_date_hour'] = data['base_date'].astype(str) + data['base_hour'].str.zfill(2)
    # data['base_date_hour'] = pd.to_datetime(data['base_date_hour'], format='%Y%m%d%H')
    road_visit_count = data[["day_of_week",'road_name','holiday','base_hour','start_node_name','end_node_name']]
    road_visit_count = road_visit_count.groupby(["day_of_week",'road_name','holiday','base_hour','start_node_name','end_node_name'],as_index=False).value_counts()
    road_visit_count = road_visit_count.rename(columns={"count": "road_visit_count"})
    data = data.merge(road_visit_count, on=["day_of_week",'road_name','holiday','base_hour','start_node_name','end_node_name'])
    # data = data.drop('base_date_hour', axis=1)
    data['base_hour'] = data['base_hour'].astype(int)
    data['base_date'] = data['base_date'].astype(int)

    return data

def month_visitor_add(data):
    additional_path = "../../additional data/visitor/"
    visitor_data_list = os.listdir(additional_path)
    res_visitor_data_df = pd.DataFrame(columns=['year', 'month', 'visitor'])

    for visitor_data in visitor_data_list:
        visitor_data_tmp = pd.read_excel(additional_path + visitor_data)[['Unnamed: 6']].iloc[2, :]
        visitor_data_tmp = pd.DataFrame.from_dict({'year': [visitor_data[0:4]],
                                        "month": [visitor_data[5:7]],
                                        "visitor": [visitor_data_tmp.values[0]]})
        res_visitor_data_df = pd.concat([res_visitor_data_df, visitor_data_tmp], ignore_index=True)

    res_visitor_data_df['month'] = res_visitor_data_df['month'].astype(int)
    res_visitor_data_df['year'] = res_visitor_data_df['year'].astype(int)
    data = data.merge(res_visitor_data_df, on=['year', 'month'])

    return data

def lgbm_loss_plot(callback):
    f, ax = plt.subplots(figsize=(10, 10))
    lgbm.plot_metric(callback, ax=ax)
    print(callback['valid_0']['l2'][-1])
    plt.savefig("./lgbm_loss.png")
    plt.show()
    plt.close()

def cat_loss_plot(data):
    f = plt.plot(figsize=(12, 6))
    print(data[-1])
    plt.plot(data)
    plt.title(f"minimum loss = {data[-1]}")
    plt.savefig("./cat_loss.png")
    plt.show()
    plt.close()

def lgbm_feature_importance(model):
    f, ax = plt.subplots(figsize=(10, 10))
    lgbm.plot_importance(model, max_num_features=25, ax=ax)
    plt.savefig("./lgbm_feature_importance.png")
    plt.show()
    plt.close()

def cat_feature_importance(model, x_val):
    f = plt.figure(figsize=(12, 6))
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(x_val.columns)[sorted_idx])
    plt.savefig("./cat_feature_importance.png")
    plt.show()
    plt.close()

def get_permutation_importance(model, X_valid, y_valid):
    result = permutation_importance(
        model, X_valid, y_valid, n_repeats=10, random_state=42
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X_valid.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (valid set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()


def lgbm_fit(data):
    callback = {}
    evals = [(data['X_val'], data["y_val"])]
    model = lgbm.LGBMRegressor(n_estimators=10,
                               learning_rate=0.05,
                               num_leaves=20,
                               min_child_samples=150,
                               max_depth=30,
                               device='gpu',
                               gpu_platform_id=1, # 1 0
                               gpu_device_id=1,
                               is_training_metric= True) #

    lgbm.early_stopping(stopping_rounds=100)
    model.fit(data["X_train"], data["y_train"], eval_metric='MAE', eval_set=evals,
              callbacks = [lgbm.record_evaluation(callback)])
    get_permutation_importance(model, data['X_val'], data["y_val"])
    lgbm_loss_plot(callback)
    lgbm_feature_importance(model)

    return model

def cat_boost_fit(trial):
    data, encoders, scalers = train_preprocessing()

    cbrm_param = {
        'iterations': 200000,
        'early_stopping_rounds': 100,
        'eval_metric': 'MAE',
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 1e-5, 100),
        'depth': trial.suggest_int('depth', 1, 16),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        'devices':'0:3',
        'task_type':'GPU',
        'random_state' : 0
    }

    evals = [(data['X_val'], data["y_val"])]
    # model = CatBoostRegressor(iterations = 200000,
    #                           early_stopping_rounds=100,
    #                            learning_rate=0.018,
    #                            min_data_in_leaf=21, # 100
    #                            max_depth=11,
    #                            eval_metric='MAE',
    #                            task_type="GPU",
    #                            devices='2:3',
    #                            reg_lambda= 14.721,
    #                           leaf_estimation_iterations=9,
    #                           bagging_temperature= 0.0226
    #                           ) #
    model = CatBoostRegressor(**cbrm_param)
    model.fit(data["X_train"], data["y_train"], eval_set=evals, verbose=True)
    # cat_loss_plot(model.evals_result_['learn']['MAE'])
    # get_permutation_importance(model, data['X_val'], data["y_val"])
    # cat_feature_importance(model, data['X_val'])

    # return model
    return model.evals_result_['validation']['MAE'][-1]

def test_preprocessing(encoders, scalers):
    data = pd.read_csv(data_path + test_file, encoding='euckr')
    data = diff_long_lat_add(data)
    data = holiday_add(data)
    # data = road_count_add(data)
    data = month_year_add(data)
    data = time_quarter_add(data)
    data = meter_sec_add(data)
    data = distance_add(data)
    # data = month_visitor_add(data)
    exist_dict = {'없음': 0, '있음': 1}
    data = data.replace({"start_turn_restricted": exist_dict, 'end_turn_restricted': exist_dict})

    for col in list(scalers.keys()):
        data[col] = scalers[col].transform(np.array(data[col]).reshape(1,-1).transpose())

    for col in list(encoders.keys()):
        data[col] = encoders[col].transform(data[col])

    data = data.drop(['id', 'multi_linked', 'connect_code', 'vehicle_restricted', 'height_restricted'], axis=1)

    return data

def predict_to_csv(model, test_data):
    pred = model.predict(test_data)
    submis = pd.read_csv(data_path + submission_file, encoding='euckr')
    submis['target'] = pred
    submis.to_csv(data_path + 'res_test.csv', index=False)

def main():
    # train_data, encoders, scalers = train_preprocessing()
    # model = partial(cat_boost_fit, train_data)
    # bayes_opt(model)
    # model = cat_boost_fit()
    sampler = TPESampler(seed=10)
    optuna_cbrm = optuna.create_study(direction='minimize', sampler=sampler)
    optuna_cbrm.optimize(cat_boost_fit, n_trials=50)

    cbrm_trial = optuna_cbrm.best_trial
    cbrm_trial_params = cbrm_trial.params
    print('Best Trial: score {},\nparams {}'.format(cbrm_trial.value, cbrm_trial_params))
    # model = lgbm_fit(train_data)

    # test_data = test_preprocessing(encoders, scalers)
    # predict_to_csv(model, test_data)
    print("good")

if __name__ == '__main__':
    main()
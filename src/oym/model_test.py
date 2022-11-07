import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, OrdinalEncoder
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import smogn
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
from sklearn.utils.class_weight import compute_class_weight

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

def cyclic_encode(data, col):
    max_val_func = lambda a: 2022 if a == 'year' else (12 if a == 'month' else (23 if a == 'base_hour' else 6))
    max_val = max_val_func(col)
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)

    return data

def smogn_apply(data):
    data = smogn.smoter(
        ## primary arguments / inputs
        data,  ## training set  (pandas dataframe)
        'target',  ## response variable y by name  (string)
        k=9,  ## positive integer (k < n)
        samp_method='extreme',  ## string ('balance' or 'extreme')

        ## phi relevance arguments
        rel_thres=0.80,  ## positive real number (0 < R < 1)
        rel_method='auto',  ## string ('auto' or 'manual')
        rel_xtrm_type='high',  ## string ('low' or 'both' or 'high')
        rel_coef=2.25  ## positive real number (0 < R)
    )

    return data

def train_preprocessing():
    data = pd.read_csv(data_path + train_file, encoding='euckr')
    data = diff_long_lat_add(data)
    data = holiday_add(data)
    data = month_year_add(data)
    data = road_count_add(data)
    data = time_quarter_add(data)
    data = meter_sec_add(data)
    data = distance_add(data)
    # data = month_visitor_add(data)

    # num_col = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude',
    #            'diff_longtitude', 'diff_latitude', 'distance']
    nominal_features = ['road_name', 'start_turn_restricted', 'end_turn_restricted',
                        'start_node_name', 'end_node_name', 'day_of_week']
    # cycling_features = ['year', 'month', 'base_hour', 'day_of_week']

    scalers = {}
    label_encoder_dict = {}

    # for col in num_col:
    #     scaler = MinMaxScaler()
    #     data[col] = scaler.fit_transform(np.array(data[col]).reshape(1,-1).transpose())
    #     scalers[col] = scaler

    for col in nominal_features:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])
        label_encoder_dict[col] = label_encoder

    # for col in cycling_features:
    #     data = cyclic_encode(data, col)

    target = data['target']
    data = data.drop(['id', 'target', 'multi_linked', 'connect_code', 'vehicle_restricted', 'height_restricted', 'base_date'], axis=1)

    group = ['road_name', 'lane_count', 'start_turn_restricted', 'road_type',
             'end_turn_restricted', 'weight_restricted', 'maximum_speed_limit','road_rating','day_of_week']
    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=0, stratify=data[group])
    # train = pd.concat([X_train, y_train])
    # train = smogn_apply(train)
    split_data = {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val}

    return split_data, label_encoder_dict, scalers

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
    data['year'] = data['base_date'].apply(lambda e: str(e)[0:4])
    data['year'] = data['year'].astype(int)
    data['month'] = data['base_date'].apply(lambda e: str(e)[4:6])
    data['month'] = data['month'].astype(int)

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
    road_visit_count = data[["base_date",'road_name','base_hour']]
    road_visit_count = road_visit_count.groupby(["base_date",'road_name','base_hour'],as_index=False).value_counts()
    road_visit_count = road_visit_count.rename(columns={"count": "road_visit_count"})
    data = data.merge(road_visit_count, on=["base_date",'road_name','base_hour'])
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

def cat_boost_fit(data):
    # data, encoders, scalers = train_preprocessing()

    # cbrm_param = {
    #     'iterations': 200000,
    #     'early_stopping_rounds': 100,
    #     'eval_metric': 'MAE',
    #     'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.1),
    #     'reg_lambda': trial.suggest_uniform('reg_lambda', 1e-5, 100),
    #     'depth': trial.suggest_int('depth', 1, 16),
    #     'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
    #     'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
    #     'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
    #     'devices':'0:3',
    #     'task_type':'GPU',
    #     'random_state' : 0
    # }

    evals = [(data['X_val'], data["y_val"])]
    model = CatBoostRegressor(iterations = 200000,
                              early_stopping_rounds=100,
                               learning_rate=0.01,
                               min_data_in_leaf=24, # 100
                               max_depth=12,
                               eval_metric='MAE',
                               task_type="GPU",
                               devices='0:3',
                               reg_lambda= 5.0244,
                              leaf_estimation_iterations=7,
                              bagging_temperature= 0.07156,
                              random_state=0
                              ) #
    # model = CatBoostRegressor(**cbrm_param)
    model.fit(data["X_train"], data["y_train"], eval_set=evals, verbose=True)
    # cat_loss_plot(model.evals_result_['learn']['MAE'])
    # get_permutation_importance(model, data['X_val'], data["y_val"])
    # cat_feature_importance(model, data['X_val'])

    return model
    # return model.evals_result_['validation']['MAE'][-1]

def test_preprocessing(label_encoders, scalers):
    data = pd.read_csv(data_path + test_file, encoding='euckr')
    data = diff_long_lat_add(data)
    data = holiday_add(data)
    data = month_year_add(data)
    data = road_count_add(data)
    data = time_quarter_add(data)
    data = meter_sec_add(data)
    data = distance_add(data)
    # data = month_visitor_add(data)

    # for col in list(scalers.keys()):
    #     data[col] = scalers[col].transform(np.array(data[col]).reshape(1,-1).transpose())

    for col in list(label_encoders.keys()):
        data[col] = label_encoders[col].transform(np.array(data[col]).reshape(1,-1).transpose())

    # for col in cycling_features:
    #     data = cyclic_encode(data, col)

    data = data.drop(['id', 'multi_linked', 'connect_code', 'vehicle_restricted', 'height_restricted', 'base_date'], axis=1)

    return data

def predict_to_csv(model, test_data):
    pred = model.predict(test_data)
    submis = pd.read_csv(data_path + submission_file, encoding='euckr')
    submis['target'] = pred
    submis.to_csv(data_path + 'res_test.csv', index=False)

def main():
    train_data, label_encoders, scalers = train_preprocessing()
    # model = partial(cat_boost_fit, train_data)
    # bayes_opt(model)
    model = cat_boost_fit(train_data)
    # sampler = TPESampler(seed=10)
    # optuna_cbrm = optuna.create_study(direction='minimize', sampler=sampler)
    # optuna_cbrm.optimize(cat_boost_fit, n_trials=50)
    #
    # cbrm_trial = optuna_cbrm.best_trial
    # cbrm_trial_params = cbrm_trial.params
    # print('Best Trial: score {},\nparams {}'.format(cbrm_trial.value, cbrm_trial_params))
    # model = lgbm_fit(train_data)
    test_data = test_preprocessing(label_encoders, scalers)
    predict_to_csv(model, test_data)
    print("good")

if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import exchange_calendars as ecals
import os
import numpy as np

data_path = '../../data/'
train_file = 'train.csv'
test_file = 'test.csv'
submission_file = 'sample_submission.csv'

def train_preprocessing():
    data = pd.read_csv(data_path + train_file, encoding='euckr')
    data = holiday_add(data)
    data = road_count_add(data)
    data = month_year_add(data)
    data = month_visitor_add(data)

    num_col = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'road_visit_count','visitor']
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
    data = data.drop(['target','id','multi_linked','connect_code','year'], axis=1)
    exist_dict = {'없음': 0, '있음': 1}
    data = data.replace({"start_turn_restricted": exist_dict, 'end_turn_restricted': exist_dict})
    group = ['road_name', 'lane_count', 'road_type', 'road_name', 'start_turn_restricted',
             'end_turn_restricted', 'weight_restricted', 'maximum_speed_limit','road_rating','day_of_week']
    X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=0, stratify=data[group])
    split_data = {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val}

    return split_data, encoders, scalers


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
    data['holiday'].value_counts()

    return data

def month_year_add(data):
    data['month'] = data['base_date'].apply(lambda e: str(e)[4:6])
    data['month'] = data['month'].astype(int)
    data['year'] = data['base_date'].apply(lambda e: str(e)[0:4])
    data['year'] = data['year'].astype(int)
    del data['base_date']

    return data

def road_count_add(data):
    data['base_hour'] = data['base_hour'].astype(str)
    data['base_date_hour'] = data['base_date'].astype(str) + data['base_hour'].str.zfill(2)
    data['base_date_hour'] = pd.to_datetime(data['base_date_hour'], format='%Y%m%d%H')
    road_visit_count = data[["base_date_hour",'road_name']]
    road_visit_count = road_visit_count.groupby(["base_date_hour","road_name"],as_index=False).value_counts()
    road_visit_count = road_visit_count.rename(columns={"count": "road_visit_count"})
    data = data.merge(road_visit_count, on=['base_date_hour','road_name'])
    data = data.drop('base_date_hour', axis=1)
    data['base_hour'] = data['base_hour'].astype(int)
    data['base_date'] = data['base_date'].astype(int)

    return data

def month_visitor_add(data):
    additional_path = "../../additional data/visitor/"
    visitor_data_list = os.listdir(additional_path)
    res_visitor_data_df = pd.DataFrame(columns=['year', 'month', 'visitor'])

    for visitor_data in visitor_data_list:
        visitor_data_tmp = pd.read_excel(additional_path + visitor_data)[['Unnamed: 6']].iloc[2, :]
        res_visitor_data_df = res_visitor_data_df.append({'year': visitor_data[0:4],
                                                          "month": visitor_data[5:7],
                                                          "visitor": visitor_data_tmp.values[0]},
                                                         ignore_index=True)

    res_visitor_data_df['month'] = res_visitor_data_df['month'].astype(int)
    res_visitor_data_df['year'] = res_visitor_data_df['year'].astype(int)
    data = data.merge(res_visitor_data_df, on=['year', 'month'])

    return data

def lgbm_loss(callback):
    f, ax = plt.subplots(figsize=(10, 10))
    lgbm.plot_metric(callback, ax=ax)
    print(callback[-1])
    plt.savefig("./lgbm_loss.png")
    plt.show()
    plt.close()

def lgbm_feature_importance(model):
    f, ax = plt.subplots(figsize=(10, 10))
    lgbm.plot_importance(model, max_num_features=25, ax=ax)
    plt.savefig("./feature_importance.png")
    plt.show()
    plt.close()

def lgbm_fit(data):
    callback = {}
    evals = [(data['X_val'], data["y_val"])]
    model = lgbm.LGBMRegressor(n_estimators=100000,
                               learning_rate=0.05,
                               num_leaves=20,
                               min_child_samples=150,
                               max_depth=20,
                               device='gpu',
                               gpu_platform_id=0, # 1 0
                               gpu_device_id=0) #
    lgbm.early_stopping(stopping_rounds=100)
    model.fit(data["X_train"], data["y_train"], eval_metric='MAE', eval_set=evals,
              callbacks = [lgbm.record_evaluation(callback)])
    lgbm_loss(callback)
    lgbm_feature_importance(model)

    return model

def test_preprocessing(encoders, scalers):
    data = pd.read_csv(data_path + test_file, encoding='euckr')
    data = holiday_add(data)
    data = month_year_add(data)
    data = road_count_add(data)
    data = month_visitor_add(data)
    exist_dict = {'없음': 0, '있음': 1}
    data = data.replace({"start_turn_restricted": exist_dict, 'end_turn_restricted': exist_dict})

    for col in list(scalers.keys()):
        data[col] = scalers[col].transform(np.array(data[col]).reshape(1,-1).transpose())

    for col in list(encoders.keys()):
        data[col] = encoders[col].transform(data[col])
    # data = data.drop(['id', 'year', 'multi_linked', 'connect_code',
    #                   'vehicle_restricted', 'height_restricted', 'start_node_name',
    #                   'start_node_name','start_latitude','start_longitude','end_node_name',
    #                   'end_latitude','end_longitude'], axis=1)
    data = data.drop(['id', 'multi_linked', 'connect_code'], axis=1)

    return data

def predict_to_csv(model, test_data):
    pred = model.predict(test_data)
    submis = pd.read_csv(data_path + submission_file, encoding='euckr')
    submis['target'] = pred
    submis.to_csv(data_path + 'res_test.csv', index=False)

def main():
    train_data, encoders, scalers = train_preprocessing()
    model = lgbm_fit(train_data)

    # test_data = test_preprocessing(encoders, scalers)
    # predict_to_csv(model, test_data)
    print("good")

if __name__ == '__main__':
    main()
import pandas as pd
import os
import tensorflow as tf
import datetime
import numpy as np
int2time = lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")
dl_path = 'D:\\DL_data'
GetFileName = lambda x: os.path.join(dl_path, x)

# setting config
uni_name = None
index_name = '000905.SH'
model_name = 'dl4_ls'
data_filename = 'all_data.pkl'
factorlist_filename = 'factor_list.csv'
label_num = 2
from_date = 20060101
to_date = 20190501
train_last_date = 20180101

def GeneratePoolData(csv_path):
    all_tradedates = pd.read_csv(os.path.join(csv_path, 'ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
    trade_dt = [x for x in all_tradedates if (x >= from_date) & (x < to_date)]
    dt_list = pd.read_csv(os.path.join(csv_path, 'DT_LIST.csv'), names=['STOCK_CODE', 'DT_LIST', 'DT_DELIST'])
    dt_list['DT_OLD'] = dt_list['DT_LIST'] + 10000
    dt_list['IN'] = True
    dt_list['OUT'] = False
    old_data = pd.pivot_table(dt_list, values='IN', index='DT_OLD', columns='STOCK_CODE').fillna(method='pad').fillna(False).loc[all_tradedates, stock_code].fillna(method='pad').fillna(False).loc[trade_dt, :]
    del_data = pd.pivot_table(dt_list, values='OUT', index='DT_DELIST', columns='STOCK_CODE').fillna(method='pad').fillna(True).loc[all_tradedates, stock_code].fillna(method='pad').fillna(True).loc[trade_dt, :]
    old_stock = old_data & del_data
    isst = pd.read_csv(os.path.join(csv_path, 'RAW_ISST.csv'), index_col=0)
    istrade = pd.read_csv(os.path.join(csv_path, 'RAW_ISTRADEDAY.csv'), index_col=0)
    pool_data = old_stock & (isst < 0.5) & (istrade > 0.5)
    return pool_data
def PoolFilter(data, universe_name, from_date, to_date):
    if universe_name is None:
        return data
    else:
        uni = pd.read_csv(GetFileName('RAW_UNIVERSE_%s.csv' % universe_name.replace('.', '_')), index_col=[0]).loc[from_date:to_date].fillna(0) > 0.5
        uni.index.name = 'FACTOR_DATE'
        uni.columns.name = 'STOCK_CODE'
        uni_info = uni.unstack().reset_index(name='UNI')
        uni_rec = uni_info.loc[uni_info['UNI']]
        filter_data = pd.merge(data, uni_rec, on=['FACTOR_DATE', 'STOCK_CODE'], how='inner')
        return filter_data
def ReadFactorListInfor(filename):
    factor_list = pd.read_csv(GetFileName(filename), names=['FACTOR_NAME', 'NEU_TYPE'])
    factor_num = len(factor_list)
    return factor_list, factor_list['FACTOR_NAME'].values.tolist(), factor_num
def LoadAllData(filename, uni_name, from_date, to_date):
    all_data = pd.read_pickle(GetFileName(filename), compression='gzip').dropna()
    all_df = PoolFilter(data=all_data, universe_name=uni_name, from_date=from_date, to_date=to_date)
    return all_df
def Ret2Tag(data):
    return (data.groupby('FACTOR_DATE')['RET'].rank(ascending=False, na_option='bottom') <= 200).astype(np.int)
def LoadData(raw_data, f_names, from_date, to_date, batch_size, repeat=None, return_labels=True):
    train_df = raw_data.loc[(raw_data['FACTOR_DATE'] < to_date) & (raw_data['FACTOR_DATE'] >= from_date)]
    if return_labels:
        dataset = tf.data.Dataset.from_tensor_slices((dict(train_df[f_names]), train_df['TAG']))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(train_df[f_names]))
    dataset = dataset.shuffle(1000).repeat(count=repeat).batch(batch_size)
    return dataset

all_df = LoadAllData(filename=data_filename, uni_name=uni_name, from_date=from_date, to_date=to_date)
all_df['TAG'] =Ret2Tag(data=all_df)

factor_list, factor_names, factor_num = ReadFactorListInfor(filename=factorlist_filename)
all_tradedates = pd.read_csv(GetFileName('ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
stock_code = pd.read_csv(GetFileName('STOCK_CODE.csv'), header=None).values.T[0].tolist()
all_pool = GeneratePoolData(csv_path=dl_path)
pool = all_pool
zz500_ret = pd.read_csv(GetFileName('INDEX_%s_RETURN.csv' % index_name), index_col=[0], header=None, names=['RETURN']).loc[from_date:to_date, 'RETURN']
GetUniverse = lambda uni, x: uni.columns[uni.loc[x]].tolist()
trade_dt = [x for x in all_tradedates if (x >= train_last_date) & (x < to_date)]
factor_dt = [trade_dt[0]] + [y for x, y in zip(trade_dt[:-1], trade_dt[1:]) if int(x / 100) != int(y / 100)]
# factor_dt = trade_dt
holding = pd.DataFrame(index=trade_dt, columns=stock_code)
# model = tf.keras.models.load_model(GetFileName(model_name))
my_feature_columns = []
for key in factor_names:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=label_num,
    model_dir=GetFileName(model_name)
)

import time
t_1 = time.time()
for date in factor_dt:
    t_2 = time.time()
    s = GetUniverse(all_pool, date)
    d_df = all_df.loc[all_df['FACTOR_DATE'] == date].set_index('STOCK_CODE').loc[s].reset_index()
    input_data = tf.data.Dataset.from_tensor_slices(dict(d_df[factor_names]))
    predictions = classifier.predict(
        input_fn=lambda: LoadData(raw_data=all_df, f_names=factor_names, from_date=date, to_date=date+1, batch_size=100, repeat=1, return_labels=False))
    select_stocks = []
    for pred_dict, expec in zip(predictions, d_df['STOCK_CODE'].values.tolist()):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        if int(class_id) < 0.5:
            select_stocks.append(expec)

    # if use_short:
    #     tag[tag['TAG'] < 0] = tag[tag['TAG'] < 0] / tag[tag['TAG'] < 0].abs().sum()
    # tag[tag['TAG'] > 0] = tag[tag['TAG'] > 0] / tag[tag['TAG'] > 0].abs().sum()
    holding_dt = trade_dt[trade_dt.index(date)+2]
    if len(select_stocks) > 0:
        holding.loc[holding_dt, select_stocks] = 1.0 / len(select_stocks)
    # holding.loc[holding_dt, tag.index] = tag['TAG']
    t_3 = time.time()
    print(date, len(select_stocks), t_3-t_2, t_3-t_1)
holding = holding.fillna(method='ffill').fillna(0)
turnover = (holding.shift(1) - holding).fillna(0).abs().sum(axis=1)
ret = pd.read_csv(GetFileName('RAW_RETURN.csv'), index_col=[0]).loc[from_date:to_date]
daily_ret = (holding * ret).fillna(0).sum(axis=1)
exc = daily_ret - zz500_ret

print((exc.mean()+1)**(250/len(exc))-1, 250**0.5*exc.mean()/exc.std())
pnl = (exc + 1).cumprod()
exc.to_csv(GetFileName("model_4_exc.csv"))
turnover.to_csv(GetFileName("model_4_turnover.csv"))
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.xticks(ticks=range(len(pnl)), labels=pnl.index)
plt.plot(pnl.values)
plt.show()
x = range(len(pnl))
y1 = pnl.values
myfig = plt.figure(21)
ax1 = myfig.add_axes([1, 1, 3, 3])
ax1.plot(x, y1, label='pnl')
ax1.legend(loc='upper left', fontsize=8)
first_index = []
for i, j in zip(range(len(pnl)-1), range(1, len(pnl))):
    if pnl.index[j] - pnl.index[i] > 50:
        first_index.append(j)
ax1.set_xticks(first_index)
ax1.set_xticklabels(pnl.index[first_index])

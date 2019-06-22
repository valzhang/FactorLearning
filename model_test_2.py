import pandas as pd
import os
import tensorflow as tf
import datetime
import numpy as np
int2time = lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")
from_date = 20180101
to_date = 20190501

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

dl_path = 'H:\\DL\\data'
model_name = 'dl2_win.h5'
use_short = False
GetFileName = lambda x: os.path.join(dl_path, x)
factor_list = pd.read_csv(GetFileName("factor_list.csv"), names=['FACTOR_NAME', 'NEU_TYPE'])
all_df = pd.read_pickle(GetFileName('all_data.pkl'), compression='gzip').dropna()
all_tradedates = pd.read_csv(GetFileName('ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
stock_code = pd.read_csv(GetFileName('STOCK_CODE.csv'), header=None).values.T[0].tolist()
all_pool = GeneratePoolData(csv_path=dl_path)
zz500_ret = pd.read_csv(GetFileName('INDEX_000905.SH_RETURN.csv'), index_col=[0], header=None, names=['RETURN']).loc[from_date:to_date, 'RETURN']
GetUniverse = lambda uni, x: uni.columns[uni.loc[x]].tolist()
trade_dt = [x for x in all_tradedates if (x >= from_date) & (x < to_date)]
factor_dt = [trade_dt[0]] + [y for x, y in zip(trade_dt[:-1], trade_dt[1:]) if int(x / 100) != int(y / 100)]
holding = pd.DataFrame(index=trade_dt, columns=stock_code)
model = tf.keras.models.load_model(GetFileName(model_name))
import time
t_1 = time.time()
for date in factor_dt:
    t_2 = time.time()
    s = GetUniverse(all_pool, date)
    d_df = all_df.loc[all_df['FACTOR_DATE'] == date].set_index('STOCK_CODE').loc[s].reset_index()
    input_data = d_df[factor_list['FACTOR_NAME']].values
    tag_pro = model.predict(input_data)
    tag = pd.DataFrame(tag_pro.argmax(axis=1), index=d_df['STOCK_CODE'], columns=['TAG'])
    if use_short:
        tag[tag['TAG'] < 0] = tag[tag['TAG'] < 0] / tag[tag['TAG'] < 0].abs().sum()
    tag[tag['TAG'] > 0] = tag[tag['TAG'] > 0] / tag[tag['TAG'] > 0].abs().sum()
    holding_dt = trade_dt[trade_dt.index(date)+2]
    holding.loc[holding_dt, tag.index] = tag['TAG']
    t_3 = time.time()
    print(date, t_3-t_2, t_3-t_1)
holding = holding.fillna(method='ffill').fillna(0)
ret = pd.read_csv(GetFileName('RAW_RETURN.csv'), index_col=[0]).loc[from_date:to_date]
daily_ret = (holding * ret).fillna(0).sum(axis=1)
exc = daily_ret - zz500_ret
print ((exc.mean()+1)**(250/len(exc))-1, 250**0.5*exc.mean()/exc.std())
pnl = (exc + 1).cumprod()
# exc.to_csv("D:\\test.csv")
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

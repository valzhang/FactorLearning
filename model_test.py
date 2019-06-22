import pandas as pd
import os
import tensorflow as tf
import datetime
int2time = lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")

dl_path = 'H:\\DL\\data'
model_name = 'model_hs300.h5'
use_short = False
GetFileName = lambda x: os.path.join(dl_path, x)
factor_list = pd.read_csv(GetFileName("factor_list.csv"), names=['FACTOR_NAME', 'NEU_TYPE'])
all_df = pd.read_pickle(GetFileName('all_data.pkl'), compression='gzip').dropna()
all_tradedates = pd.read_csv(GetFileName('ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
stock_code = pd.read_csv(GetFileName('STOCK_CODE.csv'), header=None).values.T[0].tolist()
from_date = 20180101
to_date = 20190501
hs300_uni = pd.read_csv(GetFileName('RAW_UNIVERSE_000300_SH.csv'), index_col=[0]).loc[from_date:to_date].fillna(0) > 0.5
hs300_ret = pd.read_csv(GetFileName('INDEX_000300.SH_RETURN.csv' ), index_col=[0], header=None, names=['RETURN']).loc[from_date:to_date, 'RETURN']
GetUniverse = lambda x: hs300_uni.columns[hs300_uni.loc[x]].tolist()
trade_dt = [x for x in all_tradedates if (x >= from_date) & (x < to_date)]
factor_dt = [trade_dt[0]] + [y for x, y in zip(trade_dt[:-1], trade_dt[1:]) if int(x / 100) != int(y / 100)]
holding = pd.DataFrame(index=trade_dt, columns=stock_code)
model = tf.keras.models.load_model(GetFileName(model_name))
import time
t_1 = time.time()
for date in factor_dt:
    t_2 = time.time()
    s = GetUniverse(date)
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
exc = daily_ret - hs300_ret
pnl = (exc + 1).cumprod()
# exc.to_csv("D:\\test.csv")
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import numpy as np

plt.xticks(ticks=range(len(pnl)), labels=pnl.index)
plt.plot(pnl.values)
plt.show()
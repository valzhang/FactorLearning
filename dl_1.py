import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
dl_path = 'D:/DL_data'
GetFileName = lambda x: os.path.join(dl_path, x)
factor_list = pd.read_csv(GetFileName("factor_list.csv"), names=['FACTOR_NAME', 'NEU_TYPE'])
all_df = pd.read_pickle(GetFileName('all_data.pkl'), compression='gzip').dropna()
all_df['TAG'] = all_df.groupby('FACTOR_DATE')['RET'].rank(pct=True).apply(lambda x: np.int(np.floor(x*3-1e-4)-1))
train_last_date = 20180101
train_df = all_df.loc[all_df['FACTOR_DATE'] < train_last_date]
# test_df = all_df.loc[all_df['FACTOR_DATE'] > train_last_date]
# test_data = test_df[factor_list['FACTOR_NAME']].values
# test_df.loc[test_df['TAG'] > 1, 'TAG'] = 1
# test_label = pd.get_dummies(test_df['TAG']).values
train_df.loc[train_df['TAG'] > 1, 'TAG'] = 1
train_data = train_df[factor_list['FACTOR_NAME']].values
train_label = pd.get_dummies(train_df['TAG']).values
model = tf.keras.Sequential([
    # Adds a densely-connected layer with 64 units to the model:
    layers.Dense(25, activation='relu'),
    # Add another:
    layers.Dense(64, activation='relu'),
    # Add a softmax layer with 10 output units:
    layers.Dense(3, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_label, epochs=100, batch_size=10000)

raw_path = '\\\\10.33.4.224\\rawFile'
factor_path = '\\\\10.33.4.224\\factorFile'
all_tradedates = pd.read_csv(os.path.join(factor_path, 'ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
stock_code = pd.read_csv(os.path.join(factor_path, 'STOCK_CODE.csv'), header=None).values.T[0].tolist()
from_date = 20060101
to_date = 20190501
trade_dt = [x for x in all_tradedates if (x >= from_date) & (x < to_date)]
# factor_dt为每个月第一个交易日
factor_dt = [trade_dt[0]] + [y for x, y in zip(trade_dt[:-1], trade_dt[1:]) if int(x / 100) != int(y / 100)]
holding = pd.DataFrame(index=trade_dt, columns=stock_code)
for date in factor_dt:
    print(date)
    d_df = all_df.loc[all_df['FACTOR_DATE'] == date]
    input_data = d_df[factor_list['FACTOR_NAME']].values
    # 根据输入得到预测分类概率
    tag_pro = model.predict(input_data)
    # 将概率最高的label值映射到(-1, 0, 1)
    tag = pd.DataFrame(tag_pro.argmax(axis=1), index=d_df['STOCK_CODE'], columns=['TAG']) - 1
    # 将正向持仓和反向持仓分别归一化
    tag[tag['TAG'] < 0] = tag[tag['TAG'] < 0] / tag[tag['TAG'] < 0].abs().sum()
    tag[tag['TAG'] > 0] = tag[tag['TAG'] > 0] / tag[tag['TAG'] > 0].abs().sum()
    # 计算实际持仓开始日期，为每个月第三个交易日
    holding_dt = trade_dt[trade_dt.index(date)+2]
    holding.loc[holding_dt, tag.index] = tag['TAG']
# 将非交易日权重调整为往前最新一个交易日的权重
holding = holding.fillna(method='ffill').fillna(0)
ret = pd.read_csv(os.path.join(raw_path, 'RAW_RETURN.csv'), index_col=[0])
# 将权重和当日收益相乘累加得到多空持仓日收益
daily_ret = (holding * ret.loc[from_date:to_date]).fillna(0).sum(axis=1)
pnl = (daily_ret + 1).cumprod()
from matplotlib import pyplot as plt
plt.plot(pnl)
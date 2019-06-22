import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
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
def Ret2Tag(data, use_short):
    tag = data.groupby('FACTOR_DATE')['RET'].rank(pct=True).apply(lambda x: np.int(np.floor(x*3-1e-4)-1))
    if not use_short:
        tag[tag < 0] = 0
    return tag
def WinTage(data, index):
    tmp = pd.merge(data, index, left_on=['FACTOR_DATE'], right_index=True)
    tag = (tmp['RET'] >= tmp['RETURN']).astype(np.int)
    return tag
def LoadData(raw_data, from_date, to_date):
    train_df = raw_data.loc[(all_df['FACTOR_DATE'] < to_date) & (all_df['FACTOR_DATE'] >= from_date)]
    train_df.loc[train_df['TAG'] > 1, 'TAG'] = 1
    train_data = train_df[factor_list['FACTOR_NAME']].values
    train_label = pd.get_dummies(train_df['TAG']).values
    return train_data, train_label

dl_path = 'H:\\DL\\data'
GetFileName = lambda x: os.path.join(dl_path, x)

# import data
uni_name = '000300.SH'
model_name = 'dl2_win.h5'
use_short = False
label_num = 2
from_date = 20060101
to_date = 20190501
train_last_date = 20180101
GetFileName = lambda x: os.path.join(dl_path, x)
factor_list = pd.read_csv(GetFileName("factor_list.csv"), names=['FACTOR_NAME', 'NEU_TYPE'])
factor_num = len(factor_list)
raw_df = pd.read_pickle(GetFileName('all_data.pkl'), compression='gzip').dropna()
all_df = PoolFilter(data=raw_df, universe_name=uni_name, from_date=from_date, to_date=to_date)
hs300_ret = pd.read_csv(GetFileName('INDEX_000300.SH_RETURN.csv' ), index_col=[0], header=None, names=['RETURN'])
hs300 = (1+hs300_ret).rolling(20).apply(lambda x: x.prod()-1, raw=True).shift(-19).loc[from_date:to_date]
# all_df['TAG'] =Ret2Tag(data=all_df, use_short=use_short)
all_df['TAG'] = WinTage(data=all_df, index=hs300)
train_data, train_label = LoadData(raw_data=all_df, from_date=from_date, to_date=train_last_date)
val_data, val_label = LoadData(raw_data=all_df, from_date=train_last_date, to_date=to_date)
# model init
model = tf.keras.Sequential([
    # Adds a densely-connected layer with 64 units to the model:
    layers.Dense(factor_num, activation='relu'),
    # Add another:
    layers.Dense(64, activation='relu'),
    # Add a softmax layer with 10 output units:
    layers.Dropout(0.2, noise_shape=None, seed=None),
    layers.Dense(label_num, activation='softmax')])

# model compile
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model train
model.fit(train_data, train_label, epochs=10, batch_size=1000, validation_data=(val_data, val_label))

# save model
model.save(GetFileName(model_name))
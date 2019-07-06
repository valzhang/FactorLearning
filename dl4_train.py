import tensorflow as tf
import os
import pandas as pd
import numpy as np
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
def WinTage(data, index):
    tmp = pd.merge(data, index, left_on=['FACTOR_DATE'], right_index=True)
    tag = (tmp['RET'] >= tmp['RETURN']).astype(np.int)
    return tag

def LoadData(raw_data, f_names, from_date, to_date, batch_size, repeat=None, return_labels=True):
    train_df = raw_data.loc[(raw_data['FACTOR_DATE'] < to_date) & (raw_data['FACTOR_DATE'] >= from_date)]
    if return_labels:
        dataset = tf.data.Dataset.from_tensor_slices((dict(train_df[f_names]), train_df['TAG']))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(train_df[f_names]))
    dataset = dataset.shuffle(1000).repeat(count=repeat).batch(batch_size)
    return dataset
def ReadIndexReturn(index_name):
    return pd.read_csv(GetFileName('INDEX_%s_RETURN.csv' % index_name), index_col=[0], header=None, names=['RETURN'])
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
batch_size = 1000
train_steps = 1000
def main(argv):

    # read data
    factor_list, factor_names, factor_num = ReadFactorListInfor(filename=factorlist_filename)
    all_df = LoadAllData(filename=data_filename, uni_name=uni_name, from_date=from_date, to_date=to_date)
    # all_df['TAG'] =Ret2Tag(data=all_df)
    all_df['TAG'] =WinTage(data=all_df, index=ReadIndexReturn(index_name=index_name).loc[from_date:to_date, 'RETURN'])
    # train_data, train_label = LoadData(raw_data=all_df, from_date=from_date, to_date=train_last_date)
    # val_data, val_label = LoadData(raw_data=all_df, from_date=train_last_date, to_date=to_date)

    # dnn
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
    classifier.train(
        input_fn=lambda: LoadData(raw_data=all_df, f_names=factor_names, from_date=from_date, to_date=train_last_date, batch_size=batch_size), steps=train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda: LoadData(raw_data=all_df, f_names=factor_names, from_date=train_last_date, to_date=to_date, batch_size=batch_size, repeat=1))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


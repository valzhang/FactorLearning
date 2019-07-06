import pandas as pd
import os
import tensorflow as tf
tf.set_random_seed(2)
import datetime
import numpy as np
import sklearn

dl_path = 'D:\\DL_data'
GetFileName = lambda x: os.path.join(dl_path, x)

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

def FixedTag(data, N):
    return (data.groupby('FACTOR_DATE')['RET'].rank(ascending=False, na_option='bottom') <= N).astype(np.int)

def LongShortTag(data):
    return data.groupby('FACTOR_DATE')['RET'].rank(pct=True).apply(lambda x: np.int(np.floor(x*3-1e-4)-1))

def WinTag(data, index):
    tmp = pd.merge(data, index, left_on=['FACTOR_DATE'], right_index=True)
    tag = (tmp['RET'] >= tmp['RETURN']).astype(np.int)
    return tag

def LoadData(raw_data, f_names, from_date, to_date, batch_size, repeat=None, return_labels=True):
    train_df = raw_data.loc[(raw_data['FACTOR_DATE'] < to_date) & (raw_data['FACTOR_DATE'] >= from_date)]
    if return_labels:
        dataset = tf.data.Dataset.from_tensor_slices((dict(train_df[f_names]), pd.get_dummies(train_df['TAG'])))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(train_df[f_names]))
    # dataset = dataset.shuffle(1000).repeat(count=repeat).batch(batch_size)
    dataset = dataset.repeat(count=repeat).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

def ReadIndexReturn(index_name):
    return pd.read_csv(GetFileName('INDEX_%s_RETURN.csv' % index_name), index_col=[0], header=None, names=['RETURN'])

def GeneratePoolData(csv_path, from_date, to_date, stock_code):
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

def my_model(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu)
    net = tf.layers.dense(net, units=params['linear_units'], activation=None)
    tf.Print(net, [net], summarize=10000)
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # win index use max
    # loss = tf.reduce_max(loss)
    loss = tf.reduce_sum(loss)
    # Compute evaluation metrics.
    one_dim_label = tf.argmax(labels, 1)
    accuracy = tf.metrics.accuracy(labels=one_dim_label,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def model_1(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    # net = tf.layers.dense(net, units=params['linear_units'], activation=None)
    tf.Print(net, [net], summarize=10000)
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # win index use max
    # loss = tf.reduce_max(loss)
    # loss = tf.reduce_sum(loss)
    loss = tf.reduce_mean(loss)

    # Compute evaluation metrics.
    one_dim_label = tf.argmax(labels, 1)

    precision, precision_update_op = tf.metrics.precision(labels=one_dim_label,
                                        predictions=predicted_classes,
                                        name='precision')

    recall, recall_update_op = tf.metrics.recall(labels=one_dim_label,
                                  predictions=predicted_classes,
                                  name='recall')
    f1_score, f1_update_op = tf.metrics.mean((2 * precision_update_op * recall_update_op) / (recall_update_op + recall_update_op), name='f1_score')
    tf.summary.scalar('precision', precision_update_op)
    tf.summary.scalar('recall', recall_update_op)
    tf.summary.scalar('f1_score', f1_update_op)
    eval_metric_ops = {
        "precision": (precision, precision_update_op),
        "recall": (recall, recall_update_op),
        "f1_score": (f1_score, f1_update_op)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def model_0(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    # net = tf.layers.dense(net, units=params['linear_units'], activation=None)
    tf.Print(net, [net], summarize=10000)
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # win index use max
    # loss = tf.reduce_max(loss)
    loss = tf.reduce_sum(loss)

    # Compute evaluation metrics.
    one_dim_label = tf.argmax(labels, 1)
    accuracy = tf.metrics.accuracy(labels=one_dim_label,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def model_2(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=None)
    # net = tf.layers.dense(net, units=params['linear_units'], activation=None)
    tf.Print(net, [net], summarize=10000)
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # win index use max
    # loss = tf.reduce_max(loss)
    # loss = tf.reduce_sum(loss)
    loss = tf.reduce_mean(loss)

    # Compute evaluation metrics.
    one_dim_label = tf.argmax(labels, 1)

    precision, precision_update_op = tf.metrics.precision(labels=one_dim_label,
                                        predictions=predicted_classes,
                                        name='precision')

    recall, recall_update_op = tf.metrics.recall(labels=one_dim_label,
                                  predictions=predicted_classes,
                                  name='recall')
    f1_score, f1_update_op = tf.metrics.mean((2 * precision_update_op * recall_update_op) / (recall_update_op + recall_update_op), name='f1_score')
    tf.summary.scalar('precision', precision)
    tf.summary.scalar('recall', recall)
    tf.summary.scalar('f1_score', f1_score)
    eval_metric_ops = {
        "precision": (precision, precision_update_op),
        "recall": (recall, recall_update_op),
        "f1_score": (f1_score, f1_update_op)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

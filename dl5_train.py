from Init import *
int2time = lambda x: datetime.datetime.strptime(str(x), "%Y%m%d")

# setting config
uni_name = None
index_name = '000905.SH'
model_name = 'dl5_ls'
data_filename = 'all_data.pkl'
factorlist_filename = 'factor_list.csv'
label_num = 3
from_date = 20060101
to_date = 20190501
train_last_date = 20180101
batch_size = 10000
train_steps = 10000

def main(argv):

    # read data
    factor_list, factor_names, factor_num = ReadFactorListInfor(filename=factorlist_filename)
    all_df = LoadAllData(filename=data_filename, uni_name=uni_name, from_date=from_date, to_date=to_date)
    # all_df['TAG'] =FixedTag(data=all_df, N=200)
    all_df['TAG'] =LongShortTag(data=all_df)
    print(all_df['TAG'].max(), all_df['TAG'].min())
    # all_df['TAG'] =WinTage(data=all_df, index=ReadIndexReturn(index_name=index_name).loc[from_date:to_date, 'RETURN'])
    # train_data, train_label = LoadData(raw_data=all_df, from_date=from_date, to_date=train_last_date)
    # val_data, val_label = LoadData(raw_data=all_df, from_date=train_last_date, to_date=to_date)

    # dnn
    my_feature_columns = []
    for key in factor_names:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [64, 32],
            # The model must choose between 3 classes.
            'linear_units': 16,
            'n_classes': label_num,
        },
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
from Init import *
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


all_df = LoadAllData(filename=data_filename, uni_name=uni_name, from_date=from_date, to_date=to_date)
all_df['TAG'] =FixedTag(data=all_df, N=200)

factor_list, factor_names, factor_num = ReadFactorListInfor(filename=factorlist_filename)
all_tradedates = pd.read_csv(GetFileName('ALLDATES.csv'), header=None, names=['TRADE_DT']).values.T[0].tolist()
stock_code = pd.read_csv(GetFileName('STOCK_CODE.csv'), header=None).values.T[0].tolist()
all_pool = GeneratePoolData(csv_path=dl_path, from_date=from_date, to_date=to_date, stock_code=stock_code)
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
classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [64, 32],
        # The model must choose between 3 classes.
        'linear_units': 16,
        'n_classes': 2,
    },
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
        if int(class_id) > 0.5:
            select_stocks.append(expec)

    holding_dt = trade_dt[trade_dt.index(date)+2]
    if len(select_stocks) > 0:
        holding.loc[holding_dt, select_stocks] = 1.0 / len(select_stocks)
    t_3 = time.time()
    print(date, len(select_stocks), len(d_df), t_3-t_2, t_3-t_1)
holding = holding.fillna(method='ffill').fillna(0)
turnover = (holding.shift(1) - holding).fillna(0).abs().sum(axis=1)
ret = pd.read_csv(GetFileName('RAW_RETURN.csv'), index_col=[0]).loc[from_date:to_date]
daily_ret = (holding * ret).fillna(0).sum(axis=1)
exc = daily_ret - zz500_ret

print((exc.mean()+1)**(250/len(exc))-1, 250**0.5*exc.mean()/exc.std())
pnl = (exc + 1).cumprod()
exc.to_csv(GetFileName("model_5_exc.csv"))
turnover.to_csv(GetFileName("model_5_turnover.csv"))
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

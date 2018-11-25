import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import coint

from .util import context

from forecaster import retriever
from forecaster import timeseries

df = retriever.get_data('bitcoin')

binary_headers = ['price', 'transactions']
target_header = 'today_price'

for header in df.columns.values:
    if header not in binary_headers:
        df[header] = timeseries.first_differentiate(df[header])

# Drop nan values
restricted_df = df.iloc[333:].dropna()

for header in restricted_df.columns.values:
    if header in binary_headers or header == target_header:
        continue

    # # Metrics and plots
    # timeseries.metrics(restricted_df[target_header])
    # timeseries.plot(restricted_df.index, restricted_df[target_header], '', '')

    # timeseries.metrics(restricted_df[header])
    # timeseries.plot(restricted_df.index, restricted_df[header], '', '')

    # Standalone Granger causality
    stack = np.column_stack((restricted_df[target_header], restricted_df[header]))
    res = grangercausalitytests(stack, 55, verbose=False)

    print('\n\n', header)
    for k in res.keys():
        print(res[k][0]['ssr_ftest'][1])

    # print(coint(restricted_df[price_header], restricted_df[header], maxlag=2, autolag=None))

# VAR model Granger causality
# mdata = restricted_df[['price', 'positive_reply', 'positive_topic']]
# mdata.index = pd.DatetimeIndex(restricted_df['date'])

# model = VAR(mdata)

# results = model.fit(13)

# results.summary()
# results.plot().show()

# results.plot_acorr().show()

# tres = results.test_causality('price', ['positive_reply'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['positive_topic'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['positive_reply', 'positive_topic'], kind='f')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)

# tres = results.test_causality('price', ['positive_reply', 'positive_topic'], kind='wald')
# print('tres.test_statistic:', tres.test_statistic)
# print('tres.crit_value:', tres.crit_value)
# print('tres.pvalue:', tres.pvalue)
# print('tres.df:', tres.df)
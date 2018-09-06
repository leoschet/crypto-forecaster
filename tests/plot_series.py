import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

from .util import context

from forecaster import retriever, timeseries

df = retriever.get_data('bitcoin')

start_date = pd.to_datetime('2013-12-01')
end_date = pd.to_datetime('2016-02-01')

mask = (df.index.values > start_date) & (df.index.values <= end_date)

date = df.index.values

# Price
price = df['price']
timeseries.metrics(price)
timeseries.plot(date, price, 'Price')

# # Price zscore
# price_zscore = stats.zscore(price)
# timeseries.metrics(price_zscore)
# timeseries.plot(date, price_zscore, 'Price zscore')

# Standardized price
stand_price = timeseries.standardize_laggedly(price)
df['price'] = stand_price
restricted_df = df.dropna()

timeseries.metrics(restricted_df['price'])
timeseries.plot(restricted_df.index.values, restricted_df['price'], 'Standardized price')

# # # Standardized price zscore
# # stand_price_zscore = timeseries.standardize_laggedly(price_zscore)
# # df['price'] = stand_price_zscore
# # restricted_df = df.dropna()

# # timeseries.metrics(restricted_df['price'])
# # timeseries.plot(restricted_df.index.values, restricted_df['price'], 'Standardized price zscore')

# # Stationary price
# stat_price = timeseries.stationarize(price)
# df['price'] = stat_price
# restricted_df = df.dropna()

# timeseries.metrics(restricted_df['price'])
# timeseries.plot(restricted_df.index.values, restricted_df['price'], 'Stationary price')

# # # Stationary price zscore
# # stat_price_zscore = timeseries.stationarize(price_zscore)
# # df['price'] = stat_price_zscore
# # restricted_df = df.dropna()

# # timeseries.metrics(restricted_df['price'])
# # timeseries.plot(restricted_df.index.values, restricted_df['price'], 'Stationary price zscore')

# # Diff price
# diff_price = np.log(price).diff()
# df['price'] = diff_price
# restricted_df = df.dropna()

# timeseries.metrics(restricted_df['price'])
# timeseries.plot(restricted_df.index.values, restricted_df['price'], 'Diff price')



# # Positive reply
# positive = df['positive_reply']
# timeseries.metrics(positive)
# timeseries.plot(date, positive, 'Positive reply')

# # # Positive zscore
# # positive_zscore = stats.zscore(positive)
# # timeseries.metrics(positive_zscore)
# # timeseries.plot(date, positive_zscore, 'Positive zscore')

# # Standardized positive
# stand_positive = timeseries.standardize_laggedly(positive)
# df['positive_reply'] = stand_positive
# restricted_df = df.dropna()

# timeseries.metrics(restricted_df['positive_reply'])
# timeseries.plot(restricted_df.index.values, restricted_df['positive_reply'], 'Standardized positive')

# # # Standardized positive zscore
# # stand_positive_zscore = timeseries.standardize_laggedly(positive_zscore)
# # df['positive_reply'] = stand_positive_zscore
# # restricted_df = df.dropna()

# # timeseries.metrics(restricted_df['positive_reply'])
# # timeseries.plot(restricted_df.index.values, restricted_df['positive_reply'], 'Standardized positive zscore')

# # Stationary positive
# stat_positive = timeseries.stationarize(positive)
# df['positive_reply'] = stat_positive
# restricted_df = df.dropna()

# timeseries.metrics(restricted_df['positive_reply'])
# timeseries.plot(restricted_df.index.values, restricted_df['positive_reply'], 'Stationary positive')

# # # Stationary positive zscore
# # stat_positive_zscore = timeseries.stationarize(positive_zscore)
# # df['positive_reply'] = stat_positive_zscore
# # restricted_df = df.dropna()

# # timeseries.metrics(restricted_df['positive_reply'])
# # timeseries.plot(restricted_df.index.values, restricted_df['positive_reply'], 'Stationary positive zscore')

# # Diff positive
# diff_positive = np.log(positive).diff()
# df['positive_reply'] = diff_positive
# restricted_df = df.dropna()

# timeseries.metrics(restricted_df['positive_reply'])
# timeseries.plot(restricted_df.index.values, restricted_df['positive_reply'], 'Diff positive')

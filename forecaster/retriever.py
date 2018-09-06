import os
from functools import reduce

import pandas as pd
import numpy as np

from . import settings


def get_data(cryptocurrency):
    crypto_path = os.path.join(settings.RESOURCES_DIR, cryptocurrency)

    # Currency related data frames
    price_df = _read_csv(os.path.join(crypto_path, 'price.csv'))
    _lower_headers(price_df)
    # price_df = _floaterize_prices(price_df)
    price_df['date'] = pd.to_datetime(price_df['date'])

    transactions_df = _read_csv(os.path.join(crypto_path, 'transactions.csv'))
    _lower_headers(transactions_df)
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])

    # Forum related data frames
    reply_df = _read_csv(os.path.join(crypto_path, 'reply_opinion.csv'))
    _lower_headers(reply_df)

    topic_df = _read_csv(os.path.join(crypto_path, 'topic_opinion.csv'))
    _lower_headers(topic_df)

    # Categorize vader scores
    reply_df = _transform_vader_series(reply_df, 'reply')
    topic_df = _transform_vader_series(topic_df, 'topic')

    # Drop useless columns
    _drop_inplace(reply_df, ['reply', 'vader'])
    _drop_inplace(topic_df, ['topic', 'reply', 'topiccontent', 'vader', 'opinion'])

    # Group by date and aggregate vader categorical columns
    reply_df = _fold_categorical_vader(reply_df, 'reply', by='date')
    topic_df = _fold_categorical_vader(topic_df, 'topic', by='date', agg={'views':'sum'})

    # Calculate 
    reply_df = _sum_categorical_vader(reply_df, 'reply')
    topic_df = _sum_categorical_vader(topic_df, 'topic')  

    # Merge data frames
    dfs = [reply_df, topic_df]
    forum_related = _merge_frames(dfs, on='date')
    forum_related['date'] = pd.to_datetime(forum_related['date'])

    # Merge data frames
    dfs = [price_df, transactions_df, forum_related]
    full_df = _merge_frames(dfs, on='date')

    # Sort by date
    full_df = full_df.sort_values(by='date')

    # Set today's label_headers' values as tomorrow's features
    for label in ['price', 'transactions']:
        full_df['today_' + label] = full_df[label].copy()

    # Binerize labels (price and transactions)
    full_df = _categorize_labels(full_df)

    # Set dates to index
    full_df.index = pd.DatetimeIndex(full_df['date'])
    full_df = full_df.drop(columns='date')

    return full_df

def _read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')
    
    return df

def _lower_headers(df):
    df.columns = map(str.lower, df.columns)

def _floaterize_prices(price_df):
    remove_comma = lambda text: text.replace(',', '')

    # price_df['open'] = price_df['open'].apply(remove_comma).astype(float)
    # price_df['close'] = price_df['close'].apply(remove_comma).astype(float)
    # price_df['high'] = price_df['high'].apply(remove_comma).astype(float)
    # price_df['low'] = price_df['low'].apply(remove_comma).astype(float)
    
    price_df['price'] = price_df['price'].apply(remove_comma).astype(float)

    return price_df

def _categorize_labels(df):
    labels = ['price', 'transactions']

    df[labels] = df[labels].diff().apply(np.sign).astype(str)
    df = df.dropna()

    return df

def _transform_vader_series(df, header_suffix):
    """
    Transform vader series of dataframe to categorical vader series.
    """

    categorical_vader = list(zip(*df['vader'].map(_categorize_vader)))

    categorical_columns = _get_categorical_vader(header_suffix)

    for index, header in enumerate(categorical_columns):
        df[header] = categorical_vader[index]

    return df

def _categorize_vader(score):
    """
    Transform vader score into one of the following categorical values:
    - Very negative
    - Negative
    - Neutral
    - Positive
    - Very positive

    Returns a tuple with 5 positions (one for each category)
    where one element contains 1 and the others are 0.
    """
    if score < -0.6:
        # Very negative
        return (1, 0, 0, 0, 0)
    elif score < -0.2:
        # Negative
        return (0, 1, 0, 0, 0)
    elif score < 0.2:
        # Neutral
        return (0, 0, 1, 0, 0)
    elif score < 0.6:
        # Positive
        return (0, 0, 0, 1, 0)
    else:
        # Very positive
        return (0, 0, 0, 0, 1)

def _drop_inplace(df, columns):
    df.drop(columns, inplace=True, axis=1)

def _fold_categorical_vader(df, header_suffix, by=None, agg={}):
    agg_type = {}
    categorical_columns = _get_categorical_vader(header_suffix)
    
    for header in categorical_columns:
        agg_type[header] = 'sum'

    for column, type_ in agg.items():
        agg_type[column] = type_

    return df.groupby(by).agg(agg_type).reset_index()

def _sum_categorical_vader(df, header_suffix):
    categorical_columns = _get_categorical_vader(header_suffix)
    df['total_' + header_suffix] = df[categorical_columns].sum(axis=1)
    return df

def _get_categorical_vader(header_suffix):
    very_negative = 'very_negative_' + header_suffix
    negative = 'negative_' + header_suffix
    neutral = 'neutral_' + header_suffix
    positive = 'positive_' + header_suffix
    very_positive = 'very_positive_' + header_suffix

    return [very_negative, negative, neutral, positive, very_positive]

def _merge_frames(dfs, on=None):
    return reduce(lambda left,right: pd.merge(left,right,on=on), dfs)
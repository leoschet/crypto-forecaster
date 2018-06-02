import os
from functools import reduce

import pandas as pd

from . import settings


def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)

def get_data(cryptocurrency):
    crypto_path = os.path.join(settings.RESOURSES_DIR, cryptocurrency)

    # Currency related data frames
    price_df = _read_csv(os.path.join(crypto_path, 'price.csv'))
    _lower_headers(price_df)

    transactions_df = _read_csv(os.path.join(crypto_path, 'transactions.csv'))
    _lower_headers(transactions_df)

    # Forum related data frames
    reply_df = _read_csv(os.path.join(crypto_path, 'reply_opinion.csv'))
    _lower_headers(reply_df)

    topic_df = _read_csv(os.path.join(crypto_path, 'topic_opinion.csv'))
    _lower_headers(topic_df)

    # Categorize vader scores
    reply_df = _transform_vader_series(reply_df, 'reply')
    # Drop useless columns
    _drop_inplace(reply_df, ['reply', 'vader'])
    # Group by date and aggregate vader categorical columns
    reply_df = _fold_categorical_vader(reply_df, 'reply', by='date')
    reply_df = _sum_categorical_vader(reply_df, 'reply')

    # Categorize vader scores
    topic_df = _transform_vader_series(topic_df, 'topic')
    # Drop useless columns
    _drop_inplace(topic_df, ['topic', 'reply', 'topiccontent', 'vader', 'opinion'])
    # Group by date and aggregate vader categorical columns
    topic_df = _fold_categorical_vader(topic_df, 'topic', by='date', agg={'views':'sum'})
    topic_df = _sum_categorical_vader(topic_df, 'topic')  

    dfs = [price_df, transactions_df, reply_df, topic_df]

    # Merge data frames
    full_df = _merge_frames(dfs, on='date')

    return full_df

def _read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')
    
    return df

def _lower_headers(df):
    df.columns = map(str.lower, df.columns)

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
    df[header_suffix + '_total'] = df[categorical_columns].sum(axis=1)
    return df

def _get_categorical_vader(header_suffix):
    very_negative = header_suffix + '_very_negative'
    negative = header_suffix + '_negative'
    neutral = header_suffix + '_neutral'
    positive = header_suffix + '_positive'
    very_positive = header_suffix + '_very_positive'

    return [very_negative, negative, neutral, positive, very_positive]

def _merge_frames(dfs, on=None):
    return reduce(lambda left,right: pd.merge(left,right,on=on), dfs)
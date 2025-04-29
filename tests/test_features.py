import pandas as pd
import pytest
from src.features.feature_engineering import (
    add_funding_per_round,
    add_funding_timespan,
    add_relationships_per_milestone,
    add_in_tech_hub,
    create_features,
)


def sample_df():
    return pd.DataFrame({
        'funding_total_usd': [100, 200, 0],
        'funding_rounds': [1, 2, 0],
        'age_first_funding_year': [1, 2, 3],
        'age_last_funding_year': [2, 3, 1],
        'relationships': [10, 0, 5],
        'milestones': [2, 1, 0],
        'is_CA': [1, 0, 0],
        'is_NY': [0, 1, 0],
        'is_MA': [0, 0, 0],
        'is_TX': [0, 0, 1],
    })


def test_add_funding_per_round():
    df = sample_df()
    df2 = add_funding_per_round(df)
    assert 'funding_per_round' in df2.columns
    assert df2['funding_per_round'].tolist() == [100, 100, 0]


def test_add_funding_timespan():
    df = sample_df()
    df2 = add_funding_timespan(df)
    assert 'funding_timespan' in df2.columns
    assert df2['funding_timespan'].tolist() == [1, 1, 0]


def test_add_relationships_per_milestone():
    df = sample_df()
    df2 = add_relationships_per_milestone(df)
    assert 'relationships_per_milestone' in df2.columns
    assert df2['relationships_per_milestone'].tolist() == [5, 0, 5]


def test_add_in_tech_hub_default():
    df = sample_df()
    df2 = add_in_tech_hub(df)
    assert 'in_tech_hub' in df2.columns
    assert df2['in_tech_hub'].tolist() == [1, 1, 1]


def test_create_features_chain():
    df = sample_df()
    df2 = create_features(df)
    for col in [
        'funding_per_round',
        'funding_timespan',
        'relationships_per_milestone',
        'in_tech_hub'
    ]:
        assert col in df2.columns
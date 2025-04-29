import pandas as pd
from typing import List, Optional


def add_funding_per_round(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la característica de financiación por ronda."""
    df = df.copy()
    if 'funding_total_usd' in df.columns and 'funding_rounds' in df.columns:
        df['funding_per_round'] = df['funding_total_usd'] / df['funding_rounds'].replace(0, 1)
    return df


def add_funding_timespan(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la característica de duración de financiación."""
    df = df.copy()
    cols = ['age_last_funding_year', 'age_first_funding_year']
    if all(col in df.columns for col in cols):
        timespan = df['age_last_funding_year'] - df['age_first_funding_year']
        df['funding_timespan'] = timespan.apply(lambda x: max(0, x))
    return df


def add_relationships_per_milestone(df: pd.DataFrame) -> pd.DataFrame:
    """Añade el ratio de relaciones por hito."""
    df = df.copy()
    if 'relationships' in df.columns and 'milestones' in df.columns:
        df['relationships_per_milestone'] = (
            df['relationships'] / df['milestones'].replace(0, 1)
        )
    return df


def add_in_tech_hub(df: pd.DataFrame, hub_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Añade una columna binaria si está en un hub tecnológico."""
    df = df.copy()
    if hub_columns is None:
        hub_columns = ['is_CA', 'is_NY', 'is_MA', 'is_TX']
    available = [col for col in hub_columns if col in df.columns]
    if available:
        df['in_tech_hub'] = df[available].max(axis=1)
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas las transformaciones de ingeniería de características."""
    df = add_funding_per_round(df)
    df = add_funding_timespan(df)
    df = add_relationships_per_milestone(df)
    df = add_in_tech_hub(df)
    return df

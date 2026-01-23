from typing import Optional, Tuple

import pandas as pd


def impute_with_group_median(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Optional[list] = None,
    fallback: Optional[float] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Impute missing values in a column using optional group-wise median and a fallback.

    This function fills missing values in the specified numeric column. If group columns are provided and present,
    missing values are filled with the median value within each group. If any values are still missing, the overall
    median or a specified fallback value is used.

    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        value_col (str): Column in which to impute missing values.
        group_cols (Optional[list], optional): List of columns to group by for median calculation. Defaults to None.
        fallback (Optional[float], optional): Value to use as fallback if group-wise and global median are not desired. 
            If None, uses the global median after group-level fills.

    Returns:
        Tuple[pd.Series, pd.Series]: 
            - The filled column as a pandas Series.
            - Boolean indicator Series, True where original value was missing.

    Error-handling:
        - Raises KeyError if value_col is not in df.
        - Ignores missing/invalid group_cols.
        - Does not modify the input dataframe in place.
    """
    series = df[value_col]
    indicator = series.isna()
    filled = series.copy()

    if group_cols:
        valid_groups = [col for col in group_cols if col in df.columns]
        if valid_groups:
            group_medians = df.groupby(valid_groups)[value_col].transform("median")
            filled = filled.fillna(group_medians)

    if fallback is None:
        fallback = filled.median()
    filled = filled.fillna(fallback)

    return filled, indicator


def fill_categorical(
    df: pd.DataFrame,
    col: str,
    fill_value: str = "Unknown"
) -> None:
    """
    Fill missing values in a categorical column and record imputation status.

    This function fills NA values in a categorical (object or category dtype) column
    with a specified string. It also creates a boolean indicator column reflecting
    which entries were originally missing.

    Args:
        df (pd.DataFrame): DataFrame in which to fill missing categorical values.
        col (str): Name of the categorical column to fill.
        fill_value (str, optional): Value to use to fill missing entries. Defaults to "Unknown".

    Returns:
        None. The dataframe is modified in place and a new column is added: f"{col}_was_imputed" (bool).

    Error-handling:
        - Raises KeyError if col is not in df.
        - If column is not a category/object type, no conversion is performed.
    """
    indicator = df[col].isna()
    df[col] = df[col].fillna(fill_value)
    df[f"{col}_was_imputed"] = indicator


def apply_imputation(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Optional[list] = None,
    fallback: Optional[float] = None,
    override_fallback: Optional[float] = None
) -> None:
    """
    Apply imputation to a numeric column with group-wise median and fallback options.

    This function imputes missing data in a specified numeric column using the impute_with_group_median
    function, either by fallback value or with group-wise medians as appropriate. It also writes a
    boolean indicator column reflecting imputed values.

    Args:
        df (pd.DataFrame): The DataFrame to modify in place.
        value_col (str): Name of the column to impute.
        group_cols (Optional[list], optional): Columns to use for group-wise median calculation. Defaults to None.
        fallback (Optional[float], optional): Fallback to use if all group/statistical fills are exhausted.
        override_fallback (Optional[float], optional): If provided, overrides fallback argument.

    Returns:
        None. Modifies the dataframe in place by updating value_col and creating f"{value_col}_imputed" (bool).

    Error-handling:
        - Raises KeyError if value_col is not in df.
        - Group columns not present are ignored quietly.
    """
    final_fallback = override_fallback if override_fallback is not None else fallback
    filled, indicator = impute_with_group_median(df, value_col, group_cols, final_fallback)
    df[f"{value_col}_imputed"] = indicator
    df[value_col] = filled


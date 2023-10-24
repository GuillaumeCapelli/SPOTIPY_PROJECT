import pandas as pd

def clean_tracks(df):
    """
    clean tracks, remove nulls and deduplicates rows n cells.
    
    parameters: df = the dataframe to be cleaned
    
    returns: clean dataframe, states how many row that were removed due to duplicates
    , states how many row total removed
    """
    row_cnt = df.shape[0]
    number_duplicate = df.duplicated().sum()
    df = df.dropna()
    df = df.drop_duplicates()
    final_row_cnt = df.shape[0]
    rows_removed = row_cnt - final_row_cnt
    return df, (number_duplicate, rows_removed)
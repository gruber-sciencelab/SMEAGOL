import glob
import pandas as pd


def collect_data(file_pattern, file_ids, col_with_index, col_to_collect):
    """This function is used to aggregate multiple SMEAGOL result files,
    for example on different genomes. It collects a specific column
    (col_to_collect) from all the result files and merges these columns
    into a single dataframe. The merge is based on a specific index column
    (col_with_index) that should be present in all files.

    Args:
        file_pattern (str): pattern to search for in filenames
        file_ids (list): list of IDs, one per file
        col_with_index (str): Name of the column to set as index
        col_to_collect (str): Name of the column to collect

    Returns:
        df_all (pandas DataFrame): A concatenated dataframe containing the
                                   `col_to_collect` column from all files
                                   matching the pattern `file_pattern`.
    """

    file_list = sorted(glob.glob(file_pattern))

    df_list = []

    for curr_id, curr_file in zip(file_ids, file_list):
        curr_df = pd.read_table(curr_file)
        curr_df = curr_df.set_index(col_with_index)
        curr_df = curr_df.loc[:, [col_to_collect]]
        curr_df.columns = [curr_id]
        df_list.append(curr_df)

    df_all = pd.concat(df_list, ignore_index=False, axis=1)

    return df_all

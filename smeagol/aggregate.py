import glob
import pandas as pd

def collect_data(file_pattern, 
                 col_with_index, 
                 col_to_collect):
    """Collect a specific column from all result files and merge them based on a specific index column.
    
    Args:
        file_pattern (str): pattern to search for in filenames
        col_with_index (str): column to set as index
        col_to_collect (str): column to collect
        
    Returns:
        df_all (pandas DataFrame): A concatenated dataframe containing the `col_to_collect` column from all files matching the pattern `file_pattern`.
    """
    
    file_list = sorted(glob.glob(file_pattern))

    df_list = []

    for curr_file in file_list:
        curr_df = pd.read_table(curr_file)
        curr_df = curr_df.set_index(col_with_index)
        curr_df = curr_df.loc[:,[col_to_collect]]
        curr_file_id = str(curr_file.split("/")[2])
        curr_df.columns = [curr_file_id]
        df_list.append(curr_df)

    df_all = pd.concat(df_list, ignore_index=False, axis=1)

    return(df_all)
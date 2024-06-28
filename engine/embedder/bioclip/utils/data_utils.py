import pandas as pd

def get_taxon_key_from_df(file_name, df, key='genusKey'):
    if '_crop' in file_name:
        file_name = file_name.split('_crop')[0] + '.JPG'
    if 'tile' in file_name:
        file_name = file_name.split('_tile')[0] + '.JPG'
    search_row = df[df['fileName'] == file_name]
    search_row = search_row.fillna(-1)
    return int(search_row.iloc[0][key])

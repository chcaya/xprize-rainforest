import pandas as pd
from glob import glob
import numpy as np

def clean_and_split(series):
    # Remove the square brackets and single quotes
    cleaned_series = series.str.replace("[", "").str.replace("]", "").str.replace("'", "")
    split_series = cleaned_series.str.split()
    split_list = [entry.split('_') for sublist in split_series for entry in sublist]
    return split_list


def add_names(df, mapping_df, pred_type):
        df = df.merge(mapping_df[['familyKey', 'family']], left_on=f'familyKey_{pred_type}_pred',
                      right_on='familyKey', how='left')
        df = df.rename(columns={'family': f'family_{pred_type}_pred'})
        df = df.drop(columns='familyKey')

        df = df.merge(mapping_df[['genusKey', 'genus']], left_on=f'genusKey_{pred_type}_pred', right_on='genusKey',
                      how='left')
        df = df.rename(columns={'genus': f'genus_{pred_type}_pred'})
        df = df.drop(columns='genusKey')

        return df


if __name__ == "__main__":
    result_file_patterns = 'set_*top3.csv'
    result_files = glob(result_file_patterns)
    taxon = pd.read_csv('/Users/daoud/PycharmAssets/xprize/photos_exif_taxo.csv')

    file_df_n = []
    for file in result_files:
        df = pd.read_csv(file)


        predictions = df['predictions']
        confidence_scores = df['confidence']


        # Initialize lists to store the results
        family_keys = []
        genus_keys = []

        # Process each row from list of prediction to sep family, genus key for each prediction
        for row in predictions:
            split_list = clean_and_split(pd.Series([row]))
            # Separate family keys and genus keys
            family_keys.append([item[0] for item in split_list])
            genus_keys.append([item[1] for item in split_list])

        # Create a DataFrame with the results
        result_df = pd.DataFrame({
            'familyKey_1st_pred': [fk[0] if len(fk) > 0 else None for fk in family_keys],
            'familyKey_2nd_pred': [fk[1] if len(fk) > 1 else None for fk in family_keys],
            'familyKey_3rd_pred': [fk[2] if len(fk) > 2 else None for fk in family_keys],
            'genusKey_1st_pred': [gk[0] if len(gk) > 0 else None for gk in genus_keys],
            'genusKey_2nd_pred': [gk[1] if len(gk) > 1 else None for gk in genus_keys],
            'genusKey_3rd_pred': [gk[2] if len(gk) > 2 else None for gk in genus_keys],
        })



        pred_1_scores, pred_2_scores, pred_3_scores = [], [], []
        for row in confidence_scores:
            split_conf_scores = clean_and_split(pd.Series(row))
            pred_1_scores.append(split_conf_scores[0][0].replace("[", "").replace("]", "").replace("'", ""))
            pred_2_scores.append(split_conf_scores[1][0].replace("[", "").replace("]", "").replace("'", ""))
            pred_3_scores.append(split_conf_scores[2][0].replace("[", "").replace("]", "").replace("'", ""))

        confidence_scores_df = pd.DataFrame({
            'conf_1st_pred': pred_1_scores,
            'conf_2nd_pred': pred_2_scores,
            'cond_3rd_pred': pred_3_scores
        })


        result_clean = pd.concat([df, result_df, confidence_scores_df], axis=1)


        result_clean['familyKey_1st_pred'] = result_clean['familyKey_1st_pred'].astype(float)
        result_clean['familyKey_2nd_pred'] = result_clean['familyKey_2nd_pred'].astype(float)
        result_clean['familyKey_3rd_pred'] = result_clean['familyKey_3rd_pred'].astype(float)

        result_clean['genusKey_1st_pred'] = result_clean['genusKey_1st_pred'].astype(float)
        result_clean['genusKey_2nd_pred'] = result_clean['genusKey_2nd_pred'].astype(float)
        result_clean['genusKey_3rd_pred'] = result_clean['genusKey_3rd_pred'].astype(float)

        # Add family and genus names for each prediction
        result_clean = add_names(result_clean, taxon, '1st').drop_duplicates(subset='image')
        result_clean = add_names(result_clean, taxon, '2nd').drop_duplicates(subset='image')
        result_clean = add_names(result_clean, taxon, '3rd').drop_duplicates(subset='image')

        file_df_n.append(result_clean)
        del result_clean

    all_results = pd.concat(file_df_n)
    all_results.to_csv('all_top3.csv')
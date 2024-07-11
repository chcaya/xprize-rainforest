import pandas as pd

def extract_name(df, column):
    df['name'] = df[column].str.split('zoom').str[0]
    return df

if __name__ == '__main__':
    results = pd.read_csv('all_top3.csv')
    human = pd.read_csv('/Users/daoud/PycharmAssets/xprize/20240709_xprize100ha_zoom_m3e_human_identifications - 20240709xprizefinals-zoomedpict.csv')
    human = extract_name(human, 'fileName')
    results = extract_name(results, column='image')
    human = human.add_suffix('_human')
    human = human.rename(columns={'name_human': 'name'})

    merged = pd.merge(results, human, left_on='name', right_on='name')



    def check_family_top_3(row):
        if pd.isnull(row['family_human']):
            return None
        else:
            return row['family_human'] in [row['family_1st_pred'], row['family_2nd_pred'], row['family_3rd_pred']]


    def check_family_top_1(row):
        if pd.isnull(row['family_human']):
            return None
        else:
            return row['family_human'] in [row['family_1st_pred']]


    merged['is_family_in_top1'] = merged.apply(check_family_top_1, axis=1)
    merged['is_family_in_top3'] = merged.apply(check_family_top_3, axis=1)

    merged = merged.drop(['Unnamed: 0','Unnamed: 0.1', 'source_human', 'predictions', 'confidence', 'image', 'directory_human', 'missions_human', 'fileName_human'], axis=1)
    merged['conf_1st_pred'] = merged['conf_1st_pred'].str.replace(",", "")
    merged['conf_2nd_pred'] = merged['conf_2nd_pred'].str.replace(",", "")
    columns = ['name'] + [col for col in merged.columns if col != 'name']
    merged = merged[columns]

    print(merged['is_family_in_top1'].sum() / len(merged.dropna(subset='family_human')))
    print(merged['is_family_in_top3'].sum() / len(merged.dropna(subset='family_human')))

    merged.to_csv('bioclip_vs_human_top3.csv')

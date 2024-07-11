from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    brazil_gbif_species_csv = 'brazil_trees_amazonia_sel_occs.csv'
    brazil_photos_taxonomy_csv = 'photos_exif_taxo.csv'

    brazil_gbif_species_csv_path = dir_path / brazil_gbif_species_csv
    brazil_photos_taxonomy_csv_path = dir_path / brazil_photos_taxonomy_csv

    # brazil_gbif_species = pd.read_csv(brazil_gbif_species_csv_path)
    brazil_photos_taxonomy = pd.read_csv(brazil_photos_taxonomy_csv_path)

    # Group by 'familykey' and aggregate unique counts of 'genuskey' and 'specieskey'
    result = brazil_photos_taxonomy.groupby('familyKey').agg(
        distinct_species_count=pd.NamedAgg(column='speciesKey', aggfunc='nunique'),
        distinct_genus_count=pd.NamedAgg(column='genusKey', aggfunc='nunique')
    )

    # Sort the dataframe by 'distinct_species_count' in descending order
    result_sorted = result.sort_values(by='distinct_species_count', ascending=False)
    print (result_sorted)

    # image_output_dir = dir_path / 'images/'


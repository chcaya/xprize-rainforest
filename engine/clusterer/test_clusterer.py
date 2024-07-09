import ast
import os.path

import numpy as np
import pandas as pd
import geopandas as gpd

from engine.clusterer.clusterer import Clusterer

if __name__ == '__main__':
    # embeddings_df = pd.read_pickle('C:/Users/Hugo/PycharmProjects/xprize-rainforest/engine/embedder/dinov2/embeddings_df_True_None_1720055270.0580597.pkl')
    # embeddings_df = pd.read_pickle('/home/hugo/Documents/xprize/practice/embeddings_df_dinov2_pratique100ha_True_None_1720355386.9873917.pkl')
    # embeddings_df = pd.read_csv('C:/Users/Hugo/PycharmProjects/xprize-rainforest/engine/embedder/bioclip/embeddings/bioclip_embeddings_with_filepaths.csv')
    # prefix = 'C:/Users/Hugo/Documents/XPrize/infer/20240521_zf2100ha_highres_m3m_rgb_2/classifier_tilerizer_output/20240521_zf2100ha_highres_m3m_rgb/tiles/'
    # embeddings_df.rename(columns={'labels': 'tiles_paths'}, inplace=True)
    # embeddings_df['tiles_paths'] = embeddings_df['tiles_paths'].apply(lambda x: prefix + x)

    embeddings_gdf = gpd.read_file('/home/hugo/Documents/xprize/infer/20240130_zf2transectew_m3m_rgb_TEST_2_EMBEDDERS/20240130_zf2transectew_m3m_rgb_final.gpkg')
    embeddings_df = embeddings_gdf
    # embeddings_df['embeddings'] = embeddings_df['embeddings'].apply(lambda x: ast.literal_eval(x))
    embeddings_df['embeddings_contrastive'] = embeddings_df['embeddings_contrastive'].apply(lambda x: ast.literal_eval(x))
    embeddings_df['embeddings_dinov2'] = embeddings_df['embeddings_dinov2'].apply(lambda x: ast.literal_eval(x))

    output_root_dir = '/home/hugo/Documents/xprize/cluster/cluster_PRACTICE_resnetQPE'

    n_components1 = 2
    n_components2 = None
    reduce_algo_name1 = 'umap'
    reduce_algo_name2 = None #'umap'
    visualize_algo_name = 'umap'
    use_reduced1_for_visualization = True

    clusterer = Clusterer(
        embeddings_df=embeddings_df,
        embeddings_column_name='embeddings',
        scale_embeddings=True,
        metric='euclidean',
        reduce_algo_name1=reduce_algo_name1,
        reduce_algo_name2=reduce_algo_name2,
        visualize_algo_name=visualize_algo_name,
        use_reduced1_for_visualization=use_reduced1_for_visualization,
        tsne_perplexity=30,
        umap_n_neighbors=5,
        umap_min_dist=0.001,
        min_cluster_size=3,
        n_components1=n_components1,
        n_components2=n_components2,
        output_root_dir=output_root_dir,
        n_cpus=20
    )

    clusterer.search_best_params(
        cluster_algo='hdbscan',
        dbscan_eps_list=[0.00001, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.6, 0.8, 1,
                        1.2, 1.4, 1.8, 2.2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30],
        dbscan_min_samples_list=[2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 18, 20, 25, 30, 40],
        hdbscan_min_samples_list=[2, 3, 4, 5, 6, 8, 12,
                                  # 13, 16, 17, 18, 20, 25, 30, 40
                                  ],
        hdbscan_cluster_selection_epsilon_list=[
                                                # 0.001,
            # 0.005, 0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08,
            #                                     0.01,
                                                # 0.02, 0.05,
                                                # 0.06,
                                                # 0.08,
                                                0.1,
                                                0.12, 0.14, 0.16, 0.18,
                                                0.2, 0.3, 0.4, 0.6, 0.8,
                                                # 1, 1.2, 1.4, 1.8, 2.2, 2.5, 3, 3.5, 4, 5, 6, 7, 8,
                                                # 10, 12, 14, 16, 18, 20, 25, 30
                                                ],
        hdbscan_cluster_selection_method_list=['eom', 'leaf'],
    )

    cluster_algo = 'hdbscan'
    hdbscan_min_samples = 5
    hdbscan_cluster_selection_epsilon = 0.12

    clusters = clusterer.get_clusters(
        cluster_algo=cluster_algo,
        dbscan_min_samples=None,
        dbscan_eps=None,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        hdbscan_cluster_selection_method='leaf'
    )

    merged_gdf = embeddings_gdf
    merged_gdf['cluster_labels'] = clusters
    merged_gdf['embeddings'] = merged_gdf['embeddings'].apply(lambda x: str(x))
    merged_gdf.to_file(f"./20240521_practice100ha_highres_m3m_rgb_final_clustered_resnetQPE_clusters_{reduce_algo_name1}_{n_components1}_{reduce_algo_name2}_{n_components2}_{cluster_algo}_{hdbscan_min_samples}_{str(hdbscan_cluster_selection_epsilon).replace('.', 'p')}.gpkg", driver='GPKG')





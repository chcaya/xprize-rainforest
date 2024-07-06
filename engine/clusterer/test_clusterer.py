import ast
import os.path

import pandas as pd
import geopandas as gpd

from engine.clusterer.clusterer import Clusterer

if __name__ == '__main__':
    # embeddings_df = pd.read_pickle('C:/Users/Hugo/PycharmProjects/xprize-rainforest/engine/embedder/dinov2/embeddings_df_True_None_1720055270.0580597.pkl')
    embeddings_df = pd.read_pickle('C:/Users/Hugo/PycharmProjects/xprize-rainforest/engine/embedder/dinov2/embeddings_df_True_None_1720055270.0580597.pkl')
    # embeddings_df = pd.read_csv('C:/Users/Hugo/PycharmProjects/xprize-rainforest/engine/embedder/bioclip/embeddings/bioclip_embeddings_with_filepaths.csv')
    # prefix = 'C:/Users/Hugo/Documents/XPrize/infer/20240521_zf2100ha_highres_m3m_rgb_2/classifier_tilerizer_output/20240521_zf2100ha_highres_m3m_rgb/tiles/'
    # embeddings_df.rename(columns={'labels': 'tiles_paths'}, inplace=True)
    # embeddings_df['tiles_paths'] = embeddings_df['tiles_paths'].apply(lambda x: prefix + x)
    # embeddings_df['embeddings'] = embeddings_df['embeddings'].apply(lambda x: ast.literal_eval(x))

    output_root_dir = 'C:/Users/Hugo/Documents/XPrize/cluster_2reduce'

    n_components1 = 50
    n_components2 = 3
    reduce_algo_name1 = 'pca'
    reduce_algo_name2 = 'umap'
    visualize_algo_name = 'umap'
    use_reduced1_for_visualization = True

    clusterer = Clusterer(
        embeddings_df=embeddings_df,
        embeddings_column_name='embeddings',
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

    # clusterer.search_best_params(
    #     cluster_algo='hdbscan',
    #     dbscan_eps_list=[0.00001, 0.00002, 0.00004, 0.00006, 0.00008, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.6, 0.8, 1,
    #                     1.2, 1.4, 1.8, 2.2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 25, 30],
    #     dbscan_min_samples_list=[2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 17, 18, 20, 25, 30, 40],
    #     hdbscan_min_samples_list=[2, 3, 4, 5, 6, 8, 12,
    #                               # 13, 16, 17, 18, 20, 25, 30, 40
    #                               ],
    #     hdbscan_cluster_selection_epsilon_list=[
    #                                             0.001,
    #         # 0.005, 0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08,
    #                                             # 0.02, 0.05, 0.06, 0.08,
    #                                             0.01,
    #                                             0.1,
    #                                             0.12, 0.14, 0.16, 0.18,
    #                                             0.2, 0.3, 0.4, 0.6, 0.8,
    #                                             1, 1.2, 1.4, 1.8, 2.2, 2.5, 3, 3.5, 4, 5, 6, 7, 8,
    #                                             10, 12, 14, 16, 18, 20, 25, 30
    #                                             ],
    #     hdbscan_cluster_selection_method_list=['eom', 'leaf'],
    # )

    cluster_algo = 'hdbscan'
    hdbscan_min_samples = 5
    hdbscan_cluster_selection_epsilon = 0.1

    clusters = clusterer.get_clusters(
        cluster_algo=cluster_algo,
        dbscan_min_samples=None,
        dbscan_eps=None,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        hdbscan_cluster_selection_method='leaf'
    )

    # contrastive_gdf = gpd.read_file('C:/Users/Hugo/Documents/XPrize/infer/20240521_zf2100ha_highres_m3m_rgb_clahe_final_gr0p03_infer.gpkg')
    contrastive_gdf = gpd.read_file('C:/Users/Hugo/Documents/XPrize/infer/20240521_zf2100ha_highres_m3m_rgb_2/classifier_output/20240521_zf2100ha_highres_m3m_rgb_gr0p03_inferembedderclassifier.gpkg')
    embeddings_df['cluster_labels'] = clusters

    embeddings_df['tiles_paths'] = embeddings_df['tiles_paths'].apply(lambda x: str(x).replace('\\', '/'))
    contrastive_gdf['tile_path'] = contrastive_gdf['tile_path'].apply(lambda x: str(x).replace('\\', '/'))

    print(list(embeddings_df['tiles_paths'])[0])
    print(list(contrastive_gdf['tile_path'])[0])

    embeddings_df['embeddings'] = embeddings_df['embeddings'].apply(lambda x: str(x))

    embeddings_df.rename(columns={'cluster_labels': 'cluster_labels_dinov2',
                                  'embeddings': 'embeddings_dinov2'}, inplace=True)
    print('n samples without assigned cluster:',
          len(embeddings_df.loc[embeddings_df['cluster_labels_dinov2'] == -1]))
    merged_gdf = contrastive_gdf.merge(
        embeddings_df[['cluster_labels_dinov2', 'embeddings_dinov2', 'tiles_paths']],
        left_on='tile_path', right_on='tiles_paths', how='inner')

    merged_gdf.to_file(f"./20240521_zf2100ha_highres_m3m_rgb_final_clustered_dinov2_clusters_scaled_{reduce_algo_name1}_{n_components1}_{reduce_algo_name2}_{n_components2}_{cluster_algo}_{hdbscan_min_samples}_{str(hdbscan_cluster_selection_epsilon).replace('.', 'p')}.gpkg", driver='GPKG')





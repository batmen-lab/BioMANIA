from scanpy.datasets import pbmc3k_processed
result_1 = pbmc3k_processed()
from scanpy.pp import calculate_qc_metrics
result_1.var['mito'] = result_1.var_names.str.startswith('MT-')
result_2 = calculate_qc_metrics(result_1, qc_vars=['mito'], inplace=True)
from scanpy.pp import normalize_total
result_3 = normalize_total(result_1, target_sum=1e6, max_fraction=0.05, inplace=True, copy=False)
from scanpy.tl import leiden
result_4 = leiden(result_1, key_added='louvain', copy=False)
from scanpy.pl import draw_graph
result_5 = draw_graph(result_1, color='louvain', show=True)
from scanpy.pp import filter_cells
result_3 = filter_cells(result_1, min_genes=3, inplace=True, copy=False)
result_4 = normalize_total(result_1, target_sum=1, inplace=True, copy=False)
from scanpy.pp import neighbors
result_5 = neighbors(result_1, copy=False)
from scanpy.pl import umap
result_6 = umap(result_1, color='louvain', legend_fontsize='medium', show=True)
result_2 = leiden(result_1, copy=False)
result_3 = leiden(result_1, copy=False)
result_4 = leiden(result_1, copy=False)
from scanpy.pl import scatter
scatter(result_1, x='X_umap', y='X_tsne', color='percent_mito', basis='umap', show=True)
from scanpy.pl import heatmap
import scanpy as sc

heatmap_result = sc.pl.heatmap(result_1, var_names=result_1.var_names[:10], groupby='louvain', log=True, standard_scale='var', show=True)
result_2 = filter_cells(result_1, min_genes=3, inplace=True, copy=False)
result_3 = normalize_total(result_1, target_sum=1e6, copy=False, inplace=True)
from scanpy.tl import pca
result_4 = pca(result_1, n_comps=50, copy=False)
result_5 = umap(result_1, color='louvain', layer='X_new', show=True)
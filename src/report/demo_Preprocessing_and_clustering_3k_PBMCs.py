from scanpy import read_10x_mtx
result_1 = read_10x_mtx('./tmp')
from scanpy.pl import highest_expr_genes
highest_expr_genes(result_1)
from scanpy.pp import calculate_qc_metrics
calculate_qc_metrics(result_1, inplace=True)
from scanpy.pl import violin
violin(result_1, ['n_genes_by_counts', 'total_counts'])
from scanpy.pp import normalize_total
normalize_total(result_1, inplace=True)
from scanpy.pp import log1p
log1p(result_1)
from scanpy.pp import highly_variable_genes
highly_variable_genes(result_1, inplace=True)
from scanpy.pl import highly_variable_genes
highly_variable_genes(result_1)
from scanpy.pp import regress_out
regress_out(result_1, ['total_counts'])
from scanpy.pp import scale
scale(result_1)
from scanpy.tl import pca
result_2 = pca(result_1)
from scanpy.pl import pca
pca(result_1)
from scanpy.pl import pca_variance_ratio
pca_variance_ratio(result_1)
from scanpy.pp import neighbors
neighbors(result_1)
from scanpy.tl import paga
paga(result_1)
from scanpy.pl import paga
paga(result_1)
from scanpy.tl import umap
umap(result_1)
from scanpy.pl import umap
umap(result_1)
from scanpy.tl import leiden
leiden(result_1)
from scanpy.tl import rank_genes_groups
rank_genes_groups(result_1, 'leiden')
from scanpy.pl import rank_genes_groups
rank_genes_groups(result_1)
from scanpy.pl import stacked_violin
stacked_violin(result_1, ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14','LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1', 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP'], 'leiden')
from scanpy.pl import dotplot
dotplot(result_1, ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14','LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1', 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP'], 'leiden')
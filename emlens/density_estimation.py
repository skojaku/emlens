import numpy as np
from scipy import sparse
from sklearn.neighbors import KDTree, KernelDensity
from joblib import Parallel, delayed

# KDE estimation
def estimate_density(emb, train_emb, kernel="epanechnikov", n_jobs=30):
    dh = np.power(
        train_emb.shape[0], -1.0 / (train_emb.shape[1] + 4)
    )  # Silverman's rule-of-thumb rule
    kde = KernelDensity(kernel=kernel, bandwidth=dh).fit(train_emb)

    def est(kde, emb, ids):
        p = kde.score_samples(emb[ids, :])
        return p, ids

    chunks = np.array_split(np.arange(emb.shape[0]), n_jobs)
    results = Parallel(n_jobs=n_jobs)(delayed(est)(kde, emb, ids) for ids in chunks)

    ids = np.concatenate([res[1] for res in results])
    paper_density = np.concatenate([res[0] for res in results])
    order = np.argsort(ids)
    ids = ids[order]
    paper_density = paper_density[order]
    return paper_density

from itertools import count, repeat
import numpy as np

import Levenshtein

from Orange.data import Table, Domain
from Orange.distance import Distance, DistanceModel
from Orange.misc import DistMatrix


class LevenshteinDistance(Distance):
    supports_normalization = True

    def __init__(self, callback=None, *, normalize=True, delimiter=None):
        super().__init__()
        self.normalize = normalize
        self.delimiter = delimiter
        self.callback = callback

    def fit(self, _):
        return LevenshteinModel(
            self.callback,
            delimiter=self.delimiter,
            normalize=self.normalize)


class LevenshteinModel(DistanceModel):
    def __init__(self, callback=None, *, delimiter=None, normalize=True):
        super().__init__(1)
        self.normalize = normalize
        self.delimiter = delimiter
        self.callback = callback

    def _prepare_data(self, e1, e2=None):
        idcs = np.array(
            [i for i, var in enumerate(e1.domain.metas) if var.is_string])
        if len(idcs) == 0:
            raise ValueError("No string metas")
        if len(idcs) == len(e1.domain.metas):
            idcs = None

        e1metas = e1.metas.astype(str) \
            if idcs is None else e1.metas[:, idcs].astype(str)
        if e2 is None:
            return e1metas, e1metas

        if e2.domain.metas != e1.domain.metas:
            if not ({var.name for var in e1.domain.metas if var.is_string}
                    & {var.name for var in e2.domain.metas if var.is_string}):
                raise ValueError("No common string metas")
            # Skip ordinary attributes: we don't need them
            e2 = e2.transform(Domain([], None, e1.domain.metas))

        e2metas = e2.metas.astype(str) \
            if idcs is None else e2.metas[:, idcs].astype(str)
        return e1metas, e2metas


    def __call__(self, e1, e2=None):
        if not (
            isinstance(e1, Table) and (e2 is None or isinstance(e2, Table))):
            raise ValueError(
                "Levenshtein distance can only be computed for Table objects")

        e1metas, e2metas = self._prepare_data(e1, e2)
        dist = self.compute_distances(e1metas, e2metas)
        return DistMatrix(dist, e1, e2 or e1)  # but not transformed e2!

    def compute_distances(self, x1, x2):
        func = (lambda w1, w2: 1 - Levenshtein.ratio(w1, w2)) if self.normalize \
            else Levenshtein.distance
        same = x2 is x1
        n2 = len(x2)
        dist = np.full((len(x1), len(x2)), 0, dtype=float)
        distw = np.empty((len(x1), len(x2)), dtype=float)
        nwords = x1.shape[1]
        tril = np.tril_indices(len(x1), -1) if same else (None, None)
        for col in range(nwords):
            if self.callback:
                self.callback(100 * col / nwords)
            distw[:] = 0
            words1 = np.char.lower(x1[:, col]).astype(str)
            words2 = words1 if same else np.char.lower(x2[:, col]).astype(str)
            if self.delimiter:
                words1 = \
                    [[w.strip() for w in words]
                     for words in np.char.split(words1, self.delimiter)]
                words2 = words1 if same else \
                    [[w.strip() for w in words]
                     for words in np.char.split(words2, self.delimiter)]
                for i1, rge, word1 in zip(count(),
                                          count() if same else repeat(n2),
                                          words1):
                    distw[i1, :rge] = [
                        min((func(w1, w2)
                             for w1 in word1 for w2 in word2 if w1 and w2),
                            default=np.nan)
                        for word2 in words2[:rge]]
            else:
                words1 = np.char.strip(words1)
                words2 = words1 if same else np.char.strip(words2)
                for i1, rge, word1 in zip(count(),
                                          count() if same else repeat(n2),
                                          words1):
                    if word1:
                        distw[i1, :rge] = [
                            func(word1, word2) if word2 else np.nan
                            for word2 in words2[:rge]]
                    else:
                        distw[i1, :rge] = np.nan
            if same:
                distw += distw.T
            nans = np.isnan(distw)
            num_nans = np.sum(nans)
            if num_nans == distw.size - (len(distw) if same else 0):
                continue
            elif num_nans > 0:
                if same:
                    distw[nans] = np.nanmean(distw[*tril])
                else:
                    distw[nans] = np.nanmean(distw)
            dist += distw
        return dist

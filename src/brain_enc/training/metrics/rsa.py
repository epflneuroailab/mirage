from typing import Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

import torch

class RepresentationalSimilarityAnalysis:
    """
    Representational Similarity Analysis (RSA).

    Given two representation matrices X and Y with the same number of conditions
    (rows), RSA:

    1. Computes a Representational Dissimilarity Matrix (RDM) for each:
       RDM_X[i, j] = dissimilarity(x_i, x_j)
       RDM_Y[i, j] = dissimilarity(y_i, y_j)

    2. Flattens the upper triangles of both RDMs and computes a correlation
       between them (Pearson or Spearman).
    """

    def __init__(
        self,
        dissimilarity: Literal["correlation", "euclidean", "cosine"] = "correlation",
        similarity_metric: Literal["pearson", "spearman"] = "spearman",
    ):
        self.dissimilarity = dissimilarity
        self.similarity_metric = similarity_metric

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute RSA similarity between X and Y.

        Parameters
        ----------
        X, Y : np.ndarray
            Arrays of shape (n_conditions, ...) (will be flattened along feature dims).

        Returns
        -------
        rsa_similarity : float
            Correlation between the vectorized upper triangles of the two RDMs.
        """
        return self.forward(X, Y)

    def forward(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Number of conditions must match: {X.shape[0]} vs {Y.shape[0]}"
            )

        # Flatten features
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)

        # Compute RDMs
        rdm_X = self.compute_rdm(X)
        rdm_Y = self.compute_rdm(Y)

        # Compare RDMs
        return self.compare_rdms(rdm_X, rdm_Y)

    def compute_rdm(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Representational Dissimilarity Matrix (RDM)
        for a given representation matrix X.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_conditions, n_features).

        Returns
        -------
        rdm : np.ndarray
            Array of shape (n_conditions, n_conditions) with pairwise dissimilarities.
        """
        if self.dissimilarity not in {"correlation", "euclidean", "cosine"}:
            raise ValueError(f"Unsupported dissimilarity: {self.dissimilarity}")

        # pdist returns a condensed distance vector; squareform makes it a full matrix
        distances = pdist(X, metric=self.dissimilarity)
        rdm = squareform(distances)
        return rdm

    def compare_rdms(self, rdm1: np.ndarray, rdm2: np.ndarray) -> float:
        """
        Compare two RDMs by correlating their upper triangles.
        """
        rdm1 = np.asarray(rdm1)
        rdm2 = np.asarray(rdm2)

        if rdm1.shape != rdm2.shape:
            raise ValueError(
                f"RDM shapes must match: {rdm1.shape} vs {rdm2.shape}"
            )

        if rdm1.shape[0] != rdm1.shape[1]:
            raise ValueError("RDMs must be square matrices.")

        n = rdm1.shape[0]
        i_upper = np.triu_indices(n, k=1)

        v1 = rdm1[i_upper]
        v2 = rdm2[i_upper]

        if self.similarity_metric == "pearson":
            corr, _ = pearsonr(v1, v2)
        elif self.similarity_metric == "spearman":
            corr, _ = spearmanr(v1, v2)
        else:
            raise ValueError(f"Unsupported similarity_metric: {self.similarity_metric}")

        # pearsonr/spearmanr already return float, but cast explicitly
        return float(corr)


class RepresentationalSimilarityAnalysisTorch:
    """
    Representational Similarity Analysis (RSA) in PyTorch with GPU support.

    Given two representation matrices X and Y with the same number of conditions
    (rows), RSA:

    1. Computes a Representational Dissimilarity Matrix (RDM) for each:
       RDM_X[i, j] = dissimilarity(x_i, x_j)
       RDM_Y[i, j] = dissimilarity(y_i, y_j)

    2. Flattens the upper triangles of both RDMs and computes a correlation
       (Pearson or Spearman) between them.
    """

    def __init__(
        self,
        dissimilarity: Literal["correlation", "euclidean", "cosine"] = "correlation",
        similarity_metric: Literal["pearson", "spearman"] = "spearman",
        eps: float = 1e-8,
    ):
        self.dissimilarity = dissimilarity
        self.similarity_metric = similarity_metric
        self.eps = eps

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute RSA similarity between X and Y.

        Parameters
        ----------
        X, Y : torch.Tensor
            Tensors of shape (n_conditions, ...) (flattened along feature dims).
            They can be on CPU or GPU; computation happens on their device.

        Returns
        -------
        rsa_similarity : torch.Tensor (scalar)
            Correlation between the vectorized upper triangles of the two RDMs.
        """
        return self.forward(X, Y)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Number of conditions must match: {X.shape[0]} vs {Y.shape[0]}"
            )

        # Flatten feature dimensions
        X = X.view(X.shape[0], -1)
        Y = Y.view(Y.shape[0], -1)

        # Compute RDMs
        rdm_X = self.compute_rdm(X)
        rdm_Y = self.compute_rdm(Y)

        # Compare RDMs
        return self.compare_rdms(rdm_X, rdm_Y)

    def compute_rdm(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the Representational Dissimilarity Matrix (RDM)
        for a given representation matrix X.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape (n_conditions, n_features).

        Returns
        -------
        rdm : torch.Tensor
            Tensor of shape (n_conditions, n_conditions) with pairwise dissimilarities.
        """
        if self.dissimilarity == "euclidean":
            # torch.cdist computes pairwise distances between rows
            # shape: (n, n)
            return torch.cdist(X, X, p=2.0)

        elif self.dissimilarity == "cosine":
            # Cosine distance: 1 - cosine_similarity
            X_norm = X / (X.norm(dim=1, keepdim=True) + self.eps)
            cos_sim = X_norm @ X_norm.t()
            cos_sim = cos_sim.clamp(-1.0, 1.0)
            return 1.0 - cos_sim

        elif self.dissimilarity == "correlation":
            # Correlation distance: 1 - corr(x_i, x_j)
            # 1. Center each row (condition) across features
            X_centered = X - X.mean(dim=1, keepdim=True)
            # 2. Normalize to unit norm => correlation via dot product
            X_norm = X_centered / (X_centered.norm(dim=1, keepdim=True) + self.eps)
            corr = X_norm @ X_norm.t()
            corr = corr.clamp(-1.0, 1.0)
            return 1.0 - corr

        else:
            raise ValueError(f"Unsupported dissimilarity: {self.dissimilarity}")

    def compare_rdms(self, rdm1: torch.Tensor, rdm2: torch.Tensor) -> torch.Tensor:
        """
        Compare two RDMs by correlating their upper triangles.
        Returns a scalar tensor on the same device.
        """
        if rdm1.shape != rdm2.shape:
            raise ValueError(
                f"RDM shapes must match: {rdm1.shape} vs {rdm2.shape}"
            )

        if rdm1.shape[0] != rdm1.shape[1]:
            raise ValueError("RDMs must be square matrices.")

        n = rdm1.shape[0]
        device = rdm1.device

        # Upper triangle indices excluding diagonal
        i_upper = torch.triu_indices(n, n, offset=1, device=device)
        v1 = rdm1[i_upper[0], i_upper[1]]
        v2 = rdm2[i_upper[0], i_upper[1]]

        if self.similarity_metric == "pearson":
            return self._pearson_corr(v1, v2)
        elif self.similarity_metric == "spearman":
            r1 = self._rankdata(v1)
            r2 = self._rankdata(v2)
            return self._pearson_corr(r1, r2)
        else:
            raise ValueError(f"Unsupported similarity_metric: {self.similarity_metric}")

    def _pearson_corr(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Pearson correlation between two 1D tensors.
        Returns a scalar tensor on the same device.
        """
        x = x.view(-1).float()
        y = y.view(-1).float()

        x = x - x.mean()
        y = y - y.mean()

        num = torch.sum(x * y)
        den = x.norm() * y.norm() + self.eps
        return num / den

    # def _rankdata(self, a: torch.Tensor) -> torch.Tensor:
    #     """
    #     Rank data (0-based average ranks for ties) implemented in PyTorch.

    #     This is a minimal equivalent to scipy.stats.rankdata for 1D tensors.
    #     The exact rank offset (0-based vs 1-based) does not affect Spearman,
    #     because Pearson correlation is invariant to affine transforms of ranks.
    #     """
    #     a_flat = a.view(-1)
    #     device = a_flat.device
    #     n = a_flat.shape[0]

    #     if n == 0:
    #         return a_flat

    #     # Sort values
    #     sorted_vals, sorted_idx = torch.sort(a_flat)
    #     ranks = torch.zeros_like(a_flat, dtype=torch.float, device=device)

    #     # Find boundaries where the value changes
    #     # diff[i] = True means sorted_vals[i+1] != sorted_vals[i]
    #     diff = torch.diff(sorted_vals)
    #     change_idx = torch.nonzero(diff != 0, as_tuple=False).flatten() + 1

    #     # Start and end indices for each group of equal values
    #     starts = torch.cat(
    #         [torch.tensor([0], device=device, dtype=torch.long), change_idx]
    #     )
    #     ends = torch.cat(
    #         [change_idx, torch.tensor([n], device=device, dtype=torch.long)]
    #     )

    #     # Assign average rank for each group of ties
    #     # (loop over unique values; typically fine for RDM vectors)
    #     for s, e in zip(starts.tolist(), ends.tolist()):
    #         avg_rank = (s + e - 1) / 2.0
    #         ranks[sorted_idx[s:e]] = avg_rank

    #     return ranks.view_as(a)
    def _rankdata(self, a: torch.Tensor) -> torch.Tensor:
        """
        Vectorized rankdata (0-based average ranks for ties) in PyTorch.

        Equivalent behavior to scipy.stats.rankdata(method="average"), but
        implemented without Python loops so it is much faster on GPU.
        """
        a_flat = a.view(-1)
        device = a_flat.device
        n = a_flat.numel()
        if n == 0:
            return a_flat

        # Sort values
        sorted_vals, sorted_idx = torch.sort(a_flat)  # ascending

        # Find unique values and inverse mapping
        # inv_idx[i] = group id for sorted_vals[i]
        unique_vals, inv_idx = torch.unique(sorted_vals, return_inverse=True)

        # For each group g, we want the average rank of its positions in sorted array.
        # Positions (indices in sorted array) are 0..n-1
        positions = torch.arange(n, device=device, dtype=torch.float32)

        # Sum of positions per group and count per group
        num_groups = unique_vals.shape[0]
        sums = torch.zeros(num_groups, device=device, dtype=torch.float32)
        counts = torch.zeros(num_groups, device=device, dtype=torch.float32)
        sums.scatter_add_(0, inv_idx, positions)
        counts.scatter_add_(0, inv_idx, torch.ones_like(positions))

        # Average rank per group
        avg_ranks_per_group = sums / counts  # [num_groups]

        # Map each element in sorted array to its group's avg rank
        ranks_sorted = avg_ranks_per_group[inv_idx]  # [n]

        # Unsort back to original order
        ranks = torch.empty_like(ranks_sorted)
        ranks[sorted_idx] = ranks_sorted

        return ranks.view_as(a)
    

if __name__ == "__main__":
    # Simple test
    rsa_cpu = RepresentationalSimilarityAnalysis(
        dissimilarity="correlation", similarity_metric="spearman"
    )
    
    
    rng = np.random.default_rng(42)
    X = rng.random((37, 71))
    Y = rng.random((37, 43))

    sim = rsa_cpu(X, Y)
    print(f"RSA similarity                : {sim}")
    
    rsa_gpu = RepresentationalSimilarityAnalysisTorch(
        dissimilarity="correlation", similarity_metric="spearman"
    )
    X_torch = torch.tensor(X).cuda()
    Y_torch = torch.tensor(Y).cuda()
    
    sim_torch = rsa_gpu(X_torch, Y_torch)
    print(f"RSA similarity (GPU)          : {sim_torch.item()}")
    
    diff = abs(sim - sim_torch.item())
    print(f"Difference between CPU and GPU: {diff}")
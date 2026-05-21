from typing import Optional, Literal


import numpy as np
from scipy.spatial.distance import pdist, cdist

import torch
import torch.nn as nn


class CenteredKernelAlignment:
    """
    Centered Kernel Alignment (CKA).

    Supports:
      - 'linear': 
          - unbiased=False → fast, standard (biased) linear CKA
          - unbiased=True  → unbiased HSIC-based linear CKA (Song et al.)
      - 'rbf': kernel CKA using biased HSIC (unbiased currently not implemented)

    Parameters
    ----------
    kernel_type : {'linear', 'rbf'}
        Kernel type.
    sigma : float or None
        RBF kernel width. If None for 'rbf', uses median pairwise distance.
    unbiased : bool
        If True and kernel_type == 'linear', use unbiased HSIC estimator.
        For non-linear kernels, unbiased=True raises a ValueError.
    eps : float
        Small constant for numerical stability.
    dtype: data type for computations (default: np.float64).
    """

    def __init__(
        self,
        kernel_type: Literal["linear", "rbf"] = "linear",
        sigma: Optional[float] = None,
        unbiased: bool = False,
        eps: float = 1e-8,
        dtype: np.dtype = np.float64,
    ):
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.unbiased = unbiased
        self.eps = eps
        self.dtype = dtype

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.forward(X, Y)

    def forward(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = np.asarray(X).astype(self.dtype)
        Y = np.asarray(Y).astype(self.dtype)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Batch sizes must match along axis 0: {X.shape[0]} vs {Y.shape[0]}"
            )

        # Flatten features: (n_samples, n_features)
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)

        if self.kernel_type == "linear":
            if self.unbiased:
                return self._unbiased_linear_cka(X, Y)
            else:
                # Standard (biased) fast linear CKA:
                # center each feature, then use Frobenius formula
                Xc = X - X.mean(axis=0, keepdims=True)
                Yc = Y - Y.mean(axis=0, keepdims=True)
                return self._fast_linear_cka(Xc, Yc)

        elif self.kernel_type == "rbf":
            if self.unbiased:
                raise ValueError(
                    "Unbiased CKA is only implemented for the linear kernel."
                )
            return self._kernel_cka(X, Y)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    # ---------- Biased linear CKA (fast form) ----------

    def _fast_linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Fast linear CKA using Frobenius norms:

            CKA(X, Y) = ||Yᵀ X||_F² / (||Xᵀ X||_F · ||Yᵀ Y||_F)

        Assumes X and Y are already centered across samples.
        """
        num = self._matrix_inner_product(X, Y) ** 2
        denom = self._matrix_inner_product(X, X) * self._matrix_inner_product(Y, Y)

        if denom <= self.eps:
            return 0.0
        return float(num / denom)

    @staticmethod
    def _matrix_inner_product(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Frobenius norm of Yᵀ X:
            <X, Y> = ||Yᵀ X||_F
        """
        M = Y.T @ X
        return float(np.linalg.norm(M, ord="fro"))

    # ---------- Unbiased linear CKA via unbiased HSIC ----------

    def _unbiased_linear_hsic(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Unbiased HSIC estimator (Song et al.) for linear kernel.

        X: [n, d_x]
        Y: [n, d_y]
        Returns: scalar HSIC estimate.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}")
        n = X.shape[0]
        if n < 4:
            raise ValueError(f"Unbiased HSIC requires at least 4 samples, got n={n}")

        # Linear kernel Gram matrices
        K = X @ X.T   # [n, n]
        L = Y @ Y.T   # [n, n]

        # Zero out diagonals
        K = K.copy()
        L = L.copy()
        np.fill_diagonal(K, 0.0)
        np.fill_diagonal(L, 0.0)

        ones = np.ones((n, 1), dtype=K.dtype)

        # term1 = tr(K L) = sum_ij K_ij L_ij
        term1 = float((K * L).sum())

        # term2 = (1^T K 1)(1^T L 1)
        K1 = K @ ones   # [n, 1]
        L1 = L @ ones   # [n, 1]
        term2 = float(K1.sum() * L1.sum())

        # term3 = 1^T K L 1 = sum_i K1_i L1_i
        term3 = float((K1 * L1).sum())

        coef = 1.0 / (n * (n - 3.0))
        hsic = coef * (
            term1
            + term2 / ((n - 1.0) * (n - 2.0))
            - 2.0 * term3 / (n - 2.0)
        )
        return hsic

    def _unbiased_linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Unbiased linear CKA:

            CKA_unb(X, Y) =
                HSIC_unb(X, Y) / sqrt(HSIC_unb(X, X) HSIC_unb(Y, Y))
        """
        hsic_xy = self._unbiased_linear_hsic(X, Y)
        hsic_xx = self._unbiased_linear_hsic(X, X)
        hsic_yy = self._unbiased_linear_hsic(Y, Y)

        prod = hsic_xx * hsic_yy
        if prod <= 0:
            return 0.0
        denom = np.sqrt(prod) + self.eps
        return float(hsic_xy / denom)

    # ---------- Kernel CKA (RBF, biased HSIC) ----------

    def _get_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute kernel matrix K(X, Y). Currently supports only RBF.
        """
        if Y is None:
            Y = X

        if self.kernel_type == "rbf":
            # If sigma not set, use median pairwise distance heuristic on X
            if self.sigma is None:
                n = X.shape[0]
                if n > 1:
                    dists = pdist(X, metric="euclidean")
                    if dists.size > 0:
                        self.sigma = float(np.median(dists))
                    else:
                        self.sigma = 1.0
                else:
                    self.sigma = 1.0

            sq_dists = cdist(X, Y, metric="sqeuclidean")
            K = np.exp(-sq_dists / (2.0 * (self.sigma ** 2)))
            return K

        raise ValueError(f"Unsupported kernel type for _get_kernel: {self.kernel_type}")

    @staticmethod
    def _center_gram_matrix(K: np.ndarray) -> np.ndarray:
        """
        Center a Gram matrix K using:
            K_c = H K H,  with H = I - 1/n · 11ᵀ
        """
        n = K.shape[0]
        H = np.eye(n, dtype=K.dtype) - np.ones((n, n), dtype=K.dtype) / n
        return H @ K @ H

    def _hsic_xy_biased(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Biased HSIC(X, Y) = sum(K_c ⊙ L_c).
        """
        K = self._get_kernel(X)
        L = self._get_kernel(Y)
        Kc = self._center_gram_matrix(K)
        Lc = self._center_gram_matrix(L)
        return float(np.sum(Kc * Lc))

    def _hsic_self_biased(self, X: np.ndarray) -> float:
        """
        Biased HSIC(X, X).
        """
        K = self._get_kernel(X)
        Kc = self._center_gram_matrix(K)
        return float(np.sum(Kc * Kc))

    def _kernel_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Kernel CKA using biased HSIC:

            CKA(X, Y) = HSIC_biased(X, Y) /
                        sqrt(HSIC_biased(X, X) HSIC_biased(Y, Y))
        """
        hsic_xy = self._hsic_xy_biased(X, Y)
        hsic_xx = self._hsic_self_biased(X)
        hsic_yy = self._hsic_self_biased(Y)

        prod = hsic_xx * hsic_yy
        if prod <= 0:
            return 0.0
        denom = np.sqrt(prod) + self.eps
        return float(hsic_xy / denom)
    
    


class CenteredKernelAlignmentTorch(nn.Module):
    """
    Centered Kernel Alignment (CKA) in PyTorch (GPU-friendly).

    Supports:
      - 'linear':
          - unbiased=False → fast, standard (biased) linear CKA
          - unbiased=True  → unbiased HSIC-based linear CKA
      - 'rbf':
          kernel CKA using biased HSIC (unbiased not implemented)

    X, Y are expected to be (n_samples, ...) and are flattened along feature dims.
    """

    def __init__(
        self,
        kernel_type: Literal["linear", "rbf"] = "linear",
        sigma: Optional[float] = None,
        unbiased: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Batch sizes must match along axis 0: {X.shape[0]} vs {Y.shape[0]}"
            )

        # Flatten feature dims
        X = X.view(X.shape[0], -1)
        Y = Y.view(Y.shape[0], -1)

        if self.kernel_type == "linear":
            if self.unbiased:
                return self._unbiased_linear_cka(X, Y)
            else:
                # Standard (biased) fast linear CKA
                Xc = X - X.mean(dim=0, keepdim=True)
                Yc = Y - Y.mean(dim=0, keepdim=True)
                return self._fast_linear_cka(Xc, Yc)

        elif self.kernel_type == "rbf":
            if self.unbiased:
                raise ValueError(
                    "Unbiased CKA is only implemented for the linear kernel."
                )
            return self._kernel_cka(X, Y)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    # ---------- Biased linear CKA (fast form) ----------

    def _fast_linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Fast linear CKA using Frobenius norms:

            CKA(X, Y) = ||Yᵀ X||_F² / (||Xᵀ X||_F · ||Yᵀ Y||_F)

        Assumes X and Y are already centered across samples.
        """
        num = self._matrix_inner_product(X, Y) ** 2
        denom = self._matrix_inner_product(X, X) * self._matrix_inner_product(Y, Y)
        if denom <= self.eps:
            return X.new_tensor(0.0)
        return num / denom

    @staticmethod
    def _matrix_inner_product(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Frobenius norm of Yᵀ X:
            <X, Y> = ||Yᵀ X||_F
        """
        M = Y.t() @ X
        return torch.norm(M, p="fro")

    # ---------- Unbiased linear CKA via unbiased HSIC ----------

    def _unbiased_linear_hsic(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Unbiased HSIC estimator (Song et al.) for linear kernel.

        X: [n, d_x]
        Y: [n, d_y]
        Returns: scalar HSIC estimate.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Batch sizes must match: {X.shape[0]} vs {Y.shape[0]}")
        n = X.shape[0]
        if n < 4:
            raise ValueError(f"Unbiased HSIC requires at least 4 samples, got n={n}")

        device = X.device
        dtype = X.dtype

        # Linear kernel Gram matrices
        K = X @ X.t()   # [n, n]
        L = Y @ Y.t()   # [n, n]

        # Zero out diagonals
        K = K.clone()
        L = L.clone()
        K.fill_diagonal_(0.0)
        L.fill_diagonal_(0.0)

        ones = torch.ones(n, 1, device=device, dtype=dtype)

        # term1 = tr(K L) = sum_ij K_ij L_ij
        term1 = (K * L).sum()

        # term2 = (1^T K 1)(1^T L 1)
        K1 = K @ ones   # [n, 1]
        L1 = L @ ones   # [n, 1]
        term2 = K1.sum() * L1.sum()

        # term3 = 1^T K L 1 = sum_i K1_i L1_i
        term3 = (K1 * L1).sum()

        coef = 1.0 / (n * (n - 3.0))
        hsic = coef * (
            term1
            + term2 / ((n - 1.0) * (n - 2.0))
            - 2.0 * term3 / (n - 2.0)
        )
        return hsic

    def _unbiased_linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Unbiased linear CKA:

            CKA_unb(X, Y) =
                HSIC_unb(X, Y) / sqrt(HSIC_unb(X, X) HSIC_unb(Y, Y))
        """
        hsic_xy = self._unbiased_linear_hsic(X, Y)
        hsic_xx = self._unbiased_linear_hsic(X, X)
        hsic_yy = self._unbiased_linear_hsic(Y, Y)

        prod = hsic_xx * hsic_yy
        if prod <= 0:
            return X.new_tensor(0.0)
        denom = torch.sqrt(prod) + self.eps
        return hsic_xy / denom

    # ---------- Kernel CKA (RBF, biased HSIC) ----------

    def _get_kernel(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix K(X, Y). Currently supports only RBF.
        """
        if Y is None:
            Y = X

        if self.kernel_type == "rbf":
            if self.sigma is None:
                # median heuristic on X
                n = X.shape[0]
                if n > 1:
                    # pairwise distances via cdist
                    # To avoid huge memory, you may want batch versions in practice.
                    dists = torch.cdist(X, X, p=2.0)
                    i, j = torch.triu_indices(n, n, offset=1, device=X.device)
                    vals = dists[i, j]
                    if vals.numel() > 0:
                        self.sigma = float(vals.median().item())
                    else:
                        self.sigma = 1.0
                else:
                    self.sigma = 1.0

            sq_dists = torch.cdist(X, Y, p=2.0) ** 2
            K = torch.exp(-sq_dists / (2.0 * (self.sigma ** 2)))
            return K

        raise ValueError(f"Unsupported kernel type for _get_kernel: {self.kernel_type}")

    @staticmethod
    def _center_gram_matrix(K: torch.Tensor) -> torch.Tensor:
        """
        Center a Gram matrix K using:
            K_c = H K H,  with H = I - 1/n · 11ᵀ
        """
        n = K.shape[0]
        device = K.device
        dtype = K.dtype
        H = torch.eye(n, device=device, dtype=dtype) - torch.ones(
            (n, n), device=device, dtype=dtype
        ) / n
        return H @ K @ H

    def _hsic_xy_biased(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Biased HSIC(X, Y) = sum(K_c ⊙ L_c).
        """
        K = self._get_kernel(X)
        L = self._get_kernel(Y)
        Kc = self._center_gram_matrix(K)
        Lc = self._center_gram_matrix(L)
        return (Kc * Lc).sum()

    def _hsic_self_biased(self, X: torch.Tensor) -> torch.Tensor:
        """
        Biased HSIC(X, X).
        """
        K = self._get_kernel(X)
        Kc = self._center_gram_matrix(K)
        return (Kc * Kc).sum()

    def _kernel_cka(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Kernel CKA using biased HSIC.
        """
        hsic_xy = self._hsic_xy_biased(X, Y)
        hsic_xx = self._hsic_self_biased(X)
        hsic_yy = self._hsic_self_biased(Y)

        prod = hsic_xx * hsic_yy
        if prod <= 0:
            return X.new_tensor(0.0)
        denom = torch.sqrt(prod) + self.eps
        return hsic_xy / denom
    

if __name__ == "__main__":
    # Simple test
    rng = np.random.default_rng(42)
    X = rng.random((37, 71))
    Y = rng.random((37, 43))

    cka_cpu = CenteredKernelAlignment(kernel_type="linear")
    score = cka_cpu.forward(X, Y)
    print(f"Linear CKA: {score}")

    cka_gpu = CenteredKernelAlignmentTorch(kernel_type="linear")
    X_torch = torch.tensor(X)
    Y_torch = torch.tensor(Y)
    score_torch = cka_gpu.forward(X_torch, Y_torch)
    print(f"Linear CKA (Torch): {score_torch.item()}")
    
    diff = abs(score - score_torch.item())
    print(f"Difference: {diff}")
    
    print("--"*10)
    
    
    
    cka_cpu_unbiased = CenteredKernelAlignment(kernel_type="linear", unbiased=True)
    score_unb = cka_cpu_unbiased.forward(X, Y)
    print(f"Unbiased Linear CKA: {score_unb}")
    
    cka_gpu_unbiased = CenteredKernelAlignmentTorch(kernel_type="linear", unbiased=True)
    score_torch_unb = cka_gpu_unbiased.forward(X_torch, Y_torch)
    print(f"Unbiased Linear CKA (Torch): {score_torch_unb.item()}")
    
    diff_unb = abs(score_unb - score_torch_unb.item())
    print(f"Difference (Unbiased): {diff_unb}")
    
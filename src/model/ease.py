import numpy as np


class EASE:
    def __init__(
        self,
        lambda_: float = 0.5,
    ):
        # TODO: search fr some logical lambda
        self._lambda = lambda_

    def fit(self, G: np.ndarray):
        """Fits the EASE model
        https://arxiv.org/pdf/1905.03375
        
        Args:
            G: Gram matrix of size |I|x|I|, i.e. G = X.T x X,
            where X is user-item interactions matrix of size |U|x|I| 
        Return: Matrix B of size |I|x|I|
        """
        # assert G is symmetric and square
        assert G.T == G

        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self._lambda
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diag_indices] = 0
        self.B = B

        return B


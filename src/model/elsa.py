import numpy as np


class ELSA:
    def __init__(
        self,
        lambda_: float = 200.0,
    ):
        self._lambda = lambda_

    def fit(self, G: np.ndarray):
        raise NotImplementedError
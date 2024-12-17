from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class EASE:
    def __init__(
        self,
        lambda_: float = 0.5,
        inn2id: Dict[int, int] = None,
        id2inn: Dict[int, int] = None,
    ):
        """
        inn2id and id2inn are taken from InteractionsDataset
        """

        # TODO: search for some logical lambda
        self.lambda_ = lambda_
        self.inn2id = inn2id
        self.id2inn = id2inn

    def fit(self, X: csr_matrix):
        """Fits the EASE model
        https://arxiv.org/pdf/1905.03375

        Args:
            X: User-item interactions matrix of size |U|x|I|

            TODO: MAYBE BETTER:
            G: Gram matrix of size |I|x|I|, i.e. G = X.T x X,
            where X is user-item interactions matrix of size |U|x|I|
        Return: Matrix B of size |I|x|I|
        """
        # assert G is symmetric and square
        self.X = X
        G: np.ndarray = X.T.dot(X).toarray()

        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diag_indices] = 0

        self.B = B
        self.preds = X.dot(B)

    def predict_score(self, user_inn: int, item_inn: int):
        """Prediction for a single user-item pair"""
        item_id = self.inn2id[item_inn]
        user_id = self.inn2id[user_inn]

        return self.X[user_id, :].dot(self.B[:, item_id])

    def predict_for_kt(self, user_inn: int, interactions_set: set) -> List:
        """Prediction for a single kt (user)

        args:
            interactions_set: set of inn_dt ints which the user has
            already interacted with. Passed to exclude the interactions
            from recommendations list

        Return:
            a dataframe with inn_kt (same number as input), inn_dt and score
            for all inn_dts which we have not yet interacted with
        """
        user_id = self.inn2id[user_inn]
        scores = self.X[user_id, :].dot(self.B)[0]

        # TODO: exclude those we've interacted with

        results = pd.DataFrame(
            {
                "inn_kt": [user_inn] * len(scores),
                "inn_dt": [self.id2inn[dt_id] for dt_id in range(len(scores))],
                "score": scores,
            }
        )

        return results[~results["inn_dt"].isin(interactions_set)].sort_values(
            "score", ascending=False
        )

    def predict(self, user_inns: List[int]):
        user_ids = [self.inn2id[user_inn] for user_inn in user_inns]

        results = self.X[user_ids, :].dot(self.B)
        # results now contain scores for each user-inn pair (for every user in user_inns)
        # shape is |user_inns|x|I|

        return results

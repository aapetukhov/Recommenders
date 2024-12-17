import numpy as np
import pandas as pd
from lightfm.evaluation import auc_score, precision_at_k


def calculate_auc_score(
    lightfm_model, interactions_matrix, question_features, professional_features
):
    """
    Measure the ROC AUC metric for a model.
    A perfect score is 1.0.

    Parameters
    ----------
    lightfm_model: LightFM model
        A fitted lightfm model
    interactions_matrix :
        A lightfm interactions matrix
    question_features, professional_features:
        Lightfm features

    Returns
    -------
    String containing AUC score
    """
    score = auc_score(
        lightfm_model,
        interactions_matrix,
        item_features=question_features,
        user_features=professional_features,
        num_threads=4,
    ).mean()
    return score


def calculate_precision_at_k(
    lightfm_model, interactions_matrix, question_features, professional_features
):
    """
    Measure the ROC AUC metric for a model.
    A perfect score is 1.0.

    Parameters
    ----------
    lightfm_model: LightFM model
        A fitted lightfm model
    interactions_matrix :
        A lightfm interactions matrix
    question_features, professional_features:
        Lightfm features

    Returns
    -------
    String containing AUC score
    """
    score = precision_at_k(
        lightfm_model,
        interactions_matrix,
        item_features=question_features,
        user_features=professional_features,
        num_threads=4,
    ).mean()
    return score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from faker import Faker
from typing import List, Tuple, Dict
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

# imports re for text cleaning 
import re
from datetime import datetime, timedelta, date

GOODS = [
    "мясо", "просо", "колесо", "серсо", "лассо", "молоко", "хлеб", "сыр", "кофе", "чай", "яйца", "яйцо", "масло", "сок", "йогурт", "шоколад", "овощи", "фрукты", "баклажан"
]



def make_interactions_dataset(
        num_transactions: int = 1000,
        num_inns: int = 100,
        num_groups: int = 5,
        goods: List[str] = GOODS,
        outer_prob: float = 0.2,
        raw_word_null_prob: float = 0.3,
        word_null_prob: float = 0.1,
) -> Tuple[pd.DataFrame, List[int], Dict[int, int], Dict[int, str]]:
    """
    Генерируем транзакции между фирмами +- осмысленно.
    Идея в том что заводим группы объектов внутри которых они взаимодействуют с вероятностью 0.7
    и с веростностью 0.3 с фирмой из другой группы
    """
    
    faker = Faker()
    inns = [faker.unique.random_int(1000000000, 9999999999) for _ in range(num_inns)]
    inn2group = {inn: random.randint(1, num_groups) for inn in inns}
    group2name = {group_num: faker.unique.word() for group_num in range(1, num_groups + 1)}

    transactions = []
    for i in range(num_transactions):
        group = random.randint(1, num_groups)
        cross_group = random.random() < outer_prob
        
        # если с фирмой из другой группы
        if cross_group:
            kt_group = random.randint(1, num_groups)
            while kt_group == group:
                kt_group = random.randint(1, num_groups)
            inn_kt = random.choice([inn for inn, g in inn2group.items() if g == kt_group])
        # иначе с фирммой из нашей группы
        else:
            inn_kt = random.choice([inn for inn, g in inn2group.items() if g == group])

        inn_dt = random.choice([inn for inn, g in inn2group.items() if g == group])

        # чтобы не торговали сами с собой
        while inn_dt == inn_kt:
            inn_dt = random.choice([inn for inn, g in inn2group.items() if g == group])

        raw_word = random.choice([None, "word1", "word2", "word3", "word4"]) if random.random() > raw_word_null_prob else None
        word = None if raw_word is None or random.random() < word_null_prob else f"clean_{raw_word[-1]}"

        transaction = {
            "id_trans": i + 1,
            "inn_kt": inn_kt,
            "inn_dt": inn_dt,
            "c_sum": round(random.uniform(10000, 250000), 2),
            "date": faker.date_between(
                start_date=datetime(2021, 1, 1),
                end_date=datetime(2023, 12, 31)
            ),
            # пусть назн не важен, пока пох
            "nazn": random.choice(goods),
            "kt_group_num": inn2group[inn_kt],
            "dt_group_num": inn2group[inn_dt],
            "kt_group_name": group2name[inn2group[inn_kt]],
            "dt_group_name": group2name[inn2group[inn_dt]],
            "raw_word": raw_word,
            "word": word,
        }
        transactions.append(transaction)


    return pd.DataFrame(transactions), inns, inn2group, group2name



def generate_feature_list(dataframe, features_name):
    """
    Generate features list for mapping 

    Parameters
    ----------
    dataframe: Dataframe 
    features_name : List
        List of feature columns name avaiable in dataframe. 
        
    Returns
    -------
    List of all features for mapping 
    """
    features = dataframe[features_name].apply(
        lambda x: ','.join(x.map(str)), axis=1)
    features = features.str.split(',')
    features = features.apply(pd.Series).stack().reset_index(drop=True)
    return features


def create_features(dataframe, features_name, id_col_name):
    """
    Generate features that will be ready for feeding into lightfm

    Parameters
    ----------
    dataframe: Dataframe
        Pandas Dataframe which contains features
    features_name : List
        List of feature columns name avaiable in dataframe
    id_col_name: String
        Column name which contains id of the question or
        answer that the features will map to.
        There are two possible values for this variable.
        1. questions_id_num
        2. professionals_id_num

    Returns
    -------
    Pandas Series
        A pandas series containing process features
        that are ready for feed into lightfm.
        The format of each value
        will be (user_id, ['feature_1', 'feature_2', 'feature_3'])
        Ex. -> (1, ['military', 'army', '5'])
    """

    features = dataframe[features_name].apply(
        lambda x: ','.join(x.map(str)), axis=1)
    features = features.str.split(',')
    features = list(zip(dataframe[id_col_name], features))
    return features


def calculate_auc_score(lightfm_model, interactions_matrix, 
                        question_features, professional_features): 
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
        lightfm_model, interactions_matrix, 
        item_features=question_features, 
        user_features=professional_features, 
        num_threads=4).mean()
    return score


def calculate_precision_at_k(lightfm_model, interactions_matrix, 
                        question_features, professional_features): 
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
        lightfm_model, interactions_matrix, 
        item_features=question_features, 
        user_features=professional_features, 
        num_threads=4).mean()
    return score
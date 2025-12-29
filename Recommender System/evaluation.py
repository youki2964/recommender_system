"""
evaluation.py
-------------
离线评估模块：实现 Precision@K（默认 K=5）的计算。

业务含义（非常重要，适合答辩讲解）：
--------------------------------
对某个用户 u：
    - 模型给出一份推荐列表 R_u（例如 Top-5）
    - 在测试集的“真实未来行为”中，用户真正喜欢的电影集合为 L_u
    - Precision@5(u) = |R_u ∩ L_u| / 5

对所有有测试数据的用户，计算各自的 Precision@5，然后取平均，
得到整个系统的 Precision@5。
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from recommender import Recommender


def precision_at_k(
    recommender: Recommender,
    test_interaction: pd.DataFrame,
    k: int = 5,
) -> float:
    """
    计算整体 Precision@K。

    :param recommender: 已训练好的推荐系统对象
    :param test_interaction: 测试集行为表（只包含“未来时间段”的行为）
    :param k: Top-K，这里默认为 5，即 Precision@5
    :return: 平均 Precision@K
    """
    # 1. 为每个用户构建“在测试集中真实喜欢的电影集合”
    #    这里仍然使用 rating >= 4 表示“喜欢”
    user_liked_movies: Dict[int, List[int]] = {}

    for uid, hist in test_interaction.groupby("user_id"):
        uid_int = int(uid)
        liked = hist[hist["rating"] >= 4]["movie_id"].tolist()
        if liked:
            user_liked_movies[uid_int] = liked

    if not user_liked_movies:
        print("【Eval】测试集中没有任何高分行为，无法计算 Precision@K。")
        return 0.0

    precisions: List[float] = []

    print("\n========== 开始计算 Precision@{} ==========".format(k))

    for user_id, liked_list in user_liked_movies.items():
        liked_set = set(liked_list)

        # 2. 使用推荐系统为该用户生成 Top-K 推荐
        recommended = recommender.recommend(user_id, top_n=k)
        if not recommended:
            continue

        rec_set = set(recommended[:k])

        # 3. 计算命中数 = 推荐列表与真实喜欢集合的交集大小
        hit_count = len(rec_set & liked_set)
        prec = hit_count / float(k)
        precisions.append(prec)

    if not precisions:
        print("【Eval】没有有效用户可用于计算 Precision@K。")
        return 0.0

    avg_precision = float(np.mean(precisions))
    print("【Eval】有效用户数: {}".format(len(precisions)))
    print("【Eval】Precision@{} = {:.4f}".format(k, avg_precision))

    return avg_precision



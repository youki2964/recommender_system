"""
feature_engineering.py
----------------------
特征工程模块：在这里把“原始表”转换为“画像表”和“行为表”。

目标产物：
1. user_profile（用户画像表）
   - user_id
   - age_normalized （标准化后的年龄）
   - gender_M, gender_F （二值编码）
   - occupation_* （职业 one-hot）

2. item_profile（电影画像表）
   - movie_id
   - 各类型 one-hot（直接使用 u.item 中的 0/1 列）

3. interaction（行为表）
   - user_id
   - movie_id
   - rating
   - timestamp
   - datetime （时间戳转为可读时间）

4. train / test 切分（按时间先后 80% / 20%）
"""

from typing import Tuple

import numpy as np
import pandas as pd


def build_user_profile(user_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建用户画像表 user_profile。

    步骤说明：
    1. 性别 gender：转换为 gender_M / gender_F 两个 0/1 特征
    2. 职业 occupation：做 one-hot 编码（每种职业一个 0/1 列）
    3. 年龄 age：做标准化（减去均值 / 除以标准差）
    """
    df = user_raw_df.copy()

    # 1. 性别二值编码
    df["gender_M"] = (df["gender"] == "M").astype(int)
    df["gender_F"] = (df["gender"] == "F").astype(int)

    # 2. 职业 one-hot
    occupation_dummies = pd.get_dummies(
        df["occupation"],
        prefix="occ",
        dtype=int,
    )
    df = pd.concat([df, occupation_dummies], axis=1)

    # 3. 年龄标准化
    age_mean = df["age"].mean()
    age_std = df["age"].std()
    df["age_normalized"] = (df["age"] - age_mean) / age_std

    # 4. 选择输出列
    feature_cols = ["user_id", "age_normalized", "gender_M", "gender_F"] + [
        c for c in df.columns if c.startswith("occ_")
    ]
    user_profile = df[feature_cols].reset_index(drop=True)

    return user_profile


def build_item_profile(item_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建电影画像表 item_profile。

    MovieLens 的 u.item 自带 19 个类型字段（unknown, Action, ..., Western），
    每一列已经是 0/1 编码，这里我们只需要挑出这些列即可。
    """
    df = item_raw_df.copy()

    # 除去 movie_id, title 以及其他非类型列，剩下的就是类型 one-hot
    genre_cols = [
        c
        for c in df.columns
        if c
        not in ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
    ]

    item_profile = df[["movie_id"] + genre_cols].reset_index(drop=True)
    return item_profile


def build_interaction(interaction_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建行为表 interaction。

    主要处理：
    1. 保留必要字段：user_id, movie_id, rating, timestamp
    2. 增加 datetime 字段，便于按时间排序与切分
    3. 按时间排序
    """
    df = interaction_raw_df.copy()

    # MovieLens 数据本身很干净，这里简单过滤 rating > 0 即可
    df = df[df["rating"] > 0]

    # 增加 datetime 字段
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # 按时间排序，更贴近真实场景
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def train_test_split_by_time(
    interaction: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间先后顺序切分训练集 / 测试集。

    - 前 80% 时间段的数据作为训练集
    - 后 20% 时间段的数据作为测试集

    这样可以模拟“用过去的数据训练，用未来的数据评估”的业务场景。
    """
    num_total = len(interaction)
    split_idx = int(num_total * (1.0 - test_ratio))

    train_df = interaction.iloc[:split_idx].copy()
    test_df = interaction.iloc[split_idx:].copy()

    return train_df, test_df



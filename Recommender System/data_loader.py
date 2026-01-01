"""
data_loader.py
--数据读取与基础清洗模块。

负责从 data/ 目录中读取 MovieLens 100k 的三个核心文件：
    - u.user  用户信息
    - u.item  电影信息
    - u.data  用户-电影评分行为

只做“读取 + 最小清洗”，不做特征工程（特征工程放在 feature_engineering.py 中）。
"""

from typing import Tuple

import pandas as pd


class DataLoader:
    """
    数据加载类。
    使用方法示例：
        loader = DataLoader(data_dir="data")
        user_raw, item_raw, interaction_raw = loader.load_all()
    """
    def __init__(self, data_dir: str = "ml-100k") -> None:
        """
        :param data_dir: 存放 u.user / u.item / u.data 的目录名
        """
        self.data_dir = data_dir

    # 读取三个原始数据文件
    def load_user_data(self) -> pd.DataFrame:
        """
        读取 u.user
        原始格式（以竖线 | 分隔）：
            user id | age | gender | occupation | zip code
        返回：包含上述列的 DataFrame（列名做了显式命名）
        """
        user_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
        path = f"{self.data_dir}/u.user"

        df = pd.read_csv(
            path,
            sep="|",
            names=user_cols,
            encoding="latin-1",
        )
        return df

    def load_item_data(self) -> pd.DataFrame:
        """
        读取 u.item
        原始格式（以竖线 | 分隔）：
            movie id | movie title | release date | video release date |
            IMDb URL | unknown | Action | Adventure | ... | Western
        最后 19 列是电影类型的 one-hot。
        """
        genre_cols = [
            "unknown", "Action", "Adventure", "Animation",
            "Children", "Comedy", "Crime", "Documentary",
            "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi",
            "Thriller", "War", "Western",
        ]

        item_cols = [
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
        ] + genre_cols

        path = f"{self.data_dir}/u.item"

        df = pd.read_csv(
            path,
            sep="|",
            names=item_cols,
            encoding="latin-1",
        )
        return df

    def load_interaction_data(self) -> pd.DataFrame:
        """
        读取 u.data
        原始格式（以制表符 \t 分隔）：
            user id \t item id \t rating \t timestamp
        """
        cols = ["user_id", "movie_id", "rating", "timestamp"]
        path = f"{self.data_dir}/u.data"

        df = pd.read_csv(
            path,
            sep="\t",
            names=cols,
            encoding="latin-1",
        )
        return df

    # 一次性读取全部
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        一次性读取三个原始表：
        :return: (user_raw_df, item_raw_df, interaction_raw_df)
        """
        user_df = self.load_user_data()
        item_df = self.load_item_data()
        interaction_df = self.load_interaction_data()

        return user_df, item_df, interaction_df



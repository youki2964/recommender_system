"""
数据加载模块
负责从MovieLens数据集中加载和预处理数据
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, data_dir: str = "data/ml-100k"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据文件目录路径
        """
        self.data_dir = data_dir
    
    def load_ratings(self, filename: str = "u.data") -> pd.DataFrame:
        """
        加载评分数据
        
        Args:
            filename: 评分数据文件名
            
        Returns:
            包含user_id, item_id, rating, timestamp的DataFrame
        """
        filepath = f"{self.data_dir}/{filename}"
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings = pd.read_csv(filepath, sep='\t', names=columns)
        return ratings
    
    def load_items(self, filename: str = "u.item") -> pd.DataFrame:
        """
        加载电影信息
        
        Args:
            filename: 电影信息文件名
            
        Returns:
            包含电影信息的DataFrame
        """
        filepath = f"{self.data_dir}/{filename}"
        columns = ['movie_id', 'title', 'release_date', 'video_release_date', 
                   'imdb_url'] + [f'genre_{i}' for i in range(19)]
        items = pd.read_csv(filepath, sep='|', names=columns, encoding='latin-1')
        return items
    
    def load_users(self, filename: str = "u.user") -> pd.DataFrame:
        """
        加载用户信息
        
        Args:
            filename: 用户信息文件名
            
        Returns:
            包含用户信息的DataFrame
        """
        filepath = f"{self.data_dir}/{filename}"
        columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
        users = pd.read_csv(filepath, sep='|', names=columns)
        return users
    
    def split_train_test(self, ratings: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        划分训练集和测试集
        
        Args:
            ratings: 评分数据
            test_ratio: 测试集比例
            
        Returns:
            (训练集, 测试集)
        """
        # 按用户分组，为每个用户随机划分
        train_list = []
        test_list = []
        
        for user_id, group in ratings.groupby('user_id'):
            n_test = int(len(group) * test_ratio)
            indices = group.index.tolist()
            np.random.shuffle(indices)
            test_indices = indices[:n_test]
            train_indices = indices[n_test:]
            
            train_list.append(group.loc[train_indices])
            test_list.append(group.loc[test_indices])
        
        train_df = pd.concat(train_list)
        test_df = pd.concat(test_list)
        
        return train_df, test_df


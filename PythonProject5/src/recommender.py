"""
推荐算法模块
实现各种推荐算法，包括协同过滤、基于内容的推荐等
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    """协同过滤推荐算法"""
    
    def __init__(self, method: str = 'user_based'):
        """
        初始化协同过滤推荐器
        
        Args:
            method: 推荐方法，'user_based' 或 'item_based'
        """
        self.method = method
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, ratings: pd.DataFrame):
        """
        训练模型
        
        Args:
            ratings: 评分数据，包含user_id, item_id, rating列
        """
        # 创建用户-物品矩阵
        self.user_ids = sorted(ratings['user_id'].unique())
        self.item_ids = sorted(ratings['item_id'].unique())
        
        user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        rows = [user_to_idx[uid] for uid in ratings['user_id']]
        cols = [item_to_idx[iid] for iid in ratings['item_id']]
        values = ratings['rating'].values
        
        self.user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )
        
        # 计算相似度矩阵
        if self.method == 'user_based':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        else:  # item_based
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            预测评分
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return 0.0
        
        user_idx = self.user_ids.index(user_id)
        item_idx = self.item_ids.index(item_id)
        
        if self.method == 'user_based':
            # 找到相似用户
            user_ratings = self.user_item_matrix[user_idx, :].toarray().flatten()
            similar_users = self.similarity_matrix[user_idx]
            
            # 计算加权平均
            item_ratings = self.user_item_matrix[:, item_idx].toarray().flatten()
            mask = item_ratings > 0
            if mask.sum() == 0:
                return 0.0
            
            weighted_sum = np.sum(similar_users[mask] * item_ratings[mask])
            similarity_sum = np.sum(np.abs(similar_users[mask]))
            
            if similarity_sum == 0:
                return 0.0
            
            return weighted_sum / similarity_sum
        else:  # item_based
            # 找到相似物品
            item_ratings = self.user_item_matrix[:, item_idx].toarray().flatten()
            similar_items = self.similarity_matrix[item_idx]
            
            # 计算加权平均
            user_ratings = self.user_item_matrix[user_idx, :].toarray().flatten()
            mask = user_ratings > 0
            if mask.sum() == 0:
                return 0.0
            
            weighted_sum = np.sum(similar_items[mask] * user_ratings[mask])
            similarity_sum = np.sum(np.abs(similar_items[mask]))
            
            if similarity_sum == 0:
                return 0.0
            
            return weighted_sum / similarity_sum
    
    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        为用户推荐Top-N物品
        
        Args:
            user_id: 用户ID
            n: 推荐物品数量
            
        Returns:
            [(物品ID, 预测评分), ...] 列表
        """
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix[user_idx, :].toarray().flatten()
        
        # 获取未评分的物品
        unrated_items = np.where(user_ratings == 0)[0]
        
        predictions = []
        for item_idx in unrated_items:
            item_id = self.item_ids[item_idx]
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]


class ContentBasedRecommender:
    """基于内容的推荐算法"""
    
    def __init__(self):
        """初始化基于内容的推荐器"""
        self.item_features = None
        self.user_profiles = None
        self.item_ids = None
        self.user_ids = None
    
    def fit(self, items: pd.DataFrame, ratings: pd.DataFrame):
        """
        训练模型
        
        Args:
            items: 物品特征数据
            ratings: 评分数据
        """
        # 提取物品特征（这里使用电影类型作为特征）
        genre_cols = [col for col in items.columns if col.startswith('genre_')]
        self.item_features = items[['movie_id'] + genre_cols].set_index('movie_id')
        self.item_ids = self.item_features.index.tolist()
        
        # 构建用户画像（基于用户对物品的评分加权平均）
        self.user_ids = sorted(ratings['user_id'].unique())
        self.user_profiles = {}
        
        for user_id in self.user_ids:
            user_ratings = ratings[ratings['user_id'] == user_id]
            user_items = user_ratings['item_id'].values
            user_scores = user_ratings['rating'].values
            
            # 计算加权特征向量
            user_profile = np.zeros(len(genre_cols))
            total_weight = 0
            
            for item_id, score in zip(user_items, user_scores):
                if item_id in self.item_ids:
                    item_features = self.item_features.loc[item_id].values
                    user_profile += item_features * score
                    total_weight += score
            
            if total_weight > 0:
                user_profile /= total_weight
            
            self.user_profiles[user_id] = user_profile
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        预测用户对物品的评分
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            
        Returns:
            预测评分
        """
        if user_id not in self.user_profiles or item_id not in self.item_ids:
            return 0.0
        
        user_profile = self.user_profiles[user_id]
        item_features = self.item_features.loc[item_id].values
        
        # 使用余弦相似度作为预测评分
        similarity = np.dot(user_profile, item_features) / (
            np.linalg.norm(user_profile) * np.linalg.norm(item_features) + 1e-10
        )
        
        # 将相似度映射到1-5评分范围
        return 1 + 4 * max(0, similarity)
    
    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        为用户推荐Top-N物品
        
        Args:
            user_id: 用户ID
            n: 推荐物品数量
            
        Returns:
            [(物品ID, 预测评分), ...] 列表
        """
        if user_id not in self.user_profiles:
            return []
        
        predictions = []
        for item_id in self.item_ids:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]


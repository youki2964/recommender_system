"""
评估模块
用于评估推荐系统的性能指标，如RMSE、MAE、精确率、召回率等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Evaluator:
    """推荐系统评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算均方根误差 (RMSE)
        
        Args:
            y_true: 真实评分
            y_pred: 预测评分
            
        Returns:
            RMSE值
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算平均绝对误差 (MAE)
        
        Args:
            y_true: 真实评分
            y_pred: 预测评分
            
        Returns:
            MAE值
        """
        return mean_absolute_error(y_true, y_pred)
    
    def evaluate_rating_prediction(self, test_data: pd.DataFrame, 
                                   recommender, method: str = 'predict') -> Dict[str, float]:
        """
        评估评分预测性能
        
        Args:
            test_data: 测试数据，包含user_id, item_id, rating列
            recommender: 推荐器对象
            method: 预测方法名
            
        Returns:
            包含RMSE和MAE的字典
        """
        predictions = []
        true_ratings = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            true_rating = row['rating']
            
            pred_func = getattr(recommender, method)
            pred_rating = pred_func(user_id, item_id)
            
            if pred_rating > 0:  # 只评估能预测的项
                predictions.append(pred_rating)
                true_ratings.append(true_rating)
        
        if len(predictions) == 0:
            return {'RMSE': float('inf'), 'MAE': float('inf')}
        
        predictions = np.array(predictions)
        true_ratings = np.array(true_ratings)
        
        rmse = self.calculate_rmse(true_ratings, predictions)
        mae = self.calculate_mae(true_ratings, predictions)
        
        return {'RMSE': rmse, 'MAE': mae}
    
    def evaluate_topn(self, test_data: pd.DataFrame, recommender, 
                     n: int = 10, threshold: float = 3.5) -> Dict[str, float]:
        """
        评估Top-N推荐性能
        
        Args:
            test_data: 测试数据
            recommender: 推荐器对象
            n: 推荐物品数量
            threshold: 评分阈值，高于此值视为用户喜欢
            
        Returns:
            包含精确率、召回率、F1分数的字典
        """
        user_precisions = []
        user_recalls = []
        
        for user_id in test_data['user_id'].unique():
            user_test = test_data[test_data['user_id'] == user_id]
            # 用户喜欢的物品（评分>=threshold）
            relevant_items = set(user_test[user_test['rating'] >= threshold]['item_id'].tolist())
            
            if len(relevant_items) == 0:
                continue
            
            # 获取推荐列表
            recommendations = recommender.recommend(user_id, n=n)
            recommended_items = set([item_id for item_id, _ in recommendations])
            
            if len(recommended_items) == 0:
                user_precisions.append(0.0)
                user_recalls.append(0.0)
                continue
            
            # 计算精确率和召回率
            intersection = relevant_items & recommended_items
            precision = len(intersection) / len(recommended_items) if len(recommended_items) > 0 else 0.0
            recall = len(intersection) / len(relevant_items) if len(relevant_items) > 0 else 0.0
            
            user_precisions.append(precision)
            user_recalls.append(recall)
        
        if len(user_precisions) == 0:
            return {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
        
        avg_precision = np.mean(user_precisions)
        avg_recall = np.mean(user_recalls)
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-10)
        
        return {
            'Precision': avg_precision,
            'Recall': avg_recall,
            'F1': f1
        }
    
    def evaluate_all(self, test_data: pd.DataFrame, recommender, 
                    n: int = 10, threshold: float = 3.5) -> Dict[str, float]:
        """
        综合评估推荐系统
        
        Args:
            test_data: 测试数据
            recommender: 推荐器对象
            n: 推荐物品数量
            threshold: 评分阈值
            
        Returns:
            包含所有评估指标的字典
        """
        rating_metrics = self.evaluate_rating_prediction(test_data, recommender)
        topn_metrics = self.evaluate_topn(test_data, recommender, n, threshold)
        
        return {**rating_metrics, **topn_metrics}


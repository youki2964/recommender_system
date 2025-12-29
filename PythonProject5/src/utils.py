"""
工具函数模块
提供各种辅助函数
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import json


def save_results(results: Dict[str, Any], filepath: str):
    """
    保存评估结果到JSON文件
    
    Args:
        results: 结果字典
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载结果
    
    Args:
        filepath: 文件路径
        
    Returns:
        结果字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_metrics(metrics: Dict[str, float], title: str = "评估结果"):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"{'='*50}\n")


def get_popular_items(ratings: pd.DataFrame, n: int = 10) -> List[int]:
    """
    获取热门物品列表
    
    Args:
        ratings: 评分数据
        n: 返回数量
        
    Returns:
        热门物品ID列表
    """
    item_counts = ratings.groupby('item_id').size().sort_values(ascending=False)
    return item_counts.head(n).index.tolist()


def get_user_statistics(ratings: pd.DataFrame) -> Dict[str, Any]:
    """
    获取用户统计信息
    
    Args:
        ratings: 评分数据
        
    Returns:
        统计信息字典
    """
    stats = {
        'total_users': ratings['user_id'].nunique(),
        'total_items': ratings['item_id'].nunique(),
        'total_ratings': len(ratings),
        'avg_ratings_per_user': len(ratings) / ratings['user_id'].nunique(),
        'avg_rating': ratings['rating'].mean(),
        'rating_distribution': ratings['rating'].value_counts().to_dict()
    }
    return stats


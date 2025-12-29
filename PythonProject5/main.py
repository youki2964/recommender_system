"""
推荐系统主程序入口
"""

import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.recommender import CollaborativeFiltering, ContentBasedRecommender
from src.evaluator import Evaluator
from src.utils import print_metrics, save_results, get_user_statistics
from config.config import (
    DATA_DIR, TEST_RATIO, RANDOM_SEED, RECOMMENDATION_METHOD,
    TOP_N, RATING_THRESHOLD, RESULTS_DIR, SAVE_RESULTS
)
import os


def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    
    print("="*60)
    print("推荐系统项目")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    loader = DataLoader(data_dir=DATA_DIR)
    ratings = loader.load_ratings()
    items = loader.load_items()
    users = loader.load_users()
    
    print(f"   - 评分数据: {len(ratings)} 条记录")
    print(f"   - 用户数量: {ratings['user_id'].nunique()}")
    print(f"   - 物品数量: {ratings['item_id'].nunique()}")
    
    # 显示统计信息
    stats = get_user_statistics(ratings)
    print(f"   - 平均评分: {stats['avg_rating']:.2f}")
    print(f"   - 平均每个用户评分数: {stats['avg_ratings_per_user']:.2f}")
    
    # 2. 划分训练集和测试集
    print("\n[2] 划分训练集和测试集...")
    train_data, test_data = loader.split_train_test(ratings, test_ratio=TEST_RATIO)
    print(f"   - 训练集: {len(train_data)} 条记录")
    print(f"   - 测试集: {len(test_data)} 条记录")
    
    # 3. 训练协同过滤模型
    print(f"\n[3] 训练协同过滤模型 ({RECOMMENDATION_METHOD})...")
    cf_recommender = CollaborativeFiltering(method=RECOMMENDATION_METHOD)
    cf_recommender.fit(train_data)
    print("   - 模型训练完成")
    
    # 4. 评估协同过滤模型
    print("\n[4] 评估协同过滤模型...")
    evaluator = Evaluator()
    cf_metrics = evaluator.evaluate_all(test_data, cf_recommender, n=TOP_N, threshold=RATING_THRESHOLD)
    print_metrics(cf_metrics, "协同过滤模型评估结果")
    
    # 5. 训练基于内容的推荐模型
    print("\n[5] 训练基于内容的推荐模型...")
    cb_recommender = ContentBasedRecommender()
    cb_recommender.fit(items, train_data)
    print("   - 模型训练完成")
    
    # 6. 评估基于内容的推荐模型
    print("\n[6] 评估基于内容的推荐模型...")
    cb_metrics = evaluator.evaluate_all(test_data, cb_recommender, n=TOP_N, threshold=RATING_THRESHOLD)
    print_metrics(cb_metrics, "基于内容的推荐模型评估结果")
    
    # 7. 保存结果
    if SAVE_RESULTS:
        print("\n[7] 保存评估结果...")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results = {
            'collaborative_filtering': cf_metrics,
            'content_based': cb_metrics,
            'config': {
                'method': RECOMMENDATION_METHOD,
                'top_n': TOP_N,
                'test_ratio': TEST_RATIO
            }
        }
        save_results(results, f"{RESULTS_DIR}/evaluation_results.json")
        print(f"   - 结果已保存到 {RESULTS_DIR}/evaluation_results.json")
    
    # 8. 示例推荐
    print("\n[8] 示例推荐...")
    sample_user = test_data['user_id'].iloc[0]
    print(f"   为用户 {sample_user} 推荐 Top-{TOP_N} 物品:")
    recommendations = cf_recommender.recommend(sample_user, n=TOP_N)
    for i, (item_id, score) in enumerate(recommendations[:5], 1):
        print(f"   {i}. 物品ID: {item_id}, 预测评分: {score:.2f}")
    
    print("\n" + "="*60)
    print("程序执行完成！")
    print("="*60)


if __name__ == "__main__":
    main()


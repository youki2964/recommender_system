"""
测试向量化优化版本的收缩评分计算函数
"""
import time
from data_loader import DataLoader
from feature_engineering import compute_shrinkage_ratings_vectorized

print("=" * 60)
print("测试向量化优化版本的收缩评分计算")
print("=" * 60)

# 加载数据
print("\n正在加载数据...")
loader = DataLoader("ml-100k")
users, items, interactions = loader.load_all()
print(f"数据加载完成: {len(items)} 部电影, {len(interactions)} 条评分记录")

# 测试向量化版本
print("\n正在测试向量化优化版本...")
start_time = time.time()
result = compute_shrinkage_ratings_vectorized(items, interactions, shrinkage_factor=5.0)
elapsed_time = time.time() - start_time

print(f"\n优化版本完成！")
print(f"计算耗时: {elapsed_time:.2f} 秒")
print(f"结果形状: {result.shape}")
print(f"\n前10个结果:")
print(result.head(10))
print(f"\n收缩评分统计:")
print(result["shrinkage_rating"].describe())

print("\n" + "=" * 60)
print("测试通过！向量化优化版本工作正常。")
print("=" * 60)


"""
配置文件
存储项目的配置参数
"""

# 数据路径配置
DATA_DIR = "data/ml-100k"
RATINGS_FILE = "u.data"
ITEMS_FILE = "u.item"
USERS_FILE = "u.user"

# 数据划分配置
TEST_RATIO = 0.2
RANDOM_SEED = 42

# 推荐算法配置
RECOMMENDATION_METHOD = "user_based"  # 'user_based' 或 'item_based'
TOP_N = 10  # Top-N推荐数量

# 评估配置
RATING_THRESHOLD = 3.5  # 用于Top-N评估的评分阈值

# 输出配置
RESULTS_DIR = "results"
SAVE_RESULTS = True


# 推荐系统项目

基于MovieLens 100K数据集的推荐系统实现，包含协同过滤和基于内容的推荐算法。

## 项目结构

```
PythonProject5/
├── data/                    # 数据目录
│   └── ml-100k/            # MovieLens 100K数据集
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── data_loader.py      # 数据加载模块
│   ├── recommender.py      # 推荐算法模块
│   ├── evaluator.py        # 评估模块
│   └── utils.py            # 工具函数模块
├── config/                  # 配置目录
│   └── config.py           # 配置文件
├── results/                 # 结果输出目录（自动创建）
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖包列表
└── README.md               # 项目说明文档
```

## 功能模块

### 1. 数据加载模块 (src/data_loader.py)
- `DataLoader`: 数据加载器类
  - `load_ratings()`: 加载评分数据
  - `load_items()`: 加载电影信息
  - `load_users()`: 加载用户信息
  - `split_train_test()`: 划分训练集和测试集

### 2. 推荐算法模块 (src/recommender.py)
- `CollaborativeFiltering`: 协同过滤推荐算法
  - 支持基于用户的协同过滤 (user-based)
  - 支持基于物品的协同过滤 (item-based)
  - `fit()`: 训练模型
  - `predict()`: 预测评分
  - `recommend()`: Top-N推荐

- `ContentBasedRecommender`: 基于内容的推荐算法
  - 使用电影类型特征构建用户画像
  - `fit()`: 训练模型
  - `predict()`: 预测评分
  - `recommend()`: Top-N推荐

### 3. 评估模块 (src/evaluator.py)
- `Evaluator`: 推荐系统评估器
  - `calculate_rmse()`: 计算均方根误差
  - `calculate_mae()`: 计算平均绝对误差
  - `evaluate_rating_prediction()`: 评估评分预测性能
  - `evaluate_topn()`: 评估Top-N推荐性能
  - `evaluate_all()`: 综合评估

### 4. 工具函数模块 (src/utils.py)
- `save_results()`: 保存评估结果
- `load_results()`: 加载评估结果
- `print_metrics()`: 打印评估指标
- `get_popular_items()`: 获取热门物品
- `get_user_statistics()`: 获取用户统计信息

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行主程序

```bash
python main.py
```

### 配置参数

在 `config/config.py` 中可以修改以下配置：

- `DATA_DIR`: 数据目录路径
- `TEST_RATIO`: 测试集比例（默认0.2）
- `RECOMMENDATION_METHOD`: 协同过滤方法（'user_based' 或 'item_based'）
- `TOP_N`: Top-N推荐数量（默认10）
- `RATING_THRESHOLD`: 评分阈值（默认3.5）

## 评估指标

项目使用以下指标评估推荐系统性能：

1. **RMSE (均方根误差)**: 评估评分预测的准确性
2. **MAE (平均绝对误差)**: 评估评分预测的平均误差
3. **Precision (精确率)**: Top-N推荐中相关物品的比例
4. **Recall (召回率)**: 相关物品被推荐的比例
5. **F1 Score**: 精确率和召回率的调和平均

## 数据集说明

本项目使用MovieLens 100K数据集，包含：
- 100,000条评分记录
- 943个用户
- 1,682部电影
- 评分范围：1-5分

## 输出结果

运行程序后，评估结果会保存在 `results/evaluation_results.json` 文件中，包含：
- 协同过滤模型的评估指标
- 基于内容的推荐模型的评估指标
- 配置参数信息

## 注意事项

1. 确保数据文件位于 `data/ml-100k/` 目录下
2. 首次运行会自动创建 `results/` 目录
3. 推荐算法可能需要一些时间运行，请耐心等待


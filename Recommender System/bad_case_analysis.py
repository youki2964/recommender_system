"""
bad_case_analysis.py
--------------------
Bad Case 分析：找出并展示推荐系统失败的案例。

Bad Case 定义：
- 用户真实喜欢的电影（评分>=4）没有被推荐到Top-5
- Precision@5 = 0 或很低
- 推荐结果与用户真实偏好差异很大
"""

import pandas as pd
from typing import Dict, List, Tuple

from recommender import Recommender
from data_loader import DataLoader


def analyze_bad_cases(
    recommender: Recommender,
    test_interaction: pd.DataFrame,
    item_raw_df: pd.DataFrame,
    top_n: int = 5,
    num_cases: int = 3
) -> List[Dict]:
    """
    分析并返回 Bad Case 列表。
    
    :param recommender: 已训练好的推荐系统
    :param test_interaction: 测试集行为表
    :param item_raw_df: 电影原始数据（用于获取电影标题）
    :param top_n: 推荐数量
    :param num_cases: 返回多少个失败案例
    :return: Bad Case 列表，每个元素包含用户信息、真实喜欢、推荐结果等
    """
    # 构建电影ID到标题的映射
    movie_id_to_title = dict(zip(item_raw_df['movie_id'], item_raw_df['title']))
    
    # 为每个用户构建"真实喜欢的电影集合"
    user_liked_movies: Dict[int, List[int]] = {}
    user_test_histories: Dict[int, pd.DataFrame] = {}
    
    for uid, hist in test_interaction.groupby("user_id"):
        uid_int = int(uid)
        liked = hist[hist["rating"] >= 4]["movie_id"].tolist()
        if liked:
            user_liked_movies[uid_int] = liked
            user_test_histories[uid_int] = hist
    
    # 计算每个用户的 Precision@5
    bad_cases = []
    
    for user_id, liked_list in user_liked_movies.items():
        liked_set = set(liked_list)
        
        # 获取推荐结果
        recommended = recommender.recommend(user_id, top_n=top_n)
        if not recommended:
            continue
        
        rec_set = set(recommended[:top_n])
        
        # 计算 Precision@5
        hit_count = len(rec_set & liked_set)
        precision = hit_count / float(top_n)
        
        # 找出推荐了但用户不喜欢的（在测试集中评分<4的）
        user_test_movies = set(user_test_histories[user_id]["movie_id"].tolist())
        bad_recommendations = []
        for mid in recommended[:top_n]:
            if mid in user_test_movies:
                rating = user_test_histories[user_id][
                    user_test_histories[user_id]["movie_id"] == mid
                ]["rating"].values[0]
                if rating < 4:
                    bad_recommendations.append((mid, rating))
        
        # 记录 Bad Case（Precision较低的用户）
        if precision < 0.4:  # Precision < 0.4 认为是Bad Case
            bad_cases.append({
                "user_id": user_id,
                "precision": precision,
                "hit_count": hit_count,
                "recommended": recommended[:top_n],
                "liked_movies": liked_list,
                "bad_recommendations": bad_recommendations,
                "test_history": user_test_histories[user_id]
            })
    
    # 按 Precision 排序，选择最差的几个
    bad_cases.sort(key=lambda x: x["precision"])
    return bad_cases[:num_cases]


def format_bad_case(case: Dict, movie_id_to_title: Dict, case_num: int) -> str:
    """
    格式化一个 Bad Case 的详细信息为字符串。
    """
    user_id = case["user_id"]
    precision = case["precision"]
    recommended = case["recommended"]
    liked_movies = case["liked_movies"]
    bad_recommendations = case["bad_recommendations"]
    
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Bad Case #{case_num}: 用户 {user_id}")
    lines.append(f"{'='*60}")
    lines.append(f"Precision@5: {precision:.2%} (命中 {case['hit_count']}/5)")
    
    lines.append(f"\n【用户真实喜欢的电影】（测试集中评分>=4）:")
    for i, mid in enumerate(liked_movies[:10], 1):  # 最多显示10部
        title = movie_id_to_title.get(mid, f"Movie {mid}")
        lines.append(f"  {i}. {title} (ID: {mid})")
    if len(liked_movies) > 10:
        lines.append(f"  ... 还有 {len(liked_movies) - 10} 部")
    
    lines.append(f"\n【系统推荐的电影】（Top-5）:")
    for i, mid in enumerate(recommended, 1):
        title = movie_id_to_title.get(mid, f"Movie {mid}")
        is_hit = "✓ 命中" if mid in set(liked_movies) else "✗ 未命中"
        lines.append(f"  {i}. {title} (ID: {mid}) - {is_hit}")
    
    if bad_recommendations:
        lines.append(f"\n【推荐错误】（推荐了用户不喜欢的电影）:")
        for mid, rating in bad_recommendations:
            title = movie_id_to_title.get(mid, f"Movie {mid}")
            lines.append(f"  - {title} (ID: {mid}) - 用户评分: {rating:.1f}")
    
    # 分析失败原因
    lines.append(f"\n【失败原因分析】:")
    if case['hit_count'] == 0:
        lines.append("  - 推荐列表完全没有命中用户真实喜欢的电影")
        lines.append("  - 可能原因：")
        lines.append("    1. 用户偏好比较特殊，与主流用户差异大")
        lines.append("    2. 用户历史行为数据稀疏，协同过滤效果差")
        lines.append("    3. 排序模型对该用户特征的拟合不足")
    else:
        lines.append(f"  - 仅命中 {case['hit_count']} 部电影，推荐效果不理想")
        lines.append("  - 可能原因：")
        lines.append("    1. 召回层召回了部分相关电影，但排序层排序不准确")
        lines.append("    2. 用户偏好多样性较高，推荐过于集中在某些类型")
    
    return "\n".join(lines)


def main():
    """
    主函数：执行 Bad Case 分析
    """
    print("="*60)
    print("   推荐系统 Bad Case 分析")
    print("="*60)
    
    # 1. 初始化推荐系统
    print("\n【步骤1】初始化推荐系统...")
    rec = Recommender(data_dir="ml-100k")
    
    # 2. 加载数据
    print("【步骤2】加载数据并训练模型...")
    rec.prepare_data()
    rec.train()
    
    # 3. 获取测试集和电影信息
    train_interaction, test_interaction = rec.get_train_test_interaction()
    loader = DataLoader(data_dir="ml-100k")
    _, item_raw_df, _ = loader.load_all()
    
    # 4. 分析 Bad Cases
    print("\n【步骤3】分析 Bad Cases...")
    bad_cases = analyze_bad_cases(
        recommender=rec,
        test_interaction=test_interaction,
        item_raw_df=item_raw_df,
        top_n=5,
        num_cases=3
    )
    
    if not bad_cases:
        print("未找到 Bad Cases（所有用户的 Precision@5 都 >= 0.4）")
        return
    
    # 5. 构建电影标题映射
    movie_id_to_title = dict(zip(item_raw_df['movie_id'], item_raw_df['title']))
    
    # 6. 生成分析报告并保存到文件
    print(f"\n找到 {len(bad_cases)} 个 Bad Cases，正在生成报告...")
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("   推荐系统 Bad Case 分析报告")
    report_lines.append("="*60)
    report_lines.append(f"\n分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Bad Case 数量: {len(bad_cases)}")
    report_lines.append(f"分析标准: Precision@5 < 0.4")
    report_lines.append("\n" + "="*60)
    
    for i, case in enumerate(bad_cases, 1):
        report_lines.append(format_bad_case(case, movie_id_to_title, i))
    
    report_lines.append(f"\n{'='*60}")
    report_lines.append("Bad Case 分析完成")
    report_lines.append("="*60)
    
    # 保存到文件
    report_content = "\n".join(report_lines)
    output_file = "bad_case.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n分析报告已保存到: {output_file}")
    print(f"共分析了 {len(bad_cases)} 个 Bad Cases")
    
    # 同时在控制台显示前3个案例的摘要
    print(f"\n{'='*60}")
    print("Bad Case 摘要（详细内容请查看 bad_case.txt）:")
    print("="*60)
    for i, case in enumerate(bad_cases, 1):
        print(f"\nBad Case #{i}: 用户 {case['user_id']} - Precision@5: {case['precision']:.2%} (命中 {case['hit_count']}/5)")


if __name__ == "__main__":
    main()


"""
main.py
-------
命令行 Demo 入口。

目标：
1. 一键运行推荐系统全流程：
   - 加载数据
   - 构建特征
   - 训练召回模型 + 排序模型
   - 计算冷启动 Top-5 热门电影
   - 跑一次离线评估（Precision@5），方便在课堂上展示效果

2. 提供命令行交互：
   - 用户输入 user_id
   - 输出 Top-5 推荐电影 ID 列表
   - 对新用户和老用户都可以正常工作
"""

from recommender import Recommender
from evaluation import precision_at_k


def main() -> None:
    print("==================================================")
    print("   基于 MovieLens-100k 的两阶段推荐系统 Demo")
    print("==================================================")

    # 1. 初始化推荐系统
    rec = Recommender(data_dir="ml-100k")

    # 2. 数据准备
    rec.prepare_data()

    # 3. 训练召回 + 排序 + 冷启动
    rec.train()

    # 4. 离线评估（Precision@5）
    print("\n========== 离线评估：Precision@5 ==========")
    train_interaction, test_interaction = rec.get_train_test_interaction()
    precision_at_k(rec, test_interaction, k=5)

    # 5. 命令行交互 Demo
    print("\n========== 进入命令行推荐 Demo ==========")
    print("提示：输入用户 ID（1~943），回车后输出 Top-5 推荐电影 ID 列表。")
    print("      输入 q 退出程序。")

    while True:
        user_input = input("\n请输入用户 ID（或输入 q 退出）：").strip()
        if user_input.lower() == "q":
            print("退出程序，再见！")
            break

        # 校验用户输入
        try:
            user_id = int(user_input)
        except ValueError:
            print("输入不是有效的整数，请重新输入。")
            continue

        # 调用推荐接口
        recommended_ids = rec.recommend(user_id, top_n=5)

        print(f"为用户 {user_id} 推荐的 Top-5 电影 ID 列表：")
        if recommended_ids:
            print(recommended_ids)
        else:
            print("无法生成推荐结果。")


if __name__ == "__main__":
    main()



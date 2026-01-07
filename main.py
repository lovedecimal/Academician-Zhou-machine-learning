# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # C4.5变种实现
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据集（鸢尾花，周志华教材常用示例）
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="label")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("✅ 数据集加载完成（鸢尾花，周志华教材经典案例）")

# 2. 复现C4.5决策树（周志华《机器学习》第4章）
c45_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
c45_tree.fit(X_train, y_train)
y_pred_tree = c45_tree.predict(X_test)
tree_acc = accuracy_score(y_test, y_pred_tree)

# 3. 复现AdaBoost集成学习（周志华《机器学习》第8章）
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
y_pred_boost = adaboost.predict(X_test)
boost_acc = accuracy_score(y_test, y_pred_boost)

# 4. 性能对比（周志华教材算法评估逻辑）
print("\n📊 周志华经典算法性能对比：")
print(f"C4.5决策树准确率：{tree_acc:.2f}")
print(f"AdaBoost集成学习准确率：{boost_acc:.2f}")
print("\nC4.5决策树分类报告：")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))

# 5. 可视化（算法性能对比图）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.bar(["C4.5决策树", "AdaBoost"], [tree_acc, boost_acc], color=["steelblue", "orange"])
plt.ylabel("分类准确率")
plt.title("周志华《机器学习》经典算法性能对比")
plt.ylim(0.9, 1.0)
plt.savefig("周志华算法性能对比图.png")
plt.show()
print("\n✅ 可视化图表已保存：周志华算法性能对比图.png")
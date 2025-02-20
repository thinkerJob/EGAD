import pandas as pd
from sklearn.metrics import roc_auc_score

# 读取 Excel 文件，假设第一个表格包含真实标签，第二个表格包含预测得分
def read_data(true_labels_path, predicted_scores_path):
    true_labels = pd.read_excel(true_labels_path)  # 真实标签数据，假设列名为 '真实标签'
    predicted_scores = pd.read_excel(predicted_scores_path)  # 预测得分数据，假设列名为 '预测得分'

    return true_labels, predicted_scores

# 合并数据
def merge_data(true_labels, predicted_scores):
    # 假设它们都有一个共同的ID列用于合并
    merged_data = pd.merge(true_labels, predicted_scores, on='Name')  # 根据实际的列名进行合并
    return merged_data

# 计算 AUROC
def calculate_auroc(merged_data):
    y_true = merged_data['label']  # 真实标签列名
    y_scores = merged_data['Score']  # 预测得分列名

    auc = roc_auc_score(y_true, y_scores)
    return auc

# 主函数
if __name__ == "__main__":
    # SimpleNet,cflow-ad,CRAD,destseg,msflow,RD4AD
    # 文件路径
    true_labels_path = 'TrueLabel.xlsx'  # 真实标签的Excel文件路径
    predicted_scores_path = 'CRAD.xlsx'  # 预测得分的Excel文件路径

    # 读取数据
    true_labels, predicted_scores = read_data(true_labels_path, predicted_scores_path)

    # 合并数据
    merged_data = merge_data(true_labels, predicted_scores)

    # 计算 AUROC
    auc = calculate_auroc(merged_data)

    print(f"Detection AUROC: {auc}")

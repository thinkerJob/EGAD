import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def perron_rank(adj_matrix, max_iter=100, tolerance=1e-6):
    n = adj_matrix.shape[0]
    # 初始化节点重要性向量，均匀分布
    r = np.ones(n) / n
    for _ in range(max_iter):
        # 计算新的节点重要性向量
        new_r = np.dot(adj_matrix, r)
        # 归一化
        new_r /= np.linalg.norm(new_r, 1)
        # 检查收敛性
        if np.linalg.norm(new_r - r) < tolerance:
            break
        r = new_r
    return r

def calculate_pairwise_dominance_matrix(P):
    """
    该函数根据矩阵 P 计算成对优势矩阵 F
    :param P: 成对矩阵 P
    :return: 成对优势矩阵 F
    """
    num_models = P.shape[0]
    F = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            if P[j][i]!= 0:
                F[i][j] = P[i][j] / P[j][i]
            else:
                F[i][j] = 0  # 处理 P[j][i] 为 0 的情况，避免除以 0 的错误
    return F


import numpy as np


def compute_q(F, t_max=1000):
    n = F.shape[0]
    one_vector = np.ones((n, 1))
    q = np.zeros((n, 1))
    for t in range(1, t_max + 1):
        # 计算 (F) 的 alpha 次幂
        F_alpha = np.linalg.matrix_power(F, t)
        # 计算分子部分
        numerator = np.dot(F_alpha, one_vector)
        # 计算分母部分
        denominator = np.dot(one_vector.T, np.dot(F_alpha, one_vector))[0, 0]
        # 避免分母为零
        if np.abs(denominator) < 1e-10:
            denominator = 1e-10
        # 计算本次迭代的贡献
        contribution = numerator / denominator
        # 累加贡献
        q += contribution
    q = q / t_max
    return q

def calculate_pairwise_matrix(pairwise_AUROC_dict, model_index_dict):
    """
    该函数用于根据 pairwise_AUROC_dict 计算成对矩阵 P
    :param pairwise_AUROC_dict: 存储模型对的 AUROC 成绩的字典
    :param model_index_dict: 存储模型及其对应索引的字典
    :return: 成对矩阵 P
    """
    num_models = len(model_index_dict)
    P = np.zeros((num_models, num_models))
    for model_pair, perf_pair in pairwise_AUROC_dict.items():
        (M_a, M_b) = model_pair
        (p_a, p_b) = perf_pair
        i, j = model_index_dict[M_a], model_index_dict[M_b]
        P[i, j] = p_a
        P[j, i] = p_b

    return P



class CustomDivergingNorm(Normalize):
    def __init__(self, vcenter=1, vmin=None, vmax=None):
        """
        自定义的归一化类，以 vcenter 为中心，将值映射到 [0, 0.5] 或 [0.5, 1] 范围
        :param vcenter: 中心值
        :param vmin: 最小值，默认为 None
        :param vmax: 最大值，默认为 None
        """
        super().__init__(vmin, vmax)
        self.vcenter = vcenter

    def __call__(self, value, clip=None):
        """
        自定义的归一化类，以 vcenter 为中心，将值映射到 [0, 0.5] 或 [0.5, 1] 范围
        :param value: 输入的矩阵元素值
        :param clip: 是否裁剪值
        :return: 归一化后的值
        """
        if self.vmin is None:
            self.vmin = np.min(value)
        if self.vmax is None:
            self.vmax = np.max(value)
        result = np.where(value <= self.vcenter,
                        (value - self.vmin) / (self.vcenter - self.vmin),
                        (value - self.vcenter) / (self.vmax - self.vcenter) + 0.5)
        if clip:
            result = np.clip(result, 0, 1)
        return result

def stretch_around_one(value, factor=2):
    """
    以 1 为中心拉伸元素
    :param value: 输入矩阵元素
    :param factor: 拉伸因子
    :return: 拉伸后的值
    """
    result = np.where(value < 1, 1 - (1 - value) * factor, 1 + (value - 1) * factor)
    return result



# 示例使用
if __name__ == "__main__":


    models = ['CFLOW-AD', 'CARD', 'DeSTSeg', "MSFlow", "Reverse Distillation", "SimpleNet"]
    J = len(models)
    model_index_dict = {model: index for index, model in enumerate(models)}
    """
    将模型的成绩填入pairwise_AUROC_dict中对应位置
    'CFLOW-AD', 'CARD'相对应数据的成绩中
    'CFLOW-AD'和'CARD'的成绩分别填入[random.uniform(0.5, 1), random.uniform(0.5, 1)]中第一个位和第二位
     'CFLOW-AD'和'CARD'的成绩如果分别是0.6和0.5
     ('CFLOW-AD', 'CARD'): [0.6, 0.5],
    """
    pairwise_AUROC_dict = {
        ('CFLOW-AD', 'CARD'): [0.31334,	0.60439],
        ('CFLOW-AD', 'DeSTSeg'): [0.55263,	0.57046],
        ('CFLOW-AD', 'MSFlow'): [0.58387,	0.48312],
        ('CFLOW-AD', 'Reverse Distillation'): [0.46557,	0.49328],
        ('CFLOW-AD', 'SimpleNet'): [0.33475,	0.65564],
        ('CARD', 'DeSTSeg'): [0.46988,	0.46619],
        ('CARD', 'MSFlow'): [0.46606,	0.47104],
        ('CARD', 'Reverse Distillation'): [0.398,	0.41915],
        ('CARD', 'SimpleNet'): [0.54634,	0.50426],
        ('DeSTSeg', 'MSFlow'): [0.58341,	0.4294 ],
        ('DeSTSeg', 'Reverse Distillation'): [0.48589,	0.51929],
        ('DeSTSeg', 'SimpleNet'): [0.45564,	0.57235],
        ('MSFlow', 'Reverse Distillation'): [0.46683,	0.51483],
        ('MSFlow', 'SimpleNet'): [0.42976,	0.57518],
        ('Reverse Distillation', 'SimpleNet'): [0.58151,	0.51719]
    }

    P = calculate_pairwise_matrix(pairwise_AUROC_dict, model_index_dict)
    # 可视化部分

    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
    plt.rcParams['font.size'] = 14  # 增大字体大小
    plt.figure(figsize=(10, 8))
    plt.imshow(P, cmap='viridis')
    # plt.colorbar()
    plt.xticks(range(len(models)), models, fontsize=10)
    plt.yticks(range(len(models)), models, fontsize=10)
    ax = plt.gca()
    ax.set_yticklabels(models, rotation=90, ha='center', va='center') # 修改 Y 轴标签对齐方式
    plt.title(r'Pairwise performance matrix $\boldsymbol{P}$')  # 使用 \mathbf 代替 \bm
    for i in range(len(models)):
        for j in range(len(models)):
            plt.text(j, i, r'${\mathbf{%.2f}}$' % P[i, j], ha='center', va='center', color='white')  # 使用 \mathbf 代替 \bm
    # plt.xlabel(r'${M_b}$')
    # plt.ylabel(r'${M_a}$')
    # 将图形保存为 PDF 文件
    plt.savefig('pairwise_performance_matrix.pdf', format='pdf',dpi=600)
    # plt.show()
    #
    F = calculate_pairwise_dominance_matrix(P)
    # dominance可视化部分
    # 创建自定义的 Normalize 对象，以 1 为中心
    norm = CustomDivergingNorm(vcenter=1)

    plt.figure(figsize=(10, 8))
    # 对矩阵 P 进行拉伸


    plt.imshow(P, cmap='coolwarm', norm=norm)  # 不使用 norm 进行归一化
    # plt.colorbar()
    plt.xticks(range(len(models)), models, fontsize=10)
    plt.yticks(range(len(models)), models, fontsize=10)
    ax = plt.gca()
    ax.set_yticklabels(models, rotation=90, ha='center', va='center') # 修改 Y 轴标签对齐方式
    plt.title(r'Pairwise dominance matrix $\boldsymbol{F}$')  # 使用 \mathbf 代替 \bm
    for i in range(len(models)):
        for j in range(len(models)):
            plt.text(j, i, r'${\mathbf{%.2f}}$' % F[i, j], ha='center', va='center', color='red')  # 使用 \mathbf 代替 \bm
    # plt.xlabel(r'$M_b$')
    # plt.ylabel(r'$M_a$')
    # 将图形保存为 PDF 文件
    plt.savefig('pairwise_dominance_matrix.pdf', format='pdf',dpi=600)
    q = perron_rank(F)
    # qq = compute_q(F)

    # 使用 zip 函数将 models 和 q 组合在一起，然后使用 sorted 函数进行排序
    # 排序依据是 q 中的值，reverse=True 表示降序排序
    sorted_pairs = sorted(zip(models, q), key=lambda x: x[1], reverse=True)
    print(sorted_pairs)
    # 输出排序结果
    sorted_models = [pair[0] for pair in sorted_pairs]
    print(sorted_models)
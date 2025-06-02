import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 设置使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的ResNet模型
model = models.resnet50(pretrained=True)
model = nn.Sequential(*(list(model.children())[:-1]))  # 去掉最后的全连接层
model.eval().to(device)

# 图像预处理和数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 特征提取函数
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features

# 加载文件夹中的图像并提取特征
def load_images_from_folder(folder):
    features_list = []
    image_paths = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # 支持的图片格式
            features = extract_features(img_path)
            features_list.append(features)
            image_paths.append(img_path)
    return features_list, image_paths

# 特征提取
folder_path = './images'  # 图像文件夹路径
all_features, image_paths = load_images_from_folder(folder_path)

# PCA 降维
all_features = np.array(all_features)
pca = PCA(n_components=20)  # 降到50维，可根据需要调整
reduced_features = pca.fit_transform(all_features)

# K-means聚类分析
num_clusters = 2  # 聚类数量，可以根据需要动态调整
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(reduced_features)
labels = kmeans.labels_

# 创建目标文件夹
real_image_folder = './need'
ai_image_folder = './other'
os.makedirs(real_image_folder, exist_ok=True)
os.makedirs(ai_image_folder, exist_ok=True)

# 根据聚类结果将图片保存到不同文件夹
for idx, label in enumerate(labels):
    src_path = image_paths[idx]
    if label == 0:
        dest_folder = real_image_folder
    else:
        dest_folder = ai_image_folder
    shutil.copy(src_path, dest_folder)



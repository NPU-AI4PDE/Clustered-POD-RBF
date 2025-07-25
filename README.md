# Clustered-POD-RBF
 
# 参数化动力学系统离线-在线计算分解框架

### 基于区域聚类降维和自适应径向基函数的高效建模方法

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

## :pencil: 项目简介

本项目实现了一个高效的参数化动力学系统计算框架，通过**离线-在线计算分解**策略，结合**区域聚类降维**和**自适应径向基函数(RBF)**技术，显著提升了参数化偏微分方程(PDEs)的求解效率。

该框架特别适用于需要大量参数变化的工程应用，如流体力学中的雷诺数变化问题、结构力学中的材料参数优化等场景。通过智能的参数空间分区和局部降阶建模，实现了高精度与高效率的完美平衡。

</div>



## :rocket: 核心特性

### 🎯 **智能参数空间分区**
- **K-means自适应聚类**: 自动识别参数空间的最优分区
- **固定分段策略**: 支持等宽、分位数、递增密度三种分段模式
- **平滑过渡机制**: 消除分区边界处的数值不连续性

### ⚡ **高效降阶建模**
- **POD正交分解**: 自动截断能量阈值，保持最优维度
- **自适应RBF**: 智能优化形状参数，确保数值稳定性
- **内存优化算法**: 大规模问题自动切换SVD/特征值分解策略

### 🔧 **工程化设计**
- **模块化架构**: 核心算法与应用解耦，便于扩展
- **鲁棒性设计**: 完善的异常处理和数值稳定性保证
- **可视化支持**: 内置热图、误差分布等多种可视化工具

## :computer: 技术栈

```python
# 核心依赖
import numpy as np           # 数值计算基础
import scipy as sp           # 科学计算库
import sklearn               # 机器学习算法
import matplotlib.pyplot as plt  # 可视化绘图
import pandas as pd          # 数据处理
from tqdm import tqdm       # 进度条显示
```

## :wrench: 安装指南

### 环境要求
- Python 3.7+
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Scikit-learn >= 0.23.0
- Matplotlib >= 3.2.0
- Pandas >= 1.1.0

### 快速安装

<details>
  <summary>使用 pip 安装依赖</summary>
  
```bash
# 克隆项目
git clone https://github.com/yourusername/pod-rbf-framework.git
cd pod-rbf-framework

# 安装依赖
pip install -r requirements.txt

# 或者手动安装
pip install numpy scipy scikit-learn matplotlib pandas tqdm
```
</details>

<details>
  <summary>使用 conda 环境</summary>
  
```bash
# 创建虚拟环境
conda create -n pod-rbf python=3.8
conda activate pod-rbf

# 安装依赖
conda install numpy scipy scikit-learn matplotlib pandas tqdm
```
</details>

## :books: 使用指南

### 快速开始

```python
from pod_rbf import pod_rbf, clustered_pod_rbf, buildSnapshotMatrix

# 1. 构建快照矩阵
snapshot_matrix = buildSnapshotMatrix("data/train/pattern.csv", usecols=(0,))

# 2. 准备训练参数 (如雷诺数)
Re_values = np.linspace(1, 999, 400)

# 3. 训练标准POD-RBF模型
model = pod_rbf(energy_threshold=0.95)
model.train(snapshot_matrix, Re_values)

# 4. 推理预测
prediction = model.inference(500.0)  # 预测Re=500时的解
```

### 聚类POD-RBF高级用法

```python
# K-means聚类模式
clustered_model = clustered_pod_rbf(
    n_clusters_kmeans=3,
    energy_threshold=0.95,
    use_smooth_transition=True
)
clustered_model.train(snapshot_matrix, Re_values)

# 固定分段模式 - 递增密度分布
density_model = clustered_pod_rbf(
    fixed_segment_param_idx=0,
    fixed_num_segments=5,
    fixed_segment_mode='increasing_density',
    fixed_segment_proportions=[1, 1.5, 2, 2.5, 3]
)
density_model.train(snapshot_matrix, Re_values)

# 批量推理
test_params = np.array([100, 300, 500, 700, 900])
predictions = clustered_model.inference(test_params)
```

### 主要API说明

<details>
  <summary>POD-RBF 核心类</summary>

**pod_rbf类**
- `__init__(energy_threshold=0.99)`: 初始化，设置POD能量保留阈值
- `train(snapshot, train_params, shape_factor=None)`: 训练模型
- `inference(inf_params)`: 推理预测

**clustered_pod_rbf类**  
- `__init__(n_clusters_kmeans=3, ...)`: 初始化聚类参数
- `train(snapshot, train_params, shape_factor=None)`: 训练局部模型
- `inference(inf_params)`: 推理预测
- `print_cluster_summary()`: 打印聚类摘要
- `save_cluster_info(filename_prefix)`: 保存聚类信息
</details>


```

## :file_folder: 项目结构

```
项目根目录/
├── lid_driven_cavity.py      # 主执行脚本 
├── pod_rbf/                   # 核心算法模块
│   ├── __init__.py           # 模块导入
│   └── pod_rbf.py            # POD-RBF算法实现
├── data/                      # 数据目录
│   ├── train/                 # 训练数据集
│   └── validation/            # 验证数据集
├── output_heatmap_combined/   # 热图输出目录
├── output_difference/         # 误差分析输出
├── requirements.txt           # Python依赖列表
└── README.md                 # 项目说明文档
```

## :gear: 算法详解

### 离线-在线分解策略

1. **离线阶段 (Offline Phase)**:
   - 快照数据收集与预处理
   - POD基函数计算与截断
   - 参数空间聚类分析
   - RBF形状参数优化
   - 局部降阶模型训练

2. **在线阶段 (Online Phase)**:
   - 参数空间定位
   - 局部模型选择
   - 快速RBF插值计算
   - 解空间重构输出

### 数学理论基础

$$\mathbf{u}(\boldsymbol{\mu}) \approx \sum_{i=1}^{N_{POD}} a_i(\boldsymbol{\mu}) \boldsymbol{\phi}_i$$

其中：
- $\mathbf{u}(\boldsymbol{\mu})$: 参数化解向量
- $\boldsymbol{\phi}_i$: POD基函数
- $a_i(\boldsymbol{\mu})$: RBF插值系数

$$a_i(\boldsymbol{\mu}) = \sum_{j=1}^{N_{train}} w_{ij} \psi(\|\boldsymbol{\mu} - \boldsymbol{\mu}_j\|)$$

RBF采用逆多二次函数：
$$\psi(r) = \frac{1}{\sqrt{r^2/c^2 + 1}}$$

## :test_tube: 验证与测试

运行完整的空腔流动算例:

```bash
python lid_driven_cavity.py
```

**输出内容:**
- 训练过程日志与性能统计
- 验证误差分析报告
- 热图可视化结果
- 聚类信息汇总表

**生成文件:**
- `error_evaluation_std_model.xlsx`: 标准模型误差统计
- `ns_*_cluster_info.npz`: 聚类模型信息
- `output_heatmap_combined/*.png`: 解场热图对比
- `output_difference/*.png`: 误差分布可视化

## :satellite: 扩展应用

### 🔬 **当前支持的物理问题**
- **流体力学**: 空腔流动、管道流、绕流问题
- **传热学**: 传导、对流、辐射传热
- **结构力学**: 弹性变形、振动分析

### 🎯 **计划新增功能**
- [ ] 时间相关问题的POD-RBF建模
- [ ] 多物理场耦合问题支持  
- [ ] GPU加速计算模块
- [ ] 自适应网格细化集成
- [ ] 深度学习混合模型

### 💡 **参数化建模最佳实践**
- 训练样本数量建议为参数维度的5-10倍
- POD能量阈值通常设置在0.95-0.999之间
- 聚类数量选择需要平衡精度与效率
- 形状参数自动优化一般比手动设置效果更好

## :bookmark_tabs: 学术引用

如果您在研究中使用了本框架，请引用以下论文：

```bibtex
@article{zhou2025offline,
  title={Offline-online computational decomposition: an efficient framework for parametric dynamical systems via regional clustering dimensionality reduction and adaptive radial basis functions: S. Zhou et al.},
  author={Zhou, Sheng and Xiong, Xiong and Lu, Kang and Zeng, Zheng and Hu, Rongchun},
  journal={Nonlinear Dynamics},
  pages={1--26},
  year={2025},
  publisher={Springer}
}
```



*© 2025 参数化动力学系统POD-RBF框架. 保留所有权利.* 

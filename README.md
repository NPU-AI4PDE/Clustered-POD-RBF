# Clustered-POD-RBF

## Offline-Online Computational Decomposition Framework for Parametric Dynamical Systems

### An efficient framework via regional clustering dimensionality reduction and adaptive radial basis functions

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

## :pencil: Project Description

This project implements an efficient computational framework for parametric dynamical systems through **offline-online computational decomposition** strategy, combining **regional clustering dimensionality reduction** and **adaptive radial basis functions (RBF)** techniques to significantly improve the solving efficiency of parametric partial differential equations (PDEs).

The framework is particularly suitable for engineering applications requiring extensive parameter variations, such as Reynolds number changes in fluid mechanics, material parameter optimization in structural mechanics, and other scenarios. Through intelligent parameter space partitioning and local reduced-order modeling, it achieves the perfect balance between high accuracy and high efficiency.

</div>

## :rocket: Core Features

### 🎯 **Intelligent Parameter Space Partitioning**
- **K-means Adaptive Clustering**: Automatically identifies optimal partitioning of parameter space
- **Fixed Segmentation Strategy**: Supports three segmentation modes: equal width, quantile, and increasing density
- **Smooth Transition Mechanism**: Eliminates numerical discontinuities at partition boundaries

### ⚡ **Efficient Reduced-Order Modeling**
- **POD Orthogonal Decomposition**: Automatic energy threshold truncation maintaining optimal dimensions
- **Adaptive RBF**: Intelligent shape parameter optimization ensuring numerical stability
- **Memory-Optimized Algorithms**: Large-scale problems automatically switch between SVD/eigenvalue decomposition strategies

### 🔧 **Engineering Design**
- **Modular Architecture**: Core algorithms decoupled from applications for easy extension
- **Robust Design**: Comprehensive exception handling and numerical stability guarantees
- **Visualization Support**: Built-in heatmaps, error distributions, and various visualization tools

## :computer: Technology Stack

```python
# Core Dependencies
import numpy as np           # Numerical computation foundation
import scipy as sp           # Scientific computing library
import sklearn               # Machine learning algorithms
import matplotlib.pyplot as plt  # Visualization plotting
import pandas as pd          # Data processing
from tqdm import tqdm       # Progress bar display
```

## :wrench: Installation Guide

### Environment Requirements
- Python 3.7+
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Scikit-learn >= 0.23.0
- Matplotlib >= 3.2.0
- Pandas >= 1.1.0

### Quick Installation

<details>
  <summary>Install dependencies using pip</summary>
  
```bash
# Clone the project
git clone https://github.com/yourusername/clustered-pod-rbf.git
cd clustered-pod-rbf

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy scipy scikit-learn matplotlib pandas tqdm
```
</details>

<details>
  <summary>Using conda environment</summary>
  
```bash
# Create virtual environment
conda create -n pod-rbf python=3.8
conda activate pod-rbf

# Install dependencies
conda install numpy scipy scikit-learn matplotlib pandas tqdm
```
</details>

## :books: Usage Guide

### Quick Start

```python
from pod_rbf import pod_rbf, clustered_pod_rbf, buildSnapshotMatrix

# 1. Build snapshot matrix
snapshot_matrix = buildSnapshotMatrix("data/train/pattern.csv", usecols=(0,))

# 2. Prepare training parameters (e.g., Reynolds numbers)
Re_values = np.linspace(1, 999, 400)

# 3. Train standard POD-RBF model
model = pod_rbf(energy_threshold=0.95)
model.train(snapshot_matrix, Re_values)

# 4. Inference prediction
prediction = model.inference(500.0)  # Predict solution at Re=500
```

### Advanced Usage of Clustered POD-RBF

```python
# K-means clustering mode
clustered_model = clustered_pod_rbf(
    n_clusters_kmeans=3,
    energy_threshold=0.95,
    use_smooth_transition=True
)
clustered_model.train(snapshot_matrix, Re_values)

# Fixed segmentation mode - increasing density distribution
density_model = clustered_pod_rbf(
    fixed_segment_param_idx=0,
    fixed_num_segments=5,
    fixed_segment_mode='increasing_density',
    fixed_segment_proportions=[1, 1.5, 2, 2.5, 3]
)
density_model.train(snapshot_matrix, Re_values)

# Batch inference
test_params = np.array([100, 300, 500, 700, 900])
predictions = clustered_model.inference(test_params)
```

### Main API Reference

<details>
  <summary>POD-RBF Core Classes</summary>

**pod_rbf Class**
- `__init__(energy_threshold=0.99)`: Initialize with POD energy retention threshold
- `train(snapshot, train_params, shape_factor=None)`: Train the model
- `inference(inf_params)`: Inference prediction

**clustered_pod_rbf Class**  
- `__init__(n_clusters_kmeans=3, ...)`: Initialize clustering parameters
- `train(snapshot, train_params, shape_factor=None)`: Train local models
- `inference(inf_params)`: Inference prediction
- `print_cluster_summary()`: Print clustering summary
- `save_cluster_info(filename_prefix)`: Save clustering information
</details>

## :file_folder: Project Structure

```
Project Root/
├── lid_driven_cavity.py      # Main execution script
├── pod_rbf/                   # Core algorithm module
│   ├── __init__.py           # Module import
│   └── pod_rbf.py            # POD-RBF algorithm implementation
├── data/                      # Data directory
│   ├── train/                 # Training dataset
│   └── validation/            # Validation dataset
├── output_heatmap_combined/   # Heatmap output directory
├── output_difference/         # Error analysis output
├── requirements.txt           # Python dependencies list
└── README.md                 # Project documentation
```

## :gear: Algorithm Details

### Offline-Online Decomposition Strategy

1. **Offline Phase**:
   - Snapshot data collection and preprocessing
   - POD basis function computation and truncation
   - Parameter space clustering analysis
   - RBF shape parameter optimization
   - Local reduced-order model training

2. **Online Phase**:
   - Parameter space localization
   - Local model selection
   - Fast RBF interpolation computation
   - Solution space reconstruction output

### Mathematical Theory Foundation

$$\mathbf{u}(\boldsymbol{\mu}) \approx \sum_{i=1}^{N_{POD}} a_i(\boldsymbol{\mu}) \boldsymbol{\phi}_i$$

Where:
- $\mathbf{u}(\boldsymbol{\mu})$: Parametric solution vector
- $\boldsymbol{\phi}_i$: POD basis functions
- $a_i(\boldsymbol{\mu})$: RBF interpolation coefficients

$$a_i(\boldsymbol{\mu}) = \sum_{j=1}^{N_{train}} w_{ij} \psi(\|\boldsymbol{\mu} - \boldsymbol{\mu}_j\|)$$

RBF uses inverse multiquadric function:
$$\psi(r) = \frac{1}{\sqrt{r^2/c^2 + 1}}$$

## :test_tube: Validation and Testing

Run the complete lid-driven cavity example:

```bash
python lid_driven_cavity.py
```

**Output Content:**
- Training process logs and performance statistics
- Validation error analysis reports
- Heatmap visualization results
- Clustering information summary tables

**Generated Files:**
- `error_evaluation_std_model.xlsx`: Standard model error statistics
- `ns_*_cluster_info.npz`: Clustered model information
- `output_heatmap_combined/*.png`: Solution field heatmap comparisons
- `output_difference/*.png`: Error distribution visualizations

## :satellite: Extended Applications

### 🔬 **Currently Supported Physical Problems**
- **Fluid Mechanics**: Cavity flow, pipe flow, flow around objects
- **Heat Transfer**: Conduction, convection, radiation heat transfer
- **Structural Mechanics**: Elastic deformation, vibration analysis

### 🎯 **Planned New Features**
- [ ] POD-RBF modeling for time-dependent problems
- [ ] Multi-physics coupled problem support  
- [ ] GPU acceleration computing module
- [ ] Adaptive mesh refinement integration
- [ ] Deep learning hybrid models

### 💡 **Parametric Modeling Best Practices**
- Training sample size should be 5-10 times the parameter dimension
- POD energy threshold typically set between 0.95-0.999
- Clustering number selection needs to balance accuracy and efficiency
- Automatic shape parameter optimization generally performs better than manual setting

## :bookmark_tabs: Academic Citation

If you use this framework in your research, please cite the following paper:

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

## :handshake: Contributing

We welcome contributions of all forms! Please check the following guidelines:

### 🐛 **Bug Reports**
- Use detailed titles to describe issues
- Provide reproduction steps and environment information
- Include error logs and expected behavior

### ✨ **Feature Requests** 
- Clearly describe the necessity of new features
- Provide specific usage scenario examples
- Consider backward compatibility

### 🔧 **Code Contributions**
- Fork the project and create feature branches
- Follow code style and commenting conventions
- Add corresponding test cases
- Submit detailed Pull Requests

## :scroll: License

This project is open source under the **MIT License**. For detailed information, please see the [LICENSE](LICENSE) file.

---

<div align="center">

### If this project helps you, please give us a ⭐

**Development Team** | **Technical Support** | **Academic Collaboration**
:---: | :---: | :---:
[GitHub Issues](https://github.com/yourusername/clustered-pod-rbf/issues) | [Documentation](https://github.com/yourusername/clustered-pod-rbf/wiki) | 📧 Contact Email

</div>

---

*© 2025 Clustered POD-RBF Framework. All rights reserved.* 

import numpy as np
from sklearn.cluster import KMeans  # 保留KMeans，以防用户不使用固定范围模式
import warnings
import os  # 为 buildSnapshotMatrix 和脚本逻辑添加导入
from tqdm import tqdm  # 为 buildSnapshotMatrix 添加导入
import matplotlib.pyplot as plt  # 从用户脚本导入
import pandas as pd  # 从用户脚本导入
from scipy.interpolate import griddata  # 从用户脚本导入
import time  # 从用户脚本导入
import datetime  # 从用户脚本导入


# ---------------------------------------------------------------------------
# 核心函数和类定义 (来自 Canvas，并包含 buildSnapshotMatrix)
# ---------------------------------------------------------------------------

def buildSnapshotMatrix(mypath_pattern, skiprows=1, usecols=(0,)):
    """
    从指定目录中的CSV文件组装快照矩阵。
    每个CSV文件代表一个快照，其列包含数据点。
    mypath_pattern 用于推断包含CSV文件的目录。

    参数
    ----------
    mypath_pattern : str
        用于推断CSV文件目录的路径或模式。例如 "data/train/filepattern-%d.csv"。
        实际只使用此路径的目录部分。
    skiprows : int, optional
        读取CSV时要跳过的标题行数，默认为1。
    usecols : tuple, optional
        一个元组，指定要从CSV文件中读取的列的索引，默认为 (0,)。
        如果选择多列，它们将被展平并按顺序堆叠（列优先），假设它们代表单个快照的不同部分。
        快照矩阵的每一列将包含来自一个CSV文件的所有选定列的数据。

    返回
    -------
    numpy.ndarray
        快照矩阵，其中每列是一个快照。形状为 (num_total_sample_points_per_file, num_files)。
    """
    dirpath_for_csv_files = os.path.dirname(mypath_pattern)
    if not dirpath_for_csv_files:
        dirpath_for_csv_files = "."

    if not os.path.isdir(dirpath_for_csv_files):
        raise FileNotFoundError(f"从模式 '{mypath_pattern}' 推断出的目录不存在: '{dirpath_for_csv_files}'")

    all_files_in_dir = sorted([
        f for f in os.listdir(dirpath_for_csv_files)
        if os.path.isfile(os.path.join(dirpath_for_csv_files, f))
    ])

    csv_files = [f for f in all_files_in_dir if f.lower().endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"在目录 '{dirpath_for_csv_files}' 中没有找到 .csv 文件。")

    print(f"在目录 '{dirpath_for_csv_files}' 中找到 {len(csv_files)} 个CSV文件。")

    snapshot_columns_list = []
    num_sample_points_per_snapshot = None

    for filename in tqdm(csv_files, desc="Loading snapshot CSV files"):
        filepath = os.path.join(dirpath_for_csv_files, filename)
        try:
            data_from_file = np.loadtxt(
                filepath,
                delimiter=",",
                skiprows=skiprows,
                usecols=usecols,
                unpack=False
            )
            current_snapshot_vector = data_from_file.flatten(order='F')

            if num_sample_points_per_snapshot is None:
                num_sample_points_per_snapshot = len(current_snapshot_vector)
            elif len(current_snapshot_vector) != num_sample_points_per_snapshot:
                warnings.warn(f"Number of data points in file '{filename}' ({len(current_snapshot_vector)}) "
                              f"does not match the first file ({num_sample_points_per_snapshot}). Skipping this file.")
                continue
            snapshot_columns_list.append(current_snapshot_vector)
        except Exception as e:
            warnings.warn(f"Error loading or processing file '{filepath}': {e}. Skipping this file.")
            continue

    if not snapshot_columns_list:
        raise ValueError("Failed to load data from any CSV file to build the snapshot matrix.")

    snapshot_matrix = np.array(snapshot_columns_list).T
    print(f"快照矩阵构建完成。形状: {snapshot_matrix.shape}")
    return snapshot_matrix


class pod_rbf:
    def __init__(self, energy_threshold=0.99):
        self.energy_threshold = energy_threshold
        self.snapshot = None
        self.train_params = None
        self.params_range = None
        self.shape_factor = None
        self.basis = None
        self.weights = None
        self.cumul_energy = None
        self.truncated_energy = 0.0
        self.eig_vals = None
        self.eig_vecs = None
        self.mem_limit = 16  # gigabytes

    def _calcTruncatedPODBasis(self):
        if self.snapshot is None:
            raise ValueError("Snapshot matrix is not set. Call train() first.")
        memory = self.snapshot.nbytes / 1e9
        if memory < self.mem_limit:
            U, S_singular, _ = np.linalg.svd(self.snapshot, full_matrices=False)
            if S_singular.size == 0:
                warnings.warn("SVD resulted in zero singular values or empty snapshot. Basis will be empty.")
                self.basis = np.array([]).reshape(self.snapshot.shape[0], 0)
                self.truncated_energy = 0.0
                self.cumul_energy = np.array([0.0])
                return self.basis
            energy_values = S_singular ** 2
            if np.sum(energy_values) < 1e-12:  # 检查能量和是否接近于零
                warnings.warn("Sum of squared singular values is near zero. Basis may be ill-defined or empty.")
                self.basis = np.array([]).reshape(self.snapshot.shape[0], 0)
                self.truncated_energy = 0.0
                self.cumul_energy = np.array([0.0])
                return self.basis

            self.cumul_energy = np.cumsum(energy_values) / np.sum(energy_values)

            if self.energy_threshold >= 1.0:
                trunc_id = len(S_singular) - 1
            elif self.energy_threshold <= 0 or (
                    self.cumul_energy.size > 0 and self.energy_threshold < self.cumul_energy[0]):
                trunc_id = 0
            else:
                trunc_id_candidates = np.where(self.cumul_energy >= self.energy_threshold)[0]
                if trunc_id_candidates.size > 0:
                    trunc_id = trunc_id_candidates[0]
                else:
                    trunc_id = len(S_singular) - 1

            if trunc_id < 0: trunc_id = 0
            if self.cumul_energy.size > 0:
                if trunc_id >= len(self.cumul_energy): trunc_id = len(self.cumul_energy) - 1
                self.truncated_energy = self.cumul_energy[trunc_id] if trunc_id >= 0 else 0.0  # 确保trunc_id有效
            else:
                self.truncated_energy = 0.0

            self.basis = U[:, :(trunc_id + 1)]
        else:
            print("Snapshot matrix is large, using eigenvalue decomposition method for POD basis.")
            L = np.dot(self.snapshot.T, self.snapshot)
            eig_vals_L, eig_vecs_L = np.linalg.eigh(L)
            sorted_indices = np.argsort(eig_vals_L)[::-1]
            self.eig_vals = eig_vals_L[sorted_indices]
            eig_vecs_L_sorted = eig_vecs_L[:, sorted_indices]
            positive_eig_indices = self.eig_vals > 1e-10
            self.eig_vals = self.eig_vals[positive_eig_indices]
            eig_vecs_L_sorted = eig_vecs_L_sorted[:, positive_eig_indices]

            if self.eig_vals.size == 0 or np.sum(self.eig_vals) < 1e-12:
                warnings.warn(
                    "Eigenvalue decomposition resulted in no positive eigenvalues or sum is zero. Basis will be empty.")
                self.basis = np.array([]).reshape(self.snapshot.shape[0], 0)
                self.truncated_energy = 0.0
                self.cumul_energy = np.array([0.0])
                return self.basis

            self.cumul_energy = np.cumsum(self.eig_vals) / np.sum(self.eig_vals)
            if self.energy_threshold >= 1.0:
                trunc_id = len(self.eig_vals) - 1
            elif self.energy_threshold <= 0 or (
                    self.cumul_energy.size > 0 and self.energy_threshold < self.cumul_energy[0]):
                trunc_id = 0
            else:
                trunc_id_candidates = np.where(self.cumul_energy >= self.energy_threshold)[0]
                if trunc_id_candidates.size > 0:
                    trunc_id = trunc_id_candidates[0]
                else:
                    trunc_id = len(self.eig_vals) - 1

            if trunc_id < 0: trunc_id = 0
            if self.cumul_energy.size > 0:
                if trunc_id >= len(self.cumul_energy): trunc_id = len(self.cumul_energy) - 1
                self.truncated_energy = self.cumul_energy[trunc_id] if trunc_id >= 0 else 0.0
            else:
                self.truncated_energy = 0.0

            eig_vals_truncated = self.eig_vals[:(trunc_id + 1)]
            eig_vecs_L_truncated = eig_vecs_L_sorted[:, :(trunc_id + 1)]
            sqrt_eig_vals = np.sqrt(eig_vals_truncated)
            sqrt_eig_vals[np.isclose(sqrt_eig_vals, 0)] = 1e-9
            self.basis = np.dot(self.snapshot, eig_vecs_L_truncated) / sqrt_eig_vals
        return self.basis

    def _buildCollocationMatrix(self, c_param, params_to_build_for):
        num_train_points = self.train_params.shape[1]
        num_build_points = params_to_build_for.shape[1]
        num_param_dims = self.train_params.shape[0]
        r2 = np.zeros((num_build_points, num_train_points))
        for i in range(num_param_dims):
            I_mesh, J_mesh = np.meshgrid(
                params_to_build_for[i, :], self.train_params[i, :],
                indexing='ij', copy=False
            )
            range_val = self.params_range[i]
            if np.isclose(range_val, 0):
                term = np.zeros_like(I_mesh)
            else:
                term = ((I_mesh - J_mesh) / range_val) ** 2
            r2 += term
        return 1.0 / np.sqrt(r2 / (c_param ** 2) + 1.0 + 1e-9)

    def _findOptimShapeParam(self, cond_target_range=(1e3, 1e8), max_steps=100):
        c_low = 0.01;
        c_high = 10.0
        if self.train_params is None:
            raise ValueError("_findOptimShapeParam called before train_params is set.")
        try:
            colloc_matrix_low = self._buildCollocationMatrix(c_low, self.train_params)
            colloc_matrix_high = self._buildCollocationMatrix(c_high, self.train_params)
            cond_at_c_low = np.linalg.cond(colloc_matrix_low)
            cond_at_c_high = np.linalg.cond(colloc_matrix_high)
        except np.linalg.LinAlgError:
            warnings.warn("LinAlgError in _findOptimShapeParam. Using default c=1.0.")
            return 1.0
        if cond_at_c_low < cond_at_c_high:  # Should be cond_at_c_low > cond_at_c_high for IMQ
            warnings.warn(
                f"Cond(c_low) {cond_at_c_low:.2e} < Cond(c_high) {cond_at_c_high:.2e}. Shape search might be unreliable. Default c=1.0.")
            return 1.0
        optim_c = (c_low + c_high) / 2.0
        for k in range(max_steps):
            optim_c = (c_low + c_high) / 2.0
            if np.isclose(c_low, c_high, atol=1e-4): break
            try:
                current_cond = np.linalg.cond(self._buildCollocationMatrix(optim_c, self.train_params))
            except np.linalg.LinAlgError:
                current_cond = np.inf
            if current_cond < cond_target_range[0]:
                c_high = optim_c  # Cond too low, c is too high, so new c_high is optim_c
            elif current_cond > cond_target_range[1]:
                c_low = optim_c  # Cond too high, c is too low, so new c_low is optim_c
            else:
                return optim_c
        final_cond_check = np.linalg.cond(self._buildCollocationMatrix(optim_c, self.train_params))
        warnings.warn(f"Optimal shape search ended. c={optim_c:.3e}, cond={final_cond_check:.2e}")
        return optim_c

    def train(self, snapshot, train_params, shape_factor=None):
        if train_params.ndim == 1:
            self.train_params = np.expand_dims(train_params, axis=0)
        else:
            self.train_params = train_params
        self.snapshot = snapshot
        if self.snapshot.shape[1] != self.train_params.shape[1]:
            raise ValueError("Snapshot and train_params mismatch.")
        self.params_range = np.ptp(self.train_params, axis=1)
        self.params_range[np.isclose(self.params_range, 0)] = 1.0
        if shape_factor is None:
            print("Finding optimal shape parameter c for RBF...")
            self.shape_factor = self._findOptimShapeParam()
            print(f"Found optimal shape_factor: {self.shape_factor:.6f}")
        else:
            self.shape_factor = shape_factor
        print("Calculating POD basis...")
        self.basis = self._calcTruncatedPODBasis()
        if self.basis is None or self.basis.size == 0:
            warnings.warn("POD basis empty. RBF model might fail or predict zeros.")
            self.weights = np.array([])
            return
        print(f"POD basis: {self.basis.shape}, Energy: {self.truncated_energy:.6f}")
        F = self._buildCollocationMatrix(self.shape_factor, self.train_params)
        A = np.dot(self.basis.T, self.snapshot)
        try:
            self.weights = np.linalg.solve(F, A.T).T
        except np.linalg.LinAlgError:
            warnings.warn("Singular F. Using pseudo-inverse for RBF weights.")
            self.weights = np.dot(A, np.linalg.pinv(F))

    def inference(self, inf_params):
        if self.basis is None or self.weights is None: raise RuntimeError("Model not trained.")
        if self.basis.size == 0 or self.weights.size == 0:
            num_feat = self.snapshot.shape[0] if self.snapshot is not None else 0
            if np.isscalar(inf_params) or inf_params.ndim == 1:
                return np.zeros(num_feat)
            else:
                num_inf = 1 if inf_params.ndim == 1 else inf_params.shape[1]; return np.zeros((num_feat, num_inf))
        curr_inf = inf_params;
        orig_1d = False
        if np.isscalar(curr_inf):
            curr_inf = np.array([[curr_inf]]); orig_1d = True
        elif curr_inf.ndim == 1:
            curr_inf = np.expand_dims(curr_inf, axis=1); orig_1d = True
        if curr_inf.shape[0] != self.train_params.shape[0]:
            raise ValueError(f"Inference params dim mismatch.")
        F_inf = self._buildCollocationMatrix(self.shape_factor, curr_inf)
        A_inf = np.dot(self.weights, F_inf.T)
        res = np.dot(self.basis, A_inf)
        return res[:, 0] if orig_1d else res


class clustered_pod_rbf:
    def __init__(self,
                 n_clusters_kmeans=3, energy_threshold=0.99, use_smooth_transition=True,
                 fixed_segment_param_idx=None, fixed_num_segments=5,
                 fixed_segment_mode='equal_width', fixed_segment_proportions=None):  # 新增 fixed_segment_proportions
        self.energy_threshold = energy_threshold
        self.use_smooth_transition = use_smooth_transition
        self.fixed_segment_param_idx = fixed_segment_param_idx
        self.fixed_num_segments = fixed_num_segments
        if fixed_segment_param_idx is not None and fixed_num_segments <= 0:
            raise ValueError("fixed_num_segments must be positive.")
        self.fixed_segment_mode = fixed_segment_mode
        self.fixed_segment_proportions = fixed_segment_proportions  # 存储比例
        if self.fixed_segment_mode == 'increasing_density' and self.fixed_segment_proportions is not None:
            if len(self.fixed_segment_proportions) != self.fixed_num_segments:
                raise ValueError("Length of fixed_segment_proportions must match fixed_num_segments.")
            if not all(p > 0 for p in self.fixed_segment_proportions):
                raise ValueError("All fixed_segment_proportions must be positive.")

        self.n_clusters_kmeans = n_clusters_kmeans
        if self.fixed_segment_param_idx is not None:
            self.n_clusters_effective = self.fixed_num_segments
        else:
            self.n_clusters_effective = self.n_clusters_kmeans
        self.local_models = []
        self.cluster_centers_or_representatives = None
        self.cluster_labels = None
        self.segment_boundaries = None
        self.cluster_info = {}

    def _define_fixed_segments(self, params_for_segmentation):
        min_val = np.min(params_for_segmentation);
        max_val = np.max(params_for_segmentation)
        if np.isclose(min_val, max_val):
            warnings.warn(f"Param idx {self.fixed_segment_param_idx} all same. One segment.");
            self.segment_boundaries = np.array([min_val, max_val]);  # 边界顺序改为递增
            self.n_clusters_effective = 1;
            return

        if self.fixed_segment_mode == 'equal_width':
            # 边界按递增顺序定义
            self.segment_boundaries = np.linspace(min_val, max_val, self.fixed_num_segments + 1)
        elif self.fixed_segment_mode == 'quantile':
            percentiles = np.linspace(0, 100, self.fixed_num_segments + 1)  # 0到100用于递增边界
            self.segment_boundaries = np.percentile(params_for_segmentation, percentiles)
            self.segment_boundaries = np.unique(self.segment_boundaries)  # unique会排序
            if len(self.segment_boundaries) < 2:
                warnings.warn(
                    f"Quantile for param {self.fixed_segment_param_idx} < 2 unique bounds. Fallback to equal_width.");
                self.fixed_segment_mode = 'equal_width';
                self._define_fixed_segments(params_for_segmentation);
                return
            self.n_clusters_effective = len(self.segment_boundaries) - 1
            if self.n_clusters_effective < 1:
                warnings.warn(f"Cannot create valid segment. All data one segment.");
                self.n_clusters_effective = 1;
                self.segment_boundaries = np.array([min_val, max_val])

        elif self.fixed_segment_mode == 'increasing_density':
            if self.fixed_num_segments <= 0:
                warnings.warn("fixed_num_segments must be > 0 for increasing_density mode. Defaulting to 1 segment.")
                self.n_clusters_effective = 1
                self.segment_boundaries = np.array([min_val, max_val])
                return

            sorted_params = np.sort(np.unique(params_for_segmentation))  # 使用唯一值排序
            n_unique_points = len(sorted_params)

            if self.fixed_segment_proportions is not None:
                proportions = np.array(self.fixed_segment_proportions, dtype=float)
            elif self.fixed_num_segments == 5:  # 用户特定要求：5个段，递增密度
                proportions = np.array([1, 1.5, 2, 2.5, 3], dtype=float)  # 示例比例
                # 或者更明显的递增： [1, 2, 3, 4, 5]
                # proportions = np.array([1, 2, 3, 4, 5], dtype=float)
            else:  # 其他段数的默认递增比例
                proportions = np.arange(1, self.fixed_num_segments + 1, dtype=float)

            proportions /= np.sum(proportions)  # 归一化

            self.segment_boundaries = np.zeros(self.fixed_num_segments + 1)
            self.segment_boundaries[0] = sorted_params[0]
            self.segment_boundaries[-1] = sorted_params[-1]

            current_cumulative_count = 0
            # 计算每个段的目标点数
            target_counts_per_segment = np.round(proportions * n_unique_points).astype(int)
            # 调整最后一个，确保总和为 n_unique_points
            if np.sum(target_counts_per_segment) != n_unique_points:
                target_counts_per_segment[-1] = n_unique_points - np.sum(target_counts_per_segment[:-1])

            # 确保没有段的点数为0 (除非总点数非常少)
            if any(c <= 0 for c in target_counts_per_segment) and n_unique_points > self.fixed_num_segments:
                warnings.warn(
                    "Some segments have zero or negative target counts due to rounding and proportions. Adjusting counts.")
                # 简单调整：至少为1，然后重新分配
                target_counts_per_segment = np.maximum(1, target_counts_per_segment)
                excess = np.sum(target_counts_per_segment) - n_unique_points
                while excess > 0:  # 从最大的段开始减去多余的
                    target_counts_per_segment[np.argmax(target_counts_per_segment)] -= 1
                    excess -= 1
                while excess < 0:  # 如果不够，从最小的段开始增加 (不太可能发生)
                    target_counts_per_segment[np.argmin(target_counts_per_segment)] += 1
                    excess += 1

            cumulative_point_indices = np.cumsum(target_counts_per_segment) - 1  # 每个段结束点的索引

            for i in range(self.fixed_num_segments - 1):
                idx = cumulative_point_indices[i]
                if idx + 1 < n_unique_points:
                    # 边界是当前段的最后一个点和下一个段的第一个点之间的中点
                    self.segment_boundaries[i + 1] = (sorted_params[idx] + sorted_params[idx + 1]) / 2.0
                else:  # 如果是最后一个边界之前，但点不够了
                    self.segment_boundaries[i + 1] = sorted_params[idx]  # 就用当前段的最后一个点作为边界

            # 确保边界是单调递增的，并且最终边界是最大值
            self.segment_boundaries = np.unique(np.sort(self.segment_boundaries))  # 清理并排序
            if len(self.segment_boundaries) < self.fixed_num_segments + 1:
                warnings.warn(
                    "Could not define enough unique boundaries for 'increasing_density'. Resulting segments might be fewer.")
                self.n_clusters_effective = max(1, len(self.segment_boundaries) - 1)
            # 确保第一个和最后一个边界是数据的实际最小/最大值
            self.segment_boundaries[0] = min_val
            self.segment_boundaries[-1] = max_val


        else:
            raise ValueError(f"Unknown fixed_segment_mode: {self.fixed_segment_mode}")

        actual_possible_segments = max(0,
                                       len(self.segment_boundaries) - 1) if self.segment_boundaries is not None else 0
        if self.n_clusters_effective != actual_possible_segments and self.n_clusters_effective > 0:
            if actual_possible_segments > 0:
                warnings.warn(
                    f"Effective clusters ({self.n_clusters_effective}) != actual ({actual_possible_segments}). Adjusting to {actual_possible_segments}.")
                self.n_clusters_effective = actual_possible_segments
            else:  # 如果边界定义导致没有有效段
                warnings.warn(
                    f"Boundary definition resulted in 0 segments. Defaulting to 1 segment for the whole range.")
                self.n_clusters_effective = 1
                self.segment_boundaries = np.array([min_val, max_val])

    def _assign_labels_fixed_segments(self, params_for_segmentation):
        labels = np.full(len(params_for_segmentation), -1, dtype=int)
        if self.n_clusters_effective == 0 or self.segment_boundaries is None or len(self.segment_boundaries) < 2:
            warnings.warn("Cannot assign labels: no clusters or boundaries undefined.");
            return labels

        # 使用 np.digitize，边界是递增的
        # bins 需要是内部边界点
        internal_boundaries = self.segment_boundaries[1:-1]
        if len(internal_boundaries) == 0 and self.n_clusters_effective == 1:  # 只有一个段
            labels[:] = 0
            return labels

        # right=False: bins[i-1] <= x < bins[i] -> 返回 i
        # right=True:  bins[i-1] < x <= bins[i] -> 返回 i
        # 我们希望段0是 [b0, b1], 段1是 (b1, b2], ...
        # np.digitize(x, bins) 返回x应该插入的索引以保持顺序。
        # 如果 x < bins[0], 返回0.
        # 如果 bins[i-1] <= x < bins[i], 返回i.
        # 如果 x >= bins[-1], 返回 len(bins).
        # 所以，如果bins是[b1,b2,b3,b4], 结果是0,1,2,3,4. 这正是我们想要的标签。

        labels = np.digitize(params_for_segmentation, self.segment_boundaries[1:-1], right=False)

        # 确保落在精确边界上的点被正确分配
        # 例如，如果一个点正好等于 segment_boundaries[k]，digitize(right=False) 会如何处理？
        # np.digitize(x, [b1, b2], right=False):
        # x < b1 -> 0
        # b1 <= x < b2 -> 1
        # b2 <= x -> 2
        # 这与我们的段定义 (b_k, b_{k+1}] 或 [b_k, b_{k+1}] for last 匹配。

        # 特殊处理正好等于最小边界的点，确保它在段0
        if self.segment_boundaries.size > 0:
            labels[np.isclose(params_for_segmentation, self.segment_boundaries[0])] = 0

        unassigned_indices = np.where(labels >= self.n_clusters_effective)[0]  # digitize 可能返回 len(bins)
        if unassigned_indices.size > 0:
            labels[unassigned_indices] = self.n_clusters_effective - 1  # 分配到最后一个有效簇

        if np.any(labels < 0):  # 不应发生，但作为检查
            warnings.warn("Negative labels assigned by digitize. This should not happen.")

        return labels

    def train(self, snapshot, train_params, shape_factor=None):
        original_train_params_ndim = train_params.ndim
        if train_params.ndim == 1:
            params_for_kmeans = np.expand_dims(train_params, axis=1)
            train_params_internal = np.expand_dims(train_params, axis=0)
            if self.fixed_segment_param_idx is not None and self.fixed_segment_param_idx != 0:
                raise ValueError("If train_params is 1D, fixed_segment_param_idx must be 0.")
            selected_param_for_segmentation = train_params
        else:
            params_for_kmeans = train_params.T
            train_params_internal = train_params
            if self.fixed_segment_param_idx is not None:
                if not (0 <= self.fixed_segment_param_idx < train_params_internal.shape[0]):
                    raise ValueError(
                        f"fixed_segment_param_idx {self.fixed_segment_param_idx} out of bounds for train_params dim {train_params_internal.shape[0]}.")
                selected_param_for_segmentation = train_params_internal[self.fixed_segment_param_idx, :]

        if self.fixed_segment_param_idx is not None:
            print(
                f"Using fixed segmentation mode. Parameter index: {self.fixed_segment_param_idx}, Expected segments: {self.fixed_num_segments}, Mode: {self.fixed_segment_mode}.")
            self._define_fixed_segments(selected_param_for_segmentation)
            print(f"Effective number of segments: {self.n_clusters_effective}")
            if self.segment_boundaries is not None:
                print(
                    f"Segment boundaries (ascending): {np.array2string(self.segment_boundaries, formatter={'float_kind': lambda x: '%.3f' % x})}")
            self.cluster_labels = self._assign_labels_fixed_segments(selected_param_for_segmentation)

            if self.n_clusters_effective > 0:
                self.cluster_centers_or_representatives = np.zeros(
                    (self.n_clusters_effective, train_params_internal.shape[0]))
                for i in range(self.n_clusters_effective):
                    indices_in_segment = np.where(self.cluster_labels == i)[0]
                    if len(indices_in_segment) > 0:
                        if len(self.segment_boundaries) > i + 1:
                            lower_b = self.segment_boundaries[i];
                            upper_b = self.segment_boundaries[i + 1]  # 边界现在是递增的
                            self.cluster_centers_or_representatives[i, self.fixed_segment_param_idx] = (
                                                                                                                   upper_b + lower_b) / 2.0
                        else:
                            self.cluster_centers_or_representatives[i, self.fixed_segment_param_idx] = np.mean(
                                selected_param_for_segmentation[indices_in_segment])
                        for dim_idx in range(train_params_internal.shape[0]):
                            if dim_idx != self.fixed_segment_param_idx:
                                self.cluster_centers_or_representatives[i, dim_idx] = np.mean(
                                    train_params_internal[dim_idx, indices_in_segment])
                    else:
                        if self.segment_boundaries is not None and len(self.segment_boundaries) > i + 1:
                            lower_b = self.segment_boundaries[i];
                            upper_b = self.segment_boundaries[i + 1]
                            self.cluster_centers_or_representatives[i, self.fixed_segment_param_idx] = (
                                                                                                                   upper_b + lower_b) / 2.0
                        for dim_idx in range(train_params_internal.shape[0]):
                            if dim_idx != self.fixed_segment_param_idx: self.cluster_centers_or_representatives[
                                i, dim_idx] = np.nan
            else:
                self.cluster_centers_or_representatives = np.array([])
        else:
            current_n_clusters_for_kmeans = self.n_clusters_kmeans
            if params_for_kmeans.shape[0] < current_n_clusters_for_kmeans:
                warnings.warn(
                    f"Number of training points less than n_clusters_kmeans. Adjusting clusters to number of points.")
                current_n_clusters_for_kmeans = params_for_kmeans.shape[0]
            if current_n_clusters_for_kmeans == 0: raise ValueError("K-means n_clusters is 0.")
            self.n_clusters_effective = current_n_clusters_for_kmeans
            print(f"Performing K-means clustering. Number of clusters: {self.n_clusters_effective}...")
            kmeans = KMeans(n_clusters=self.n_clusters_effective, random_state=42, n_init='auto')
            self.cluster_labels = kmeans.fit_predict(params_for_kmeans)
            self.cluster_centers_or_representatives = kmeans.cluster_centers_

        self.cluster_info = {
            'n_clusters_effective': self.n_clusters_effective,
            'cluster_centers_or_representatives': self.cluster_centers_or_representatives.copy() if self.cluster_centers_or_representatives is not None else None,
            'segment_boundaries': self.segment_boundaries.copy() if self.segment_boundaries is not None else None,
            'fixed_segment_param_idx': self.fixed_segment_param_idx, 'fixed_segment_mode': self.fixed_segment_mode,
            'fixed_segment_proportions': self.fixed_segment_proportions,  # 保存比例
            'local_pod_dims': [], 'local_shape_params': [], 'local_energy_captured': [],
            'cluster_sizes': [], 'param_ranges_local': []
        }
        self.local_models = []
        if self.n_clusters_effective > 0:
            print(f"\nTraining {self.n_clusters_effective} local models...")
            for i in range(self.n_clusters_effective):
                indices = np.where(self.cluster_labels == i)[0]
                if not indices.size:
                    self.cluster_info['local_pod_dims'].append(0);
                    self.cluster_info['local_shape_params'].append(None);
                    self.cluster_info['local_energy_captured'].append(None);
                    self.cluster_info['cluster_sizes'].append(0);
                    self.cluster_info['param_ranges_local'].append(None);
                    self.local_models.append(None);
                    continue

                snap = snapshot[:, indices];
                params_loc = train_params_internal[:, indices]
                local_model_inst = pod_rbf(energy_threshold=self.energy_threshold)
                try:
                    sf_local = None
                    if isinstance(shape_factor, list) and i < len(shape_factor):
                        sf_local = shape_factor[i]
                    elif not isinstance(shape_factor, list):
                        sf_local = shape_factor

                    local_model_inst.train(snap, params_loc, sf_local)
                    self.local_models.append(local_model_inst)
                    self.cluster_info['local_pod_dims'].append(local_model_inst.basis.shape[
                                                                   1] if local_model_inst.basis is not None and local_model_inst.basis.size > 0 else 0)
                    self.cluster_info['local_shape_params'].append(local_model_inst.shape_factor)
                    self.cluster_info['local_energy_captured'].append(local_model_inst.truncated_energy)
                    print(
                        f"  Cluster/Segment {i + 1}: Trained. POD Dim: {self.cluster_info['local_pod_dims'][-1]}, Shape: {self.cluster_info['local_shape_params'][-1]:.4f}, Energy: {self.cluster_info['local_energy_captured'][-1]:.4f}")

                except Exception as e:
                    print(f"  Cluster/Segment {i + 1} training failed: {e}");
                    self.local_models.append(None)
                    self.cluster_info['local_pod_dims'].append(0);
                    self.cluster_info['local_shape_params'].append(None);
                    self.cluster_info['local_energy_captured'].append(None);
                self.cluster_info['cluster_sizes'].append(len(indices))
        self.print_cluster_summary()

    def _get_cluster_for_inf_param_fixed(self, inf_param_val_for_segment):
        if self.segment_boundaries is None or self.n_clusters_effective == 0:
            warnings.warn("Fixed segment boundaries undefined or no clusters. Defaulting to index 0.");
            return 0

        # 使用 np.digitize 处理递增的边界
        # self.segment_boundaries 是 [b0, b1, ..., bN] (N+1 个点)
        # 我们需要内部的 N-1 个边界点作为 bins
        internal_bins = self.segment_boundaries[1:-1]

        if len(internal_bins) == 0:  # 只有一个段
            return 0

            # right=False: bins[i-1] <= x < bins[i] -> 返回 i
        # x < bins[0] -> 0
        # x >= bins[-1] -> len(bins)
        # 标签范围是 0 to len(bins) (即 0 to n_clusters_effective-1)
        label = np.digitize(inf_param_val_for_segment, internal_bins, right=False)

        # 确保标签在有效范围内 [0, n_clusters_effective - 1]
        label = np.clip(label, 0, self.n_clusters_effective - 1)
        return label

    def inference(self, inf_params):
        if not self.local_models or all(m is None for m in self.local_models):
            raise RuntimeError("Model not trained or all local models failed.")

        original_inf_ndim = inf_params.ndim;
        is_scalar_input = np.isscalar(inf_params)
        current_inf_params_proc = inf_params
        num_train_dims_check = -1
        if self.cluster_centers_or_representatives is not None and self.cluster_centers_or_representatives.ndim > 1:
            num_train_dims_check = self.cluster_centers_or_representatives.shape[1]
        elif self.fixed_segment_param_idx is not None and self.cluster_centers_or_representatives is not None and self.cluster_centers_or_representatives.ndim > 0:
            num_train_dims_check = self.cluster_centers_or_representatives.shape[1]
        else:
            for model_inst in self.local_models:
                if model_inst and model_inst.train_params is not None:
                    num_train_dims_check = model_inst.train_params.shape[0];
                    break

        if is_scalar_input:
            if num_train_dims_check != -1 and num_train_dims_check != 1:
                raise ValueError(
                    f"Scalar inference param for multi-dim model (trained with {num_train_dims_check} dims).")
            current_inf_params_proc = np.array([[inf_params]])
        elif current_inf_params_proc.ndim == 1:
            if num_train_dims_check != -1 and len(current_inf_params_proc) != num_train_dims_check:
                raise ValueError(
                    f"1D inference param dim ({len(current_inf_params_proc)}) mismatch train dim ({num_train_dims_check}).")
            current_inf_params_proc = np.expand_dims(current_inf_params_proc, axis=1)

        num_inf_points = current_inf_params_proc.shape[1]
        all_predictions_list = []

        for i_pt in range(num_inf_points):
            single_inf_param_vector = current_inf_params_proc[:, i_pt]
            target_indices_for_smoothing = [];
            weights_for_smoothing = []

            if self.fixed_segment_param_idx is not None:
                inf_val_for_segment_dim = single_inf_param_vector[self.fixed_segment_param_idx]
                if not self.use_smooth_transition or self.n_clusters_effective == 1:
                    target_cluster_idx = self._get_cluster_for_inf_param_fixed(inf_val_for_segment_dim)
                    target_indices_for_smoothing.append(target_cluster_idx);
                    weights_for_smoothing.append(1.0)
                else:
                    # 平滑过渡：基于到代表点的距离（对于分段参数维度）
                    param_centers_1d = self.cluster_centers_or_representatives[:, self.fixed_segment_param_idx]
                    distances_1d = np.abs(param_centers_1d - inf_val_for_segment_dim)
                    distances_1d[np.isclose(distances_1d, 0)] = 1e-9
                    sorted_dist_indices = np.argsort(distances_1d)
                    valid_model_count = 0
                    for k_idx in sorted_dist_indices:
                        if k_idx < len(self.local_models) and self.local_models[k_idx] is not None:
                            target_indices_for_smoothing.append(k_idx);
                            weights_for_smoothing.append(1.0 / distances_1d[k_idx])
                            valid_model_count += 1;
                            if len(target_indices_for_smoothing) >= 2: break
                    if not target_indices_for_smoothing:
                        primary_idx = self._get_cluster_for_inf_param_fixed(inf_val_for_segment_dim)
                        if primary_idx < len(self.local_models) and self.local_models[primary_idx] is not None:
                            target_indices_for_smoothing.append(primary_idx);
                            weights_for_smoothing.append(1.0)
                        else:
                            raise RuntimeError(
                                f"Fixed seg mode: No valid local models for point {single_inf_param_vector}.")
            else:
                distances_kmeans = np.linalg.norm(self.cluster_centers_or_representatives - single_inf_param_vector,
                                                  axis=1)
                distances_kmeans[np.isclose(distances_kmeans, 0)] = 1e-9
                if not self.use_smooth_transition or self.n_clusters_effective == 1:
                    nearest_cluster_idx = np.argmin(distances_kmeans)
                    target_indices_for_smoothing.append(nearest_cluster_idx);
                    weights_for_smoothing.append(1.0)
                else:
                    sorted_dist_indices = np.argsort(distances_kmeans)
                    valid_model_count = 0
                    for k_idx in sorted_dist_indices:
                        if k_idx < len(self.local_models) and self.local_models[k_idx] is not None:
                            target_indices_for_smoothing.append(k_idx);
                            weights_for_smoothing.append(1.0 / distances_kmeans[k_idx])
                            valid_model_count += 1;
                            if len(target_indices_for_smoothing) >= 2: break
                    if not target_indices_for_smoothing:
                        nearest_idx = np.argmin(distances_kmeans)
                        if nearest_idx < len(self.local_models) and self.local_models[nearest_idx] is not None:
                            target_indices_for_smoothing.append(nearest_idx);
                            weights_for_smoothing.append(1.0)
                        else:
                            raise RuntimeError(
                                f"K-means mode: No valid local models for point {single_inf_param_vector}.")

            current_point_prediction = 0;
            sum_of_weights = np.sum(weights_for_smoothing)
            if np.isclose(sum_of_weights, 0) or not target_indices_for_smoothing:
                if target_indices_for_smoothing and target_indices_for_smoothing[0] < len(self.local_models) and \
                        self.local_models[target_indices_for_smoothing[0]] is not None:
                    chosen_idx = target_indices_for_smoothing[0]
                    current_point_prediction = self.local_models[chosen_idx].inference(single_inf_param_vector)
                    if np.isclose(sum_of_weights, 0): warnings.warn(
                        "Smooth transition weights sum to zero. Using nearest single model.")
                else:
                    raise RuntimeError(f"No target models for inference of {single_inf_param_vector}.")
            else:
                normalized_weights = np.array(weights_for_smoothing) / sum_of_weights
                for model_idx, weight in zip(target_indices_for_smoothing, normalized_weights):
                    if model_idx >= len(self.local_models) or self.local_models[model_idx] is None:
                        warnings.warn(
                            f"Model index {model_idx} for smoothing is invalid/unavailable. Its weight ignored.")
                        continue
                    pred_from_local_model = self.local_models[model_idx].inference(single_inf_param_vector)
                    current_point_prediction += weight * pred_from_local_model
            all_predictions_list.append(current_point_prediction)

        if not all_predictions_list:
            num_features = 0
            for model_inst in self.local_models:
                if model_inst and hasattr(model_inst,
                                          'basis') and model_inst.basis is not None and model_inst.basis.size > 0:
                    num_features = model_inst.basis.shape[0];
                    break
            if num_features == 0:
                for model_inst in self.local_models:
                    if model_inst and hasattr(model_inst, 'snapshot') and model_inst.snapshot is not None:
                        num_features = model_inst.snapshot.shape[0];
                        break
            if is_scalar_input or original_inf_ndim == 1:
                return np.zeros(num_features)
            else:
                return np.zeros((num_features, num_inf_points))

        final_predictions_array = np.stack(all_predictions_list, axis=1)
        if is_scalar_input or original_inf_ndim == 1: return final_predictions_array[:, 0]
        return final_predictions_array

    def print_cluster_summary(self):
        if not self.cluster_info or self.n_clusters_effective == 0:
            print("No cluster/segment info available or effective clusters is 0. Train model first.")
            return
        print("\n" + "=" * 90);
        print("Clustered POD-RBF Model Summary");
        print("=" * 90)
        mode = "K-means Clustering" if self.fixed_segment_param_idx is None else \
            f"Fixed Segmentation (Param Index: {self.fixed_segment_param_idx}, Mode: {self.fixed_segment_mode})"
        print(f"Clustering/Segmentation Mode: {mode}")
        print(f"Effective Number of Clusters/Segments (K_eff): {self.cluster_info.get('n_clusters_effective', 'N/A')}")
        print(f"POD Energy Threshold: {self.energy_threshold}")
        print(f"Use Smooth Transition for Inference: {self.use_smooth_transition}")
        if self.fixed_segment_param_idx is not None and self.cluster_info.get('segment_boundaries') is not None:
            boundaries_str = np.array2string(self.cluster_info['segment_boundaries'],
                                             formatter={'float_kind': lambda x: '%.3f' % x})
            # 边界现在是递增的
            print(f"Segment Boundaries (ascending): {boundaries_str}")
        print("\nCluster/Segment Details:");
        header_format = "{:>12} {:>6} {:>8} {:>12} {:>10} {}";
        row_format = "{:>12} {:>6} {:>8} {:>12} {:>10} {}"
        print("-" * 90);
        print(header_format.format('Cluster/SegID', 'Size', 'POD Dim', 'Shape Param', 'Energy',
                                   'Local Param Range / Rep. Point'));
        print("-" * 90)
        num_items_to_print = len(self.cluster_info.get('cluster_sizes', []))
        for i in range(num_items_to_print):
            cluster_id_str = f"{i + 1}"  # 标签0对应段1，以此类推
            if self.local_models and i < len(self.local_models) and self.local_models[
                i] is None: cluster_id_str += " (Failed)"
            size = self.cluster_info['cluster_sizes'][i] if i < len(self.cluster_info['cluster_sizes']) else 'N/A'
            pod_dim = self.cluster_info['local_pod_dims'][i] if i < len(self.cluster_info['local_pod_dims']) else 'N/A'
            shape_param_val = self.cluster_info['local_shape_params'][i] if i < len(
                self.cluster_info['local_shape_params']) else None
            energy_val = self.cluster_info['local_energy_captured'][i] if i < len(
                self.cluster_info['local_energy_captured']) else None
            shape_param_str = f"{shape_param_val:.6f}" if shape_param_val is not None else "N/A"
            energy_str = f"{energy_val:.6f}" if energy_val is not None else "N/A"
            desc_str = ""
            if self.fixed_segment_param_idx is None:
                if self.cluster_info.get('cluster_centers_or_representatives') is not None and i < \
                        self.cluster_info['cluster_centers_or_representatives'].shape[0]:
                    center = self.cluster_info['cluster_centers_or_representatives'][i, :]
                    center_str = np.array2string(center, formatter={'float_kind': lambda x: '%.3f' % x}, separator=', ')
                    desc_str = f"Center: {center_str}"
                else:
                    desc_str = "Center: N/A"
            else:  # 固定分段，边界是递增的
                if self.cluster_info.get('segment_boundaries') is not None and len(
                        self.cluster_info['segment_boundaries']) > i + 1:
                    lower_b = self.cluster_info['segment_boundaries'][i];
                    upper_b = self.cluster_info['segment_boundaries'][i + 1]
                    desc_str = f"Seg Param P{self.fixed_segment_param_idx} Defined Range: [{lower_b:.3f}, {upper_b:.3f}]"  # 段的范围
                    if self.cluster_info.get('cluster_centers_or_representatives') is not None and i < \
                            self.cluster_info['cluster_centers_or_representatives'].shape[0]:
                        if self.fixed_segment_param_idx < self.cluster_info['cluster_centers_or_representatives'].shape[
                            1]:
                            rep_pt_val = self.cluster_info['cluster_centers_or_representatives'][
                                i, self.fixed_segment_param_idx]
                            desc_str += f", Rep. Value: {rep_pt_val:.3f}"
                        else:
                            desc_str += ", Rep. Value: N/A (Index Error)"
                else:
                    desc_str = "Segment Range: N/A"
            print(row_format.format(cluster_id_str, size, pod_dim, shape_param_str, energy_str, desc_str))
        print("-" * 90)

    def save_cluster_info(self, filename_prefix="cluster_info"):
        if not self.cluster_info or self.n_clusters_effective == 0:
            print("No cluster/segment info to save. Train model first.")
            return
        npz_filename = filename_prefix + ".npz";
        txt_filename = filename_prefix + ".txt"
        save_data = {
            'n_clusters_effective': self.n_clusters_effective, 'energy_threshold': self.energy_threshold,
            'use_smooth_transition': self.use_smooth_transition,
            'fixed_segment_param_idx': self.fixed_segment_param_idx,
            'fixed_num_segments': self.fixed_num_segments, 'fixed_segment_mode': self.fixed_segment_mode,
            'fixed_segment_proportions': self.fixed_segment_proportions,  # 保存比例
            'cluster_labels': self.cluster_labels,
            'cluster_centers_or_representatives': np.array(
                self.cluster_info.get('cluster_centers_or_representatives', [])),
            'segment_boundaries': np.array(self.cluster_info.get('segment_boundaries', [])),
            'local_pod_dims': np.array(self.cluster_info.get('local_pod_dims', [])),
            'local_shape_params': np.array(self.cluster_info.get('local_shape_params', [])),
            'local_energy_captured': np.array(self.cluster_info.get('local_energy_captured', [])),
            'cluster_sizes': np.array(self.cluster_info.get('cluster_sizes', [])),
        }
        if save_data['local_shape_params'].dtype == object:
            save_data['local_shape_params'] = np.array(
                [x if x is not None else np.nan for x in save_data['local_shape_params'].tolist()], dtype=float)
        if save_data['local_energy_captured'].dtype == object:
            save_data['local_energy_captured'] = np.array(
                [x if x is not None else np.nan for x in save_data['local_energy_captured'].tolist()], dtype=float)
        try:
            np.savez(npz_filename, **save_data)
            print(f"\nCluster/segment data saved to: {npz_filename}")
        except Exception as e:
            print(f"Error saving .npz file: {e}")
        try:
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("Clustered/Segmented POD-RBF Model Information\n");
                f.write("=" * 50 + "\n\n")
                f.write(
                    f"Clustering/Segmentation Mode: {'K-means Clustering' if self.fixed_segment_param_idx is None else f'Fixed Segmentation (Param Index: {self.fixed_segment_param_idx}, Mode: {self.fixed_segment_mode})'}\n")
                if self.fixed_segment_mode == 'increasing_density' and self.fixed_segment_proportions is not None:
                    f.write(f"Custom Segment Proportions: {self.fixed_segment_proportions}\n")
                f.write(f"Effective Number of Clusters/Segments (K_eff): {self.n_clusters_effective}\n")
                f.write(f"POD Energy Threshold: {self.energy_threshold}\n")
                f.write(f"Use Smooth Transition for Inference: {self.use_smooth_transition}\n")
                if self.fixed_segment_param_idx is not None:
                    f.write(f"Fixed Segmentation Parameter Index: {self.fixed_segment_param_idx}\n")
                    f.write(f"Expected Number of Segments: {self.fixed_num_segments}\n")
                    f.write(f"Segmentation Mode: {self.fixed_segment_mode}\n")
                    if self.segment_boundaries is not None:
                        f.write(
                            f"Segment Boundaries (ascending): {np.array2string(self.segment_boundaries, formatter={'float_kind': lambda x: '%.3f' % x})}\n")  # 递增

                f.write("\nCluster/Segment Details:\n");
                f.write("-" * 80 + "\n")
                f.write(
                    f"{'Cluster/SegID':>12} {'Size':>6} {'POD Dim':>8} {'Shape Param':>12} {'Energy':>10} {'Rep. Point/Center (Seg Param Val.)':>30}\n")
                f.write("-" * 80 + "\n")
                for i in range(self.n_clusters_effective):
                    id_str = f"{i + 1}";
                    sf_str = "N/A";
                    en_str = "N/A";
                    rep_str = "N/A"
                    if self.local_models and i < len(self.local_models) and self.local_models[
                        i] is None: id_str += " (Failed)"
                    size = self.cluster_info['cluster_sizes'][i] if i < len(
                        self.cluster_info['cluster_sizes']) else 'N/A'
                    pod = self.cluster_info['local_pod_dims'][i] if i < len(
                        self.cluster_info['local_pod_dims']) else 'N/A'
                    sf_val = self.cluster_info['local_shape_params'][i] if i < len(
                        self.cluster_info['local_shape_params']) else None
                    if sf_val is not None: sf_str = f"{sf_val:.6f}"
                    en_val = self.cluster_info['local_energy_captured'][i] if i < len(
                        self.cluster_info['local_energy_captured']) else None
                    if en_val is not None: en_str = f"{en_val:.6f}"
                    if self.cluster_centers_or_representatives is not None and i < \
                            self.cluster_centers_or_representatives.shape[0]:
                        if self.fixed_segment_param_idx is not None:
                            rep_val = self.cluster_centers_or_representatives[i, self.fixed_segment_param_idx]
                            rep_str = f"Seg Param Rep. Value: {rep_val:.3f}"
                        else:
                            center_vals = self.cluster_centers_or_representatives[i, :]
                            rep_str = f"Center: {np.array2string(center_vals, formatter={'float_kind': lambda x: '%.2f' % x}, max_line_width=50)}"
                    f.write(f"{id_str:>12} {size:>6} {pod:>8} {sf_str:>12} {en_str:>10} {rep_str}\n")
                f.write("-" * 80 + "\n")
            print(f"Human-readable summary saved to: {txt_filename}")
        except Exception as e:
            print(f"Error saving .txt file: {e}")

    def load_cluster_info(self, filename_prefix="cluster_info"):
        npz_filename = filename_prefix + ".npz"
        if not os.path.exists(npz_filename):
            print(f"Error: .npz file '{npz_filename}' not found. Cannot load info.")
            return False
        try:
            data = np.load(npz_filename, allow_pickle=True)
            self.n_clusters_effective = int(
                data['n_clusters_effective']) if 'n_clusters_effective' in data else self.n_clusters_kmeans
            self.energy_threshold = float(data['energy_threshold']) if 'energy_threshold' in data else 0.99
            self.use_smooth_transition = bool(
                data['use_smooth_transition']) if 'use_smooth_transition' in data else True
            self.fixed_segment_param_idx = data[
                'fixed_segment_param_idx'].item() if 'fixed_segment_param_idx' in data and data[
                'fixed_segment_param_idx'].size == 1 and data['fixed_segment_param_idx'].ndim == 0 else (
                data['fixed_segment_param_idx'] if 'fixed_segment_param_idx' in data and data[
                    'fixed_segment_param_idx'] is not None else None)
            self.fixed_num_segments = int(data['fixed_num_segments']) if 'fixed_num_segments' in data else 5
            self.fixed_segment_mode = str(data['fixed_segment_mode']) if 'fixed_segment_mode' in data else 'equal_width'
            self.fixed_segment_proportions = data.get('fixed_segment_proportions')  # 加载比例

            self.cluster_labels = data.get('cluster_labels')
            self.cluster_centers_or_representatives = data.get('cluster_centers_or_representatives')
            self.segment_boundaries = data.get('segment_boundaries')
            self.cluster_info = {
                'n_clusters_effective': self.n_clusters_effective,
                'cluster_centers_or_representatives': self.cluster_centers_or_representatives,
                'segment_boundaries': self.segment_boundaries,
                'fixed_segment_param_idx': self.fixed_segment_param_idx,
                'fixed_segment_mode': self.fixed_segment_mode,
                'fixed_segment_proportions': self.fixed_segment_proportions,
                'local_pod_dims': list(data.get('local_pod_dims', [])),
                'local_shape_params': list(data.get('local_shape_params', [])),
                'local_energy_captured': list(data.get('local_energy_captured', [])),
                'cluster_sizes': list(data.get('cluster_sizes', []))
            }
            print(f"Cluster/segment info loaded from '{npz_filename}'.")
            self.print_cluster_summary()
            return True
        except Exception as e:
            print(f"Error loading cluster/segment info from '{npz_filename}': {e}")
            return False


# ---------------------------------------------------------------------------
# 用户脚本的辅助函数
# ---------------------------------------------------------------------------
def current_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_time(message, log_file):
    try:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"{current_time_str()} - {message}\n")
    except Exception as e:
        print(f"Failed to write to log file {log_file}: {e}")


def round_to_nearest_05(x):
    return round(x / 0.05) * 0.05


# ---------------------------------------------------------------------------
# 主执行逻辑 (来自用户的 lid_driven_cavity.py 脚本)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------
    # 0) 基本设置和路径
    # -----------------------------
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_script_dir, "data")
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")
    output_heatmap_dir = os.path.join(current_script_dir, "output_heatmap_combined")
    output_diff2d_dir = os.path.join(current_script_dir, "output_difference")

    for folder in [output_heatmap_dir, output_diff2d_dir]:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder); print(f"Created folder: {folder}")
            except OSError as e:
                print(f"Failed to create folder {folder}: {e}");
    time_log_file = os.path.join(output_heatmap_dir, "time_log.txt")
    log_time("Program started", time_log_file)

    # -----------------------------
    # 1) 训练模型
    # -----------------------------
    print("\n" + "=" * 60);
    print("Step 1: Model Training");
    print("=" * 60)
    Re = np.linspace(1, 999, num=400);
    Re[0] = 1
    if not os.path.isdir(train_dir):
        print(f"Error: Training data directory '{train_dir}' not found.");
        exit()

    coords_example_file = "Re-1.0.csv"
    coords_path = os.path.join(train_dir, coords_example_file)
    if not os.path.exists(coords_path):
        coords_example_file_alt = f"Re-{Re[0]:.1f}.csv"
        if not os.path.exists(os.path.join(train_dir, coords_example_file_alt)):
            coords_example_file_alt = f"Re-{int(Re[0])}.csv"
        coords_path = os.path.join(train_dir, coords_example_file_alt)
    if not os.path.exists(coords_path):
        print(
            f"Error: Example file for coordinates '{coords_example_file}' (or alternative) not found in '{train_dir}'.")
        try:
            first_csv = sorted([f for f in os.listdir(train_dir) if f.lower().endswith('.csv')])[0]
            coords_path = os.path.join(train_dir, first_csv)
            print(f"Warning: Using first found CSV '{first_csv}' to load coordinates.")
        except IndexError:
            print(f"Error: No CSV files in '{train_dir}' to load coordinates."); exit()
    try:
        x, y = np.loadtxt(coords_path, delimiter=",", skiprows=1, usecols=(1, 2), unpack=True, )
        print(f"Coordinates loaded from '{coords_path}'. X points: {len(x)}, Y points: {len(y)}")
    except Exception as e:
        print(f"Error loading coordinates from '{coords_path}': {e}"); exit()

    try:
        train_snapshot_pattern = os.path.join(train_dir, "dummy_pattern_for_dir_extraction.csv")
        train_snapshot = buildSnapshotMatrix(train_snapshot_pattern, usecols=(0,))
    except Exception as e:
        print(f"Error building training snapshot matrix: {e}"); exit()

    print("\nTraining standard POD-RBF model...")
    model_std_pod_rbf = pod_rbf(energy_threshold=0.95)
    model_std_pod_rbf.train(train_snapshot, Re)
    print(
        f"Standard POD-RBF model trained. Energy kept after truncation = {model_std_pod_rbf.truncated_energy * 100:.2f}%")

    print("\n" + "=" * 60);
    print("Training Clustered POD-RBF model (K-means, 3 clusters)");
    print("=" * 60)
    clustered_model_kmeans = clustered_pod_rbf(
        n_clusters_kmeans=3, energy_threshold=0.95, use_smooth_transition=True, fixed_segment_param_idx=None
    )
    clustered_model_kmeans.train(train_snapshot, Re)
    clustered_model_kmeans.print_cluster_summary()
    kmeans_cluster_info_file_prefix = os.path.join(current_script_dir, "ns_kmeans_cluster_info")
    clustered_model_kmeans.save_cluster_info(kmeans_cluster_info_file_prefix)

    print("\n" + "=" * 60);
    print("Training Clustered POD-RBF model (Fixed 5 increasing density ranges)");
    print("=" * 60)
    # 为 'increasing_density' 模式定义比例，例如点数大致按 1:1.5:2:2.5:3 递增
    # 或者，如果不提供 fixed_segment_proportions，则在 fixed_segment_mode='increasing_density' 且 fixed_num_segments=5 时，
    # _define_fixed_segments 内部会使用默认的递增比例。
    custom_proportions = [1, 1.5, 2, 2.5, 3]  # 示例：低Re段点数少，高Re段点数多
    clustered_model_fixed_density = clustered_pod_rbf(
        fixed_segment_param_idx=0,
        fixed_num_segments=5,
        fixed_segment_mode='increasing_density',
        fixed_segment_proportions=custom_proportions,  # 可以传递自定义比例
        energy_threshold=0.95,
        use_smooth_transition=True
    )
    clustered_model_fixed_density.train(train_snapshot, Re)
    clustered_model_fixed_density.print_cluster_summary()
    fixed_density_cluster_info_file_prefix = os.path.join(current_script_dir, "ns_fixed_density_cluster_info")
    clustered_model_fixed_density.save_cluster_info(fixed_density_cluster_info_file_prefix)

    # -----------------------------
    # 2) 验证和可视化设置
    # -----------------------------
    print("\n" + "=" * 60);
    print("Step 2: Validation and Visualization");
    print("=" * 60)
    if not os.path.isdir(validation_dir):
        print(f"Warning: Validation data directory '{validation_dir}' not found. Skipping validation.");
        validation_files = []
    else:
        validation_files = [f for f in os.listdir(validation_dir) if f.lower().endswith('.csv')]
        if not validation_files: print(
            f"Warning: Validation data directory '{validation_dir}' is empty or no CSVs. Skipping validation.")

    error_data_std_model = {"Parameter": [], "Relative L2 Error": [], "Max Error": []}
    total_l2_error_std = 0;
    total_max_error_std = 0;
    num_val_files_processed = 0

    if x.size > 0 and y.size > 0:
        global_xmin, global_xmax = np.min(x), np.max(x);
        global_ymin, global_ymax = np.min(y), np.max(y)
        mid_x = (global_xmin + global_xmax) / 2.0;
        mid_y = (global_ymin + global_ymax) / 2.0
        x_ticks_coords = [global_xmin, mid_x, global_xmax];
        y_ticks_coords = [global_ymin, mid_y, global_ymax]
        x_tick_labels_coords = [f"{round_to_nearest_05(val):.2f}" for val in x_ticks_coords]
        y_tick_labels_coords = [f"{round_to_nearest_05(val):.2f}" for val in y_ticks_coords]
        grid_x = np.linspace(global_xmin, global_xmax, 200);
        grid_y = np.linspace(global_ymin, global_ymax, 200)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    else:
        print("Warning: Coordinates x,y not loaded. Plotting will be skipped."); validation_files = []

    # -----------------------------
    # 3) 遍历验证文件 & 可视化 (针对标准POD-RBF模型)
    # -----------------------------
    program_start_time = time.time()
    for val_file in validation_files:
        file_start_time = time.time()
        try:
            val_path = os.path.join(validation_dir, val_file)
            raw_data_val = np.loadtxt(val_path, delimiter=",", skiprows=1)
            if raw_data_val.ndim == 1:
                val_target_solution = raw_data_val
            else:
                val_target_solution = raw_data_val[:, 0]
            try:
                re_value_str = val_file.split('-')[1].split('.csv')[0];
                re_value_float = float(re_value_str)
            except (IndexError, ValueError):
                print(f"Warning: Cannot extract Re value from '{val_file}'. Skipping."); continue

            pred_solution_std = model_std_pod_rbf.inference(re_value_float)
            if pred_solution_std.shape != val_target_solution.shape:
                print(
                    f"Warning: Shape mismatch for {val_file}. Pred: {pred_solution_std.shape}, True: {val_target_solution.shape}. Skipping error calc.")
            else:
                l2_error = np.linalg.norm(pred_solution_std - val_target_solution) / np.linalg.norm(val_target_solution)
                max_error = np.max(np.abs(pred_solution_std - val_target_solution))
                print(f"[Re={re_value_str}] Standard POD-RBF: L2 Error = {l2_error:.4e}, Max Error = {max_error:.4e}")
                error_data_std_model["Parameter"].append(re_value_str);
                error_data_std_model["Relative L2 Error"].append(l2_error)
                error_data_std_model["Max Error"].append(max_error);
                total_l2_error_std += l2_error;
                total_max_error_std += max_error
                num_val_files_processed += 1

            if x.size > 0 and y.size > 0:
                grid_sol_pred = griddata((x, y), pred_solution_std, (grid_X, grid_Y), method='linear')
                grid_val_target = griddata((x, y), val_target_solution, (grid_X, grid_Y), method='linear')
                diff_pred_target = np.abs(pred_solution_std - val_target_solution)
                grid_diff_display = griddata((x, y), diff_pred_target, (grid_X, grid_Y), method='linear')

                fig_h, (ax1_h, ax2_h) = plt.subplots(1, 2, figsize=(16, 7))
                fig_h.suptitle(f"Re = {re_value_str}", fontsize=18, fontweight='bold')
                common_vmin = min(np.nanmin(grid_sol_pred), np.nanmin(grid_val_target));
                common_vmax = max(np.nanmax(grid_sol_pred), np.nanmax(grid_val_target))
                mid_val = (common_vmin + common_vmax) / 2.0;
                rmin, rmid, rmax = round_to_nearest_05(common_vmin), round_to_nearest_05(mid_val), round_to_nearest_05(
                    common_vmax)
                common_ticks = sorted(list(set([rmin, rmid, rmax])));
                tick_labels = [f"{val:.2f}" for val in common_ticks]

                im1 = ax1_h.imshow(grid_sol_pred, extent=(global_xmin, global_xmax, global_ymin, global_ymax),
                                   origin='lower', cmap='jet', aspect='auto', vmin=common_vmin, vmax=common_vmax)
                ax1_h.set_title(f"Predicted Solution (Standard POD-RBF)", fontsize=14)
                ax1_h.set_xlabel("X");
                ax1_h.set_ylabel("Y")
                ax1_h.set_xticks(x_ticks_coords);
                ax1_h.set_xticklabels(x_tick_labels_coords, fontweight='bold')
                ax1_h.set_yticks(y_ticks_coords);
                ax1_h.set_yticklabels(y_tick_labels_coords, fontweight='bold')
                cbar1 = fig_h.colorbar(im1, ax=ax1_h, orientation='vertical', fraction=0.046, pad=0.04)
                cbar1.set_ticks(common_ticks);
                cbar1.ax.set_yticklabels(tick_labels, fontweight='bold')

                im2 = ax2_h.imshow(grid_val_target, extent=(global_xmin, global_xmax, global_ymin, global_ymax),
                                   origin='lower', cmap='jet', aspect='auto', vmin=common_vmin, vmax=common_vmax)
                ax2_h.set_title(f"True Solution", fontsize=14)
                ax2_h.set_xlabel("X");
                ax2_h.set_ylabel("Y")
                ax2_h.set_xticks(x_ticks_coords);
                ax2_h.set_xticklabels(x_tick_labels_coords, fontweight='bold')
                ax2_h.set_yticks(y_ticks_coords);
                ax2_h.set_yticklabels(y_tick_labels_coords, fontweight='bold')
                cbar2 = fig_h.colorbar(im2, ax=ax2_h, orientation='vertical', fraction=0.046, pad=0.04)
                cbar2.set_ticks(common_ticks);
                cbar2.ax.set_yticklabels(tick_labels, fontweight='bold')

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                combined_filename = os.path.join(output_heatmap_dir, f"Heatmap_Re_{re_value_str}.png")
                plt.savefig(combined_filename, dpi=300);
                plt.close(fig_h)

                fig_d, ax_d = plt.subplots(figsize=(9, 7))
                im_d = ax_d.imshow(grid_diff_display, extent=(global_xmin, global_xmax, global_ymin, global_ymax),
                                   origin='lower', cmap='jet', aspect='auto')
                diff_vmin = 0;
                diff_vmax = np.nanmax(grid_diff_display) if np.any(~np.isnan(grid_diff_display)) else 0.2
                diff_vmax = max(diff_vmax, 0.01);
                im_d.set_clim(diff_vmin, diff_vmax)
                diff_ticks_vals = np.linspace(diff_vmin, diff_vmax, 5);
                diff_tick_labels_vals = [f"{val:.2e}" for val in diff_ticks_vals]
                cbar_d = fig_d.colorbar(im_d, ax=ax_d, orientation='vertical', fraction=0.046, pad=0.04)
                cbar_d.set_ticks(diff_ticks_vals);
                cbar_d.ax.set_yticklabels(diff_tick_labels_vals, fontweight='bold')
                cbar_d.set_label("Absolute Error", fontsize=12)

                ax_d.set_title(f"Absolute Error |Predicted - True| (Re={re_value_str})", fontsize=16)
                ax_d.set_xlabel("X");
                ax_d.set_ylabel("Y")
                ax_d.set_xticks(x_ticks_coords);
                ax_d.set_xticklabels(x_tick_labels_coords, fontweight='bold')
                ax_d.set_yticks(y_ticks_coords);
                ax_d.set_yticklabels(y_tick_labels_coords, fontweight='bold')
                plt.tight_layout()
                diff2d_filename = os.path.join(output_diff2d_dir, f"Difference_Re_{re_value_str}.png")
                plt.savefig(diff2d_filename, dpi=300);
                plt.close(fig_d)
        except Exception as e_file:
            print(f"Error processing file {val_file}: {e_file}")  # 英文
            log_time(f"Error processing file {val_file}: {e_file}", time_log_file)  # 英文
            continue
        file_end_time = time.time()
        elapsed_time_file = file_end_time - file_start_time
        log_msg_file = f"Processed file {val_file}, time taken {elapsed_time_file:.2f} seconds"  # 英文
        log_time(log_msg_file, time_log_file)
        print(log_msg_file)

    # -----------------------------
    # 4) 保存误差数据 & 输出平均误差
    # -----------------------------
    if num_val_files_processed > 0:
        avg_l2_error_std = total_l2_error_std / num_val_files_processed
        avg_max_error_std = total_max_error_std / num_val_files_processed
        error_data_std_model["Parameter"].append("Average")
        error_data_std_model["Relative L2 Error"].append(avg_l2_error_std)
        error_data_std_model["Max Error"].append(avg_max_error_std)
        print(f"\nStandard POD-RBF Model Average L2 Relative Error = {avg_l2_error_std:.4e}")  # 英文
        print(f"Standard POD-RBF Model Average Max Absolute Error = {avg_max_error_std:.4e}")  # 英文
    else:
        print("\nNo validation files processed, cannot calculate average error.")  # 英文

    excel_filename = os.path.join(current_script_dir, "error_evaluation_std_model.xlsx")
    try:
        df_std = pd.DataFrame(error_data_std_model)
        df_std.to_excel(excel_filename, index=False)
        print(f"Standard POD-RBF model error data saved to: {excel_filename}")  # 英文
    except Exception as e_excel:
        print(f"Failed to save error data to Excel: {e_excel}")  # 英文

    program_end_time = time.time()
    total_elapsed_program = program_end_time - program_start_time
    log_msg_total_program = f"All results generated. Total time taken: {total_elapsed_program:.2f} seconds"  # 英文
    log_time(log_msg_total_program, time_log_file)
    print(f"\n{log_msg_total_program}")


import numpy as np
from sklearn.cluster import KMeans # 保留KMeans，以防用户不使用固定范围模式
import warnings
import os # 为 buildSnapshotMatrix 添加导入
from tqdm import tqdm # 为 buildSnapshotMatrix 添加导入

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
    dirpath_for_csv_files = os.path.dirname(mypath_pattern) # 从模式中提取目录路径
    if not dirpath_for_csv_files: # 如果 mypath_pattern 只是当前目录中的文件名
        dirpath_for_csv_files = "." # 使用当前目录
    
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

    for filename in tqdm(csv_files, desc="加载快照CSV文件"):
        filepath = os.path.join(dirpath_for_csv_files, filename)
        try:
            # 加载指定列的数据
            data_from_file = np.loadtxt(
                filepath,
                delimiter=",",
                skiprows=skiprows,
                usecols=usecols,
                unpack=False 
            )
            
            # 将加载的数据展平为单个列向量 (如果usecols选择了多列，则按列顺序F展平)
            current_snapshot_vector = data_from_file.flatten(order='F') 

            if num_sample_points_per_snapshot is None:
                num_sample_points_per_snapshot = len(current_snapshot_vector)
            elif len(current_snapshot_vector) != num_sample_points_per_snapshot:
                warnings.warn(f"文件 '{filename}' 中的数据点数 ({len(current_snapshot_vector)}) "
                              f"与第一个文件 ({num_sample_points_per_snapshot}) 不一致。跳过此文件。")
                continue
            
            snapshot_columns_list.append(current_snapshot_vector)

        except Exception as e:
            warnings.warn(f"加载或处理文件 '{filepath}' 时出错: {e}。跳过此文件。")
            continue
    
    if not snapshot_columns_list:
        raise ValueError("未能从任何CSV文件成功加载数据以构建快照矩阵。")

    # 将列的列表转换为快照矩阵 (转置使得每列是一个快照)
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
        self.mem_limit = 16  # gigabytes, 内存限制，用于切换SVD算法

    def _calcTruncatedPODBasis(self):
        """
        计算截断的POD基。
        """
        if self.snapshot is None:
            raise ValueError("Snapshot matrix is not set. Call train() first.")

        memory = self.snapshot.nbytes / 1e9 # 计算快照矩阵的内存占用

        if memory < self.mem_limit:
            # 内存占用较小时，直接使用SVD
            U, S_singular, _ = np.linalg.svd(self.snapshot, full_matrices=False)
            if S_singular.size == 0:
                warnings.warn("SVD resulted in zero singular values or empty snapshot. Basis will be empty.")
                self.basis = np.array([]).reshape(self.snapshot.shape[0], 0)
                self.truncated_energy = 0.0
                self.cumul_energy = np.array([0.0])
                return self.basis
            
            # 能量按奇异值的平方计算 (更常见) 或奇异值本身 (取决于定义)
            # 原代码使用S，这里用S**2更符合能量定义
            energy_values = S_singular**2 
            self.cumul_energy = np.cumsum(energy_values) / np.sum(energy_values)
            
            if self.energy_threshold >= 1.0: # 保留所有能量/模式
                trunc_id = len(S_singular) -1 
            elif self.energy_threshold <= 0 or (self.cumul_energy.size > 0 and self.energy_threshold < self.cumul_energy[0]):
                trunc_id = 0
            else:
                # 找到满足能量阈值的最小模式数
                trunc_id_candidates = np.where(self.cumul_energy >= self.energy_threshold)[0]
                if trunc_id_candidates.size > 0:
                    trunc_id = trunc_id_candidates[0]
                else: # 如果没有模式达到阈值 (例如阈值过高或能量分散)
                    trunc_id = len(S_singular) - 1 # 保留所有模式
            
            self.truncated_energy = self.cumul_energy[trunc_id]
            self.basis = U[:, :(trunc_id + 1)]
        else:
            # 内存占用较大时，使用特征值分解方法 ("method of snapshots")
            print("Snapshot matrix is large, using eigenvalue decomposition method for POD basis.")
            # 计算协方差矩阵 C = snapshot.T @ snapshot
            L = np.dot(self.snapshot.T, self.snapshot)
            eig_vals_L, eig_vecs_L = np.linalg.eigh(L) # eigh 用于对称/厄米矩阵

            # 特征值从大到小排序
            sorted_indices = np.argsort(eig_vals_L)[::-1]
            self.eig_vals = eig_vals_L[sorted_indices]
            eig_vecs_L_sorted = eig_vecs_L[:, sorted_indices]

            # 移除负的或非常小的特征值 (数值误差可能导致)
            positive_eig_indices = self.eig_vals > 1e-10 # 容差
            self.eig_vals = self.eig_vals[positive_eig_indices]
            eig_vecs_L_sorted = eig_vecs_L_sorted[:, positive_eig_indices]

            if self.eig_vals.size == 0:
                warnings.warn("Eigenvalue decomposition resulted in no positive eigenvalues. Basis will be empty.")
                self.basis = np.array([]).reshape(self.snapshot.shape[0], 0)
                self.truncated_energy = 0.0
                self.cumul_energy = np.array([0.0])
                return self.basis

            self.cumul_energy = np.cumsum(self.eig_vals) / np.sum(self.eig_vals)

            if self.energy_threshold >= 1.0:
                trunc_id = len(self.eig_vals) - 1
            elif self.energy_threshold <= 0 or (self.cumul_energy.size > 0 and self.energy_threshold < self.cumul_energy[0]):
                trunc_id = 0
            else:
                trunc_id_candidates = np.where(self.cumul_energy >= self.energy_threshold)[0]
                if trunc_id_candidates.size > 0:
                    trunc_id = trunc_id_candidates[0]
                else:
                    trunc_id = len(self.eig_vals) - 1

            self.truncated_energy = self.cumul_energy[trunc_id]
            
            # 保留截断数量的特征值和对应的特征向量
            eig_vals_truncated = self.eig_vals[:(trunc_id + 1)]
            eig_vecs_L_truncated = eig_vecs_L_sorted[:, :(trunc_id + 1)]
            
            # 计算POD基 U_k = S @ V_k @ (Lambda_k)^(-1/2)
            sqrt_eig_vals = np.sqrt(eig_vals_truncated)
            # 避免除以零
            sqrt_eig_vals[np.isclose(sqrt_eig_vals, 0)] = 1e-9 
            self.basis = np.dot(self.snapshot, eig_vecs_L_truncated) / sqrt_eig_vals
        
        return self.basis

    def _buildCollocationMatrix(self, c_param, params_to_build_for):
        """
        构建RBF配置矩阵 (训练时) 或推理矩阵的一部分。
        params_to_build_for: (num_param_dims, num_points)
        """
        num_train_points = self.train_params.shape[1]
        num_build_points = params_to_build_for.shape[1]
        num_param_dims = self.train_params.shape[0]

        r2 = np.zeros((num_build_points, num_train_points))

        for i in range(num_param_dims):
            I_mesh, J_mesh = np.meshgrid(
                params_to_build_for[i, :],
                self.train_params[i, :],
                indexing='ij',
                copy=False
            )
            
            range_val = self.params_range[i]
            if np.isclose(range_val, 0): 
                term = np.zeros_like(I_mesh) 
            else:
                term = ((I_mesh - J_mesh) / range_val)**2
            r2 += term
        
        return 1.0 / np.sqrt(r2 / (c_param**2) + 1.0 + 1e-9)


    def _findOptimShapeParam(self, cond_target_range=(1e3, 1e8), max_steps=100):
        """
        寻找RBF最优形状参数c。
        目标是使配置矩阵的条件数落在 cond_target_range 内。
        """
        c_low = 0.01  
        c_high = 10.0 
        
        # 确保 self.train_params 在调用此方法前已设置
        if self.train_params is None:
            raise ValueError("_findOptimShapeParam called before train_params is set.")

        try:
            # 构建用于条件数计算的配置矩阵 (训练点 vs 训练点)
            colloc_matrix_low = self._buildCollocationMatrix(c_low, self.train_params)
            colloc_matrix_high = self._buildCollocationMatrix(c_high, self.train_params)
            cond_at_c_low = np.linalg.cond(colloc_matrix_low)
            cond_at_c_high = np.linalg.cond(colloc_matrix_high)
        except np.linalg.LinAlgError:
            warnings.warn("LinAlgError during initial condition number calculation in _findOptimShapeParam. Using default c=1.0.")
            return 1.0

        # IMQ RBF: c 越大, RBF越平滑, 条件数通常越小。所以 cond(c_low) > cond(c_high)
        if cond_at_c_low < cond_at_c_high:
            warnings.warn(f"Condition number at c_low ({cond_at_c_low:.2e}) is not greater than at c_high ({cond_at_c_high:.2e}). "
                          "Shape parameter search might be unreliable. Using default c=1.0.")
            return 1.0 # 返回一个默认值

        optim_c = (c_low + c_high) / 2.0 # 初始化
        for k in range(max_steps):
            optim_c = (c_low + c_high) / 2.0
            if np.isclose(c_low, c_high, atol=1e-4): break # 如果搜索区间足够小则停止
            
            try:
                current_colloc_matrix = self._buildCollocationMatrix(optim_c, self.train_params)
                current_cond = np.linalg.cond(current_colloc_matrix)
            except np.linalg.LinAlgError: 
                current_cond = np.inf 

            if current_cond < cond_target_range[0]: # 条件数太小 (c可能过大)
                c_high = optim_c 
            elif current_cond > cond_target_range[1]: # 条件数太大 (c可能过小)
                c_low = optim_c
            else: # 条件数在目标范围内
                return optim_c
        
        final_cond_check = np.linalg.cond(self._buildCollocationMatrix(optim_c, self.train_params))
        warnings.warn(f"Optimal shape parameter search did not converge to target range within {max_steps} steps. "
                      f"Returning c={optim_c:.3e} with condition number={final_cond_check:.2e} (Target: {cond_target_range})")
        return optim_c

    def train(self, snapshot, train_params, shape_factor=None):
        if train_params.ndim == 1:
            self.train_params = np.expand_dims(train_params, axis=0)
        else:
            self.train_params = train_params
        
        self.snapshot = snapshot
        if self.snapshot.shape[1] != self.train_params.shape[1]:
            raise ValueError("Number of snapshots and training parameter points do not match.")

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
            warnings.warn("POD basis is empty. RBF model will predict zeros or fail.")
            self.weights = np.array([]) 
            return

        print(f"POD basis calculated. Shape: {self.basis.shape}, Truncated energy: {self.truncated_energy:.6f}")

        # 配置矩阵 F (N_train_points x N_train_points)
        F = self._buildCollocationMatrix(self.shape_factor, self.train_params) 
        
        A = np.dot(self.basis.T, self.snapshot) 
        
        try:
            # 求解 W @ F.T = A.T  =>  F @ W.T = A.T
            # 或者 W @ F = A (如果F是对称的，F=F.T)
            # 如果 F 是 collocation matrix C_ij = phi(||x_i - x_j||) 那么 F 是对称的。
            # A = W @ F => W = A @ inv(F)
            self.weights = np.linalg.solve(F, A.T).T # A.T (N_train, N_modes), F (N_train, N_train) -> solve(F, A.T) gives W.T (N_train, N_modes)
                                                    # So W is (N_modes, N_train)
        except np.linalg.LinAlgError:
            warnings.warn("Singular collocation matrix F. Using pseudo-inverse for RBF weights.")
            self.weights = np.dot(A, np.linalg.pinv(F))


    def inference(self, inf_params):
        if self.basis is None or self.weights is None:
            raise RuntimeError("Model is not trained yet.")
        if self.basis.size == 0 or self.weights.size == 0:
            num_features = self.snapshot.shape[0] if self.snapshot is not None else 0
            if np.isscalar(inf_params) or inf_params.ndim == 1:
                return np.zeros(num_features)
            else: 
                num_inf_points = 1 if inf_params.ndim == 1 else inf_params.shape[1]
                return np.zeros((num_features, num_inf_points))

        current_inf_params = inf_params
        original_inf_ndim_is_1 = False
        if np.isscalar(current_inf_params): 
            current_inf_params = np.array([[current_inf_params]]) 
            original_inf_ndim_is_1 = True
        elif current_inf_params.ndim == 1: 
            current_inf_params = np.expand_dims(current_inf_params, axis=1) 
            original_inf_ndim_is_1 = True
        
        if current_inf_params.shape[0] != self.train_params.shape[0]:
            raise ValueError(f"Inference params dim {current_inf_params.shape[0]} != train params dim {self.train_params.shape[0]}")

        # 推理RBF矩阵 F_inf (num_inf_points x num_train_points)
        F_inf = self._buildCollocationMatrix(self.shape_factor, current_inf_params)
        
        # A_inf = W @ F_inf.T
        # W: (num_pod_modes, num_train_points)
        # F_inf.T: (num_train_points, num_inf_points)
        A_inf = np.dot(self.weights, F_inf.T)
        
        inference_result = np.dot(self.basis, A_inf)
        
        if original_inf_ndim_is_1:
            return inference_result[:, 0] 
        return inference_result


class clustered_pod_rbf:
    def __init__(self,
                 n_clusters_kmeans=3,
                 energy_threshold=0.99,
                 use_smooth_transition=True,
                 fixed_segment_param_idx=None, 
                 fixed_num_segments=5,
                 fixed_segment_mode='equal_width'):
        self.energy_threshold = energy_threshold
        self.use_smooth_transition = use_smooth_transition
        self.fixed_segment_param_idx = fixed_segment_param_idx
        self.fixed_num_segments = fixed_num_segments
        if fixed_segment_param_idx is not None and fixed_num_segments <= 0:
            raise ValueError("使用固定分段时，fixed_num_segments必须为正数。")
        self.fixed_segment_mode = fixed_segment_mode
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
        min_val = np.min(params_for_segmentation)
        max_val = np.max(params_for_segmentation)
        if np.isclose(min_val, max_val): 
            warnings.warn(f"用于分段的参数索引 {self.fixed_segment_param_idx} 的所有值都相同。将只创建一个分段。")
            self.segment_boundaries = np.array([max_val, min_val]) 
            self.n_clusters_effective = 1 
            return

        if self.fixed_segment_mode == 'equal_width':
            self.segment_boundaries = np.linspace(max_val, min_val, self.fixed_num_segments + 1)
        elif self.fixed_segment_mode == 'quantile':
            percentiles = np.linspace(100, 0, self.fixed_num_segments + 1)
            self.segment_boundaries = np.percentile(params_for_segmentation, percentiles)
            unique_boundaries = np.unique(self.segment_boundaries)
            self.segment_boundaries = unique_boundaries[::-1] 
            
            if len(self.segment_boundaries) < 2 : 
                 warnings.warn(f"参数 {self.fixed_segment_param_idx} 的分位数分段导致少于2个唯一边界。可能所有数据点都非常接近。将尝试使用等宽分段。")
                 self.fixed_segment_mode = 'equal_width' 
                 self._define_fixed_segments(params_for_segmentation) 
                 return
            self.n_clusters_effective = len(self.segment_boundaries) - 1
            if self.n_clusters_effective < 1: 
                warnings.warn(f"无法创建至少一个有效分段，即使在回退之后。将所有数据视为一个分段。")
                self.n_clusters_effective = 1
                self.segment_boundaries = np.array([max_val, min_val])
        else:
            raise ValueError(f"未知的 fixed_segment_mode: {self.fixed_segment_mode}")
        
        if self.n_clusters_effective != max(0, len(self.segment_boundaries) - 1) and self.n_clusters_effective > 0 : # max(0,...) to handle len=1 case
             actual_possible_segments = max(0, len(self.segment_boundaries) - 1)
             warnings.warn(f"有效簇数 ({self.n_clusters_effective}) 与根据边界计算的数量 ({actual_possible_segments}) 不匹配。将根据边界调整为 {actual_possible_segments}。")
             self.n_clusters_effective = actual_possible_segments
             if self.n_clusters_effective == 0 and len(self.segment_boundaries) >=2: # Should not happen if len >=2
                 self.n_clusters_effective = 1 # Ensure at least one if boundaries allow


    def _assign_labels_fixed_segments(self, params_for_segmentation):
        num_points = len(params_for_segmentation)
        labels = np.full(num_points, -1, dtype=int) 
        if self.n_clusters_effective == 0: 
            warnings.warn("有效簇数为0，无法分配标签。")
            return labels
        if self.segment_boundaries is None or len(self.segment_boundaries) < 2:
            warnings.warn("分段边界未正确定义，无法分配标签。")
            return labels

        for k in range(self.n_clusters_effective):
            upper_b = self.segment_boundaries[k]
            lower_b = self.segment_boundaries[k+1]
            if k == self.n_clusters_effective - 1: 
                condition = (params_for_segmentation >= lower_b - 1e-9) & (params_for_segmentation <= upper_b + 1e-9) # 包含边界，加容差
            else: 
                condition = (params_for_segmentation > lower_b + 1e-9) & (params_for_segmentation <= upper_b + 1e-9) # (lower, upper]
            labels[condition] = k
        
        if self.segment_boundaries.size > 0 : # 确保最大值点在簇0
            max_val_points_condition = np.isclose(params_for_segmentation, self.segment_boundaries[0])
            labels[max_val_points_condition] = 0

        unassigned_count = np.sum(labels == -1)
        if unassigned_count > 0:
            warnings.warn(f"{unassigned_count} 个点在固定分段中未能分配标签。将尝试分配给最近的段的中点。")
            for idx in np.where(labels == -1)[0]:
                val = params_for_segmentation[idx]
                closest_segment = -1
                min_dist_to_midpoint = np.inf
                if self.n_clusters_effective > 0: # 确保有簇可以分配
                    for k_label in range(self.n_clusters_effective):
                        # 检查 self.segment_boundaries 是否足够长
                        if len(self.segment_boundaries) > k_label + 1:
                            seg_upper = self.segment_boundaries[k_label]
                            seg_lower = self.segment_boundaries[k_label+1]
                            midpoint = (seg_upper + seg_lower) / 2.0
                            dist = np.abs(val - midpoint)
                            if dist < min_dist_to_midpoint:
                                min_dist_to_midpoint = dist
                                closest_segment = k_label
                    if closest_segment != -1:
                        labels[idx] = closest_segment
                    elif self.n_clusters_effective == 1: # 如果只有一个簇
                        labels[idx] = 0 
        return labels

    def train(self, snapshot, train_params, shape_factor=None):
        original_train_params_ndim = train_params.ndim
        if train_params.ndim == 1:
            params_for_kmeans = np.expand_dims(train_params, axis=1) 
            train_params_internal = np.expand_dims(train_params, axis=0) 
            if self.fixed_segment_param_idx is not None and self.fixed_segment_param_idx != 0:
                raise ValueError("如果 train_params 是1D, fixed_segment_param_idx 必须是0。")
            selected_param_for_segmentation = train_params 
        else: 
            params_for_kmeans = train_params.T 
            train_params_internal = train_params
            if self.fixed_segment_param_idx is not None:
                if not (0 <= self.fixed_segment_param_idx < train_params_internal.shape[0]):
                    raise ValueError(f"fixed_segment_param_idx {self.fixed_segment_param_idx} 超出 train_params 维度 {train_params_internal.shape[0]} 的范围。")
                selected_param_for_segmentation = train_params_internal[self.fixed_segment_param_idx, :] 

        if self.fixed_segment_param_idx is not None:
            print(f"使用固定分段模式，参数索引: {self.fixed_segment_param_idx}，期望分段数: {self.fixed_num_segments}，模式: {self.fixed_segment_mode}。")
            self._define_fixed_segments(selected_param_for_segmentation)
            print(f"有效分段数: {self.n_clusters_effective}")
            if self.segment_boundaries is not None:
                 print(f"分段边界 (降序): {np.array2string(self.segment_boundaries, formatter={'float_kind':lambda x: '%.3f' % x})}")
            self.cluster_labels = self._assign_labels_fixed_segments(selected_param_for_segmentation)

            if self.n_clusters_effective > 0:
                self.cluster_centers_or_representatives = np.zeros((self.n_clusters_effective, train_params_internal.shape[0]))
                for i in range(self.n_clusters_effective):
                    indices_in_segment = np.where(self.cluster_labels == i)[0]
                    if len(indices_in_segment) > 0:
                        if len(self.segment_boundaries) > i + 1: # 确保边界存在
                            upper_b = self.segment_boundaries[i]
                            lower_b = self.segment_boundaries[i+1]
                            self.cluster_centers_or_representatives[i, self.fixed_segment_param_idx] = (upper_b + lower_b) / 2.0
                        else: # 边界不足，可能发生在n_clusters_effective=1但segment_boundaries意外很短
                            self.cluster_centers_or_representatives[i, self.fixed_segment_param_idx] = np.mean(selected_param_for_segmentation[indices_in_segment])


                        for dim_idx in range(train_params_internal.shape[0]):
                            if dim_idx != self.fixed_segment_param_idx:
                                self.cluster_centers_or_representatives[i, dim_idx] = np.mean(train_params_internal[dim_idx, indices_in_segment])
                    else: 
                        if self.segment_boundaries is not None and len(self.segment_boundaries) > i+1 : 
                            upper_b = self.segment_boundaries[i]
                            lower_b = self.segment_boundaries[i+1]
                            self.cluster_centers_or_representatives[i, self.fixed_segment_param_idx] = (upper_b + lower_b) / 2.0
                        for dim_idx in range(train_params_internal.shape[0]):
                            if dim_idx != self.fixed_segment_param_idx: self.cluster_centers_or_representatives[i, dim_idx] = np.nan
            else: 
                self.cluster_centers_or_representatives = np.array([])
        else: 
            current_n_clusters_for_kmeans = self.n_clusters_kmeans
            if params_for_kmeans.shape[0] < current_n_clusters_for_kmeans:
                warnings.warn(f"训练点数 ({params_for_kmeans.shape[0]}) 少于 K-means 簇数 ({current_n_clusters_for_kmeans})。"
                              f"将 K-means 簇数设为 {params_for_kmeans.shape[0]}。")
                current_n_clusters_for_kmeans = params_for_kmeans.shape[0]
            if current_n_clusters_for_kmeans == 0:
                raise ValueError("无法使用0个簇进行K-means训练。")
            self.n_clusters_effective = current_n_clusters_for_kmeans
            print(f"执行 K-means 聚类，簇数: {self.n_clusters_effective}...")
            kmeans = KMeans(n_clusters=self.n_clusters_effective, random_state=42, n_init='auto')
            self.cluster_labels = kmeans.fit_predict(params_for_kmeans)
            self.cluster_centers_or_representatives = kmeans.cluster_centers_ 

        self.cluster_info = {
            'n_clusters_effective': self.n_clusters_effective,
            'cluster_centers_or_representatives': self.cluster_centers_or_representatives.copy() if self.cluster_centers_or_representatives is not None else None,
            'segment_boundaries': self.segment_boundaries.copy() if self.segment_boundaries is not None else None,
            'fixed_segment_param_idx': self.fixed_segment_param_idx,
            'fixed_segment_mode': self.fixed_segment_mode,
            'local_pod_dims': [], 'local_shape_params': [], 'local_energy_captured': [],
            'cluster_sizes': [], 'param_ranges_local': [] 
        }
        self.local_models = []

        if self.n_clusters_effective > 0:
            print(f"\n为 {self.n_clusters_effective} 个局部 POD-RBF 模型进行训练...")
            print("=" * 50)
            for i in range(self.n_clusters_effective):
                indices_in_cluster = np.where(self.cluster_labels == i)[0]
                if len(indices_in_cluster) == 0:
                    print(f"警告: 簇/段 {i+1} 为空。跳过。")
                    self.cluster_info['local_pod_dims'].append(0); self.cluster_info['local_shape_params'].append(None); 
                    self.cluster_info['local_energy_captured'].append(None); self.cluster_info['cluster_sizes'].append(0);
                    self.cluster_info['param_ranges_local'].append(None); self.local_models.append(None)
                    continue

                cluster_snapshot = snapshot[:, indices_in_cluster]
                current_cluster_params_for_podrbf = train_params_internal[:, indices_in_cluster]
                local_range_info = []
                if current_cluster_params_for_podrbf.size > 0: # 确保参数非空
                    for d_idx in range(current_cluster_params_for_podrbf.shape[0]):
                        min_p, max_p = np.min(current_cluster_params_for_podrbf[d_idx,:]), np.max(current_cluster_params_for_podrbf[d_idx,:])
                        local_range_info.append((min_p, max_p))
                self.cluster_info['param_ranges_local'].append(local_range_info)

                current_shape_factor_for_local_model = None
                if isinstance(shape_factor, (list, tuple)): 
                    if i < len(shape_factor): current_shape_factor_for_local_model = shape_factor[i]
                    else: warnings.warn(f"提供的形状因子列表长度不足。簇 {i+1} 将自动寻找形状因子。")
                elif isinstance(shape_factor, (int, float)): 
                    current_shape_factor_for_local_model = shape_factor

                print(f"\n簇/段 {i+1}/{self.n_clusters_effective}:"); print(f"  - 样本数: {len(indices_in_cluster)}")
                if self.fixed_segment_param_idx is not None and selected_param_for_segmentation.size > 0 and indices_in_cluster.size > 0:
                    seg_p_vals_in_cluster = selected_param_for_segmentation[indices_in_cluster]
                    if seg_p_vals_in_cluster.size > 0: # 再次检查确保非空
                         print(f"  - 分段参数实际范围: Min={np.min(seg_p_vals_in_cluster):.3f}, Max={np.max(seg_p_vals_in_cluster):.3f}")
                    if self.segment_boundaries is not None and len(self.segment_boundaries) > i+1:
                        print(f"  - 定义的分段范围: ({self.segment_boundaries[i+1]:.3f}, {self.segment_boundaries[i]:.3f}]")
                
                local_model_instance = pod_rbf(energy_threshold=self.energy_threshold)
                try:
                    local_model_instance.train(cluster_snapshot, current_cluster_params_for_podrbf, current_shape_factor_for_local_model)
                    self.local_models.append(local_model_instance)
                    pod_dim = local_model_instance.basis.shape[1] if hasattr(local_model_instance, 'basis') and local_model_instance.basis is not None and local_model_instance.basis.size > 0 else 0
                    actual_sf = local_model_instance.shape_factor if hasattr(local_model_instance, 'shape_factor') else None
                    energy_cap = local_model_instance.truncated_energy if hasattr(local_model_instance, 'truncated_energy') else None
                    self.cluster_info['local_pod_dims'].append(pod_dim); self.cluster_info['local_shape_params'].append(actual_sf); self.cluster_info['local_energy_captured'].append(energy_cap)
                    print(f"  - 局部POD维度: {pod_dim}")
                    if actual_sf is not None: print(f"  - 使用的形状因子: {actual_sf:.6f}")
                    if energy_cap is not None: print(f"  - 捕获能量: {energy_cap:.6f}")
                except Exception as e:
                    print(f"为簇/段 {i+1} 训练模型时出错: {e}")
                    self.local_models.append(None); 
                    self.cluster_info['local_pod_dims'].append(0); self.cluster_info['local_shape_params'].append(None); self.cluster_info['local_energy_captured'].append(None)
                self.cluster_info['cluster_sizes'].append(len(indices_in_cluster))
        else: 
            warnings.warn("有效簇/段数为0，没有训练任何局部模型。")

        num_successful_models = sum(1 for m in self.local_models if m is not None)
        print(f"\n完成训练。成功训练了 {num_successful_models} 个局部POD-RBF模型 (共 {self.n_clusters_effective} 个簇/段)。")
        self.print_cluster_summary()

    def _get_cluster_for_inf_param_fixed(self, inf_param_val_for_segment):
        if self.segment_boundaries is None or self.n_clusters_effective == 0:
            warnings.warn("固定分段边界未定义或簇数为0，无法确定推理段。返回索引0。")
            return 0
        if inf_param_val_for_segment > self.segment_boundaries[0] + 1e-9: 
            return 0 
        if inf_param_val_for_segment < self.segment_boundaries[-1] - 1e-9: 
            return self.n_clusters_effective - 1 
        for k in range(self.n_clusters_effective):
            upper_b = self.segment_boundaries[k]
            lower_b = self.segment_boundaries[k+1]
            if k == self.n_clusters_effective - 1: 
                if inf_param_val_for_segment >= lower_b - 1e-9 and inf_param_val_for_segment <= upper_b + 1e-9: return k
            else: 
                if inf_param_val_for_segment > lower_b + 1e-9 and inf_param_val_for_segment <= upper_b + 1e-9: return k
        if np.isclose(inf_param_val_for_segment, self.segment_boundaries[0]): return 0
        warnings.warn(f"推理参数值 {inf_param_val_for_segment:.4f} (固定分段) 未清晰落入任何分段。默认分配给分段0。")
        return 0 

    def inference(self, inf_params):
        if not self.local_models or all(m is None for m in self.local_models):
            raise RuntimeError("模型尚未训练，或所有局部模型都训练失败。")

        original_inf_ndim = inf_params.ndim; is_scalar_input = np.isscalar(inf_params)
        current_inf_params_proc = inf_params
        num_train_dims_check = -1
        if self.cluster_centers_or_representatives is not None and self.cluster_centers_or_representatives.ndim > 1:
            num_train_dims_check = self.cluster_centers_or_representatives.shape[1]
        elif self.local_models and self.local_models[0] and self.local_models[0].train_params is not None:
             num_train_dims_check = self.local_models[0].train_params.shape[0]


        if is_scalar_input:
            if num_train_dims_check != -1 and num_train_dims_check != 1:
                raise ValueError(f"标量推理参数提供给了一个多维参数空间模型 (训练时维度: {num_train_dims_check})。")
            current_inf_params_proc = np.array([[inf_params]]) 
        elif current_inf_params_proc.ndim == 1: 
            if num_train_dims_check != -1 and len(current_inf_params_proc) != num_train_dims_check:
                 raise ValueError(f"1D推理参数的维度 ({len(current_inf_params_proc)}) 与训练时维度 ({num_train_dims_check}) 不匹配。")
            current_inf_params_proc = np.expand_dims(current_inf_params_proc, axis=1) 
        
        num_inf_points = current_inf_params_proc.shape[1]
        all_predictions_list = []

        for i_pt in range(num_inf_points):
            single_inf_param_vector = current_inf_params_proc[:, i_pt] 
            target_indices_for_smoothing = []; weights_for_smoothing = []

            if self.fixed_segment_param_idx is not None:
                inf_val_for_segment_dim = single_inf_param_vector[self.fixed_segment_param_idx]
                if not self.use_smooth_transition or self.n_clusters_effective == 1:
                    target_cluster_idx = self._get_cluster_for_inf_param_fixed(inf_val_for_segment_dim)
                    target_indices_for_smoothing.append(target_cluster_idx); weights_for_smoothing.append(1.0)
                else: 
                    param_centers_1d = self.cluster_centers_or_representatives[:, self.fixed_segment_param_idx]
                    distances_1d = np.abs(param_centers_1d - inf_val_for_segment_dim)
                    distances_1d[np.isclose(distances_1d, 0)] = 1e-9 
                    sorted_dist_indices = np.argsort(distances_1d)
                    valid_model_count = 0
                    for k_idx in sorted_dist_indices:
                        if k_idx < len(self.local_models) and self.local_models[k_idx] is not None:
                            target_indices_for_smoothing.append(k_idx); weights_for_smoothing.append(1.0 / distances_1d[k_idx])
                            valid_model_count += 1
                            if valid_model_count >= 2: break 
                    if not target_indices_for_smoothing: 
                        primary_idx = self._get_cluster_for_inf_param_fixed(inf_val_for_segment_dim)
                        if primary_idx < len(self.local_models) and self.local_models[primary_idx] is not None:
                            target_indices_for_smoothing.append(primary_idx); weights_for_smoothing.append(1.0)
                        else: raise RuntimeError(f"固定分段模式下，点 {single_inf_param_vector} 附近没有可用的局部模型。")
            else: 
                distances_kmeans = np.linalg.norm(self.cluster_centers_or_representatives - single_inf_param_vector.reshape(1, -1), axis=1)
                distances_kmeans[np.isclose(distances_kmeans, 0)] = 1e-9
                if not self.use_smooth_transition or self.n_clusters_effective == 1:
                    nearest_cluster_idx = np.argmin(distances_kmeans)
                    target_indices_for_smoothing.append(nearest_cluster_idx); weights_for_smoothing.append(1.0)
                else: 
                    sorted_dist_indices = np.argsort(distances_kmeans)
                    valid_model_count = 0
                    for k_idx in sorted_dist_indices:
                        if k_idx < len(self.local_models) and self.local_models[k_idx] is not None:
                            target_indices_for_smoothing.append(k_idx); weights_for_smoothing.append(1.0 / distances_kmeans[k_idx])
                            valid_model_count +=1
                            if valid_model_count >=2: break
                    if not target_indices_for_smoothing:
                        nearest_idx = np.argmin(distances_kmeans)
                        if nearest_idx < len(self.local_models) and self.local_models[nearest_idx] is not None:
                            target_indices_for_smoothing.append(nearest_idx); weights_for_smoothing.append(1.0)
                        else: raise RuntimeError(f"K-means模式下，点 {single_inf_param_vector} 附近没有可用的局部模型。")
            
            current_point_prediction = 0; sum_of_weights = np.sum(weights_for_smoothing)
            if np.isclose(sum_of_weights, 0) or not target_indices_for_smoothing:
                if target_indices_for_smoothing and target_indices_for_smoothing[0] < len(self.local_models) and self.local_models[target_indices_for_smoothing[0]] is not None:
                    chosen_idx = target_indices_for_smoothing[0]
                    current_point_prediction = self.local_models[chosen_idx].inference(single_inf_param_vector)
                    if np.isclose(sum_of_weights, 0): warnings.warn("平滑过渡权重和为零，回退到使用最近的单个模型。")
                else: raise RuntimeError(f"没有目标模型用于推理 {single_inf_param_vector}。")
            else:
                normalized_weights = np.array(weights_for_smoothing) / sum_of_weights
                for model_idx, weight in zip(target_indices_for_smoothing, normalized_weights):
                    if model_idx >= len(self.local_models) or self.local_models[model_idx] is None:
                        warnings.warn(f"用于平滑过渡的模型索引 {model_idx} 不可用或超出范围。其权重将被忽略。")
                        continue 
                    pred_from_local_model = self.local_models[model_idx].inference(single_inf_param_vector)
                    current_point_prediction += weight * pred_from_local_model
            all_predictions_list.append(current_point_prediction)

        if not all_predictions_list:
            num_features = 0
            # 尝试从一个有效的局部模型获取特征数
            for model_inst in self.local_models:
                if model_inst and model_inst.basis is not None:
                    num_features = model_inst.basis.shape[0]
                    break
            if num_features == 0 and self.local_models and self.local_models[0] and self.local_models[0].snapshot is not None: # Fallback
                 num_features = self.local_models[0].snapshot.shape[0]


            if is_scalar_input or original_inf_ndim == 1: return np.zeros(num_features)
            else: return np.zeros((num_features, num_inf_points))

        final_predictions_array = np.stack(all_predictions_list, axis=1) 
        if is_scalar_input or original_inf_ndim == 1: return final_predictions_array[:, 0] 
        return final_predictions_array

    def print_cluster_summary(self):
        if not self.cluster_info or self.n_clusters_effective == 0:
            print("没有簇/段信息可用或有效簇数为0。请先训练模型。")
            return
        print("\n" + "=" * 90); print("聚类POD-RBF模型摘要"); print("=" * 90)
        mode = "K-means聚类" if self.fixed_segment_param_idx is None else \
               f"固定分段 (参数索引: {self.fixed_segment_param_idx}, 模式: {self.fixed_segment_mode})"
        print(f"聚类/分段模式: {mode}"); print(f"有效簇/段数 (K_eff): {self.cluster_info.get('n_clusters_effective', 'N/A')}")
        print(f"POD能量阈值: {self.energy_threshold}"); print(f"推理时使用平滑过渡: {self.use_smooth_transition}")
        if self.fixed_segment_param_idx is not None and self.cluster_info.get('segment_boundaries') is not None:
            boundaries_str = np.array2string(self.cluster_info['segment_boundaries'], formatter={'float_kind':lambda x: '%.3f' % x})
            print(f"分段边界 (降序): {boundaries_str}")
        print("\n簇/段详情:"); header_format = "{:>12} {:>6} {:>8} {:>12} {:>10} {}"; row_format =    "{:>12} {:>6} {:>8} {:>12} {:>10} {}"
        print("-" * 90); print(header_format.format('簇/段ID', '大小', 'POD维数', '形状因子', '能量', '局部参数范围 / 代表点')); print("-" * 90)
        num_items_to_print = len(self.cluster_info.get('cluster_sizes', []))
        for i in range(num_items_to_print):
            cluster_id_str = f"{i+1}"
            if self.local_models and i < len(self.local_models) and self.local_models[i] is None: cluster_id_str += " (失败)"
            size = self.cluster_info['cluster_sizes'][i] if i < len(self.cluster_info['cluster_sizes']) else 'N/A'
            pod_dim = self.cluster_info['local_pod_dims'][i] if i < len(self.cluster_info['local_pod_dims']) else 'N/A'
            shape_param_val = self.cluster_info['local_shape_params'][i] if i < len(self.cluster_info['local_shape_params']) else None
            energy_val = self.cluster_info['local_energy_captured'][i] if i < len(self.cluster_info['local_energy_captured']) else None
            shape_param_str = f"{shape_param_val:.6f}" if shape_param_val is not None else "N/A"
            energy_str = f"{energy_val:.6f}" if energy_val is not None else "N/A"
            desc_str = ""
            if self.fixed_segment_param_idx is None: 
                if self.cluster_info.get('cluster_centers_or_representatives') is not None and i < self.cluster_info['cluster_centers_or_representatives'].shape[0]:
                    center = self.cluster_info['cluster_centers_or_representatives'][i, :]
                    center_str = np.array2string(center, formatter={'float_kind':lambda x: '%.3f' % x}, separator=', ')
                    desc_str = f"中心点: {center_str}"
                else: desc_str = "中心点: N/A"
            else: 
                if self.cluster_info.get('segment_boundaries') is not None and len(self.cluster_info['segment_boundaries']) > i+1:
                    upper_b = self.cluster_info['segment_boundaries'][i]; lower_b = self.cluster_info['segment_boundaries'][i+1]
                    desc_str = f"分段参数P{self.fixed_segment_param_idx} 定义范围: ({lower_b:.3f}, {upper_b:.3f}]"
                    if self.cluster_info.get('cluster_centers_or_representatives') is not None and i < self.cluster_info['cluster_centers_or_representatives'].shape[0]:
                        rep_pt_val = self.cluster_info['cluster_centers_or_representatives'][i, self.fixed_segment_param_idx]
                        desc_str += f", 代表值: {rep_pt_val:.3f}"
                else: desc_str = "分段范围: N/A"
            print(row_format.format(cluster_id_str, size, pod_dim, shape_param_str, energy_str, desc_str))
        print("-" * 90)


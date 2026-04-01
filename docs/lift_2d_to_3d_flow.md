# Lift 2D-to-3D 处理流程详解

## 整体流程图

```
2D Change Masks (H,W)  ──→  深度估计  ──→  3D点云  ──→  kNN匹配  ──→  评分投票  ──→  3D Mask
    (每相机)              DepthAnything       BackProj        BallTree         多视图融合
```

## 详细步骤

### 1. 输入
- `change_masks`: 每张相机的 2D 变化掩码 (H×W, bool)
- `cameras`: 相机列表 (72个相机实例)
- `gaussians`: 3D Gaussian 模型 (3.8M 个)

### 2. 对每个相机进行处理

```python
for view_id, (cam, mask) in enumerate(zip(cameras, change_masks)):
```

**Step 2a: 深度估计**
```python
depth = base_lifter.estimate_depth(obs)  # H×W 深度图 (米)
```

**Step 2b: 获取变化像素**
```python
pos_pixels = (mask > 0.5) & torch.isfinite(depth) & (depth > 0)
# 从约 250K 变化像素中采样 2048 个
```

**Step 2c: 深度 -> 3D 点云 (反投影)**
```python
# 像素坐标 (u,v) + 深度 d -> 相机坐标系 (x_cam, y_cam, z_cam)
x_cam = (u - cx) / fx * d
y_cam = (v - cy) / fy * d
z_cam = d

# 相机坐标 -> 世界坐标 (使用 Twc 矩阵)
p_world_h = p_cam @ Twc.T
p_world = p_world_h[..., :3] / p_world_h[..., 3:]
# 得到 2048 个 3D 点
```

**Step 2d: kNN 查找 (BallTree)**
```python
# 对每个 3D 点，找最近的 k=8 个 Gaussians
knn_dists, knn_idx = ball_tree.query(p_world, k=8)
# knn_dists: (2048, 8) 每个点到最近8个Gaussian的距离
# knn_idx: (2048, 8) 最近8个Gaussian的索引
```

**Step 2e: Scale-Normalized 距离检查**
```python
# 计算每个 Gaussian 的局部尺度
local_scale = scales[knn_idx]  # (2048, 8, 3)
denom = local_scale.norm(dim=-1)  # (2048, 8) 每个Gaussian的尺度

# 距离 / 尺度 = 尺度归一化距离
d_local = knn_dists / denom  # (2048, 8)

# 如果 d_local < 2.5，认为该Gaussian"靠近"这个像素
close_mask = d_local < 2.5  # (2048, 8) bool
```

**Step 2f: 深度一致性检查**
```python
# 把 Gaussian 中心投影到相机空间，检查深度是否匹配
z_knn = project_to_cam(gaussians[knn_idx])  # Gaussian在相机系的深度
depth_ok = |z_knn - depth_pixel| < tolerance  # 绝对+相对容差
valid = close_mask & depth_ok
```

**Step 2g: 累积评分**
```python
# 对每个 Gaussian，累加它收到的投票数
for each valid (pixel, gaussian) pair:
    seed_score[gaussian] += weight  # weight基于距离
    positive_views[gaussian] += 1   # 收到多少个视图的投票
```

### 3. 多视图融合

```python
# 计算最终分数
pos = lambda_seed * seed_score      # 2.0 * 正投票
neg = lambda_neg * neg_score         # 0.25 * 负投票
score = pos / (pos + neg + 1e-8)    # 归一化分数 [0,1]

# 多视图一致性过滤
keep = (visible_views >= 2)          # 至少被2个视图看到
     & (positive_views >= 1)         # 至少1个视图认为是变化
     & (seed_views >= 1)            # 至少1个视图贡献了种子
     & (ratio >= 0.05)              # positive/visible > 5%

score = torch.where(keep, score, 0)  # 不满足条件的置零

# 阈值分割
changed = score > 0.6
```

### 4. 关键阈值汇总

| 阈值 | 默认值 | 含义 |
|------|--------|------|
| `max_pos` | 2048 | 每张图采样的最大像素数 |
| `k_nn` | 8 | 每个像素找最近多少个Gaussians |
| `local_radius_thresh` | 2.5 | 尺度归一化距离阈值 (× Gaussian大小) |
| `depth_tol_abs` | 0.05m | 深度容差绝对值 |
| `depth_tol_rel` | 0.05 | 深度容差相对值 (× 深度) |
| `min_visible_views` | 2 | Gaussian至少被几个视图看到 |
| `min_positive_views` | 2 | Gaussian至少被几个视图标记为positive |
| `min_seed_views` | 1 | 至少几个视图贡献了种子 |
| `min_positive_ratio` | 0.3 | positive/visible 的最小比例 |
| `final_thresh` | 0.6 | 最终分数阈值 |
| `lambda_seed` | 2.0 | positive 权重 |
| `lambda_neg` | 0.25 | negative 权重 |

### 5. 内存优化: BallTree

使用 sklearn 的 BallTree 而不是直接计算完整距离矩阵:

```python
# 错误方式 (会 OOM)
dists = torch.cdist(pixels_xyz, gaussians)  # (2048, 3.8M) = 30GB

# 正确方式 (BallTree)
ball_tree = BallTree(gaussians.cpu().numpy(), leaf_size=50)
knn_dists, knn_idx = ball_tree.query(pixels_xyz.cpu().numpy(), k=8)
```

BallTree 使用双树算法，不需要计算完整距离矩阵，大大节省内存。

### 6. 调试信息

运行 debug script 后会输出:

```
[DEBUG] score stats: min=0.0000, max=0.9636, mean=0.0000
[DEBUG] cond1 (visible>=2): 57
[DEBUG] cond2 (positive>=1): 38
[DEBUG] cond3 (seed>=1): 38
[DEBUG] cond4 (ratio>=0.05): 38
[DEBUG] keep mask: 6 / 3866162
[DEBUG] positive_views distribution: max=18, >0: 38
```

含义:
- `score max=0.9636`: 最高分数为 0.9636
- `positive_views >0: 38`: 只有 38 个 Gaussians 被至少 1 个视图标记为变化
- `keep mask: 6 / 3866162`: 只有 6 个 Gaussians 通过所有过滤条件

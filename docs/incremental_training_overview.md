# scripts/run_incremental.py 增量训练流程分析

## 一、所需输入文件

### 1. 预训练模型
```
model_path/                    # --model_path 参数
└── point_cloud/
    └── iteration_XXX/       # 自动加载最新 iteration
        └── point_cloud.ply  # 高斯点云模型
```

### 2. 源数据集（Colmap 格式）
```
source_path/                   # --source_path 参数
├── sparse/0/
│   ├── images.bin
│   ├── cameras.bin
│   └── points3D.ply
└── images/                   # --images 参数 (默认 "images")
    └── *.png/jpg
```

---

## 二、命令行参数

```bash
python scripts/run_incremental.py \
    --model_path <预训练模型路径> \
    --source_path <数据集路径> \
    [其他选项]
```

主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 预训练 GaussianModel 路径 (iteration folder) | **必须** |
| `--source_path` | 源数据路径 (COLMAP) | **必须** |
| `--images` | 图像文件夹名 | "images" |
| `--output_dir` | 输出目录 | "./output_incremental" |
| `--iters` | 每时间步迭代数 | 500 |
| `--lr` | 学习率 | 1e-3 |
| `--detect_changes` | 启用变化检测 | True |
| `--change_threshold` | 变化检测阈值 | 0.8 |
| `--densify` | 启用密集化 | False |
| `--use_sg` | 使用 SG features | True |
| `--use_sg_guidance` | 使用 SG 提升引导 | True |

---

## 三、增量训练流程

### 1. 模型加载阶段 (`load_gaussian_model()`)

```
load_gaussian_model()
├── 解析 model_path 和 source_path 参数
├── 创建 GaussianModel (sh_degree=3, sg_degree=3)
├── 创建 Scene 对象
│   └── 调用 Scene.__init__(load_iteration=-1)
│       └── 自动加载 model_path/point_cloud/iteration_XXX/point_cloud.ply
└── 调用 gaussians.training_setup() 配置优化器
```

### 2. 增量训练器初始化 (`IncrementalTrainer`)

```
IncrementalTrainer.__init__()
├── 创建 GGSGaussianAdapter (高斯模型适配器)
├── 创建 GGSRenderAdapter (渲染适配器)
├── 初始化 DinoV2Detector (变化检测)
│   └── 使用 DINOv2 特征比较渲染图与目标图的差异
├── 初始化 SGAwareLifter (2D->3D 提升器)
│   └── 将 2D 变化掩码提升到 3D 空间
└── 创建输出目录
```

### 3. 核心增量更新流程 (`run_incremental_update()`)

```
run_incremental_update()
│
├── Step 1: 初始化 active_mask (所有高斯点设为 active)
│   └── active_mask = torch.ones(N, dtype=torch.bool)
│
├── Step 2: 变化检测 (detect_changes)
│   ├── for each camera:
│   │   ├── render_camera() → 渲染当前模型
│   │   ├── detect_changes() → 比较渲染图 vs 目标图
│   │   │   └── 使用 DINOv2 detector 预测 change_mask
│   │   └── 保存 change_mask
│   │
│   └── 输出: List[change_mask] (每张图像的变化区域)
│
├── Step 3: 2D变化掩码 → 3D高斯掩码 (lift_changes_to_3d)
│   ├── 调用 sg_lifter.lift_with_sg()
│   │   └── 结合深度估计和多视角几何
│   └── 输出: active_mask (仅变化区域的高斯点为 True)
│
├── Step 4: 训练一个时间步 (train_timestep)
│   │
│   └── for i in range(iters_per_timestep):
│       ├── 采样一个相机
│       ├── train_step():
│       │   ├── render_camera() → 渲染图像
│       │   ├── compute_render_loss() → MSE loss
│       │   ├── loss.backward() → 反向传播
│       │   ├── _apply_gradient_mask() → 仅更新 active 高斯
│       │   └── optimizer.step() → 更新参数
│       │
│       └── 定期 densify_and_prune() (训练后)
│
└── Step 5: 保存最终模型
    └── gaussian_model.save_ply("final_model.ply")
```

### 4. 关键机制

#### 变化检测 (Change Detection)
```python
# 使用 DINOv2 特征检测变化
change_mask = detector.predict_change_mask(
    rendered_image.permute(1, 2, 0),  # (H, W, C)
    target_image.permute(1, 2, 0),
)
```

#### 2D→3D 提升 (Lifting)
```python
# 将 2D 变化掩码提升到 3D
result = sg_lifter.lift_with_sg(
    adapter,      # 高斯模型适配器
    cameras,      # 相机列表
    change_masks, # 2D 变化掩码
)
active_mask = result.positive_mask  # 3D 变化掩码
```

#### 梯度掩码 (Gradient Masking)
```python
# 仅对 active 高斯点计算梯度
active_mask = ...  # 变化区域的高斯点

# 梯度清零非 active 区域
param.grad = param.grad * active_mask.view(-1, 1).expand_as(param.grad)
```

#### 密集化与剪枝 (Densify & Prune)
```python
# 仅对 active 区域进行密集化
_densify_active_only(active_indices)

# 剪枝 inactive 区域中透明度低的高斯
prune_mask = (opacity < threshold) & (~active_mask)
```

---

## 四、与普通训练的区别

| 特性 | 普通训练 (train.py) | 增量训练 (run_incremental.py) |
|------|---------------------|-------------------------------|
| 目标 | 从头训练整个场景 | 更新已训练模型的特定区域 |
| 数据 | 完整数据集 | 新图像/变化区域 |
| 更新范围 | 所有高斯点 | 仅 active_mask 覆盖的高斯点 |
| 变化检测 | 无 | 使用 DINOv2 检测变化区域 |
| 2D→3D 提升 | 无 | 使用 Depth-Anything + SG guidance |
| 应用场景 | 初始训练 | 场景更新、增量学习 |

---

## 五、典型运行命令

```bash
# 基本用法
python scripts/run_incremental.py \
    --model_path ./output/my_model/point_cloud/iteration_30000 \
    --source_path /data/DTU/scan1 \
    --iters 500

# 启用密集化
python scripts/run_incremental.py \
    --model_path ./output/my_model \
    --source_path /data/DTU/scan1 \
    --densify \
    --densify_grad_thresh 0.0002

# 自定义参数
python scripts/run_incremental.py \
    --model_path ./output/my_model \
    --source_path /data/DTU/scan1 \
    --iters 1000 \
    --lr 1e-4 \
    --detect_changes True \
    --change_threshold 0.5
```

---

## 六、输出文件

```
output_incremental/
└── final_model.ply   # 更新后的高斯点云模型
```

# train.py 执行流程分析

## 训练流程概览

### 1. 入口 (`if __name__ == "__main__")`)

```
命令行参数解析
    ├── ModelParams: 数据集路径、模型路径、SH degree 等
    ├── OptimizationParams: 学习率、迭代次数、正则化参数等
    └── PipelineParams: 渲染管线参数
```

### 2. 初始化阶段

| 步骤 | 说明 |
|------|------|
| `safe_state()` | 初始化 RNG 状态 |
| `network_gui.init()` | 启动 GUI 服务器（可选） |
| `GaussianModel(sh_degree, sg_degree)` | 创建 3D Gaussian 模型 |
| `Scene(dataset, gaussians)` | 加载场景数据（相机、点云） |
| `gaussians.training_setup(opt)` | 配置优化器（Adam） |

### 3. 核心训练循环 (`for iteration in range(...)`)

```
┌─────────────────────────────────────────────────────────────┐
│  for iteration in range(first_iter, opt.iterations + 1):   │
├─────────────────────────────────────────────────────────────┤
│  1. 更新学习率 (gaussians.update_learning_rate)              │
│                                                             │
│  2. SH degree 递增 (每 1000 iterations)                     │
│     └── gaussians.oneupSHdegree()                          │
│                                                             │
│  3. 随机采样一个相机 viewpoint_cam                          │
│                                                             │
│  4. 渲染 (render 函数)                                      │
│     └── 输入: 相机 + Gaussians → 输出: 图像 + 深度 + 法向量 │
│                                                             │
│  5. 计算损失函数                                            │
│     ├── L1_loss_appearance (RGB 重建损失)                   │
│     ├── depth_normal_loss (深度-法向量一致性)               │
│     ├── ncc_loss (多视角 NCC 损失)                         │
│     └── geo_loss (多视角几何损失)                          │
│     loss = rgb_loss + λ_depth * depth_normal + ...          │
│                                                             │
│  6. 反向传播 (loss.backward())                              │
│                                                             │
│  7. 稠密化与剪枝 (Densification, iteration < densify_until)│
│     ├── gaussians.add_densification_stats()                │
│     ├── gaussians.densify_and_prune()                      │
│     └── gaussians.reset_opacity() (定期重置不透明度)        │
│                                                             │
│  8. 更新 3D Filter (compute_3D_filter)                      │
│                                                             │
│  9. 优化器步骤 (gaussians.optimizer.step())                 │
│                                                             │
│  10. 保存模型 (iteration in saving_iterations)              │
│  11. 保存 checkpoint (iteration in checkpoint_iterations)    │
└─────────────────────────────────────────────────────────────┘
```

### 4. Gaussian Model 的核心参数

每个 3D Gaussian 由以下参数定义（可学习）：

| 参数 | 维度 | 说明 |
|------|------|------|
| `_xyz` | (N, 3) | 3D 位置 |
| `_features_dc` | (N, 3, 1) | DC 颜色特征 |
| `_features_rest` | (N, 3, SH) | Spherical Harmonics |
| `_sg_axis` | (N, SG, 3) | Gaussian 轴方向 |
| `_sg_sharpness` | (N, SG) | Gaussian 锐度 |
| `_sg_color` | (N, SG, 3) | Gaussian 颜色 |
| `_scaling` | (N, 3) | 缩放矩阵 |
| `_rotation` | (N, 4) | 四元数旋转 |
| `_opacity` | (N, 1) | 不透明度 |

### 5. 数据来源

- **Colmap 模式**: 从 `source_path/sparse` 读取稀疏重建结果
- **Blender 模式**: 从 `transforms_train.json` 读取

### 6. 输出

- **Point Cloud**: `point_cloud/iteration_*/point_cloud.ply`
- **Checkpoints**: `chkpnt*.pth`
- **TensorBoard 日志**: 用于可视化训练过程

---

这是一个基于 **3D Gaussian Splatting** 的项目，通过迭代优化 Gaussian 参数来重建场景，最终输出可渲染的 Gaussian 模型。

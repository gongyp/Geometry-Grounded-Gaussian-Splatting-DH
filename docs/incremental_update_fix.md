# 增量更新训练 Bug 修复与简化方案

## 问题描述

增量更新训练在第 20 步左右崩溃，报错：

```
RuntimeError: The size of tensor a (1613705) must match the size of tensor b (1619056) at non-singleton dimension 0
```

**根本原因**：`filter_3D` 与优化器参数不同步，剪枝时 `_prune_optimizer` 可能部分修改优化器状态后失败，导致状态不一致。

## 修复方案

### 1. `scene/gaussian_model.py` - prune_points 设备同步

添加防御性形状检查和设备同步：

```python
def prune_points(self, mask):
    # DEFENSIVE: 确保所有辅助张量形状正确
    if hasattr(self, 'xyz_gradient_accum') and self.xyz_gradient_accum.shape[0] != model_size:
        self.xyz_gradient_accum = torch.zeros((model_size, 1), device=first_param.device)
    # ... 类似检查 filter_3D, denom, max_radii2D ...

    # 修剪优化器参数
    optimizable_tensors = self._prune_optimizer(valid_points_mask)

    # 确保所有非优化器张量设备一致
    if hasattr(self, 'filter_3D'):
        if self.filter_3D.device != first_param.device:
            self.filter_3D = self.filter_3D.to(device=first_param.device)
        self.filter_3D = self.filter_3D[valid_points_mask]
```

### 2. `scene/gaussian_model.py` - densification_postfix 设备修复

修复硬编码 `"cuda"` 问题：

```python
# 修改前
device = "cuda"  # 硬编码

# 修改后
device = self.get_xyz.device  # 动态获取
```

### 3. `incremental/trainer.py` - 训练循环外剪枝

将剪枝/稠密化从循环内部移到循环结束后：

```python
def train_timestep(self, cameras, target_images):
    all_metrics = []

    # 训练循环 - 内部不进行任何剪枝/稠密化
    for i in range(self.cfg.iters_per_timestep):
        metrics = self.train_step(camera, target)
        all_metrics.append(metrics)

    # 剪枝和稠密化在训练循环结束后执行
    if self.cfg.prune_enabled and self._global_step % self.cfg.prune_every == 0:
        self._prune_only()

    if self.cfg.densify_enabled:
        self.densify_and_prune()

    return all_metrics
```

**原因**：在训练循环内部修改优化器参数会破坏 autograd 图。

### 4. `incremental/trainer.py` - 简化架构

直接使用 `gaussian_model` 而非 adapter：

| 方法 | 修改 |
|------|------|
| `render_camera` | 直接调用 `gaussian_renderer.render` |
| `train_step` | 直接使用 `gaussian_model.optimizer.step()` |
| `densify_and_prune` | 直接调用 `gaussian_model.densify_and_prune()` |
| `_prune_only` | 直接调用 `gaussian_model.prune_points()` |

## 文件修改清单

| 文件 | 方法 | 修改内容 |
|------|------|----------|
| `scene/gaussian_model.py` | `prune_points` | 添加设备同步和防御性形状检查 |
| `scene/gaussian_model.py` | `densification_postfix` | 修复硬编码 "cuda" 设备问题 |
| `incremental/trainer.py` | `train_timestep` | 将剪枝从循环内部移到循环结束后 |
| `incremental/trainer.py` | `render_camera` | 直接调用 gaussian_renderer |
| `incremental/trainer.py` | `train_step` | 直接使用 gaussian_model |
| `incremental/trainer.py` | `densify_and_prune` | 直接调用 gaussian_model.densify_and_prune() |
| `incremental/trainer.py` | `_prune_only` | 直接调用 gaussian_model.prune_points() |

## 当前状态

- ✅ 剪枝正常工作
- ✅ 稠密化正常工作
- ✅ 训练循环外执行避免 autograd 图损坏
- ⚠️ 变化检测仍使用 cl-splats（DinoV2 + DepthAnything）

## 测试结果

```
iters=100, prune_every=100
Final Gaussians: 1613702 (原 1619056)
Pruned 66 Gaussians
Densification/pruning complete.
Training complete! 无错误
```

## 训练命令

### 全量训练

```bash
python train.py -s <source_path> -m <model_output_path> --Iterations <iterations>
```

**示例**：
```bash
python train.py -s eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate -m output/Meetingroom-Localupdate --Iterations 30000
```

### 增量更新训练

```bash
python scripts/run_incremental.py \
    --model_path <pretrained_model_path> \
    --source_path <source_data_path> \
    --iters <iterations_per_timestep> \
    --densify \
    --densify_grad_thresh <grad_threshold> \
    --use_sg \
    --use_sg_guidance
```

**参数说明**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必需 | 预训练模型路径 |
| `--source_path` | 必需 | 源数据路径 |
| `--iters` | 500 | 每个时间步的迭代次数 |
| `--densify` | False | 启用稠密化 |
| `--densify_grad_thresh` | 0.0002 | 稠密化梯度阈值 |
| `--use_sg` | False | 使用 SG 特征 |
| `--use_sg_guidance` | False | 使用 SG 引导 |
| `--detect_changes` | True | 启用变化检测 |
| `--change_threshold` | 0.8 | 变化检测阈值 |

**示例**：
```bash
# 基础增量更新（无稠密化/剪枝）
python scripts/run_incremental.py \
    --model_path output/Meetingroom-Localupdate \
    --source_path eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate \
    --iters 500

# 启用稠密化和变化检测
python scripts/run_incremental.py \
    --model_path output/Meetingroom-Localupdate \
    --source_path eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate \
    --iters 500 \
    --densify \
    --densify_grad_thresh 0.0002

# 自定义迭代次数
python scripts/run_incremental.py \
    --model_path output/Meetingroom-Localupdate \
    --source_path eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate \
    --iters 200 \
    --densify
```

## 注意事项

1. **增量更新仍依赖 cl-splats**：变化检测（DinoV2）和深度提升（DepthAnything）仍使用 cl-splats 实现
2. **稠密化/剪枝在循环外执行**：避免破坏 autograd 图
3. **filter_3D 自动处理**：`densification_postfix` 会自动扩展 filter_3D

## 日期

- 2026-03-25: 首次修复（禁用稠密化/剪枝）
- 2026-03-25: 尝试启用稠密化/剪枝（失败）
- 2026-03-26: 最终修复成功（循环外执行 + 设备同步）

# CL-Splats 项目架构与设计思路

## 一、项目目标

**核心问题**：如何在3D Gaussian Splatting (3DGS) 场景中实现增量更新（Continual Learning），即：
- 给定t0时刻的场景表示
- 当场景发生变化时（添加/删除/移动物体），只更新变化区域
- 不重新训练整个场景，保持t0的表示不变

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CL-Splats Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │   Dataset    │───▶│   Trainer    │───▶│   Output     │                │
│  │  (场景数据)   │    │  (主控制器)   │    │ (高斯/渲染)  │                │
│  └──────────────┘    └──────────────┘    └──────────────┘                │
│         │                   │                                              │
│         ▼                   ▼                                              │
│  ┌──────────────────────────────────────────┐                              │
│  │            Core Components              │                              │
│  │  ┌────────────┐ ┌────────────┐ ┌──────┴─────┐                      │
│  │  │   DinoV2   │ │  Depth    │ │  Primitives │                      │
│  │  │  Detector  │ │ Anything   │ │ (几何约束)   │                      │
│  │  │(2D变化检测)│ │ Lifter    │ │             │                      │
│  │  └────────────┘ └────────────┘ └─────────────┘                      │
│  │        │              │               │                              │
│  │        ▼              ▼               ▼                              │
│  │  ┌─────────────────────────────────────────┐                          │
│  │  │          CLGaussians (3DGS)             │                          │
│  │  │  positions, scales, quats, sh, opacity   │                          │
│  │  └─────────────────────────────────────────┘                          │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、增量更新核心流程

```
时间 t0 (基础训练)                    时间 t1+ (增量更新)
┌─────────────────────┐            ┌─────────────────────────────┐
│ 1. 加载点云          │            │ 1. 加载新时刻相机            │
│ 2. 初始化Gaussians  │            │ 2. 渲染当前Gaussians       │
│ 3. 标准3DGS训练     │            │ 3. DinoV2变化检测          │──────┐
│    (所有Gaussian)    │            │    (rendered vs GT)       │      │
└─────────────────────┘            │ 4. Depth-Anything提升       │      │
                                   │    (2D mask → 3D Gaussian)  │──────┼──▶ 变化高斯
                                   │ 5. 拟合几何原语(球/OBB)     │      │      掩码
                                   │ 6. 局部优化 + 约束          │      │
                                   │    (只优化变化区域)         │      │
                                   │ 7. 稠密化 + 剪枝           │      │
                                   └─────────────────────────────┘      │
                                                                      │
                              ┌────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ 最终输出:            │
                    │ - 基础区域保持不变   │
                    │ - 变化区域更新       │
                    └─────────────────────┘
```

---

## 四、核心组件详解

### 1. Trainer (trainer.py) - 中央控制器

**职责**：
- 协调整个pipeline
- 管理训练循环
- 调用变化检测，提升、约束

**关键方法**：

```
├── prepare_timestep(t)    # 准备时刻t的训练
│   ├── 如果t==0: 初始化所有Gaussian
│   └── 如果t>0: 执行detect→lift→constrain
├── _render_camera()        # 使用gsplat渲染
├── _train_step()          # 单步训练
│   ├── 渲染 +光度损失
│   ├── 几何约束损失(可选)
│   ├── 梯度掩码(只更新active)
│   └── 稠密化+剪枝
└── evaluate()             # PSNR/SSIM评估
```

### 2. CLGaussians (cl_gaussians.py) - 3DGS表示

**Gaussian参数**：
```
├── positions  (N,3)  位置
├── scales     (N,3)  尺度 (log空间)
├── quats      (N,4)  旋转四元数
├── sh_features (N,3,K) 球谐特征
└── opacity    (N,1)  不透明度 (logit空间)
```

**关键方法**：
```
├── prune_gaussians(mask)     # 剪枝
├── split_gaussians(mask)    # 稠密化(分裂+克隆)
└── export_ply()            # 导出ply文件
```

### 3. DinoV2Detector (dinov2_detector.py) - 2D变化检测

**原理**：比较渲染图和真实图的DINOv2特征相似度

**流程**：
1. 图像预处理 (resize到14倍数 + ImageNet归一化)
2. 提取DINOv2特征
3. 计算余弦相似度
4. 相似度 < 阈值 → 变化像素

**输入**：rendered_image [H,W,3], observation [H,W,3]
**输出**：change_mask [H,W] (bool)

### 4. DepthAnythingLifter (depth_anything_lifter.py) - 2D→3D提升

**问题**：2D变化检测只知道哪些像素变化，不知道哪些3D结构变化

**解决方案**：多视角几何提升

**流程**：
1. 深度估计 (Depth-Anything V2)
2. 2D→3D反投影 (像素 → 相机坐标 → 世界坐标)
3. kNN搜索 (找到最近的K个Gaussian)
4. 深度一致性检查 (投影深度 vs Gaussian深度)
5. 证据累积 (多视角投票)
6. 多视角一致性过滤

**输出**：changed_gaussians (N,) bool - 哪些Gaussian受影响

### 5. Primitives (primitives.py) - 几何约束

**作用**：限制变化区域的几何范围，防止过度漂移

**方法**：
```
├── fit_sphere(points)        # 拟合球体
├── fit_obb(points)          # 拟合有向包围盒
├── group_active_gaussians() # 高斯分组(连通分量)
├── fit_primitives_for_active() # 为active高斯拟合原语
├── distance_to_primitive()   # 点到原语距离
└── union_distance()         # 到所有原语的最小距离
```

**在训练中**：
- 计算active Gaussian到原语的距离
- 添加距离约束 loss
- 超出阈值多次后剪枝

---

## 五、文件关系图

```
                          ┌─────────────────────────┐
                          │     run_test_scene.py   │
                          │     (入口脚本)          │
                          └───────────┬─────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────┐
                          │     dataset_reader.py    │
                          │  - SceneInfo            │
                          │  - CameraInfo          │
                          │  - BasicPointCloud     │
                          │  - readColmapSceneInfo │
                          │  - readNerfSyntheticInfo│
                          └───────────┬─────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────┐
                          │       trainer.py         │
                          │    CLSplatsTrainer      │
                          │  ◄───────────────│      │
                          │         │        │      │
                          │         ▼        │      │
                          │  ┌──────────────┴───┐  │
                          │  │                  │  │
                          │  ▼                  ▼  │
            ┌─────────────┴──┐    ┌───────────────┴──┐
            │    cameras.py   │    │ cl_gaussians.py  │
            │    Camera类     │    │  CLGaussians    │
            │  - R, T        │    │  - params       │
            │  - K (内参)    │    │  - optimizer    │
            │  - image       │    │  - prune()     │
            │  - world_view_ │    │  - split()      │
            │    transform    │    │  - export_ply()│
            └─────────────────┘    └─────────────────┘
                    │                       ▲
                    │                       │
                    ▼                       │
┌─────────────────────────────────────────────────────────────┐
│              change_detection/                              │
│  ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │ base_detector.py    │    │  dinov2_detector.py     │  │
│  │ (抽象基类)          │───▶│  DinoV2Detector        │  │
│  │ predict_change_mask │    │  - predict_change_mask   │  │
│  └─────────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              lifter/                                         │
│  ┌─────────────────────┐    ┌───────────────────────────┐ │
│  │ base_lifter.py      │    │ depth_anything_lifter.py  │ │
│  │ (抽象基类)          │───▶│ DepthAnythingLifter      │ │
│  │ lift()              │    │ - estimate_depth()       │ │
│  └─────────────────────┘    │ - lift()                 │ │
│                             │   (2D→3D提升)            │ │
│                             └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              constraints/                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    primitives.py                       │  │
│  │  - SpherePrimitive, OBBPrimitive                      │  │
│  │  - fit_sphere(), fit_obb()                           │  │
│  │  - group_active_gaussians()                          │  │
│  │  - fit_primitives_for_active()                      │  │
│  │  - distance_to_primitive(), union_distance()          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

gsplat (第三方库)
├── rasterization()  # 渲染
├── Strategy         # 策略
└── exporter        # 导出
```

---

## 六、各文件作用汇总

| 文件 | 作用 | 关键类/函数 |
|------|------|------------|
| `trainer.py` | 中央控制器，协调整个训练流程 | `CLSplatsTrainer` |
| `cl_gaussians.py` | 3D Gaussian表示和管理 | `CLGaussians`, `GaussianParams` |
| `cameras.py` | 相机数据结构 | `Camera` |
| `dataset_reader.py` | 数据加载(Colmap/NeRF格式) | `SceneInfo`, `CameraInfo`, `readNerfSyntheticInfo` |
| `dinov2_detector.py` | 2D变化检测 | `DinoV2Detector` |
| `depth_anything_lifter.py` | 2D→3D提升 | `DepthAnythingLifter` |
| `primitives.py` | 几何约束(球/OBB) | `fit_primitives_for_active`, `union_distance` |
| `config.py` | 配置管理 | `CLSplatsConfig` |
| `run_test_scene.py` | 入口脚本 | `main()` |

---

## 七、增量更新关键设计

### 1. 变化检测 (Detect)
```python
rendered = render(gaussians, camera)      # 渲染当前估计
gt = camera.original_image                # 真实图像
mask = detector(rendered, gt)            # DinoV2变化检测
```

### 2. 提升 (Lift)
```python
depth = estimate_depth(observation)      # Depth-Anything估计深度
changed_3d = lift(mask, depth, cameras)  # 2D mask → 3D Gaussian mask
```

### 3. 约束 (Constrain)
```python
primitives = fit_primitives(positions[changed_3d])  # 拟合几何原语
loss = photometric_loss + λ * distance_to_primitives  # 添加几何约束
```

### 4. 局部优化 (Optimize)
```python
# 只优化active Gaussian
grad *= active_mask  # 梯度掩码
optimizer.step()     # 更新参数
```

### 5. 稠密化+剪枝 (Densify & Prune)
```python
split_gaussians(active_mask)           # 在变化区域稠密化
prune_gaussians(outside_primitives)      # 剪枝漂移的高斯
```

---

## 八、核心设计思想

这个架构的核心思想是：**只更新变化区域，保持基础场景不变**，通过2D变化检测→3D提升→几何约束的流程实现高效的增量学习。

**关键创新点**：

1. **DinoV2特征相似度**：使用预训练ViT的特征相似度而非像素级差异，更鲁棒
2. **Depth-Anything多视角提升**：将2D变化准确映射到3D Gaussian
3. **几何原语约束**：防止变化区域过度漂移
4. **局部优化**：只更新变化区域，保持基础场景质量

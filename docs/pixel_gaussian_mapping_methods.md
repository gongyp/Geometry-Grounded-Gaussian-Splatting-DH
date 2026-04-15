# 像素到 3D Gaussian 映射方法与增量更新方案（修订版）

本文档总结从 2D 图像变化区域定位对应 3D Gaussian 的方法，分析项目原始实现 `incremental/lifter/depth_anything_lifter.py` 的思路与局限，并给出更适合真实场景增量更新的推荐方案。

---

## 1. 问题背景

在增量更新（Incremental Update）场景中，通常需要完成以下步骤：

1. 对比渲染图像与真实图像，检测变化区域
2. 将 2D 变化掩码提升到 3D 空间
3. 判断哪些已有 Gaussian 需要更新
4. 判断哪些变化区域无法由现有 Gaussian 解释，从而触发 densification / 新 Gaussian 增加

核心问题不是“每个像素唯一对应哪个高斯”，而是：

> **一个变化像素在当前视图下，主要由哪些 Gaussian 负责？各自责任有多大？**

这一定义比“唯一匹配”更符合 Gaussian Splatting 的真实渲染机制，因为一个像素往往由多个 Gaussian 共同贡献。

---

## 2. 问题重述：我们真正想要什么

对于变化像素集合 \(M\)，理想输出应包括两部分：

1. **已有 Gaussian 的责任分配**
   - 哪些 Gaussian 对这些变化像素负责
   - 每个 Gaussian 的累积责任分数是多少

2. **无法解释的变化区域**
   - 当前模型没有足够高置信度 Gaussian 能解释的像素
   - 这些区域不应强行绑定到旧 Gaussian，而应作为新增几何候选

因此，pixel-to-Gaussian mapping 更准确地说应是：

> **pixel → Top-K contributing Gaussians + responsibility weights**

而不是：

> pixel → one Gaussian

---

## 3. 方法体系重组

相比把所有方法简单并列，更合理的组织方式是分为 4 类。

### A. 粗筛类方法
用于快速缩小候选 Gaussian 集合，但不直接给出精确归因。

1. 渲染器可见性筛选
2. 椭圆投影覆盖法
3. 深度反投影 + 邻近搜索

### B. 精确归因类方法
真正输出像素到 Gaussian 的贡献或责任关系。

4. 渲染器中间量导出 / Contributor Buffer
5. 解析责任度计算（考虑 compositing）

### C. 辅助证据类方法
不是 pixel-to-Gaussian 的主映射，而是帮助确定哪些 Gaussian 更值得更新。

6. 差分反向传播 / 梯度归因
7. 残差驱动的重要性评分

### D. 数据结构与加速类方法
用于多次查询、增量迭代和大规模场景优化。

8. 倒排索引 / Tile-based incidence map
9. Top-K contributor cache

---

## 4. 当前常见 6 种方法的修订分析

下面对已有文档中列出的 6 种方法进行修订。

---

## 4.1 方案 1：渲染器可见性筛选

### 原理
利用渲染器输出的可见性信息，例如 `radii > 0`、frustum culling 或其他 rasterization 阶段的可见标志，得到当前视图下可能相关的 Gaussian 集合。

### 作用定位
这不是严格意义上的 pixel-to-Gaussian 映射，而是：

> **view-level candidate pruning（视图级候选裁剪）**

### 优点
- 简单
- 快速
- 与当前视角一致
- 很适合作为所有后续方法的前置过滤

### 缺点
- 不能提供像素级对应关系
- 无法区分“几何上可见”与“实际对像素有主要贡献”

### 结论
应保留，但只应作为粗筛步骤，而不应单独作为最终映射方法。

---

## 4.2 方案 2：唯一 ID 编码

### 原理
为每个 Gaussian 分配唯一标识，并尝试通过渲染后像素值恢复对应 Gaussian ID。

### 需要区分两种情况

#### 情况 A：普通颜色编码 + 常规混合渲染
若只是把每个 Gaussian 的颜色替换成唯一 RGB 编码，再按普通 Gaussian Splatting 流程渲染，那么像素值通常是多个 Gaussian 混合的结果。

此时：
- 颜色会混合
- 多个 ID 会叠加
- 无法稳定反解出唯一 Gaussian

因此这种做法**不严格可靠**。

#### 情况 B：修改渲染器输出 ID Buffer / Contributor Buffer
若修改 rasterizer，使其直接输出：
- top-1 contributor ID
- top-K contributor IDs
- 或每像素 contributor 列表

则这是一个很强的工程方案。

### 结论
“唯一 ID 编码”本身并不天然精确；
真正可行的是：

> **渲染器显式输出 contributor index buffer**

---

## 4.3 方案 3：解析贡献度/责任度计算

### 原理
根据 Gaussian 在当前视图中的投影、opacity、深度顺序和 alpha compositing 过程，计算每个 Gaussian 对每个像素的实际贡献。

### 关键点
如果只按二维 Gaussian 核值判断：

\[
\text{kernel}(p,g) = \exp\left(-\frac{1}{2} d^2_{mah}(p,g)\right)
\]

那么得到的更接近“几何覆盖近似”。

更合理的责任应考虑 compositing：

\[
w_{p,g} \propto T^{before}_{p,g} \cdot \alpha_{p,g}
\]

其中：
- \(\alpha_{p,g}\) 为该像素处 Gaussian 的 alpha
- \(T^{before}_{p,g}\) 为该 Gaussian compositing 前的累计透射率

再归一化可得责任度：

\[
r_{p,g} = \frac{w_{p,g}}{\sum_{g' \in \mathcal{G}(p)} w_{p,g'} + \epsilon}
\]

### 优点
- 与真实渲染过程最接近
- 适合做 soft assignment
- 数学定义清晰

### 缺点
- 实现难度较高
- 计算量大
- 若在渲染器外部重建 compositing，成本较高

### 结论
这是理论上最合理的方法之一，但最好结合 rasterizer 内部信息实现，而不是完全在外部重算。

---

## 4.4 方案 4：差分反向传播 / 梯度归因

### 原理
对当前渲染与 GT 之间的残差定义损失，对 Gaussian 参数求梯度，依据梯度大小判断哪些 Gaussian 与变化更相关。

### 本质
它回答的是：

> 哪些 Gaussian 对当前误差更敏感？

而不是：

> 某个像素由哪些 Gaussian 组成？

### 优点
- 可微
- 可与训练/优化流程自然结合

### 缺点
- 梯度依赖损失定义
- 受当前参数状态影响
- 不能直接等价为像素覆盖或真实贡献

### 结论
适合作为辅助优先级估计，不适合作为主 mapping 方法。

---

## 4.5 方案 5：尺度校准的单目深度估计

### 原理
用单目深度网络估计像素深度，做尺度校准后反投影到 3D，再与 Gaussian 做邻近匹配。

### 现实问题
即使做了尺度校准，单目深度仍常存在：
- 全局 scale 偏差
- shift 偏差
- 局部非线性误差
- 边缘错位
- 遮挡错误

因此它更像是：

> **几何弱先验**

而不是精确 pixel-to-Gaussian 映射依据。

### 结论
可以作为辅助约束或排序依据，不建议作为主路径。

---

## 4.6 方案 6：椭圆投影覆盖法

### 原理
将 3D Gaussian 投影到当前图像平面，得到二维椭圆 support region；判断像素是否落在其一定 Mahalanobis 距离阈值内。

### 数学形式
Gaussian 投影中心记为 \(\mu_{2d}\)，协方差为 \(\Sigma_{2d}\)。对像素 \(p\)：

\[
d^2_{mah}(p,g) = (p - \mu_{2d})^T \Sigma_{2d}^{-1} (p - \mu_{2d})
\]

若 \(d^2_{mah}(p,g) < \tau\)，则视为该 Gaussian 覆盖该像素。

### 本质定位
这是：

> **几何覆盖候选生成方法**

不是严格的最终责任归因。

### 优点
- 不依赖单目深度尺度
- 与当前视角的投影几何一致
- 非常适合作为候选集生成

### 缺点
- 容易过选
- 不考虑真实遮挡与 compositing 时，会把“几何上覆盖但视觉上几乎不贡献”的 Gaussian 也包含进来

### 结论
方案 6 很重要，但更适合作为高质量候选生成器，而不是终极精确方案。

---

## 5. 原始实现 `depth_anything_lifter.py` 分析

项目原始实现路径为：

1. 用 Depth-Anything 预测单目深度
2. 对变化像素反投影得到 3D 点
3. 在 Gaussian 均值上做 kNN
4. 用局部尺度归一化距离加权投票
5. 多视图累计正负证据

这一路径的优点是：
- 保留了 soft voting
- 支持多视图聚合
- 有正负证据对冲机制

但其问题也非常明显。

---

## 5.1 原始实现的真实本质

它并不是在做“真实渲染空间的像素归因”，而是在做：

\[
\text{pixel} \rightarrow \hat{X}_{3D}^{mono} \rightarrow kNN(\mu_g)
\]

即：
- 先把像素提升为一个由单目深度决定的 3D 点
- 再找最近的 Gaussian 中心

而不是：

\[
\text{pixel} \rightarrow \{\text{当前视图下真正负责该像素的 Gaussian}\}
\]

因此它更像是：

> **单目深度驱动的 3D 近邻投票器**

而不是渲染一致的 pixel-to-Gaussian 归因器。

---

## 5.2 原始实现存在的关键问题

### 问题 1：把单目深度当作几何 lifting 主坐标系
在真实场景中，单目深度经常存在 scale / shift / local distortion / edge misalignment 等问题。
一旦反投影点偏了，后续 kNN 会系统性选错 Gaussian。

### 问题 2：depth consistency 实际被关闭
代码中明确写了由于深度尺度/偏移问题，depth consistency check 被禁用，最终等价于：

- 用单目深度生成 3D 点
- 但不再验证其与当前 Gaussian 几何是否一致

这样会把大量错误 3D 位置直接传递给 kNN 过程。

### 问题 3：kNN 是对 Gaussian 中心做的，而不是对真实贡献做的
Gaussian 对像素的责任依赖于：
- 2D 投影位置
- footprint / covariance
- opacity
- 深度排序
- compositing

而不是简单的 3D 中心最近。

### 问题 4：缺少视图可见性与投影约束
当前实现没有优先用“当前视图下可见且投影相关”的 Gaussian 做候选，而是直接在全局 Gaussian 均值上做近邻查询。

### 问题 5：负样本设计过粗
原实现从整张非变化区域随机采样负像素，这会导致：
- 与变化区域无关的背景像素也给出负证据
- 负证据稀释正证据
- 对小变化区域尤其不友好

更合理的负样本应来自变化边界附近或竞争候选高斯对应区域。

### 问题 6：`visible_views` 并不是真正的可见视图计数
它本质记录的是“被 query 命中过多少次”，而不是“在多少视图中真实可见”。
因此后续的多视图过滤条件统计含义并不严格。

### 问题 7：FAISS 距离的数值语义需要特别检查
FAISS 的 L2 索引通常返回的是平方 L2 距离。若直接把它当普通欧氏距离使用，则：
- 距离阈值失真
- 权重衰减过快
- `local_radius_thresh` 的物理意义被破坏

这是原始实现中非常值得优先确认的数值问题。

### 问题 8：失败时“少于 50 个 changed 就全选”会掩盖真实效果
该 fallback 会把“没检测到”与“全体误报”混在一起，不利于分析实际性能。

---

## 5.3 为什么在 `Real-World/cone_pinhole` 上效果容易差

对于真实拍摄数据，以下因素会进一步放大上述问题：

1. 单目深度在边缘、阴影、弱纹理、遮挡区域容易不稳定
2. 新出现/消失物体往往位于边界区域，最容易发生前后景错绑
3. pinhole 透视下深度误差引起的 3D 反投影偏移更明显
4. 某些变化区域本质上是“当前模型无法解释的新几何”，不应强行绑定到旧 Gaussian

因此原始算法在真实场景中失败，并不只是因为参数不合适，而是主假设本身太脆弱。

---

## 6. 重新定义推荐目标

结合原始实现经验，推荐目标不应再是：

> 用单目深度反投影得到 3D 点，再找最近 Gaussian

而应改为：

> 在当前视图的渲染/投影空间中，先找到真正可能负责该像素的 Gaussian，再做多视图证据聚合。

也就是说：
- **保留原始 lifter 的多视图 evidence aggregation 框架**
- **替换 seed 生成机制**

---

## 7. 推荐方案（最终建议）

# 7.1 总体推荐

### 最优工程方案

> **可见性粗筛 + 渲染器内部导出每像素 Top-K contributor / responsibility map + 多视图正负证据聚合 + unexplained change 分支**

这是当前最推荐的方案。

---

## 7.2 核心思想

保留原始实现中合理的部分：
- 多视图正负证据累计
- soft assignment
- per-Gaussian 评分
- 多视图一致性过滤

替换原始实现中不合理的 seed 生成机制：
- 不再依赖“单目深度反投影点 → 3D kNN”
- 改为“当前视图下真实或近似 contributor 查询”

---

## 7.3 推荐方案 A：Renderer-aware Multi-view Responsibility Lifter（首选）

### 步骤 1：可见性粗筛
在当前视图中先用 rasterizer 获得：
- 可见 Gaussian 集合
- 或至少 `radii > 0` / frustum-visible 集合

作为候选集剪枝。

### 步骤 2：前向渲染时输出 contributor buffer
建议在 rasterization / compositing 过程中，记录每像素：
- `topk_gaussian_ids[p, 1:K]`
- `topk_weights[p, 1:K]`
- 可选 `pixel_depth`
- 可选 `pixel_alpha`
- 可选 per-pixel residual

其中权重建议定义为：

\[
w_{p,g} = T^{before}_{p,g} \cdot \alpha_{p,g}
\]

### 步骤 3：构建 responsibility map
对每个变化像素 \(p\)：

\[
r_{p,g} = \frac{w_{p,g}}{\sum_{j=1}^{K} w_{p,g_j} + \epsilon}
\]

然后对 Gaussian 累积：

\[
S_g = \sum_{p \in M} r_{p,g}
\]

### 步骤 4：局部负证据
负证据不再从整图随机采样，而改用：
- 变化边界外一圈 ring 作为 hard negative
- 或当前模型高置信解释的未变化像素

### 步骤 5：多视图汇总
沿用原始 lifter 的思想，对各视图累计：
- `seed_score`
- `neg_score`
- `positive_views`
- `visible_views`

并计算：

\[
score_g = \frac{\lambda_{seed} S^{+}_g}{\lambda_{seed} S^{+}_g + \lambda_{neg} S^{-}_g + \epsilon}
\]

### 步骤 6：unexplained change 分支
若某些变化像素满足：
- 候选 Gaussian 的总责任太低
- 或多视图始终不稳定
- 或在当前模型中没有高置信 contributor

则不要强行绑定到旧 Gaussian，而是标记为：
- unexplained change
- densification / new Gaussian proposal 区域

### 优点
- 与真实渲染过程一致
- 避免单目深度主导几何定位
- 支持一个像素对应多个 Gaussian
- 天然适合增量更新
- 适合真实场景数据

---

## 7.4 推荐方案 B：不改渲染器时的次优方案

若短期内不方便修改 rasterizer，可采用：

> **可见性筛选 + 2D 椭圆投影候选 + 软责任分配 + 单目深度弱约束**

### 做法
1. 先筛出当前视图下可见 Gaussian
2. 将可见 Gaussian 投影到 2D 图像平面
3. 对每个变化像素，只在覆盖该像素的投影 Gaussian 中找候选
4. 用以下近似权重：

\[
w_{p,g} =
\exp\left(-\frac12 d^2_{mah}(p,g)\right)
\cdot \alpha_g
\cdot \exp\left(-\lambda_z |z_g - \hat{z}_p|\right)
\]

其中：
- \(d^2_{mah}(p,g)\)：2D 椭圆 Mahalanobis 距离
- \(\alpha_g\)：Gaussian opacity
- \(z_g\)：Gaussian 在当前相机下深度
- \(\hat{z}_p\)：单目深度估计，仅作弱约束

5. 对候选 Gaussian 做归一化，得到 soft responsibility
6. 保留多视图聚合与局部负证据机制

### 关键思想
在这个方案中，单目深度不再负责“定义 3D query 点”，而只是参与排序或过滤。

这比原始 `depth_anything_lifter.py` 稳定得多。

---

## 7.5 不推荐继续作为主方案的路径

以下路径不建议继续作为主线：

1. **纯单目深度反投影 + 3D kNN**
2. **普通颜色编码 ID 渲染**
3. **纯梯度归因替代映射**
4. **整图随机负样本的正负对冲**

---

## 8. 分阶段落地建议

---

## 8.1 P0：先修正原始实现中的关键问题

若暂时还沿用 `depth_anything_lifter.py`，建议优先处理：

1. 检查 FAISS 距离是否为平方 L2，必要时开方
2. 去掉“changed 数量 < 50 就全选”的 fallback
3. 将负样本从“全图随机采样”改成“局部 ring negative”
4. 将 `visible_views` 改为真实可见性统计，而不是 query hit 次数

这些修改不能从根本上解决问题，但能避免明显的数值与统计偏差。

---

## 8.2 P1：保留 Depth-Anything，但降级为弱先验

推荐从：
- 单目深度反投影成世界点 → 全局 kNN

改为：
- 2D 投影空间先找候选 Gaussian
- 再用单目深度做深度一致性 gating 或 soft ranking

这一步是从“depth-driven lifting”转向“projection-driven lifting”的关键。

---

## 8.3 P2：最终升级到 renderer-aware responsibility map

这是最推荐的长期方向：
- 在 renderer 内部直接记录 contributor 信息
- 输出 top-K IDs 与权重
- 与现有多视图 evidence aggregation 结合

此时整个系统将从“近似几何匹配”升级为“渲染一致责任归因”。

---

## 9. 最终结论

### 结论 1：原始 6 种方法需要修订
主要问题在于：
- 过度高估了 ID 编码和椭圆覆盖法的“精确性”
- 混淆了“几何覆盖”和“真实渲染贡献”
- 低估了 rasterizer 内部中间量导出的重要性

### 结论 2：`depth_anything_lifter.py` 的核心问题不是调参，而是主假设错误
其本质是“单目深度驱动的 3D 近邻投票”，而不是“渲染一致的像素归因”，因此在真实场景数据上容易系统性失效。

### 结论 3：最优方案不是继续强化深度 lifting，而是替换为渲染/投影空间归因
最优建议为：

> **可见性粗筛 + 渲染器内部 Top-K contributor / responsibility map + 多视图正负证据聚合 + unexplained change 分支**

若短期内无法修改渲染器，则推荐：

> **可见性筛选 + 2D 椭圆投影候选 + soft responsibility + 单目深度弱约束**

这比当前原始实现更贴合 Gaussian Splatting 的真实渲染机制，也更适合真实场景的增量更新。

---

## 10. 推荐实施摘要

### 短期可做
- 修正 `depth_anything_lifter.py` 中的数值与统计问题
- 单目深度从主定位器降级为弱先验
- 用 2D 投影候选替代全局 3D kNN

### 中期推荐
- 建立 per-pixel Top-K 候选 Gaussian 缓存
- 使用局部负证据替代整图随机负样本
- 加入 unexplained change 输出

### 长期最优
- 修改 rasterizer，直接输出 contributor / responsibility buffer
- 与现有多视图 evidence aggregation 框架融合

---

## 11. 一句话总结

> **增量更新中的关键，不是把变化像素“抬升成一个 3D 点再找最近 Gaussian”，而是在当前视图的渲染空间中，找出真正对该像素负责的 Top-K Gaussian，并把无法解释的变化区域单独分流。**

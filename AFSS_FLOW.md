# AFSS 防遗忘采样策略 — 完整流程说明

> **论文**: "Does YOLO Really Need to See Every Training Image in Every Epoch?" (CVPR 2026)
>
> **核心思想**: 传统训练每个 epoch 所有图像都参与；AFSS 让已学会的 Easy 图像少看，还没学会的 Hard 图像多看，从而加速训练且不损失精度。

---

## 0. 与原论文对比：新增与调整

下表对比了原论文设计与本项目实际实现的差异，标注了每项改动的动机。

| 维度 | 原论文 (CVPR 2026) | 本项目实现 | 改动动机 |
|------|-------------------|-----------|----------|
| **训练框架** | 独立训练循环 | 集成到 Ultralytics YOLOv13 框架 | 论文代码基于自定义训练器，需适配 Ultralytics 的 `InfiniteDataLoader`、`_RepeatSampler`、DDP 等机制 |
| **检测头兼容** | 仅标准 Detect | 同时支持 `Detect` 和 `Detect_NMSFree` | YOLOv13 新增 NMSFree 检测头，输出为 DFL 原始张量，需额外解码后才能做 IoU 匹配 |
| **GT 设备对齐** | 未涉及（假设数据在正确设备） | `preprocess_batch` 中显式 `.to(device)` 迁移 GT 框/类别 | Ultralytics 默认只迁移 img 到 GPU，bboxes/cls 留在 CPU，导致 `box_iou` 设备不匹配 |
| **采样比例** | 固定值：Hard 100%, Moderate 40%, Easy 2% | 支持手动设置 + 自动推荐 + warmup 后自适应，三者可叠加 | 论文在 COCO（118k 图像）上验证，直接套用到中小数据集会导致活跃集过小 |
| **参数初始化** | 人工调参 | `suggest_afss_params()` 根据数据集规模自动计算初始值 | 不同规模数据集需要不同的采样策略，手工调参成本高 |
| **自适应调参** | 无（全程固定比例） | `adapt_ratios()` warmup 后基于真实分布修正比例 | 论文假设 Easy/Moderate/Hard 分布约 70/20/10，但实际分布因任务而异，固定比例可能过激或不足 |
| **活跃集保护** | 无显式约束 | 强制约束活跃集 ≥ 25% 且 ≥ batch_size × 16 | 防止采样比例过于激进导致训练信息严重丢失（实测 1013 张图活跃集仅 16%，精度下降） |
| **小数据集安全退出** | 无 | `num_images < 500` 自动禁用 AFSS | 极小数据集不适合采样加速，强行采样反而损害模型收敛 |
| **评估缓存** | 每次更新重新扫描 | `_afss_eval_dataset` 首次创建后缓存复用 | Ultralytics 标签扫描较慢，避免每次更新重复扫描 |
| **矩形推理限制** | 未涉及 | 评估时强制 `rect=False` | Ultralytics 矩形模式下 batch 内图像尺寸不一致，导致 `torch.stack` 报错 |
| **配置方式** | 代码硬编码 | `default.yaml` 暴露 8 个配置项 + CLI 参数 | 便于用户通过 `yolo train afss=True` 一行命令启用 |
| **多进程兼容** | 未涉及 | `RANK in {-1, 0}` 初始化 + FlashAttention 回退 + worker 日志去重 | Ultralytics DDP 下多进程/多 worker 场景需额外处理 |

### 核心改进：双重自适应机制

原论文使用**固定参数**，本项目引入了**双重自适应**，两者完全解耦：

```
┌─────────────────────────────────────────────────────────────────┐
│  第一层：suggest_afss_params()（afss_auto_tune=True 时生效）      │
│  ─────────────────────────────────────────────────────────────  │
│  时机：训练开始前（_setup_train）                                   │
│  依据：数据集规模（N）+ batch size + epochs                       │
│  输出：warmup_epochs, easy_ratio, moderate_ratio, update_interval │
│  性质：基于经验的启发式公式，用假设分布 70/20/10 估算               │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  第二层：adapt_ratios()（所有 AFSS 场景均生效）                     │
│  ─────────────────────────────────────────────────────────────  │
│  时机：warmup 结束后首次 AFSS 更新                                │
│  依据：compute_per_image_metrics() 产出的真实 Easy/Mod/Hard 分布   │
│  输出：修正后的 easy_ratio, moderate_ratio                        │
│  性质：基于真实观测值的精确修正，只执行一次                         │
└─────────────────────────────────────────────────────────────────┘
```

**典型场景**：

| 用户配置 | 第一层 | 第二层 | 最终参数来源 |
|---------|--------|--------|------------|
| `afss_auto_tune=True` | ✅ 基于数据规模计算 | ✅ 基于真实分布修正 | 自动推荐 + 真实修正 |
| `afss_auto_tune=False` | ✖️ 跳过 | ✅ 基于真实分布修正 | 手动/默认值 + 真实修正 |
| `afss=False` | ✖️ | ✖️ | 不启用 AFSS |

---

## 1. 核心指标：学习充分性

对每张训练图像 i，定义学习充分性：

```
S_i = min(Precision_i, Recall_i)
```

- **P 高 R 低** = 漏检多（该学的没学到）
- **P 低 R 高** = 误检多（学到了噪声）
- **P 和 R 都高** = 学会了（Easy 图像）

---

## 2. 三级图像分类

| 级别 | 条件 | 每 epoch 采样比例 | 采样机制 |
|------|------|-------------------|----------|
| **Hard** | S_i < 0.3 | 100% 全量 | 全部参与 |
| **Moderate** | 0.3 ≤ S_i < 0.8 | ~40% | 强制覆盖（≥3 epoch 未参与）+ 随机补充 |
| **Easy** | S_i ≥ 0.8 | ~2% | 强制复习（≥10 epoch 未参与）+ 随机多样性 |

---

## 3. 训练时间线

### 3.1 Warmup 阶段（Epoch 0 ~ 9）

```
Epoch 0 ─────────── Warmup（全量训练）─────────────────── Epoch 9
  │                    所有图像 100% 参与                   │
  │                    行为完全等同普通训练                   │
  │                                                       │
  ▼                                                       ▼
```

此阶段 AFSS 不激活，所有图像全量参与每个 epoch，确保模型获得足够的基础学习。

### 3.2 第一次 AFSS 更新（Epoch 10 结束时）

```
Epoch 10 训练完毕
  │
  ├─ 1. AFSSManager.should_update(10) == True
  │
  ├─ 2. compute_per_image_metrics()
  │     → EMA 模型切换为 eval 模式
  │     → 构建 val-mode 训练集（无增强，rect=False）
  │     → 对每张图做推理（conf=0.25, IoU=0.5）
  │     → 贪心 IoU 匹配预测框 vs GT 框
  │     → 统计每张图的 TP / FP / FN
  │     → 计算 Precision_i = TP/(TP+FP), Recall_i = TP/(TP+FN)
  │
  ├─ 3. AFSSManager.update_metrics()
  │     → 更新所有图像的 P, R, S_i
  │     → 按 S_i 分为 Easy / Moderate / Hard
  │
  ├─ 4. 日志输出
  │       AFSS: Easy=1200(24.0%) Moderate=2300(46.0%) Hard=1500(30.0%) MeanS=0.352
  │
  └─ 5. 自适应调参（仅首次更新）
        AFSSManager.adapt_ratios()
        → 基于真实分布调整 easy_ratio / moderate_ratio
        → 确保活跃集 ≥ 25%
        AFSS: Adapted ratios based on real distribution (Easy 1200, Moderate 2300, Hard 1500)
          easy_ratio: 0.020 → 0.085
          moderate_ratio: 0.400 → 0.620
          active set: 2,620 / 5,000 (52%)
```

### 3.3 AFSS 采样训练阶段（Epoch 11 ~ 14）

```
每个 epoch 开始时：
  │
  ├─ AFSSManager.current_epoch = epoch
  │
  ├─ AFSSBatchSampler._compute_active()
  │   → 检测到 epoch 变化
  │   → 调用 AFSSManager.sample_indices(epoch)
  │   → 三级采样：
  │       Hard 1500 张 → 全量参与
  │       Moderate 2300 张 → ~40% ≈ 920 张
  │       Easy 1200 张 → ~2% ≈ 24 张
  │   → 活跃集 ≈ 2444 张（原来 5000 张的 49%）
  │
  ├─ DataLoader 只加载这些图像
  │   → 每 epoch 的 batch 数从 64 降到 ~38
  │   → 训练速度提升约 40%
  │
  └─ 训练正常进行...
```

### 3.4 后续 AFSS 更新（每 5 个 epoch）

```
Epoch 15 → 第二次更新 → Easy 更多 → 活跃集更小 → 训练更快
Epoch 20 → 第三次更新 → ...
Epoch 25 → 第四次更新 → ...
...
直到训练结束
```

随着训练推进，模型越来越好 → Easy 图像越来越多 → 每 epoch 需训练的图像越来越少 → **训练越来越快**。

---

## 4. 采样细节示例

以 epoch 11 为例，假设分级结果：

```
总分级：Hard=500, Moderate=300, Easy=213
```

### Hard 图像（500 张）

```
500 张 → 全部 500 张参与
└─ 更新 last_used_epoch = 11
```

### Moderate 图像（300 张，目标 300 × 40% = 120 张）

```
├─ 强制覆盖：last_used_epoch 距今 ≥ 3 epoch 的
│   → 假设 80 张满足条件
│   → 80 张全部参与，更新 last_used_epoch
│
├─ 随机补充：从剩余 220 张中随机选 40 张
│   → 更新 last_used_epoch
│
└─ 共 120 张参与
```

### Easy 图像（213 张，目标 213 × 2% = 4 张）

```
├─ 强制复习：last_used_epoch 距今 ≥ 10 epoch 的
│   → 假设 2 张满足条件（防遗忘）
│   → 全部参与
│
├─ 随机多样性：从剩余中随机选 2 张
│
└─ 共 4 张参与
```

### 汇总

```
总活跃集 = 500 + 120 + 4 = 624 张（原来 1013 张的 62%）
```

---

## 5. 完整数据流向图

```
┌──────────────────────────────────────────────────────────────┐
│  训练开始 (epoch 0)                                            │
│                                                               │
│  train_loader ← AFSSBatchSampler (warmup 模式)                 │
│  → 所有图像全量参与，行为等同普通训练                              │
└────────────────────────┬─────────────────────────────────────┘
                         │ epoch 0~9 正常训练
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  epoch 10: should_update(10) == True                           │
│                                                               │
│  ① AFSSManager.current_epoch = 10                             │
│  ② AFSSBatchSampler.__iter__() 中 while True 循环              │
│     → sample_indices(10) → warmup 期间返回全量                  │
│  ③ epoch 10 训练完毕                                           │
│  ④ compute_per_image_metrics()                                │
│     → EMA 模型 eval()                                         │
│     → 构建 val-mode 训练集（无增强, rect=False）                 │
│     → 对每张图推理 + IoU 匹配 → P_i, R_i                       │
│  ⑤ AFSSManager.update_metrics(indices, P, R, epoch)           │
│     → sufficiency[i] = min(P_i, R_i)                          │
│     → 自动分级 Easy / Moderate / Hard                          │
│  ⑥ 日志输出分布统计                                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  epoch 11 开始                                                 │
│                                                               │
│  AFSSManager.current_epoch = 11                               │
│  AFSSBatchSampler._compute_active()                           │
│    → 检测到 epoch 变化 → 调用 sample_indices(11)                │
│    → 三级采样：Hard 全量 + Moderate 40% + Easy 2%               │
│    → 返回活跃索引列表                                           │
│                                                               │
│  __iter__() 内 while True:                                    │
│    → shuffle 活跃索引                                          │
│    → 按 batch_size 切分 → yield 批次                            │
│    → 循环直到 epoch 14 结束                                     │
│                                                               │
│  效果：batch 数从 64 降到 ~38                                   │
│  → 每个 epoch 训练量减少约 40%                                  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  epoch 15: should_update(15) == True                           │
│  → 重复步骤 ①~⑥                                               │
│  → 此时模型更好了，Easy 图像变多                                 │
│  → 活跃集更小 → 训练更快                                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 各文件职责分工

| 文件 | 核心类/函数 | 职责 |
|------|-------------|------|
| `ultralytics/utils/afss.py` | `AFSSManager` | 状态管理：存储每图的 P/R/S_i，分级，采样逻辑 |
| `ultralytics/utils/afss.py` | `AFSSBatchSampler` | DataLoader 适配：无限循环迭代，epoch 变化时更换采样集 |
| `ultralytics/utils/afss.py` | `suggest_afss_params` | **参数自动推荐**: 根据数据集规模计算最优 AFSS 超参数 |
| `ultralytics/utils/afss.py` | `adapt_ratios` | **自适应调参**: 首次更新后基于真实难度分布调整采样比例 |
| `ultralytics/data/build.py` | `build_dataloader` | 支持传入自定义 `batch_sampler` 参数 |
| `ultralytics/engine/trainer.py` | `_setup_train` | 训练前初始化 AFSS Manager 和 BatchSampler，支持自动推荐分支 |
| `ultralytics/engine/trainer.py` | `_do_train` | 训练循环：epoch 开始设 current_epoch，epoch 结束触发更新 + 首次自适应调参 |
| `ultralytics/models/yolo/detect/train.py` | `compute_per_image_metrics` | 推理评估：对训练集计算每图的 Precision 和 Recall |
| `ultralytics/cfg/default.yaml` | AFSS 配置块 | 8 个配置参数控制 AFSS 行为（含 `afss_auto_tune` 开关） |
| `ultralytics/nn/modules/block.py` | FlashAttention 回退 | `warning` → `debug` 抑制多 worker 重复日志 |

---

## 7. 配置参数一览

```yaml
# AFSS (Anti-Forgetting Sampling Strategy) - CVPR 2026
afss: False              # (bool) 是否启用 AFSS
afss_auto_tune: False    # (bool) 自动推荐参数（覆盖手动值）
afss_easy_thresh: 0.8    # (float) 充分性 ≥ 0.8 分类为 Easy
afss_hard_thresh: 0.3    # (float) 充分性 < 0.3 分类为 Hard
afss_easy_ratio: 0.02    # (float) Easy 图像每 epoch 采样比例 ~2%
afss_moderate_ratio: 0.4 # (float) Moderate 图像每 epoch 采样比例 ~40%
afss_update_interval: 5  # (int) 状态更新间隔（epoch）
afss_warmup_epochs: 10   # (int) AFSS 启用前的 warmup epoch 数
```

### `afss_auto_tune=True` 时各参数如何计算

| 参数 | 计算公式 | 设计思路 |
|------|---------|--------|
| `warmup_epochs` | `max(10, min(50, N/40))` | 小数据集需更长 warmup 建立稳定模型 |
| `easy_ratio` | `0.02 × √(118k/N)`，上限 0.50 | 参照 COCO 基准，数据越少→采样越保守 |
| `moderate_ratio` | `0.40 × √(√(118k/N))`，上限 0.90 | 中等难度图像也需保守采样 |
| `update_interval` | `max(3, min(20, N/200))` | 平衡评估开销与更新频率 |

**强制约束**:
- 活跃集占比 ≥ 25%，不够则自动提升采样比例
- 活跃集张数 ≥ `batch_size × 16`
- `num_images < 500` 时自动禁用 AFSS

### 不同数据集规模推荐示例

| 数据集规模 | easy_ratio | moderate_ratio | warmup | 活跃集占比 | 预估加速 |
|----------|-----------|---------------|--------|----------|--------|
| 1,000 张 | ~22% | ~87% | 25 | ~34% | ~2.9x |
| 5,000 张 | ~10% | ~70% | 50 | ~29% | ~3.4x |
| 20,000 张 | ~5% | ~55% | 50 | ~25% | ~4.0x |
| 118,000 张 (COCO) | ~2% | ~40% | 50 | ~16% | ~6.3x |

---

## 8. 使用方法

### 方式一：自动推荐参数（推荐）

```bash
# 根据数据集规模自动计算最优 AFSS 参数
yolo detect train model=yolov13n.pt data=coco128.yaml epochs=100 \
    afss=True afss_auto_tune=True
```

Python 脚本：
```python
model.train(
    data='cfg/datasets/my_data.yaml',
    epochs=3000, imgsz=640, batch=16, workers=12,
    afss=True,
    afss_auto_tune=True,  # 自动推荐参数
)
```

### 方式二：手动设置参数

```bash
yolo detect train model=yolov13n.pt data=coco128.yaml epochs=100 \
    afss=True \
    afss_easy_ratio=0.2 \
    afss_moderate_ratio=0.7 \
    afss_warmup_epochs=30
```

### 不启用（默认行为，完全兼容）

```bash
yolo detect train model=yolov13n.pt data=coco128.yaml epochs=100
```

### 训练日志示例

**自动推荐模式**（`afss_auto_tune=True`）:

```
============================================================
  AFSS Auto-Tune Recommendations
============================================================
  Dataset        : 1,013 images, 10 classes
  Batch size     : 16
  Epochs         : 3000
------------------------------------------------------------
  easy_thresh    : 0.8   (≥ this → Easy)
  hard_thresh    : 0.3   (< this → Hard)
  easy_ratio     : 21.6%  (sample rate for Easy images)
  moderate_ratio : 86.7%  (sample rate for Moderate images)
  update_interval: 5 epochs
  warmup_epochs  : 25 epochs
------------------------------------------------------------
  Est. active set     : ~345 / 1,013 (34%)
  Est. training speedup: ~2.9x per epoch
  Total AFSS updates : ~595 times during training
============================================================

AFSS enabled: 1013 images, easy_thresh=0.8, hard_thresh=0.3, easy_ratio=0.216, moderate_ratio=0.867, update every 5 epochs, warmup 25 epochs

...（前 25 epoch 正常训练）

AFSS: Updating image states at epoch 25...
AFSS: Computing per-image metrics: 100%|██████████| 32/32 [00:08<00:00, 4.00it/s]
AFSS: Easy=243(24.0%) Moderate=466(46.0%) Hard=304(30.0%) MeanS=0.352

...（后续 epoch 每 epoch 的 batch 数减少，训练加速）
```

**手动模式**（`afss_auto_tune=False`）:

```
AFSS enabled: 1013 images, easy_thresh=0.8, hard_thresh=0.3, easy_ratio=0.02, moderate_ratio=0.4, update every 5 epochs, warmup 10 epochs
```

---

## 9. 设计要点

1. **向后兼容**: `afss=False`（默认）时训练流程完全不受影响
2. **InfiniteDataLoader 兼容**: `AFSSBatchSampler.__iter__()` 使用 `while True` 无限循环，适配 Ultralytics `_RepeatSampler` 只调用一次 `iter()` 的机制
3. **检测头兼容**: `compute_per_image_metrics()` 同时处理 `Detect_NMSFree`（DFL 原始输出需 `decode_bboxes` 解码）和标准 `Detect`（已解码输出直接 NMS）
4. **GPU 设备一致**: GT 框和类别通过 `.to(device)` 显式迁移到 GPU，避免 `box_iou` 设备不匹配
5. **DDP 兼容**: AFSS 仅在 `RANK in {-1, 0}` 时初始化
6. **评估缓存**: `_afss_eval_dataset` 首次创建后缓存复用，避免每次更新重复扫描标签
7. **rect=False**: 评估数据集不使用矩形模式，避免同一 batch 内图像尺寸不一致导致 `torch.stack` 报错
8. **自动推荐安全约束**: `suggest_afss_params()` 保证活跃集 ≥ 25%，`num_images < 500` 时自动禁用 AFSS 并警告
9. **自动推荐可覆盖**: `afss_auto_tune=True` 时仍可传入 `easy_thresh` 和 `hard_thresh` 作为参考值，其余参数由算法计算
10. **自适应调参**: `adapt_ratios()` 在首次 AFSS 更新后利用真实难度分布调整 `easy_ratio` / `moderate_ratio`，替代假设的 70/20/10 分布，确保活跃集 ≥ 25% 且 ≥ batch_size × 16，只执行一次

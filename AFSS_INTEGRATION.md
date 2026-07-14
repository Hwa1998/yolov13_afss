# AFSS 防遗忘采样策略集成记录

> **论文参考**: "Does YOLO Really Need to See Every Training Image in Every Epoch?" (CVPR 2026)
>
> **核心思想**: 通过评估每张训练图像的学习充分性 S_i = min(P_i, R_i)，将图像分为 Easy / Moderate / Hard 三级，并按差异化策略采样参与训练，从而在不损失精度的前提下加速训练（约 1.5x）。

---

## 文件变更总览

| 文件 | 操作 | 行号范围 | 说明 |
|------|------|----------|------|
| `ultralytics/utils/afss.py` | **新建** | 1-361, 363-525 | AFSS 核心管理器 + 自定义 BatchSampler + 参数自动推荐 |
| `ultralytics/engine/trainer.py` | **修改** | 149-150, 292-370, 391-393, 404-405, 492-504 | 训练循环集成 AFSS（含自动推荐分支） |
| `ultralytics/models/yolo/detect/train.py` | **修改** | 8, 17, 155-309 | 新增 `compute_per_image_metrics()` 方法 |
| `ultralytics/data/build.py` | **修改** | 135, 145-146, 154-164 | `build_dataloader` 支持自定义 `batch_sampler` |
| `ultralytics/cfg/default.yaml` | **修改** | 36-44 | 新增 AFSS 配置参数（含 `afss_auto_tune` 开关） |

---

## 1. 新建文件: `ultralytics/utils/afss.py`

**目的**: 实现 AFSS 的核心算法逻辑，包括图像充分性评估、三级分类、差异化采样。

### 1.1 `AFSSManager` 类 (L33-L283)

管理每张训练图像的学习状态，决定哪些图像参与当前 epoch 的训练。

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__` | L41-L95 | 初始化所有图像为 Hard（sufficiency=0），保证训练初期全量参与 |
| `update_metrics` | L97-L123 | 更新指定图像的 P/R，重算 sufficiency = min(P, R) |
| `classify_images` | L125-L150 | 按阈值划分 Easy(≥0.8) / Moderate / Hard(<0.3) |
| `sample_indices` | L152-L239 | **核心采样逻辑**: 三级策略返回当前 epoch 的活跃索引 |
| `should_update` | L241-L256 | 判断是否到达状态更新节点（warmup 后每 N epoch） |
| `mark_used` | L258-L268 | 标记图像在指定 epoch 被使用 |
| `get_stats` | L270-L287 | 返回 Easy/Moderate/Hard 分布统计 |
| `adapt_ratios` | L289-L363 | **自适应调参**: 首次更新后基于真实分布调整采样比例 |

**三级采样策略** (`sample_indices`):
- **Hard 图像**: 100% 全量参与
- **Moderate 图像**: ~40% 采样 = 强制覆盖（超过 3 epoch 未参与）+ 随机补充
- **Easy 图像**: ~2% 采样 = 强制复习（超过 10 epoch 未参与）+ 随机多样性
- **Warmup 期间**（前 10 epoch）: 所有图像 100% 参与，不启用 AFSS

**自适应调参** (`adapt_ratios`):
- **触发时机**: 首次 AFSS 更新后（warmup 结束后第一次 `compute_per_image_metrics` 之后）
- **目的**: 利用 warmup 后真实的 Easy/Moderate/Hard 分布，替代 `suggest_afss_params()` 中假设的 70/20/10 分布
- **算法**:
  1. 获取实际分布: `n_easy`, `n_moderate`, `n_hard`
  2. 计算当前活跃集: `active = n_hard + n_mod × moderate_ratio + n_easy × easy_ratio`
  3. 约束: 活跃集 ≥ 25% 且 ≥ `batch_size × 16`
  4. 不满足约束时迭代提升 `easy_ratio`（×1.15）和 `moderate_ratio`（×1.10）
- **只执行一次**: 通过 `_adapted` 标志确保只调整一次
- **示例**: 若真实分布为 Easy=78%, Moderate=13%, Hard=9%，默认参数（2%/40%）只激活 ~16% 图像，`adapt_ratios` 会自动提升到 ~22%/87% 使活跃集达到 ~35%

### 1.2 `AFSSBatchSampler` 类 (L286-L361)

兼容 Ultralytics `InfiniteDataLoader` / `_RepeatSampler` 的自定义 BatchSampler。

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__` | L296-L308 | 初始化，含 epoch 缓存机制避免重复调用 `sample_indices` |
| `_compute_active` | L310-L316 | 仅在 epoch 变化时重新计算活跃索引 |
| `__iter__` | L318-L342 | **无限循环生成器**，适配 `_RepeatSampler` 只调用一次 `iter()` 的机制 |
| `__len__` | L344-L350 | 返回当前 epoch 的批次数 |
| `set_epoch` | L352-L361 | 设置当前 epoch |

**关键设计**: Ultralytics 的 `_RepeatSampler` 仅调用 `iter(sampler)` 一次并 `yield from` 永久循环。因此 `AFSSBatchSampler.__iter__()` 必须使用 `while True` 无限循环，在循环内部检测 epoch 变化并重新计算活跃集，确保每个 epoch 的采样策略不同。

### 1.3 `suggest_afss_params()` 函数 (L363-L525)

根据数据集特征自动推荐 AFSS 超参数的工具函数。

**输入参数**: `num_images`（训练图像数）、`batch_size`、`epochs`、`num_classes`、`easy_thresh`、`hard_thresh`

**核心算法**:

| 参数 | 计算公式 | 设计思路 |
|------|---------|--------|
| `warmup_epochs` | `max(10, min(50, N/40))` | 小数据集需更长 warmup 建立稳定模型 |
| `easy_ratio` | `0.02 × √(118k/N)`，上限 0.50 | 参照 COCO 基准，数据越少→采样越保守 |
| `moderate_ratio` | `0.40 × √(√(118k/N))`，上限 0.90 | 中等难度图像也需保守采样 |
| `update_interval` | `max(3, min(20, N/200))` | 平衡评估开销与更新频率 |

**强制约束**:
- 活跃集占比 ≥ 25%（不够则自动提升采样比例）
- 活跃集张数 ≥ `batch_size × 16`（确保有足够 batch）
- 若 `num_images < 500`，直接返回 `{"afss": False}` 并警告数据集太小

**输出**: 包含推荐参数的 dict，含诊断信息 `_active_ratio`（活跃集占比）、`_speedup`（预估加速比）、`_num_updates`（总更新次数）。

---

## 2. 修改文件: `ultralytics/engine/trainer.py`

**目的**: 在 BaseTrainer 的训练循环中集成 AFSS 采样逻辑。共 4 处修改。

### 2.1 初始化 AFSS 管理器属性 (L149-L150)

```python
# AFSS (Anti-Forgetting Sampling Strategy)
self.afss_manager = None
```

**位置**: `BaseTrainer.__init__()` 方法内部  
**目的**: 为所有 Trainer 实例添加 `afss_manager` 属性，默认 `None` 表示不启用 AFSS，保证向后兼容。

### 2.2 AFSS 初始化 + 重建 DataLoader (L292-L370)

```python
if getattr(self.args, "afss", False) and RANK in {-1, 0}:
    from ultralytics.utils.afss import AFSSManager, AFSSBatchSampler, suggest_afss_params
    from ultralytics.data.build import build_dataloader as _build_dataloader

    n_train_images = len(self.train_loader.dataset)

    # --- Auto-tune or manual parameters ---
    if getattr(self.args, "afss_auto_tune", False):
        # 自动推荐模式：根据数据集特征计算参数
        nc = 10  # 从 dataset 或 model 获取类别数
        tuned = suggest_afss_params(
            num_images=n_train_images, batch_size=batch_size,
            epochs=self.epochs, num_classes=nc, ...
        )
        if not tuned.get("afss", True):
            # 数据集太小，自动禁用 AFSS
            self.afss_manager = None
        else:
            easy_thresh = tuned["afss_easy_thresh"]
            # ... 其他参数覆盖 ...
    else:
        # 手动模式：使用用户传入或默认参数
        easy_thresh = getattr(self.args, "afss_easy_thresh", 0.8)
        # ... 其他参数 ...

    # 仅当 afss_manager 不为 None 时初始化 AFSS
    if self.afss_manager is None and getattr(self.args, "afss_auto_tune", False):
        pass  # 自动推荐判定数据集太小，跳过 AFSS
    else:
        self.afss_manager = AFSSManager(...)
        afss_batch_sampler = AFSSBatchSampler(...)
        self.train_loader = _build_dataloader(..., batch_sampler=afss_batch_sampler)
        LOGGER.info(f"AFSS enabled: ...")
```

**位置**: `BaseTrainer._setup_train()` 方法内部，在原始 `train_loader` 创建之后  
**目的**:
- 支持两种模式：`afss_auto_tune=True` 自动推荐 vs 手动设置参数
- 自动推荐模式会调用 `suggest_afss_params()`，根据数据集规模、batch size、epochs 计算最优参数
- 若自动推荐判定数据集太小（<500 张），自动禁用 AFSS 并打印警告
- 手动模式行为与之前完全一致
- 日志同时输出 `easy_ratio` 和 `moderate_ratio`（新增），方便确认实际使用的采样比例

### 2.3 Epoch 开始时更新 AFSS 状态 (L391-L405)

```python
# AFSS: set current epoch for sampler
if self.afss_manager:
    self.afss_manager.current_epoch = epoch

...

# AFSS: recalculate nb since active set size changes
nb = len(self.train_loader)  # number of batches (may change with AFSS)
```

**位置**: `BaseTrainer._do_train()` 方法内部，每个 epoch 循环开始处（`on_train_epoch_start` 回调之后）  
**目的**:
- 将当前 epoch 通知 AFSS 管理器，使采样器知道当前是哪个 epoch
- 重新计算 `nb`（批次数），因为 AFSS 采样后活跃集大小会变化，影响进度条显示和 warmup 步数计算

### 2.4 Epoch 结束时更新图像充分性 (L492-L520)

```python
if self.afss_manager and hasattr(self, "compute_per_image_metrics"):
    if self.afss_manager.should_update(epoch + 1):
        LOGGER.info(f"\nAFSS: Updating image states at epoch {epoch + 1}...")
        indices, precisions, recalls = self.compute_per_image_metrics()
        self.afss_manager.update_metrics(indices, precisions, recalls, epoch)
        stats = self.afss_manager.get_stats()
        LOGGER.info(
            f"AFSS: Easy={stats['easy']}({stats['easy_pct']:.1f}%) "
            f"Moderate={stats['moderate']}({stats['moderate_pct']:.1f}%) "
            f"Hard={stats['hard']}({stats['hard_pct']:.1f}%) "
            f"MeanS={stats['mean_sufficiency']:.3f}"
        )

        # AFSS: Adapt ratios after first update based on real distribution
        adapt_info = self.afss_manager.adapt_ratios(self.batch_size)
        if adapt_info:
            LOGGER.info(
                f"AFSS: Adapted ratios based on real distribution ..."
                f"  easy_ratio: {adapt_info['old_easy_ratio']:.3f} → {adapt_info['new_easy_ratio']:.3f}\n"
                f"  moderate_ratio: ... → ...\n"
                f"  active set: ... / ... (...)"
            )
```

**位置**: `BaseTrainer._do_train()` 方法内部，每个 epoch 的 batch 循环结束后、Scheduler 更新前  
**目的**:
- 通过 `should_update()` 判断是否到达更新节点（warmup 后每 5 epoch）
- 调用 `compute_per_image_metrics()` 对训练集做一轮轻量推理
- 将结果传入 `AFSSManager.update_metrics()` 更新每张图的 P/R/充分性
- 日志输出 Easy/Moderate/Hard 分布和平均充分性，方便监控训练过程
- **首次更新后自适应调参**: 调用 `adapt_ratios()` 根据真实分布调整 `easy_ratio` / `moderate_ratio`，确保活跃集 ≥ 25%

---

## 3. 修改文件: `ultralytics/models/yolo/detect/train.py`

**目的**: 在 `DetectionTrainer` 中新增 `compute_per_image_metrics()` 方法，为 AFSS 提供每图 P/R 数据。

### 3.1 新增 import (L8, L17)

```python
import torch                    # L8 (原文件已有，确认存在)
from ultralytics.utils.ops import non_max_suppression  # L17 (新增 import)
```

**目的**: `torch.no_grad()` 用于推理时禁用梯度，`non_max_suppression` 用于后处理模型输出。

### 3.2 新增 `compute_per_image_metrics()` 方法 (L155-L309)

```python
def compute_per_image_metrics(self, iou_thresh=0.5, conf_thresh=0.25):
```

**位置**: `DetectionTrainer` 类内部  
**目的**: 对训练集做一轮轻量推理（无增强、无反向传播），计算每张图的 Precision 和 Recall。

**工作流程**:
1. **L170-L172**: 使用 EMA 模型（更稳定的预测），切换为 eval 模式
2. **L174-L189**: 构建 val-mode 数据集（无增强、rect=True），用 2x batch_size 的简单 DataLoader。**已优化**: 使用 `_afss_eval_dataset` 缓存，首次调用后复用，避免每 5 epoch 重复扫描标签缓存
   - 注意 `build_dataloader` 参数为位置参数 `batch`，非 `batch_size`
   - 注意 `non_max_suppression` 参数为 `iou_thres`，非 `iou_thresh`
3. **L192-L305**: 对每个 batch 进行推理和 P/R 计算:
   - **L201-L221 (NMSFree 检测头处理)**: `Detect_NMSFree` 的输出是 `{"one2one": (tuple)}` 格式，内部包含 DFL 原始通道。使用 `model_head.decode_bboxes()` 解码得到 xywh 坐标，再用 `non_max_suppression` 统一处理
   - **L222-L233 (标准 Detect 检测头处理)**: 输出已是 `(batch, 4+nc, A)` 格式，直接 `non_max_suppression`
   - **L236-L303 (逐图像 P/R 计算)**:
     - 提取 GT（归一化 xywh → 像素 xyxy）
     - 贪心 IoU 匹配（阈值 0.5），统计 TP / FP / FN
     - 考虑类别匹配（同 IoU 框但类别不同不计为 TP）
     - 边界情况: 无 GT 无预测 → P=R=1；有 GT 无预测 → P=1, R=0
4. **L307-L309**: 恢复模型为 train 模式，返回 `(image_indices, precisions, recalls)`

---

## 4. 修改文件: `ultralytics/data/build.py`

**目的**: 让 `build_dataloader()` 支持传入自定义 `batch_sampler`，使 AFSS 采样器能替换默认的 shuffle + batch_size 方式。

### 4.1 函数签名新增参数 (L135)

```python
def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, batch_sampler=None):
```

**位置**: 函数定义行  
**改动**: 新增 `batch_sampler=None` 可选参数  
**目的**: 向后兼容，默认 `None` 时行为不变。

### 4.2 新增 batch_sampler 分支 (L154-L164)

```python
if batch_sampler is not None:
    return InfiniteDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=nw,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
```

**位置**: `build_dataloader()` 函数内部，`sampler` 创建之前  
**目的**: 当传入 `batch_sampler` 时，直接使用它创建 `InfiniteDataLoader`，跳过默认的 shuffle + DistributedSampler 逻辑。

### 4.3 原有逻辑不变 (L166-L177)

非 AFSS 模式（`batch_sampler=None`）走原有路径，行为完全不受影响。

---

## 5. 修改文件: `ultralytics/cfg/default.yaml`

**目的**: 添加 AFSS 相关配置参数，用户可通过命令行或 YAML 覆盖来控制 AFSS 行为。

### 新增配置项 (L36-L44)

```yaml
# AFSS (Anti-Forgetting Sampling Strategy) - CVPR 2026
afss: False              # (bool) 是否启用 AFSS
afss_auto_tune: False    # (bool) 自动推荐 AFSS 参数（覆盖手动值）
afss_easy_thresh: 0.8    # (float) 充分性 ≥ 0.8 分类为 Easy
afss_hard_thresh: 0.3    # (float) 充分性 < 0.3 分类为 Hard
afss_easy_ratio: 0.02    # (float) Easy 图像每 epoch 采样比例 ~2%
afss_moderate_ratio: 0.4 # (float) Moderate 图像每 epoch 采样比例 ~40%
afss_update_interval: 5  # (int) 状态更新间隔（epoch）
afss_warmup_epochs: 10   # (int) AFSS 启用前的 warmup epoch 数
```

**位置**: `resume` 和 `amp` 之间  
**新增参数**: `afss_auto_tune` — 设为 `True` 时，忽略手动设置的 `easy_ratio`、`moderate_ratio` 等，由 `suggest_afss_params()` 根据数据集规模自动计算  
**目的**: 用户可通过 `yolo train afss=True afss_auto_tune=True` 一键启用 AFSS 并获得适配当前数据集的推荐参数。

---

## 使用方法

### 方式一：自动推荐参数（推荐）

```bash
# 根据数据集规模自动计算最优 AFSS 参数
yolo detect train model=yolov13n.pt data=coco128.yaml epochs=100 \
    afss=True afss_auto_tune=True
```

Python 脚本方式：
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
# 命令行手动指定参数
yolo detect train model=yolov13n.pt data=coco128.yaml epochs=100 \
    afss=True \
    afss_easy_ratio=0.2 \
    afss_moderate_ratio=0.7 \
    afss_warmup_epochs=30
```

### 不启用 AFSS（默认行为）

```bash
# 不传 afss 参数或 afss=False，训练流程完全不受影响
yolo detect train model=yolov13n.pt data=coco128.yaml epochs=100
```

### 日志输出示例

**自动推荐模式日志**:

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
```

**手动模式日志**:

```
AFSS enabled: 5000 images, easy_thresh=0.8, hard_thresh=0.3, easy_ratio=0.02, moderate_ratio=0.4, update every 5 epochs, warmup 10 epochs

...（前 10 epoch 正常训练，所有图像参与）

AFSS: Updating image states at epoch 15...
AFSS: Computing per-image metrics: 100%|██████████| 157/157 [00:12<00:00]
AFSS: Easy=1200(24.0%) Moderate=2300(46.0%) Hard=1500(30.0%) MeanS=0.352
AFSS: Adapted ratios based on real distribution (Easy 1200, Moderate 2300, Hard 1500)
  easy_ratio: 0.020 → 0.085
  moderate_ratio: 0.400 → 0.620
  active set: 2,620 / 5,000 (52%)

...（后续 epoch 采样数减少，训练加速）
```

---

## 设计注意事项

1. **向后兼容**: `afss=False`（默认）时，所有修改不影响原始训练流程
2. **InfiniteDataLoader 兼容**: `AFSSBatchSampler` 使用 `while True` 无限循环适配 `_RepeatSampler`
3. **NMSFree 兼容**: `compute_per_image_metrics()` 同时处理 `Detect_NMSFree`（DFL 原始输出）和标准 `Detect`（已解码输出）两种检测头
4. **DDP 兼容**: AFSS 仅在 `RANK in {-1, 0}` 时初始化，非主进程不受影响
5. **内存效率**: 使用 numpy 数组存储 per-image 状态，万级图像仅占几十 KB
6. **自动推荐安全约束**: `suggest_afss_params()` 保证活跃集 ≥ 25%，数据量 < 500 时自动禁用 AFSS
7. **自动推荐可覆盖**: `afss_auto_tune=True` 时仍可传入 `easy_thresh` 和 `hard_thresh` 作为参考值，其余参数由算法计算
8. **自适应调参**: `adapt_ratios()` 在首次 AFSS 更新后利用真实难度分布调整采样比例，替代 `suggest_afss_params()` 中假设的 70/20/10 分布，确保活跃集 ≥ 25% 且 ≥ batch_size × 16

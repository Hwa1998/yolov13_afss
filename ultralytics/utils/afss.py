# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
AFSS: Anti-Forgetting Sampling Strategy for Efficient YOLO Training (CVPR 2026)

Reference:
    "Does YOLO Really Need to See Every Training Image in Every Epoch?"

Core idea:
    - Compute per-image learning sufficiency: S_i = min(Precision_i, Recall_i)
    - Classify images as Easy / Moderate / Hard based on sufficiency
    - Sample each difficulty level at different rates:
        * Hard: 100% every epoch
        * Moderate: ~40% (forced coverage + random)
        * Easy: ~2% (forced review + random diversity)
    - Update states every N epochs to avoid overhead

Usage:
    from ultralytics.utils.afss import AFSSManager, AFSSBatchSampler

    manager = AFSSManager(num_images=10000)
    sampler = AFSSBatchSampler(manager, batch_size=16)
    # In training loop, call manager.update_metrics() and manager.should_update()
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import BatchSampler

from ultralytics.utils import LOGGER


class AFSSManager:
    """
    Manages the Anti-Forgetting Sampling Strategy for YOLO training.

    Tracks per-image learning sufficiency and determines which images
    should participate in each training epoch based on difficulty level.
    """

    def __init__(
        self,
        num_images: int,
        easy_thresh: float = 0.8,
        hard_thresh: float = 0.3,
        easy_ratio: float = 0.02,
        moderate_ratio: float = 0.4,
        update_interval: int = 5,
        easy_review_interval: int = 10,
        moderate_review_interval: int = 3,
        warmup_epochs: int = 10,
    ):
        """
        Initialize AFSS manager.

        Args:
            num_images: Total number of training images.
            easy_thresh: Sufficiency threshold above which an image is Easy.
            hard_thresh: Sufficiency threshold below which an image is Hard.
            easy_ratio: Target fraction of Easy images sampled per epoch (~2%).
            moderate_ratio: Target fraction of Moderate images sampled per epoch (~40%).
            update_interval: Number of epochs between state updates.
            easy_review_interval: Max epochs an Easy image can go without being reviewed.
            moderate_review_interval: Max epochs a Moderate image can go without being reviewed.
            warmup_epochs: Number of initial epochs where all images participate (no AFSS).
        """
        self.num_images = num_images
        self.easy_thresh = easy_thresh
        self.hard_thresh = hard_thresh
        self.easy_ratio = easy_ratio
        self.moderate_ratio = moderate_ratio
        self.update_interval = update_interval
        self.easy_review_interval = easy_review_interval
        self.moderate_review_interval = moderate_review_interval
        self.warmup_epochs = warmup_epochs

        # Per-image state arrays (numpy for efficiency)
        # Initialize all as Hard (sufficiency=0) so all images participate initially
        self.precision = np.zeros(num_images, dtype=np.float32)
        self.recall = np.zeros(num_images, dtype=np.float32)
        self.sufficiency = np.zeros(num_images, dtype=np.float32)  # min(P, R)
        self.last_used_epoch = np.full(num_images, -1, dtype=np.int32)  # -1 means never used

        # Current epoch tracker (set externally by trainer)
        self.current_epoch: int = 0

        # Cache for difficulty classification
        self._easy_indices: List[int] = []
        self._moderate_indices: List[int] = []
        self._hard_indices: List[int] = []
        self._classify_dirty = True  # needs reclassification after update

        # Adaptive tuning: adjust ratios after first real distribution is known
        self._adapted = False

    def update_metrics(
        self,
        image_indices: np.ndarray,
        precisions: np.ndarray,
        recalls: np.ndarray,
        current_epoch: int,
    ) -> None:
        """
        Update precision/recall for specified images and recompute sufficiency.

        Args:
            image_indices: Array of dataset indices that were evaluated.
            precisions: Per-image precision values (same length as image_indices).
            recalls: Per-image recall values (same length as image_indices).
            current_epoch: The epoch at which this evaluation was performed.
        """
        image_indices = np.asarray(image_indices, dtype=np.int64)
        precisions = np.asarray(precisions, dtype=np.float32)
        recalls = np.asarray(recalls, dtype=np.float32)

        self.precision[image_indices] = precisions
        self.recall[image_indices] = recalls
        self.sufficiency[image_indices] = np.minimum(precisions, recalls)

        # Mark images that have been used (participated in this epoch's evaluation pass)
        # Note: last_used_epoch is updated in sample_indices for images that actually trained
        self._classify_dirty = True

    def classify_images(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Classify all images into Easy / Moderate / Hard based on sufficiency.

        Returns:
            Tuple of (easy_indices, moderate_indices, hard_indices).
        """
        if not self._classify_dirty:
            return self._easy_indices, self._moderate_indices, self._hard_indices

        easy, moderate, hard = [], [], []
        for i in range(self.num_images):
            s = self.sufficiency[i]
            if s >= self.easy_thresh:
                easy.append(i)
            elif s < self.hard_thresh:
                hard.append(i)
            else:
                moderate.append(i)

        self._easy_indices = easy
        self._moderate_indices = moderate
        self._hard_indices = hard
        self._classify_dirty = False

        return easy, moderate, hard

    def sample_indices(self, current_epoch: int) -> List[int]:
        """
        Determine which image indices should participate in this epoch's training.

        During warmup (epoch < warmup_epochs), all images participate.
        Otherwise, applies three-tiered sampling:
            - Hard: 100% full coverage
            - Moderate: ~40% (forced coverage + random fill)
            - Easy: ~2% (forced review + random diversity)

        Args:
            current_epoch: The current training epoch number (0-based).

        Returns:
            List of image indices to use for training this epoch.
        """
        self.current_epoch = current_epoch

        # During warmup: all images participate, mark as used
        if current_epoch < self.warmup_epochs:
            all_indices = list(range(self.num_images))
            self.last_used_epoch[:] = current_epoch
            return all_indices

        easy, moderate, hard = self.classify_images()
        selected = []

        # --- Hard images: 100% full coverage ---
        selected.extend(hard)
        for idx in hard:
            self.last_used_epoch[idx] = current_epoch

        # --- Moderate images: ~40% with short-term coverage ---
        if moderate:
            n_moderate_target = max(1, int(len(moderate) * self.moderate_ratio))

            # Forced coverage: images not used in last `moderate_review_interval` epochs
            forced = [
                idx for idx in moderate
                if (current_epoch - self.last_used_epoch[idx]) >= self.moderate_review_interval
            ]
            # Cap forced to target
            if len(forced) > n_moderate_target:
                forced = random.sample(forced, n_moderate_target)

            selected.extend(forced)
            for idx in forced:
                self.last_used_epoch[idx] = current_epoch

            # Random fill from remaining moderate
            remaining = [idx for idx in moderate if idx not in set(forced)]
            n_random = max(0, n_moderate_target - len(forced))
            if remaining and n_random > 0:
                n_random = min(n_random, len(remaining))
                random_fill = random.sample(remaining, n_random)
                selected.extend(random_fill)
                for idx in random_fill:
                    self.last_used_epoch[idx] = current_epoch

        # --- Easy images: ~2% with continuous review ---
        if easy:
            n_easy_target = max(1, int(len(easy) * self.easy_ratio))
            # Split target: half forced review, half random diversity
            n_forced = max(1, n_easy_target // 2)
            n_random = max(1, n_easy_target - n_forced)

            # Forced review: images not used in last `easy_review_interval` epochs
            forced = [
                idx for idx in easy
                if (current_epoch - self.last_used_epoch[idx]) >= self.easy_review_interval
            ]
            if len(forced) > n_forced:
                forced = random.sample(forced, n_forced)

            selected.extend(forced)
            for idx in forced:
                self.last_used_epoch[idx] = current_epoch

            # Random diversity from remaining easy
            remaining = [idx for idx in easy if idx not in set(forced)]
            if remaining and n_random > 0:
                n_random = min(n_random, len(remaining))
                random_fill = random.sample(remaining, n_random)
                selected.extend(random_fill)
                for idx in random_fill:
                    self.last_used_epoch[idx] = current_epoch

        return selected

    def should_update(self, current_epoch: int) -> bool:
        """
        Check if AFSS states should be updated at this epoch.

        Update happens every `update_interval` epochs, but only after warmup.

        Args:
            current_epoch: The current epoch number (0-based).

        Returns:
            True if metrics should be recomputed this epoch.
        """
        if current_epoch < self.warmup_epochs:
            return False
        # Update at warmup_epochs, warmup_epochs+interval, warmup_epochs+2*interval, ...
        return (current_epoch - self.warmup_epochs) % self.update_interval == 0

    def mark_used(self, indices: np.ndarray, epoch: int) -> None:
        """
        Mark given image indices as used at the specified epoch.
        Called after each batch to track actual usage.

        Args:
            indices: Array of image indices that participated in training.
            epoch: The epoch in which they were used.
        """
        indices = np.asarray(indices, dtype=np.int64)
        self.last_used_epoch[indices] = epoch

    def get_stats(self) -> dict:
        """
        Get current distribution statistics of Easy/Moderate/Hard images.

        Returns:
            Dict with counts and percentages for each difficulty level.
        """
        easy, moderate, hard = self.classify_images()
        total = self.num_images
        return {
            "easy": len(easy),
            "moderate": len(moderate),
            "hard": len(hard),
            "easy_pct": 100.0 * len(easy) / total if total > 0 else 0,
            "moderate_pct": 100.0 * len(moderate) / total if total > 0 else 0,
            "hard_pct": 100.0 * len(hard) / total if total > 0 else 0,
            "mean_sufficiency": float(self.sufficiency.mean()),
        }

    def adapt_ratios(self, batch_size: int = 16) -> Optional[Dict]:
        """
        Adjust easy_ratio / moderate_ratio based on actual difficulty distribution.

        Should be called once, after the first AFSS update (when real P/R values
        are available from compute_per_image_metrics).  Uses the actual
        Easy/Moderate/Hard counts to ensure the active set is large enough.

        Args:
            batch_size: Training batch size, used for minimum-batches constraint.

        Returns:
            Dict describing the adaptation if ratios were changed, else None.
        """
        if self._adapted:
            return None

        easy, moderate, hard = self.classify_images()
        n_easy = len(easy)
        n_mod = len(moderate)
        n_hard = len(hard)
        n_total = n_easy + n_mod + n_hard
        if n_total == 0:
            return None

        old_er = self.easy_ratio
        old_mr = self.moderate_ratio

        # Current active set with real distribution
        active = n_hard + int(n_mod * self.moderate_ratio) + int(n_easy * self.easy_ratio)
        active_ratio = active / n_total

        # Minimum constraints
        min_active_ratio = 0.25
        min_active_count = max(batch_size * 16, batch_size * 5)

        # Iteratively bump ratios until constraints are met
        max_iter = 50
        for _ in range(max_iter):
            if active_ratio >= min_active_ratio and active >= min_active_count:
                break
            bumped = False
            if self.easy_ratio < 0.80:
                self.easy_ratio = min(0.80, self.easy_ratio * 1.15)
                bumped = True
            if self.moderate_ratio < 0.95:
                self.moderate_ratio = min(0.95, self.moderate_ratio * 1.10)
                bumped = True
            # Recalculate
            active = n_hard + int(n_mod * self.moderate_ratio) + int(n_easy * self.easy_ratio)
            active_ratio = active / n_total
            if not bumped:
                break

        self._adapted = True

        # Only return info if ratios actually changed
        changed = (abs(self.easy_ratio - old_er) > 0.001 or
                   abs(self.moderate_ratio - old_mr) > 0.001)
        if not changed:
            return None

        self._classify_dirty = True  # force reclassification on next sample

        return {
            "old_easy_ratio": old_er,
            "old_moderate_ratio": old_mr,
            "new_easy_ratio": self.easy_ratio,
            "new_moderate_ratio": self.moderate_ratio,
            "n_easy": n_easy,
            "n_moderate": n_mod,
            "n_hard": n_hard,
            "active": active,
            "active_ratio": active_ratio,
        }


class AFSSBatchSampler:
    """
    Batch sampler that uses AFSS to determine which images to include per epoch.

    Replaces the standard shuffle-based batch sampler during training when AFSS is enabled.
    Designed to work with Ultralytics' InfiniteDataLoader / _RepeatSampler, which calls
    iter() only once and then yields from that generator forever.  We therefore run an
    infinite loop inside __iter__ and re-compute the active set whenever the epoch changes.
    """

    def __init__(self, afss_manager: AFSSManager, batch_size: int, drop_last: bool = False):
        """
        Args:
            afss_manager: The AFSSManager instance controlling sampling strategy.
            batch_size: Number of images per batch.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.afss = afss_manager
        self.batch_size = batch_size
        self.drop_last = drop_last
        # Cache so __len__ does not call sample_indices (which has side-effects)
        self._last_epoch: int = -1
        self._last_active_indices: List[int] = []

    def _compute_active(self) -> List[int]:
        """Recompute active indices if the epoch has changed since last call."""
        epoch = self.afss.current_epoch
        if epoch != self._last_epoch:
            self._last_epoch = epoch
            self._last_active_indices = self.afss.sample_indices(epoch)
        return self._last_active_indices

    def __iter__(self):
        """
        Infinite generator compatible with _RepeatSampler.

        _RepeatSampler calls iter() once and then ``yield from`` forever.
        We therefore loop indefinitely, recomputing the active set each time
        the manager's current_epoch changes.
        """
        while True:
            active_indices = list(self._compute_active())
            if not active_indices:
                return  # nothing to yield — should not happen in practice

            random.shuffle(active_indices)

            n = len(active_indices)
            if self.drop_last:
                n_batches = n // self.batch_size
                for i in range(n_batches):
                    start = i * self.batch_size
                    yield active_indices[start:start + self.batch_size]
            else:
                for i in range(0, n, self.batch_size):
                    batch = active_indices[i:i + self.batch_size]
                    yield batch

    def __len__(self) -> int:
        """Return the number of batches for the current epoch's active set."""
        active = self._compute_active()
        n = len(active)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for the underlying AFSS manager.
        Called by trainer at the start of each epoch.

        Args:
            epoch: The current epoch number.
        """
        self.afss.current_epoch = epoch


# ---------------------------------------------------------------------------
# Auto-tune: suggest AFSS hyperparameters based on dataset characteristics
# ---------------------------------------------------------------------------

def suggest_afss_params(
    num_images: int,
    batch_size: int = 16,
    epochs: int = 300,
    num_classes: int = 10,
    easy_thresh: float = 0.8,
    hard_thresh: float = 0.3,
) -> Dict:
    """
    Automatically recommend AFSS hyperparameters based on dataset characteristics.

    Core principle:
        Ensure that the *active set* (images participating per epoch) is large
        enough to form meaningful batches, while still providing a training speedup.
        A minimum active-set ratio of ~25% of the dataset is enforced.

    Args:
        num_images:    Total number of training images.
        batch_size:    Training batch size.
        epochs:        Total training epochs.
        num_classes:   Number of detection classes.
        easy_thresh:   Sufficiency threshold for Easy (higher = stricter).
        hard_thresh:   Sufficiency threshold for Hard (lower = stricter).

    Returns:
        Dict with recommended AFSS parameters and diagnostic info.
        Returns {"afss": False} with a warning if dataset is too small.
    """
    # ------------------------------------------------------------------
    # 1. Dataset scale classification
    # ------------------------------------------------------------------
    if num_images < 500:
        LOGGER.warning(
            "AFSS auto-tune: dataset too small (%d images). "
            "AFSS is NOT recommended — use afss=False for full training.",
            num_images,
        )
        return {"afss": False, "reason": "dataset_too_small"}

    # ------------------------------------------------------------------
    # 2. Warmup epochs
    #    - Small datasets need longer warmup for a stable baseline model.
    #    - Large datasets can start AFSS sooner.
    # ------------------------------------------------------------------
    warmup_epochs = max(10, min(50, round(num_images / 40)))

    # ------------------------------------------------------------------
    # 3. Sampling ratios — scale with 1/sqrt(N)
    #    More images → more redundancy → can skip more aggressively.
    # ------------------------------------------------------------------
    # Reference: COCO (118k images) uses easy=0.02, moderate=0.40
    # For smaller datasets, ratios should be much higher.
    scale = math.sqrt(118000 / max(num_images, 1))  # >1 for small datasets

    # Cap scale to avoid extreme values
    scale = min(scale, 8.0)

    easy_ratio = min(0.50, 0.02 * scale)      # COCO baseline 0.02
    moderate_ratio = min(0.90, 0.40 * math.sqrt(scale))  # COCO baseline 0.40

    # ------------------------------------------------------------------
    # 4. Estimate active set size (assumes typical post-warmup split:
    #    ~70% Easy, ~20% Moderate, ~10% Hard based on YOLO convergence)
    # ------------------------------------------------------------------
    est_easy_pct = 0.70
    est_mod_pct = 0.20
    est_hard_pct = 0.10

    n_easy = int(num_images * est_easy_pct)
    n_mod = int(num_images * est_mod_pct)
    n_hard = int(num_images * est_hard_pct)

    active = (
        n_hard
        + int(n_mod * moderate_ratio)
        + int(n_easy * easy_ratio)
    )
    active_ratio = active / num_images if num_images > 0 else 0

    # ------------------------------------------------------------------
    # 5. Enforce minimum active-set constraints
    # ------------------------------------------------------------------
    min_active_ratio = 0.25  # at least 25% of images per epoch
    min_active_batches = max(5, batch_size)  # enough for ≥5 batches ideally

    # Bump ratios if active set is too small
    while active_ratio < min_active_ratio or active < batch_size * min_active_batches:
        easy_ratio = min(0.80, easy_ratio * 1.15)
        moderate_ratio = min(0.95, moderate_ratio * 1.10)
        # Recalculate
        active = (
            n_hard
            + int(n_mod * moderate_ratio)
            + int(n_easy * easy_ratio)
        )
        active_ratio = active / num_images if num_images > 0 else 0
        # Safety: if already maxed out, break to avoid infinite loop
        if easy_ratio >= 0.80 and moderate_ratio >= 0.95:
            break

    # ------------------------------------------------------------------
    # 6. Update interval
    #    - Frequent enough to track learning, but not wasteful.
    #    - Eval cost ≈ 1 full pass; balance against training cost.
    # ------------------------------------------------------------------
    update_interval = max(3, min(20, round(num_images / 200)))

    # ------------------------------------------------------------------
    # 7. Round and format
    # ------------------------------------------------------------------
    easy_ratio = round(easy_ratio, 3)
    moderate_ratio = round(moderate_ratio, 3)
    easy_thresh = round(easy_thresh, 3)
    hard_thresh = round(hard_thresh, 3)

    # Effective speedup estimate (vs. full training)
    speedup = 1.0 / active_ratio if active_ratio > 0 else 1.0
    n_updates = max(1, (epochs - warmup_epochs) // update_interval) if epochs > warmup_epochs else 0

    config = {
        "afss": True,
        "afss_easy_thresh": easy_thresh,
        "afss_hard_thresh": hard_thresh,
        "afss_easy_ratio": easy_ratio,
        "afss_moderate_ratio": moderate_ratio,
        "afss_update_interval": update_interval,
        "afss_warmup_epochs": warmup_epochs,
        # Diagnostic (not passed to AFSSManager, for logging only)
        "_active_ratio": round(active_ratio * 100, 1),
        "_speedup": round(speedup, 2),
        "_num_updates": n_updates,
    }

    # ------------------------------------------------------------------
    # 8. Print recommendation summary
    # ------------------------------------------------------------------
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("  AFSS Auto-Tune Recommendations")
    LOGGER.info("=" * 60)
    LOGGER.info(f"  Dataset        : {num_images:,} images, {num_classes} classes")
    LOGGER.info(f"  Batch size     : {batch_size}")
    LOGGER.info(f"  Epochs         : {epochs}")
    LOGGER.info("-" * 60)
    LOGGER.info(f"  easy_thresh    : {easy_thresh}  (≥ this → Easy)")
    LOGGER.info(f"  hard_thresh    : {hard_thresh}  (< this → Hard)")
    LOGGER.info(f"  easy_ratio     : {easy_ratio:.1%}  (sample rate for Easy images)")
    LOGGER.info(f"  moderate_ratio : {moderate_ratio:.1%}  (sample rate for Moderate images)")
    LOGGER.info(f"  update_interval: {update_interval} epochs")
    LOGGER.info(f"  warmup_epochs  : {warmup_epochs} epochs")
    LOGGER.info("-" * 60)
    LOGGER.info(f"  Est. active set     : ~{active:,} / {num_images:,} ({active_ratio:.0%})")
    LOGGER.info(f"  Est. training speedup: ~{speedup:.1f}x per epoch")
    LOGGER.info(f"  Total AFSS updates : ~{n_updates} times during training")
    LOGGER.info("=" * 60 + "\n")

    return config

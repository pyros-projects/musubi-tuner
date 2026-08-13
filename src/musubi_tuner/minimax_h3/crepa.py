"""CREPA (Cross-frame Representation Alignment) for MiniMax H3 LoRA training.

Backbone mode, ported from the LTX-2 branch implementation (itself based on
arXiv 2506.09229 / SimpleTuner LayerSync): hidden states of a shallow
"student" DiT block are projected through a small MLP and pulled toward the
detached hidden states of a deeper "teacher" block — for the same latent
frame AND its temporal neighbors (exponentially decayed weights). The
gradient flows through the student path into every LoRA module below it,
pressuring early layers toward temporally shared representations
(anti-shimmer), at zero extra forward passes.

H3 adaptations vs the LTX original:
- No forward hooks: under non-reentrant gradient checkpointing a hook fires
  during the unrecorded first pass and captures a graph-disconnected tensor.
  The H3 model records block *outputs* directly in its block loops (see
  ``MiniMaxH3Model.set_crepa_capture``) — checkpoint boundaries are exactly
  what stays graph-connected.
- The packed ``[text|audio|video]`` sequence is sliced to the video segment
  via the recorded ``layout.video_slice`` before frame reshaping (H3 video
  rows are frame-contiguous, T-major).
- Image batches (``latent_t < 2``) are skipped — there is nothing to align.

Only the projector trains from this loss besides the LoRA; it is registered
with Accelerate checkpointing (state dir) and is not needed at inference.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class CREPAConfig:
    student_block_idx: int = 16
    teacher_block_idx: int = 32
    lambda_crepa: float = 0.1
    tau: float = 1.0
    num_neighbors: int = 2
    schedule: str = "constant"  # constant | linear | cosine
    warmup_steps: int = 0
    max_steps: int = 0
    normalize: bool = True

    def validate(self, num_blocks: int) -> None:
        if not 0 <= self.student_block_idx < num_blocks:
            raise ValueError(f"crepa student_block_idx {self.student_block_idx} out of range (0..{num_blocks - 1})")
        if not self.student_block_idx < self.teacher_block_idx < num_blocks:
            raise ValueError(
                f"crepa teacher_block_idx {self.teacher_block_idx} must be > student ({self.student_block_idx}) "
                f"and < {num_blocks}"
            )
        if self.schedule not in ("constant", "linear", "cosine"):
            raise ValueError(f"crepa schedule must be constant|linear|cosine, got {self.schedule!r}")
        if self.num_neighbors < 1:
            raise ValueError("crepa num_neighbors must be >= 1")


def parse_crepa_args(raw_args: list[str] | None) -> CREPAConfig:
    values: dict[str, str] = {}
    for item in raw_args or []:
        if "=" not in item:
            raise ValueError(f"--crepa_args entries must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()

    config = CREPAConfig()
    casts = {
        "student_block_idx": int,
        "teacher_block_idx": int,
        "lambda_crepa": float,
        "tau": float,
        "num_neighbors": int,
        "schedule": str,
        "warmup_steps": int,
        "max_steps": int,
        "normalize": lambda v: v.lower() in ("1", "true", "yes"),
    }
    for key, value in values.items():
        if key not in casts:
            raise ValueError(f"Unknown --crepa_args key: {key} (known: {', '.join(casts)})")
        setattr(config, key, casts[key](value))
    return config


class CREPAProjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CREPAModule:
    def __init__(self, config: CREPAConfig, hidden_dim: int):
        self.config = config
        self.projector = CREPAProjector(hidden_dim)
        self._current_lambda = 0.0 if config.warmup_steps > 0 else config.lambda_crepa
        self.last_loss: float | None = None
        self._skip_logged = False

    def on_step(self, global_step: int) -> None:
        cfg = self.config
        if cfg.warmup_steps > 0 and global_step < cfg.warmup_steps:
            self._current_lambda = cfg.lambda_crepa * (global_step / cfg.warmup_steps)
            return
        if cfg.schedule == "constant" or cfg.max_steps <= 0:
            self._current_lambda = cfg.lambda_crepa
            return
        progress = min((global_step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 1.0)
        if cfg.schedule == "linear":
            self._current_lambda = cfg.lambda_crepa * (1.0 - progress)
        else:  # cosine
            self._current_lambda = cfg.lambda_crepa * 0.5 * (1.0 + math.cos(math.pi * progress))

    def compute_loss(self, capture: dict) -> torch.Tensor | None:
        """Consume a capture produced by the model's block loops; returns the
        weighted CREPA loss or None (image batch, disarmed, or lambda 0)."""
        self.last_loss = None
        student = capture.get("student")
        teacher = capture.get("teacher")
        video_slice = capture.get("video_slice")
        latent_t = int(capture.get("latent_t") or 0)
        if student is None or teacher is None or video_slice is None:
            return None
        if self._current_lambda == 0.0:
            return None
        if latent_t < 2:
            if not self._skip_logged:
                logger.info("CREPA: latent_t=%d — image batches carry no temporal signal, skipping", latent_t)
                self._skip_logged = True
            return None

        if student.ndim == 2:  # serial path [S, D]
            student = student[None]
            teacher = teacher[None]
        start, stop = video_slice
        student = student[:, start:stop, :]
        teacher = teacher[:, start:stop, :]

        batch, video_tokens, dim = student.shape
        if video_tokens % latent_t:
            logger.warning("CREPA: %d video tokens not divisible by T=%d, skipping", video_tokens, latent_t)
            return None
        tokens_per_frame = video_tokens // latent_t

        projected = self.projector(student.to(self.projector.net[0].weight.dtype))
        proj_frames = projected.reshape(batch, latent_t, tokens_per_frame, dim).mean(dim=2)
        teach_frames = teacher.float().reshape(batch, latent_t, tokens_per_frame, dim).mean(dim=2)
        loss = self._similarity_loss(proj_frames.float(), teach_frames, latent_t)
        if loss is not None:
            self.last_loss = float(loss.detach())
        return loss

    def _similarity_loss(self, proj_frames: torch.Tensor, teach_frames: torch.Tensor, t: int) -> torch.Tensor | None:
        cfg = self.config
        if cfg.normalize:
            proj_frames = F.normalize(proj_frames, dim=-1)
            teach_frames = F.normalize(teach_frames, dim=-1)
        sim = torch.bmm(proj_frames, teach_frames.transpose(1, 2))  # [B, T, T]

        loss = torch.zeros(sim.shape[0], device=sim.device, dtype=sim.dtype)
        num_terms = t
        for f in range(t):
            loss = loss - sim[:, f, f]
            for delta in range(1, cfg.num_neighbors + 1):
                weight = math.exp(-delta / cfg.tau)
                if f - delta >= 0:
                    loss = loss - weight * sim[:, f, f - delta]
                    num_terms += 1
                if f + delta < t:
                    loss = loss - weight * sim[:, f, f + delta]
                    num_terms += 1
        loss = loss.mean() / max(num_terms / t, 1.0)
        crepa_loss = loss * self._current_lambda
        if not torch.isfinite(crepa_loss):
            logger.warning("CREPA loss non-finite, skipping")
            return None
        return crepa_loss

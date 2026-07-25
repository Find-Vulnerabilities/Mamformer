"""
Multi-Path Parallel Thinking Mode for Mamformer
=================================================
True parallel reasoning: N independent thinking paths branch from
the same prompt state, then a summary synthesizes them into the
final answer.

Unlike sequential chain-of-thought, each path reasons independently
(blind to other paths), enabling diverse perspectives that are
synthesized in a summary phase.

Architecture:
    Prompt ─┬─ <|think_start|> path_1 tokens... <|think_end|> ─┐
            ├─ <|think_start|> path_2 tokens... <|think_end|> ─┤
            └─ <|think_start|> path_3 tokens... <|think_end|> ─┘
            → <|summary_start|> → [synthesis] → <|answer_start|> → answer

Each path reuses the prompt's KV cache, so they all branch from
the same understanding. The summary + answer see all paths.

Modes (inspired by SABER + ParaThinker):
  - NoThink:    Standard generation, no thinking
  - FastThink:  2 paths × 128 tokens each
  - CoreThink:  3 paths × 341 tokens each
  - DeepThink:  5 paths × 819 tokens each

Usage:
    from mamformer.thinking import ThinkingConfig, ThinkingMode, MultiPathController

    cfg = ThinkingConfig.from_preset("CoreThink")
    output = model.generate(input_ids, thinking_config=cfg)
    # output["all_paths"]     — list of per-path thinking token lists
    # output["summary_ids"]   — summary synthesis tokens
    # output["answer_ids"]    — final answer tokens

Reference:
  "ParaThinker: Native Parallel Reasoning" (Wen et al., 2025)
  "SABER: Switchable and Balanced Training" (arXiv 2508.10026, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════
# Thinking Mode
# ═══════════════════════════════════════════════════════════════════════

class ThinkingMode(Enum):
    """Available thinking intensity levels with parallel path counts."""

    NoThink = "NoThink"        # No thinking, standard generation
    FastThink = "FastThink"    # 2 paths, ~128 tokens each
    CoreThink = "CoreThink"    # 3 paths, ~341 tokens each
    DeepThink = "DeepThink"    # 5 paths, ~819 tokens each

    @property
    def num_paths(self) -> int:
        """Number of parallel reasoning paths for this mode."""
        return {
            ThinkingMode.NoThink: 0,
            ThinkingMode.FastThink: 2,
            ThinkingMode.CoreThink: 3,
            ThinkingMode.DeepThink: 5,
        }[self]

    @property
    def default_budget(self) -> int:
        """Default thinking token budget PER PATH for this mode."""
        return {
            ThinkingMode.NoThink: 0,
            ThinkingMode.FastThink: 128,
            ThinkingMode.CoreThink: 341,   # ~1024 total across 3 paths
            ThinkingMode.DeepThink: 819,   # ~4096 total across 5 paths
        }[self]

    @property
    def default_summary_budget(self) -> int:
        """Default summary synthesis token budget."""
        return {
            ThinkingMode.NoThink: 0,
            ThinkingMode.FastThink: 64,
            ThinkingMode.CoreThink: 128,
            ThinkingMode.DeepThink: 256,
        }[self]

    @classmethod
    def from_string(cls, s: str) -> "ThinkingMode":
        """Parse from string, case-insensitive."""
        for mode in cls:
            if mode.value.lower() == s.lower():
                return mode
        raise ValueError(
            f"Unknown thinking mode '{s}'. "
            f"Available: {[m.value for m in cls]}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Thinking Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ThinkingConfig:
    """
    Configuration for multi-path parallel thinking mode.

    Args:
        mode: Thinking intensity level (default: NoThink)
        num_paths: Number of parallel reasoning paths (0 = use mode default)
        think_budget: Max thinking tokens per path (0 = use mode default)
        summary_budget: Max tokens for summary synthesis (0 = use mode default)
        enabled: Master toggle — False disables thinking entirely
        show_thinking: Include thinking tokens in decoded output
        early_stop: Allow model to end thinking/path early via think_end token
        think_start_token_id: Token ID for <|think_start|> (3)
        think_end_token_id: Token ID for <|think_end|> (4)
        answer_start_token_id: Token ID for <|answer_start|> (5)
        answer_end_token_id: Token ID for <|answer_end|> (6)
        summary_start_token_id: Token ID for <|summary_start|> (7)
    """

    mode: ThinkingMode = ThinkingMode.NoThink
    num_paths: int = 0
    think_budget: int = 0
    summary_budget: int = 0
    enabled: bool = False
    show_thinking: bool = False
    early_stop: bool = True

    # ── Token IDs (defaults from MamformerTokenizer constants) ────
    think_start_token_id: int = 3
    think_end_token_id: int = 4
    answer_start_token_id: int = 5
    answer_end_token_id: int = 6
    summary_start_token_id: int = 7

    # ── Resolved properties ──────────────────────────────────────

    @property
    def effective_num_paths(self) -> int:
        """Resolved number of parallel paths."""
        if self.num_paths > 0:
            return self.num_paths
        return self.mode.num_paths

    @property
    def effective_budget(self) -> int:
        """Resolved thinking budget per path."""
        if self.think_budget > 0:
            return self.think_budget
        return self.mode.default_budget

    @property
    def effective_summary_budget(self) -> int:
        """Resolved summary synthesis budget."""
        if self.summary_budget > 0:
            return self.summary_budget
        return self.mode.default_summary_budget

    @property
    def is_active(self) -> bool:
        """Whether thinking mode is actively generating reasoning paths."""
        return self.enabled and self.mode != ThinkingMode.NoThink

    @property
    def total_think_budget(self) -> int:
        """Total thinking tokens across all paths."""
        return self.effective_num_paths * self.effective_budget

    # ── Factory methods ──────────────────────────────────────────

    @classmethod
    def disabled(cls) -> "ThinkingConfig":
        """Create a fully-disabled thinking config (default)."""
        return cls(mode=ThinkingMode.NoThink, enabled=False)

    @classmethod
    def from_preset(
        cls,
        mode: str = "NoThink",
        budget: int = 0,
        num_paths: int = 0,
        summary_budget: int = 0,
        show_thinking: bool = False,
    ) -> "ThinkingConfig":
        """
        Create from a string preset name.

        Args:
            mode: "NoThink", "FastThink", "CoreThink", or "DeepThink"
            budget: Custom per-path budget (0 = use mode default)
            num_paths: Custom path count (0 = use mode default)
            summary_budget: Custom summary budget (0 = use mode default)
            show_thinking: Include thinking tokens in output
        """
        thinking_mode = ThinkingMode.from_string(mode)
        return cls(
            mode=thinking_mode,
            think_budget=budget,
            num_paths=num_paths,
            summary_budget=summary_budget,
            enabled=(thinking_mode != ThinkingMode.NoThink),
            show_thinking=show_thinking,
        )

    def validate(self) -> None:
        """Validate config consistency."""
        if self.think_budget < 0:
            raise ValueError(f"think_budget must be >= 0, got {self.think_budget}")
        if self.num_paths < 0:
            raise ValueError(f"num_paths must be >= 0, got {self.num_paths}")
        if self.enabled and self.mode == ThinkingMode.NoThink:
            raise ValueError(
                "thinking enabled=True but mode=NoThink. "
                "Set mode to FastThink/CoreThink/DeepThink or enabled=False."
            )

    def summary(self) -> str:
        """Human-readable summary."""
        n = self.effective_num_paths
        b = self.effective_budget
        sb = self.effective_summary_budget
        return (
            f"ThinkingConfig(mode={self.mode.value}, paths={n}, "
            f"budget_per_path={b}, total_think={n*b}, summary_budget={sb})"
        )


# ═══════════════════════════════════════════════════════════════════════
# Multi-Path Thinking Controller
# ═══════════════════════════════════════════════════════════════════════

class MultiPathController:
    """
    Manages multi-path parallel thinking lifecycle.

    Phases:
      prompt → path_0 → path_1 → ... → path_N-1 → summary → answer → done

    Each path is generated independently from the prompt's KV cache,
    so paths are blind to each other (true parallel reasoning).

    After all paths complete, a summary phase synthesizes insights,
    then the answer is generated with full context.

    Usage:
        ctrl = MultiPathController(config)
        # Phase: path generation
        for path_idx in range(ctrl.num_paths):
            ctrl.start_path(path_idx)
            while not ctrl.path_done:
                next_token = model.sample(logits)
                event = ctrl.record_path_token(path_idx, next_token)
        # Phase: summary
        ctrl.start_summary()
        while not ctrl.summary_done:
            next_token = model.sample(logits)
            event = ctrl.record_summary_token(next_token)
        # Phase: answer
        ctrl.start_answer()
        while not ctrl.answer_done:
            next_token = model.sample(logits)
            event = ctrl.record_answer_token(next_token)
        result = ctrl.finalize()
    """

    def __init__(self, config: ThinkingConfig):
        self.config = config
        self.num_paths = config.effective_num_paths

        # Per-path thinking tokens: list of lists
        self._path_tokens: List[List[int]] = [[] for _ in range(self.num_paths)]
        self._path_counts: List[int] = [0] * self.num_paths
        self._path_budget_forced: List[bool] = [False] * self.num_paths

        # Summary tokens
        self._summary_tokens: List[int] = []
        self._summary_count: int = 0
        self._summary_budget_forced: bool = False

        # Answer tokens
        self._answer_tokens: List[int] = []

        # Phase tracking
        self._phase: str = "prompt"  # prompt → path:N → summary → answer → done
        self._current_path: int = -1
        self._all_paths_done: bool = False

    # ── Phase properties ─────────────────────────────────────────

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def current_path(self) -> int:
        return self._current_path

    @property
    def is_done(self) -> bool:
        return self._phase == "done"

    @property
    def path_done(self) -> bool:
        """Whether the current path has finished generating.
        Returns True only after a path has been started AND completed.
        Returns False while a path is actively generating."""
        return (not self._phase.startswith("path:")) and self._phase != "prompt"

    @property
    def summary_done(self) -> bool:
        """Whether summary phase is complete.
        Returns True once summary phase has finished (not during generation)."""
        return self._phase not in ("summary", "prompt")

    @property
    def answer_done(self) -> bool:
        """Whether answer phase is complete.
        Returns True once answer phase has finished (not during generation)."""
        return self._phase not in ("answer", "prompt")

    # ── Phase transitions ────────────────────────────────────────

    def start_path(self, path_idx: int) -> None:
        """Begin generating path `path_idx`."""
        if path_idx >= self.num_paths:
            return
        self._phase = f"path:{path_idx}"
        self._current_path = path_idx

    def _end_current_path(self) -> None:
        """Mark current path as done and advance."""
        idx = self._current_path
        # Check if all paths are done
        if idx + 1 >= self.num_paths:
            self._all_paths_done = True
            self._phase = "paths_done"
        else:
            self._phase = "between_paths"

    def start_summary(self) -> None:
        """Begin summary synthesis phase."""
        self._phase = "summary"

    def start_answer(self) -> None:
        """Begin answer generation phase."""
        self._phase = "answer"

    # ── Path token recording ─────────────────────────────────────

    def record_path_token(self, path_idx: int, token_id: int) -> Optional[str]:
        """
        Record a token during path generation.

        Returns:
            None: continue generating this path
            "end_path": model emitted think_end, path complete
            "budget_exceeded": budget limit reached for this path
        """
        if token_id == self.config.think_end_token_id:
            self._path_tokens[path_idx].append(token_id)
            self._end_current_path()
            return "end_path"

        self._path_tokens[path_idx].append(token_id)
        self._path_counts[path_idx] += 1

        if self._path_counts[path_idx] >= self.config.effective_budget:
            self._path_budget_forced[path_idx] = True
            return "budget_exceeded"

        return None

    # ── Summary token recording ──────────────────────────────────

    def record_summary_token(self, token_id: int) -> Optional[str]:
        """
        Record a token during summary phase.

        Returns:
            None: continue summary
            "end_summary": summary budget exceeded or model done
        """
        self._summary_tokens.append(token_id)
        self._summary_count += 1

        if self._summary_count >= self.config.effective_summary_budget:
            self._summary_budget_forced = True
            self._phase = "summary_done"
            return "end_summary"

        return None

    # ── Answer token recording ───────────────────────────────────

    def record_answer_token(self, token_id: int) -> Optional[str]:
        """
        Record a token during answer phase.

        Returns:
            None: continue answer
            "end_answer": EOS or answer_end marker encountered
        """
        # Stop on answer_end marker or EOS token (ID 2)
        if token_id in (self.config.answer_end_token_id, 2):
            self._phase = "done"
            return "end_answer"
            self._phase = "done"
            return "end_answer"

        self._answer_tokens.append(token_id)
        return None

    # ── Token injection helpers ──────────────────────────────────

    def get_think_start_token(self) -> int:
        return self.config.think_start_token_id

    def get_think_end_token(self) -> int:
        return self.config.think_end_token_id

    def get_summary_start_token(self) -> int:
        return self.config.summary_start_token_id

    def get_answer_start_token(self) -> int:
        return self.config.answer_start_token_id

    # ── Finalization ─────────────────────────────────────────────

    def finalize(self) -> dict:
        """Build final output dict."""
        return {
            "all_paths": self._path_tokens,
            "path_counts": self._path_counts,
            "path_budget_forced": self._path_budget_forced,
            "summary_tokens": self._summary_tokens,
            "summary_count": self._summary_count,
            "summary_budget_forced": self._summary_budget_forced,
            "answer_tokens": self._answer_tokens,
            "total_think_tokens": sum(self._path_counts),
            "phase": self._phase,
        }

    def __repr__(self) -> str:
        return (
            f"MultiPathController(phase={self._phase}, "
            f"paths={self._path_counts}/{self.num_paths}, "
            f"summary={self._summary_count}/{self.config.effective_summary_budget}, "
            f"answer={len(self._answer_tokens)})"
        )

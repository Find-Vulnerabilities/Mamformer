"""
Mamformer Configuration System
===========================
Flexible dataclass-based configuration with tiered presets
from 7B to 671B parameters, supporting configurable context
length, output length, and active parameter counts.

Tiers:
  - ultra-7b:   ~39B total, ~7.5B active,   8K context,   4K output
  - ultra-37b:  ~200B total, ~37B active,  128K context,  32K output
  - ultra-671b: ~671B total, ~37B active,    1M context, 163K output (MAX)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ═══════════════════════════════════════════════════════════════════════
# Sub-Configs
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MambaConfig:
    """Configuration for the Mamba-2 SSM block within each hybrid layer."""

    expand: int = 1
    d_state: int = 128
    d_conv: int = 4
    dt_rank: str | int = "auto"

    def __post_init__(self):
        # dt_rank resolution happens in MamformerConfig.__post_init__
        # when used standalone, keep "auto" and warn
        pass


@dataclass
class RopeConfig:
    """Configuration for Rotary Position Embeddings with YaRN."""

    theta: float = 10000.0
    use_yarn: bool = False
    yarn_scale: float = 1.0  # 1.0 = no scaling, 128.0 = 8K->1M
    yarn_original_max_seq_len: int = 8192  # Original training context length
    yarn_beta_fast: int = 32
    yarn_beta_slow: int = 1


@dataclass
class MoEConfig:
    """DeepSeek-style Mixture of Experts FFN config."""

    enabled: bool = False
    n_shared_experts: int = 2
    shared_expert_intermediate_dim: int = 2304
    n_routed_experts: int = 64
    top_k: int = 8
    expert_intermediate_dim: int = 576
    routed_expert_intermediate_dim: int = 0  # Alias
    router_temperature: float = 1.0
    aux_loss_free: bool = True
    bias_update_speed: float = 0.001
    target_expert_load: float = 1.0
    expert_dropout: float = 0.0

    def __post_init__(self):
        if self.routed_expert_intermediate_dim == 0:
            self.routed_expert_intermediate_dim = self.expert_intermediate_dim


@dataclass
class DSAConfig:
    """Differential State-Aware Attention config."""

    enabled: bool = False
    lambda_init: float = 0.8        # Initial λ = σ(0.8) ≈ 0.69, clamped to [0, 0.99]
    use_state_injection: bool = True
    state_injection_dim: int = 64
    num_attn_groups: int = 2


@dataclass
class MTPConfig:
    """Multi-Token Prediction config."""

    enabled: bool = False
    depth: int = 2
    loss_weight: float = 0.3
    mtp_d_model: int = 0


@dataclass
class STMoEConfig:
    """
    Space-Time MoE configuration.

    Couples Mamba-2's temporal hidden state with MoE routing for
    temporally coherent expert selection.

    Core formula:
        Logits_t = W_g · x_t + λ · (W_h · h_t)

    Safety mechanisms:
      - Residual Decoupling: λ ≤ lambda_max (default 0.3)
      - Dynamic Balance Lock: prevents expert over-specialization
    """

    enabled: bool = False
    lambda_init: float = 0.2            # Initial temporal guidance weight
    lambda_max: float = 0.3             # Safety clamp (residual decoupling)
    learnable_lambda: bool = True       # Whether λ is a learnable parameter
    use_balance_lock: bool = True       # Enable dynamic balance lock
    balance_lock_threshold: int = 50    # Max consecutive expert activations


@dataclass
class KDADiffConfig:
    """
    KDA-Diff: Kernelized Differential Attention with Dynamic Interleaving.

    Fuses Kimi K3's KDA interleaving efficiency with Mamformer's DSA
    (differential + SSM state injection). Uses linear attention for 75% of
    layers and full differential attention for 25%, with optional dynamic
    SSM-driven ratio control.

    KV cache reduction: ~85% vs pure DSA at 1M context.
    """

    enabled: bool = False
    linear_ratio: int = 3              # 3:1 interleaving (3 linear : 1 full)
    kernel_dim: int = 128              # Feature map dimension for linear attention
    latent_dim: int = 512              # MLA compression for full attention KV
    use_dynamic_ratio: bool = True     # SSM-state-driven dynamic interleaving


@dataclass
class CommunicativeMoEConfig:
    """
    Cross-Expert Communication config.

    Wraps a base MoE (DeepSeekMoE or SpaceTimeMoE) with cross-attention
    among selected expert outputs, enabling experts to share information
    before gate-weighted combination.

    Inspired by Kimi K3's expert collaboration mechanism.
    """

    enabled: bool = False
    n_comm_heads: int = 4          # Communication attention heads
    comm_depth: int = 1             # Communication layers (1 = lightweight)
    comm_dropout: float = 0.0       # Dropout in communication layer


@dataclass
class InterleaveConfig:
    """
    Layer-level attention interleaving configuration.

    Controls which layers run attention, SSM, or both (fusion).

    Pattern "attn_every_k" (original):
        Places one HYBRID layer (Attention ∥ SSM in parallel) every K layers.
        The hybrid layer runs both pathways simultaneously with gated fusion.

    Pattern "cross_layer" (recommended):
        Cross-layer interleaving — attention and SSM run in DIFFERENT layers.
        SSM layers pass their hidden states forward to the next attention layer
        via cross-layer state injection (SSM h_states → DSA/KDA-Diff K/V).
        Fusion layers (specified by fusion_layers) keep the original parallel
        design for final-stage deep fusion.

        Advantages over original attn_every_k hybrid:
          - 40-50% lower FLOPs per layer (each layer does one thing)
          - Eliminates signal redundancy (attention and SSM see different
            representational depths)
          - Preserves SSM→Attention cross-pollination via cross-layer state
          - More attention layers at same FLOP budget

    Pattern "custom":
        Explicit list of layer indices that should have attention.
    """

    enabled: bool = False
    pattern: str = "attn_every_k"       # "attn_every_k" | "cross_layer" | "custom"
    attn_every_k: int = 4               # Hybrid/attention layer every K layers
    first_layer_attn: bool = True       # Layer 0 always has attention
    last_layers_dense: int = 2          # Last N layers all have attention
    attention_layers: list[int] = field(default_factory=list)  # "custom" explicit list
    # Cross-layer / fusion settings
    fusion_layers: list[int] = field(default_factory=list)  # Layers keeping parallel fusion

    def resolve_attention_layers(self, n_layers: int) -> list[int]:
        """
        Compute which layer indices have attention based on the pattern.

        For "attn_every_k": hybrid layers (attention + SSM in parallel).
        For "cross_layer": attention-only layers (SSM injected from previous SSM layer).
        For "custom": explicit list.

        Returns a sorted list of layer indices with attention.
        """
        if self.pattern == "custom":
            layers = set(self.attention_layers)
            layers = {i for i in layers if 0 <= i < n_layers}
            if not layers:
                raise ValueError(
                    "InterleaveConfig.pattern='custom' but attention_layers "
                    f"is empty or contains no valid indices for n_layers={n_layers}"
                )
            return sorted(layers)

        if self.pattern == "cross_layer":
            return self._resolve_cross_layer_attention(n_layers)

        # "attn_every_k" pattern (original hybrid)
        layers: set[int] = set()
        start = 0 if self.first_layer_attn else self.attn_every_k
        for i in range(start, n_layers, self.attn_every_k):
            layers.add(i)
        if self.last_layers_dense > 0:
            for i in range(max(0, n_layers - self.last_layers_dense), n_layers):
                layers.add(i)
        return sorted(layers)

    def _resolve_cross_layer_attention(self, n_layers: int) -> list[int]:
        """
        Resolve attention layers for cross_layer pattern.

        Logic:
          1. Place attention layers every K layers (starting at layer 0)
          2. Last N layers get attention (for output quality)
          3. Fusion layers are the LAST entries in fusion_layers (parallel attn+SSM)
             and are automatically included in attention layers
          4. All other layers are SSM-only

        Example (n_layers=52, attn_every_k=4, last_layers_dense=2,
                fusion_layers=[48,49,50,51]):
          Attention: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 49, 50, 51
          Fusion:    48, 49, 50, 51 (last 4 are parallel fusion)
          SSM-only:  everything else
        """
        layers: set[int] = set()

        # Every K layers, respecting first_layer_attn flag
        start = 0 if self.first_layer_attn else self.attn_every_k
        for i in range(start, n_layers, self.attn_every_k):
            layers.add(i)

        # Last N layers dense attention
        if self.last_layers_dense > 0:
            for i in range(max(0, n_layers - self.last_layers_dense), n_layers):
                layers.add(i)

        # Fusion layers are always attention layers
        for i in self.fusion_layers:
            if 0 <= i < n_layers:
                layers.add(i)

        return sorted(layers)

    def resolve_fusion_layers(self, n_layers: int) -> set[int]:
        """
        Return the set of layer indices that should run parallel fusion
        (Attention ∥ SSM + gate). Only meaningful for cross_layer pattern.

        For attn_every_k: all attention layers are hybrid (fusion).
        For cross_layer: only explicit fusion_layers are hybrid.
        """
        if self.pattern == "cross_layer":
            return {i for i in self.fusion_layers if 0 <= i < n_layers}
        # In attn_every_k mode, all attention layers are hybrid = fusion
        return set(self.resolve_attention_layers(n_layers))

    def resolve_layer_types(self, n_layers: int) -> list[dict]:
        """
        Resolve the complete layer-type configuration.

        Returns a list of dicts, one per layer:
          {"has_attention": bool, "has_ssm": bool, "is_fusion": bool}

        cross_layer pattern:
          - Attention layers: has_attention=True, has_ssm=False (except fusion)
          - SSM-only layers: has_attention=False, has_ssm=True
          - Fusion layers: has_attention=True, has_ssm=True (parallel)

        attn_every_k pattern:
          - Hybrid layers: has_attention=True, has_ssm=True (both are fusion)
          - SSM-only layers: has_attention=False, has_ssm=True
        """
        attn_set = set(self.resolve_attention_layers(n_layers))
        fusion_set = self.resolve_fusion_layers(n_layers)

        result = []
        for i in range(n_layers):
            has_attn = i in attn_set
            is_fusion = i in fusion_set
            # In cross_layer: SSM-only layers have SSM, attention-only don't (unless fusion)
            # In attn_every_k: all attention layers have SSM (hybrid), SSM-only layers have SSM
            if self.pattern == "cross_layer":
                has_ssm = (not has_attn) or is_fusion
            else:
                has_ssm = True  # All layers have SSM in attn_every_k mode

            result.append({
                "has_attention": has_attn,
                "has_ssm": has_ssm,
                "is_fusion": is_fusion,
            })
        return result


@dataclass
class GenerationConfig:
    """
    Model-level generation limits and defaults (stored in model config).

    Each model tier records its supported context window and output length
    here. For runtime generation parameters (temperature, top_k, etc.),
    see generation.py's GenerationConfig (runtime-level).
    """

    max_context: int = 8192          # Maximum sequence length the model supports
    max_output_tokens: int = 4096    # Maximum new tokens to generate by default
    default_temperature: float = 0.7  # Default sampling temperature
    default_top_k: int = 50           # Default top-k
    default_top_p: float = 0.9        # Default top-p
    repetition_penalty: float = 1.0   # Default repetition penalty

    # ── Thinking mode defaults ──────────────────────────────────────
    thinking_enabled: bool = False     # Enable thinking mode by default
    thinking_mode: str = "NoThink"    # Default thinking intensity
    thinking_budget: int = 0          # Default thinking token budget per path (0=mode default)
    thinking_num_paths: int = 0       # Default number of parallel paths (0=mode default)
    thinking_summary_budget: int = 0  # Default summary synthesis budget (0=mode default)


# ═══════════════════════════════════════════════════════════════════════
# Main Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MamformerConfig:
    """
    Mamformer hybrid LLM configuration.

    Supports flexible parameter counts, context lengths, and output
    limits through tiered presets and MoE scaling.

    Usage:
        # Tier presets
        c = MamformerConfig.from_preset("ultra-7b")
        c = MamformerConfig.from_preset("ultra-37b")
        c = MamformerConfig.from_preset("ultra-671b")  # MAX

        # From YAML
        c = MamformerConfig.from_yaml("configs/ultra-671b-max.yaml")

        # Programmatic
        c = MamformerConfig(d_model=7168, n_layers=48, ...)
    """

    # ── Core dimensions ───────────────────────────────────────────────
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 9216
    vocab_size: int = 128000
    max_seq_len: int = 8192
    tie_word_embeddings: bool = True

    # ── Sliding Window ────────────────────────────────────────────────
    use_sliding_window: bool = False
    sliding_window: int = 4096

    # ── Sub-configs ───────────────────────────────────────────────────
    mamba: MambaConfig = field(default_factory=MambaConfig)
    rope: RopeConfig = field(default_factory=RopeConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    dsa: DSAConfig = field(default_factory=DSAConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    kda_diff: KDADiffConfig = field(default_factory=KDADiffConfig)
    st_moe: STMoEConfig = field(default_factory=STMoEConfig)
    communicative_moe: CommunicativeMoEConfig = field(default_factory=CommunicativeMoEConfig)
    interleave: InterleaveConfig = field(default_factory=InterleaveConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # ── Regularization ────────────────────────────────────────────────
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # ── Metadata ──────────────────────────────────────────────────────
    name: str = "Mamformer"
    model_type: str = "Mamformer"
    description: str = ""

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim != self.d_model // self.n_heads:
            raise ValueError(f"head_dim ({self.head_dim}) must equal d_model//n_heads ({self.d_model//self.n_heads})")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff ({self.d_ff}) must be positive")

        if isinstance(self.mamba, MambaConfig) and self.mamba.dt_rank == "auto":
            self.mamba.dt_rank = math.ceil(self.d_model / 32)

        # Sync GenerationConfig.max_context with max_seq_len
        if self.generation.max_context == 8192 and self.max_seq_len != 8192:
            self.generation.max_context = self.max_seq_len

        # Cross-validate CommunicativeMoE
        if self.communicative_moe.enabled:
            if not (self.moe.enabled or self.st_moe.enabled):
                raise ValueError(
                    "CommunicativeMoE requires MoE or ST-MoE to be enabled. "
                    "Set moe.enabled=True or st_moe.enabled=True."
                )

    # ── Derived properties ───────────────────────────────────────────

    @property
    def n_head_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def d_inner(self) -> int:
        return self.d_model * self.mamba.expand

    @property
    def total_context_length(self) -> int:
        """Effective context window with YaRN scaling."""
        if self.rope.use_yarn:
            return int(self.rope.yarn_original_max_seq_len * self.rope.yarn_scale)
        return self.max_seq_len

    @property
    def max_output_tokens(self) -> int:
        """Default max new tokens for generation."""
        return self.generation.max_output_tokens

    def _attn_params(self) -> int:
        """Per-layer attention parameters."""
        if self.dsa.enabled:
            p = (2 * self.d_model * self.n_heads * self.head_dim  # Q1, Q2
                 + 2 * self.d_model * self.n_kv_heads * self.head_dim  # K, V
                 + self.d_model * self.n_heads * self.head_dim)  # O
            if self.dsa.use_state_injection:
                p += (2 * self.d_model * self.dsa.state_injection_dim
                      + 2 * self.dsa.state_injection_dim * self.n_kv_heads * self.head_dim)
            return p
        return (self.d_model * self.n_heads * self.head_dim
                + 2 * self.d_model * self.n_kv_heads * self.head_dim
                + self.d_model * self.n_heads * self.head_dim)

    def _ssm_params(self) -> int:
        """Per-layer Mamba-2 parameters."""
        return (2 * self.d_model * self.d_inner * 2
                + self.d_inner * self.d_model
                + self.d_inner * self.mamba.d_conv
                + self.d_model * self.mamba.d_state
                + 2 * self.d_model * self.mamba.d_state
                + self.mamba.d_state + self.d_inner)

    def _ffn_total_params(self) -> int:
        """Per-layer FFN total parameters (MoE or dense)."""
        if self.moe.enabled:
            return (self.moe.n_shared_experts * 3 * self.d_model * self.moe.shared_expert_intermediate_dim
                    + self.moe.n_routed_experts * 3 * self.d_model * self.moe.routed_expert_intermediate_dim
                    + self.d_model * self.moe.n_routed_experts
                    + self.moe.n_routed_experts)
        return 3 * self.d_model * self.d_ff

    def _ffn_active_params(self) -> int:
        """Per-layer FFN active parameters."""
        if self.moe.enabled or self.st_moe.enabled:
            return (self.moe.n_shared_experts * 3 * self.d_model * self.moe.shared_expert_intermediate_dim
                    + self.moe.top_k * 3 * self.d_model * self.moe.routed_expert_intermediate_dim
                    + self.d_model * self.moe.n_routed_experts
                    + self.moe.n_routed_experts)
        return 3 * self.d_model * self.d_ff

    def _mtp_params(self) -> int:
        """MTP module total parameters."""
        if not self.mtp.enabled:
            return 0
        mtp_dim = self.mtp.mtp_d_model if self.mtp.mtp_d_model > 0 else self.d_model
        per_depth = (mtp_dim  # embedding
                     + self._attn_params()
                     + self._ssm_params()
                     + 3 * mtp_dim * (self.d_ff // 8)
                     + self.d_model * mtp_dim
                     + self.d_model)
        return self.mtp.depth * per_depth

    @property
    def num_parameters(self) -> int:
        """Total parameter count."""
        per_layer = (self._attn_params() + self._ssm_params()
                     + self._ffn_total_params()
                     + self.d_model  # gate
                     + 2 * self.d_model * 2)  # norms
        total = self.n_layers * per_layer + self.vocab_size * self.d_model + self.d_model
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.d_model
        total += self._mtp_params()
        return total

    @property
    def num_active_parameters(self) -> int:
        """Active (per-token) parameter count."""
        per_layer = (self._attn_params() + self._ssm_params()
                     + self._ffn_active_params()
                     + self.d_model + 2 * self.d_model * 2)
        total = self.n_layers * per_layer + self.vocab_size * self.d_model + self.d_model
        # MTP not active during main forward
        return total

    @property
    def num_parameters_billions(self) -> float:
        return self.num_parameters / 1e9

    @property
    def num_active_parameters_billions(self) -> float:
        return self.num_active_parameters / 1e9

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize config to a flat dictionary. Uses _SUB_CONFIG_MAPS
        so adding a field only requires updating the map, not this method."""
        d = {
            "name": self.name, "model_type": self.model_type,
            "description": self.description,
            "d_model": self.d_model, "n_layers": self.n_layers,
            "n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim, "d_ff": self.d_ff,
            "vocab_size": self.vocab_size, "max_seq_len": self.max_seq_len,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
            "dropout": self.dropout, "rms_norm_eps": self.rms_norm_eps,
            "initializer_range": self.initializer_range,
        }
        # Flatten sub-configs using shared map
        for attr_name, _prefix, field_map in _SUB_CONFIG_MAPS:
            sub = getattr(self, attr_name)
            for src_key, flat_key in field_map.items():
                d[flat_key] = getattr(sub, src_key)
        return d

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "MamformerConfig":
        """Deserialize from flat dict. Sub-config field extraction uses
        _SUB_CONFIG_MAPS — adding a field only requires updating the map."""

        # Build sub-configs from flat keys using shared field map
        sub_configs = {}
        # Mapping: attr_name → config class
        _CFG_CLASSES = {
            "mamba": MambaConfig, "rope": RopeConfig, "moe": MoEConfig,
            "dsa": DSAConfig, "kda_diff": KDADiffConfig, "mtp": MTPConfig,
            "st_moe": STMoEConfig, "communicative_moe": CommunicativeMoEConfig,
            "interleave": InterleaveConfig, "generation": GenerationConfig,
        }
        for attr_name, _prefix, field_map in _SUB_CONFIG_MAPS:
            cfg_cls = _CFG_CLASSES[attr_name]
            # Read flat-key → constructor arg
            kwargs = {}
            for src_key, flat_key in field_map.items():
                kwargs[src_key] = d.get(flat_key, _default_for_field(cfg_cls, src_key))
            sub_configs[attr_name] = cfg_cls(**kwargs)

        return cls(
            name=d.get("name", "Mamformer"),
            model_type=d.get("model_type", "Mamformer"),
            d_model=d.get("d_model", 4096),
            n_layers=d.get("n_layers", 32),
            n_heads=d.get("n_heads", 32),
            n_kv_heads=d.get("n_kv_heads", 8),
            head_dim=d.get("head_dim", 128),
            d_ff=d.get("d_ff", 9216),
            vocab_size=d.get("vocab_size", 128000),
            max_seq_len=d.get("max_seq_len", 8192),
            tie_word_embeddings=d.get("tie_word_embeddings", True),
            use_sliding_window=d.get("use_sliding_window", False),
            sliding_window=d.get("sliding_window", 4096),
            dropout=d.get("dropout", 0.0),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            initializer_range=d.get("initializer_range", 0.02),
            description=d.get("description", ""),
            **sub_configs,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MamformerConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if "model" in raw:
            d = raw["model"]
        else:
            d = raw

        # Flatten nested sub-configs from YAML using shared field map
        for nested_key, _prefix, field_map in _SUB_CONFIG_MAPS:
            _flatten_nested(d, nested_key, field_map)

        return cls.from_dict(d)

    # ── Tier Presets ──────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str = "7b") -> "MamformerConfig":
        """
        Create config from a named preset.

        Standard:
          - "7b":     ~7B dense, 8K context
          - "1b":     ~1B dense, 4K context
          - "300m":   ~300M dense, 2K context
          - "debug":  Tiny, for testing

        Ultra (MoE + DSA + MTP):
          - "ultra-7b":   ~39B total / ~7.5B active, 8K context, 4K output
          - "ultra-37b":  ~200B total / ~37B active, 128K context, 32K output
          - "ultra-371b": ~371B total / ~28B active, 256K context, 65K output
          - "ultra-671b": ~671B total / ~37B active, 1M context, 163K output (MAX)
        """
        if name in ("7b", "1b", "300m", "debug"):
            return _make_dense_preset(name)
        elif name == "ultra-7b":
            return _make_ultra_7b()
        elif name == "ultra-37b":
            return _make_ultra_37b()
        elif name == "ultra-371b":
            return _make_ultra_371b()
        elif name == "ultra-671b":
            return _make_ultra_671b()
        else:
            raise ValueError(f"Unknown preset '{name}'. "
                             f"Available: 7b, 1b, 300m, debug, "
                             f"ultra-7b, ultra-37b, ultra-371b, ultra-671b")

    def summary(self) -> str:
        """Human-readable configuration summary."""
        sep = "=" * 60
        lines = [
            sep,
            f"  {self.name} — {self.description}" if self.description else f"  {self.name}",
            sep,
            f"  d_model:         {self.d_model}",
            f"  n_layers:        {self.n_layers}",
            f"  n_heads:         {self.n_heads} (GQA, kv={self.n_kv_heads})",
            f"  head_dim:        {self.head_dim}",
            f"  d_ff (base):     {self.d_ff}",
            f"  vocab_size:      {self.vocab_size}",
            sep,
            f"  Context:         {self.total_context_length:,} tokens",
            f"  Max Output:      {self.max_output_tokens:,} tokens",
            sep,
        ]
        if self.moe.enabled:
            lines += [
                f"  MoE:             ENABLED",
                f"    Shared:        {self.moe.n_shared_experts} x dim {self.moe.shared_expert_intermediate_dim}",
                f"    Routed:        {self.moe.n_routed_experts} experts x dim {self.moe.routed_expert_intermediate_dim}",
                f"    Active:        top-{self.moe.top_k}",
                f"    Load balance:  {'aux-loss-free' if self.moe.aux_loss_free else 'auxiliary loss'}",
            ]
        if self.dsa.enabled:
            lines.append(f"  DSA:             ENABLED (lambda={self.dsa.lambda_init})")
        if self.kda_diff.enabled:
            lines.append(f"  KDA-Diff:        ENABLED ({self.kda_diff.linear_ratio}:1 linear:full, "
                         f"kernel={self.kda_diff.kernel_dim}, latent={self.kda_diff.latent_dim}, "
                         f"dynamic={'on' if self.kda_diff.use_dynamic_ratio else 'off'})")
        if self.mtp.enabled:
            lines.append(f"  MTP:             ENABLED (depth={self.mtp.depth})")
        if self.st_moe.enabled:
            lines.append(f"  ST-MoE:          ENABLED (λ={self.st_moe.lambda_init}, max={self.st_moe.lambda_max}, balance_lock)")
        if self.communicative_moe.enabled:
            lines.append(f"  CommunicativeMoE:ENABLED ({self.communicative_moe.n_comm_heads} heads, depth={self.communicative_moe.comm_depth})")
        if self.interleave.enabled:
            attn_layers = self.interleave.resolve_attention_layers(self.n_layers)
            layer_types = self.interleave.resolve_layer_types(self.n_layers)
            n_fusion = sum(1 for lt in layer_types if lt["is_fusion"])
            n_attn_only = sum(1 for lt in layer_types if lt["has_attention"] and not lt["is_fusion"])
            n_ssm_only = sum(1 for lt in layer_types if not lt["has_attention"] and lt["has_ssm"])
            parts = [f"{len(attn_layers)}/{self.n_layers} attn"]
            if n_fusion > 0:
                parts.append(f"{n_fusion} fusion (parallel)")
            if n_attn_only > 0:
                parts.append(f"{n_attn_only} attn-only (cross-layer)")
            parts.append(f"{n_ssm_only} SSM-only")
            lines.append(f"  Interleave:       ENABLED ({', '.join(parts)}, "
                         f"pattern={self.interleave.pattern})")
        if self.rope.use_yarn:
            lines.append(f"  YaRN:            scale={self.rope.yarn_scale}x, theta={self.rope.theta}")
        lines += [
            sep,
            f"  Total params:    {self.num_parameters_billions:.1f}B",
            f"  Active params:   {self.num_active_parameters_billions:.1f}B",
            f"  Expansion ratio: {self.num_parameters / max(1, self.num_active_parameters):.1f}x",
            sep,
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Preset Builders
# ═══════════════════════════════════════════════════════════════════════

# ── Shared field maps for serialization ──────────────────────────────────
# Each sub-config defines its flat-key prefix and a dict mapping
# config attributes → flat YAML keys. Adding a new config field only
# requires updating the relevant map below (not to_dict, from_dict, AND
# _flatten_nested separately).

_SUB_CONFIG_MAPS = [
    ("mamba", "mamba", {
        "expand": "mamba_expand", "d_state": "mamba_d_state",
        "d_conv": "mamba_d_conv", "dt_rank": "mamba_dt_rank",
    }),
    ("rope", "rope", {
        "theta": "rope_theta", "use_yarn": "rope_use_yarn",
        "yarn_scale": "rope_yarn_scale",
        "yarn_original_max_seq_len": "rope_yarn_original_max_seq_len",
        "yarn_beta_fast": "rope_yarn_beta_fast",
        "yarn_beta_slow": "rope_yarn_beta_slow",
    }),
    ("moe", "moe", {
        "enabled": "moe_enabled",
        "n_shared_experts": "moe_n_shared_experts",
        "shared_expert_intermediate_dim": "moe_shared_expert_intermediate_dim",
        "n_routed_experts": "moe_n_routed_experts",
        "top_k": "moe_top_k",
        "expert_intermediate_dim": "moe_expert_intermediate_dim",
        "router_temperature": "moe_router_temperature",
        "aux_loss_free": "moe_aux_loss_free",
        "bias_update_speed": "moe_bias_update_speed",
        "expert_dropout": "moe_expert_dropout",
        "target_expert_load": "moe_target_expert_load",
    }),
    ("dsa", "dsa", {
        "enabled": "dsa_enabled", "lambda_init": "dsa_lambda_init",
        "use_state_injection": "dsa_use_state_injection",
        "state_injection_dim": "dsa_state_injection_dim",
        "num_attn_groups": "dsa_num_attn_groups",
    }),
    ("kda_diff", "kda_diff", {
        "enabled": "kda_diff_enabled",
        "linear_ratio": "kda_diff_linear_ratio",
        "kernel_dim": "kda_diff_kernel_dim",
        "latent_dim": "kda_diff_latent_dim",
        "use_dynamic_ratio": "kda_diff_use_dynamic_ratio",
    }),
    ("mtp", "mtp", {
        "enabled": "mtp_enabled", "depth": "mtp_depth",
        "loss_weight": "mtp_loss_weight", "mtp_d_model": "mtp_d_model",
    }),
    ("st_moe", "st_moe", {
        "enabled": "st_moe_enabled",
        "lambda_init": "st_moe_lambda_init",
        "lambda_max": "st_moe_lambda_max",
        "learnable_lambda": "st_moe_learnable_lambda",
        "use_balance_lock": "st_moe_use_balance_lock",
        "balance_lock_threshold": "st_moe_balance_lock_threshold",
    }),
    ("communicative_moe", "comm_moe", {
        "enabled": "comm_moe_enabled",
        "n_comm_heads": "comm_moe_n_heads",
        "comm_depth": "comm_moe_depth",
        "comm_dropout": "comm_moe_dropout",
    }),
    ("interleave", "interleave", {
        "enabled": "interleave_enabled",
        "pattern": "interleave_pattern",
        "attn_every_k": "interleave_attn_every_k",
        "first_layer_attn": "interleave_first_layer_attn",
        "last_layers_dense": "interleave_last_layers_dense",
        "attention_layers": "interleave_attention_layers",
        "fusion_layers": "interleave_fusion_layers",
    }),
    ("generation", "gen", {
        "max_context": "gen_max_context",
        "max_output_tokens": "gen_max_output_tokens",
        "default_temperature": "gen_default_temperature",
        "default_top_k": "gen_default_top_k",
        "default_top_p": "gen_default_top_p",
        "repetition_penalty": "gen_repetition_penalty",
        "thinking_enabled": "gen_thinking_enabled",
        "thinking_mode": "gen_thinking_mode",
        "thinking_budget": "gen_thinking_budget",
        "thinking_num_paths": "gen_thinking_num_paths",
        "thinking_summary_budget": "gen_thinking_summary_budget",
    }),
]


def _default_for_field(cls, field_name: str):
    """Get the default value for a dataclass field by instantiating the class."""
    instance = cls()
    return getattr(instance, field_name)


def _flatten_nested(d: dict, key: str, mapping: dict) -> None:
    """Flatten a nested dict key into top-level keys using mapping."""
    if key in d and isinstance(d[key], dict):
        nested = d.pop(key)
        for nested_key, flat_key in mapping.items():
            if nested_key in nested:
                d[flat_key] = nested[nested_key]


def _make_dense_preset(name: str) -> MamformerConfig:
    """Build dense presets (7b, 1b, 300m, debug)."""
    presets = {
        "7b": dict(d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
                    head_dim=128, d_ff=9216, vocab_size=128000, max_seq_len=8192),
        "1b": dict(d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
                    head_dim=128, d_ff=5632, vocab_size=64000, max_seq_len=4096),
        "300m": dict(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4,
                      head_dim=64, d_ff=2816, vocab_size=32000, max_seq_len=2048),
        "debug": dict(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2,
                       head_dim=64, d_ff=512, vocab_size=1000, max_seq_len=128),
    }
    p = presets[name]
    return MamformerConfig(
        name=f"Mamformer-{name}",
        mamba=MambaConfig(expand=1, d_state=128 if name != "debug" else 32, d_conv=4, dt_rank="auto"),
        generation=GenerationConfig(max_context=p["max_seq_len"], max_output_tokens=p["max_seq_len"] // 2),
        **p,
    )


def _make_ultra_7b() -> MamformerConfig:
    """Tier 1: ~39B total, ~7.5B active, 8K context, 4K output."""
    return MamformerConfig(
        name="Mamformer-Ultra-7B",
        description="~39B total / ~7.5B active | 8K context | 4K output",
        d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
        d_ff=9216, vocab_size=128000, max_seq_len=8192,
        use_sliding_window=True, sliding_window=4096,
        mamba=MambaConfig(expand=1, d_state=128, d_conv=4),
        rope=RopeConfig(theta=1000000.0, use_yarn=True, yarn_scale=1.0),
        moe=MoEConfig(enabled=True, n_shared_experts=2, shared_expert_intermediate_dim=2304,
                       n_routed_experts=128, top_k=8, expert_intermediate_dim=576,
                       aux_loss_free=True, bias_update_speed=0.001),
        dsa=DSAConfig(enabled=False, lambda_init=0.8, use_state_injection=True, state_injection_dim=64),
        kda_diff=KDADiffConfig(enabled=True, linear_ratio=3, kernel_dim=128, latent_dim=512, use_dynamic_ratio=True),
        mtp=MTPConfig(enabled=True, depth=2, loss_weight=0.3),
        interleave=InterleaveConfig(enabled=True, pattern="cross_layer", attn_every_k=4,
                                     first_layer_attn=True, last_layers_dense=2,
                                     fusion_layers=[30, 31]),
        generation=GenerationConfig(max_context=8192, max_output_tokens=4096,
                                     default_temperature=0.7, default_top_k=50, default_top_p=0.9),
    )


def _make_ultra_37b() -> MamformerConfig:
    """Tier 2: ~200B total, ~37B active, 128K context, 32K output."""
    return MamformerConfig(
        name="Mamformer-Ultra-37B",
        description="~200B total / ~37B active | 128K context | 32K output",
        d_model=6144, n_layers=40, n_heads=48, n_kv_heads=8, head_dim=128,
        d_ff=12288, vocab_size=128000, max_seq_len=131072,
        use_sliding_window=True, sliding_window=16384,
        mamba=MambaConfig(expand=1, d_state=128, d_conv=4),
        rope=RopeConfig(theta=10000000.0, use_yarn=True, yarn_scale=16.0,
                        yarn_original_max_seq_len=8192),
        moe=MoEConfig(enabled=True, n_shared_experts=2, shared_expert_intermediate_dim=3072,
                       n_routed_experts=256, top_k=8, expert_intermediate_dim=768,
                       aux_loss_free=True, bias_update_speed=0.001),
        dsa=DSAConfig(enabled=False, lambda_init=0.8, use_state_injection=True, state_injection_dim=64),
        kda_diff=KDADiffConfig(enabled=True, linear_ratio=3, kernel_dim=128, latent_dim=512, use_dynamic_ratio=True),
        mtp=MTPConfig(enabled=True, depth=2, loss_weight=0.3),
        interleave=InterleaveConfig(enabled=True, pattern="cross_layer", attn_every_k=4,
                                     first_layer_attn=True, last_layers_dense=2,
                                     fusion_layers=[38, 39]),
        generation=GenerationConfig(max_context=131072, max_output_tokens=32768,
                                     default_temperature=0.7, default_top_k=50, default_top_p=0.9),
    )


def _make_ultra_371b() -> MamformerConfig:
    """Tier 3: ~371B total, ~28B active, 256K context, 65K output."""
    return MamformerConfig(
        name="Mamformer-Ultra-371B",
        description="371B total / 28B active | 256K context | 65K output",
        d_model=7168, n_layers=46, n_heads=56, n_kv_heads=8, head_dim=128,
        d_ff=14336, vocab_size=128000, max_seq_len=262144,  # 256K
        use_sliding_window=True, sliding_window=16384,  # 16K sliding window
        mamba=MambaConfig(expand=1, d_state=128, d_conv=4),
        rope=RopeConfig(theta=20000000.0, use_yarn=True, yarn_scale=32.0,
                        yarn_original_max_seq_len=8192, yarn_beta_fast=32, yarn_beta_slow=1),
        moe=MoEConfig(enabled=True, n_shared_experts=2, shared_expert_intermediate_dim=3584,
                       n_routed_experts=384, top_k=8, expert_intermediate_dim=896,
                       aux_loss_free=True, bias_update_speed=0.001),
        dsa=DSAConfig(enabled=False, lambda_init=0.8, use_state_injection=True, state_injection_dim=64),
        kda_diff=KDADiffConfig(enabled=True, linear_ratio=3, kernel_dim=128, latent_dim=512, use_dynamic_ratio=True),
        mtp=MTPConfig(enabled=True, depth=2, loss_weight=0.3),
        interleave=InterleaveConfig(enabled=True, pattern="cross_layer", attn_every_k=4,
                                     first_layer_attn=True, last_layers_dense=2,
                                     fusion_layers=[44, 45]),
        generation=GenerationConfig(max_context=262144, max_output_tokens=65536,
                                     default_temperature=0.7, default_top_k=50, default_top_p=0.9),
    )


def _make_ultra_671b() -> MamformerConfig:
    """Tier MAX: ~671B total, ~37B active, 1M context, 163K output."""
    return MamformerConfig(
        name="Mamformer-Ultra-671B",
        description="671B total / 37B active | 1M context | 163K output [MAX]",
        d_model=7168, n_layers=52, n_heads=56, n_kv_heads=8, head_dim=128,
        d_ff=14336, vocab_size=128000, max_seq_len=1048576,  # 1M
        use_sliding_window=True, sliding_window=32768,  # 32K sliding window
        mamba=MambaConfig(expand=1, d_state=128, d_conv=4),
        rope=RopeConfig(theta=50000000.0, use_yarn=True, yarn_scale=128.0,
                        yarn_original_max_seq_len=8192, yarn_beta_fast=32, yarn_beta_slow=1),
        moe=MoEConfig(enabled=True, n_shared_experts=2, shared_expert_intermediate_dim=3584,
                       n_routed_experts=640, top_k=8, expert_intermediate_dim=896,
                       aux_loss_free=True, bias_update_speed=0.001),
        dsa=DSAConfig(enabled=False, lambda_init=0.8, use_state_injection=True, state_injection_dim=64),
        kda_diff=KDADiffConfig(enabled=True, linear_ratio=3, kernel_dim=128, latent_dim=512, use_dynamic_ratio=True),
        mtp=MTPConfig(enabled=True, depth=2, loss_weight=0.3),
        interleave=InterleaveConfig(enabled=True, pattern="cross_layer", attn_every_k=4,
                                     first_layer_attn=True, last_layers_dense=2,
                                     fusion_layers=[48, 49, 50, 51]),
        generation=GenerationConfig(max_context=1048576, max_output_tokens=163800,
                                     default_temperature=0.7, default_top_k=50, default_top_p=0.9),
    )

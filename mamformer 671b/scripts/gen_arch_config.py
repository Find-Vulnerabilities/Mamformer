"""Generate _arch_verify.yaml — small model with ALL Ultra features enabled."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig, MambaConfig, RopeConfig, MoEConfig
from mamformer.config import KDADiffConfig, MTPConfig, InterleaveConfig, GenerationConfig

c = MamformerConfig(
    name="ArchVerify",
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=2,
    head_dim=64,
    d_ff=2048,
    vocab_size=8000,
    max_seq_len=512,
    mamba=MambaConfig(expand=1, d_state=64, d_conv=4),
    rope=RopeConfig(theta=10000.0),
    moe=MoEConfig(
        enabled=True,
        n_shared_experts=2,
        shared_expert_intermediate_dim=512,
        n_routed_experts=16,
        top_k=4,
        expert_intermediate_dim=256,
        aux_loss_free=True,
        bias_update_speed=0.001,
    ),
    kda_diff=KDADiffConfig(
        enabled=True,
        linear_ratio=3,
        kernel_dim=64,
        latent_dim=256,
        use_dynamic_ratio=True,
    ),
    mtp=MTPConfig(enabled=True, depth=2, loss_weight=0.3),
    interleave=InterleaveConfig(
        enabled=True,
        pattern="cross_layer",
        attn_every_k=4,
        first_layer_attn=True,
        last_layers_dense=1,
        fusion_layers=[7],
    ),
    generation=GenerationConfig(max_context=512, max_output_tokens=256),
)

c.to_yaml('configs/_arch_verify.yaml')
print(f"Generated: {c.summary()}")

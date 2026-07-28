"""Generate _1b_ultra.yaml — 1B model with ALL Ultra features enabled."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamformer.config import MamformerConfig, MambaConfig, RopeConfig, MoEConfig
from mamformer.config import KDADiffConfig, MTPConfig, InterleaveConfig, GenerationConfig

c = MamformerConfig(
    name="Mamformer-1B-Ultra",
    d_model=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    head_dim=128,
    d_ff=5632,
    vocab_size=64000,
    max_seq_len=4096,
    mamba=MambaConfig(expand=1, d_state=128, d_conv=4),
    rope=RopeConfig(theta=10000.0),
    moe=MoEConfig(
        enabled=True,
        n_shared_experts=2,
        shared_expert_intermediate_dim=1024,
        n_routed_experts=32,
        top_k=4,
        expert_intermediate_dim=384,
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
        last_layers_dense=2,
        fusion_layers=[22, 23],
    ),
    generation=GenerationConfig(max_context=4096, max_output_tokens=2048),
)

c.to_yaml('configs/_1b_ultra.yaml')
print(c.summary())

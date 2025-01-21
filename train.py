import os
import torch
from tqdm import trange
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from torch.distributed.tensor import init_device_mesh
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.models.granitemoe.modeling_granitemoe import load_balancing_loss_func
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from functools import partial

# hack in a section in the config
# "mlp_cfg": {
#       "n_expert": 4,
#       "load_balancing_loss": true
#   }
def main(
    model_name='/home/flim/data/mamba2-370m',
    dtype: str='bfloat16',
    device: str='cuda',
    total_train_steps: int = 100,
    per_device_train_batch_size: int = 4,
    max_seq_length: int = 128,
    learning_rate: float = 1e-4,
):

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    if world_size > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    model = MambaLMHeadModel.from_pretrained(
        model_name,
        device=device,
        dtype=getattr(torch, dtype)
    )
    model.train()

    # for i, block in enumerate(model.backbone.layers):
    #     from torch.distributed.algorithms._checkpoint import checkpoint_wrapper
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
    from mamba_ssm.modules.block import Block
    apply_activation_checkpointing(
        model, check_fn=lambda mod: isinstance(mod, Block)
    )


    device_mesh = None
    if world_size > 1:
        device_mesh = init_device_mesh(
            device, 
            (world_size,), mesh_dim_names=('data_parallel', )
        )

        # mamba_ssm will keep D params in float32, which will cause 
        # problems
        for name, param in model.named_parameters():
            if str(param.dtype) != f"torch.{dtype}":
                param.data = param.data.to(getattr(torch, dtype))

        # lazy to use FSDP2, use FSDP1 for now
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            ignored_modules=None,
            use_orig_params=False,
            forward_prefetch=False,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block},
            ),
            device_id=rank,
        )

    # - create optimizer (after sharding)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # some easy to debug dummy data
    input_ids = torch.ones(
        (per_device_train_batch_size, max_seq_length), 
        dtype=torch.long, 
        device=device,
    )
    labels = input_ids

    ave_loss = 0.
    for step in trange(total_train_steps, disable=rank>0):
        optimizer.zero_grad()
        out = model(input_ids)
        loss = ForCausalLMLoss(
            out.logits, labels, out.logits.shape[-1]
        )


        if out.aux_outputs is not None:
            aux_loss = load_balancing_loss_func(
                out.aux_outputs, 
                num_experts=model.config.mlp_cfg['n_expert'],
                top_k=model.config.mlp_cfg.get('top_k',2)
            )

            loss += 0.2 * aux_loss

        ave_loss = (
            step / (step+1) * ave_loss + loss.detach().item() / (step + 1)
        )

        loss.backward()
        optimizer.step()

        print ({"step": step+1, "loss": ave_loss})


if __name__ == '__main__':
    # import fire
    # fire.Fire(main)
    main()
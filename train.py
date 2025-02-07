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
import time
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
    print_itvl: int = 20,
    low_cpu_mem_mode: bool = False,
    n_expert: int = 2,
    freeze_data: bool = True,
    warmup_steps: int = 10,
):

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    if world_size > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


    from accelerate.big_modeling import init_empty_weights, init_on_device

    if not low_cpu_mem_mode:
        # from contextlib import nullcontext
        loading_context = partial(init_on_device, device=device)
    elif rank == 0:
        loading_context = partial(init_on_device, device='cpu')
    else:
        loading_context = partial(init_empty_weights, include_buffers=False)

    with loading_context():
        # TODO: weight initialization not done properly
        # need to pass in initializer_cfg to init the weights

        # config_kwargs will be passed to config
        model = MambaLMHeadModel.from_pretrained(
            model_name,
            dtype=getattr(torch, dtype),
            low_cpu_mem_mode=rank > 0 and low_cpu_mem_mode,
            config_kwargs={
                "mlp_cfg": {
                    "n_expert": n_expert,
                    "load_balancing_loss": True
                }
            }
        )

    model.train()

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
    from mamba_ssm.modules.block import Block
    apply_activation_checkpointing(
        model, check_fn=lambda mod: isinstance(mod, Block)
    )

    device_mesh = None
    if world_size > 1:
        # TODO: add device mesh to ScatterMoE
        device_mesh = init_device_mesh(
            device, 
            (world_size,), mesh_dim_names=('data_parallel', )
        )

        # mamba_ssm will keep D params in float32, which will cause 
        # problems
        for name, param in model.named_parameters():
            if str(param.dtype) != f"torch.{dtype}":
                param.data = param.data.to(getattr(torch, dtype))

        # somehow the buffers in RotaryEmb causing alot of problems
        # - so we need to ignore them in FSDP
        # - also we delete the buffers and just attach them as tensors and 
        #  move them to device manually
        for name, mod in model.named_modules():
            if 'rotary_emb' in name:
                inv_freq = mod._buffers['inv_freq']
                del mod._buffers['inv_freq']
                mod.inv_freq = inv_freq.to(device)

        from accelerate.utils.fsdp_utils import ensure_weights_retied

        # lazy to use FSDP2, use FSDP1 for now
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=False,
            forward_prefetch=False,
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block},
            ),
            device_id=rank,
            sync_module_states=True,
            param_init_fn=(
                None if rank == 0 else ensure_weights_retied(
                    lambda x: x.to_empty(device=device), 
                    model,
                    device
                )
            ),
            ignored_modules=[
                p for name, p in model.named_modules()
                if 'rotary_emb' in name
            ]
        )

    stats = {}
    if rank == 0:
        # print for debug
        print(model)
        # print ("Number parameters per device: ", sum([p.numel() for p in model.parameters()]))
        stats['num_parameters'] = sum([p.numel() for p in model.parameters()])
        torch.cuda.empty_cache()
        stats['memory_after_model_load'] = torch.cuda.memory_allocated()
        # print ("Memory after model loading", torch.cuda.memory_allocated())

    # - create optimizer (after sharding)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    def generate_data(static: bool = False):

        func = (
            partial(torch.randint, high=model.config.vocab_size)
            if not static else torch.ones
        )
        # some easy to debug dummy data
        input_ids = func(
            size=(per_device_train_batch_size, max_seq_length), 
            dtype=torch.long, 
            device=device,
        )
        labels = input_ids
        return input_ids, labels

    assert warmup_steps < total_train_steps

    ave_loss = 0.
    for step in trange(total_train_steps, disable=rank>0):

        if (
            (freeze_data and step==0) or not freeze_data
        ):
            input_ids, labels = generate_data()

        if rank == 0 and step == warmup_steps:
            t1 = time.time()

        optimizer.zero_grad()
        out = model(input_ids)

        if rank == 0 and step == 0:
            # print ("Memory after model forward", torch.cuda.memory_allocated())
            stats['memory_after_model_forward'] = torch.cuda.memory_allocated()

        loss = ForCausalLMLoss(
            out.logits, labels, out.logits.shape[-1]
        )

        distribution = None
        if out.aux_outputs is not None:
            top_k = model.config.mlp_cfg.get('top_k',2)
            aux_loss = load_balancing_loss_func(
                out.aux_outputs, 
                num_experts=model.config.mlp_cfg['n_expert'],
                top_k=top_k,
            )

            # just do on the top
            _, distribution = out.aux_outputs[0].topk(
                top_k
            ).indices.unique(return_counts=True)
            distribution = distribution.detach().cpu().tolist()

            loss += 0.2 * aux_loss

        ave_loss = (
            step / (step+1) * ave_loss + loss.detach().item() / (step + 1)
        )

        loss.backward()
        if rank == 0 and step == 0:
            # print ("Memory after model backward", torch.cuda.memory_allocated())
            stats['memory_after_model_backward'] = torch.cuda.memory_allocated()

        optimizer.step()

        if rank == 0 and (step % print_itvl) == 0:
            print ({"step": step+1, "loss": ave_loss, "distribution": distribution})

    torch.cuda.synchronize()

    if rank == 0:
        t1 = time.time() - t1
        stats['time_taken'] = t1
        print (stats)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
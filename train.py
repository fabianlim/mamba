import os
import torch
from tqdm import trange
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from torch.distributed.tensor import init_device_mesh
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.models.granitemoe.modeling_granitemoe import load_balancing_loss_func
from torch.optim import AdamW

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

    model = MambaLMHeadModel.from_pretrained(
        model_name,
        device=device,
        dtype=getattr(torch, dtype)
    )
    model.train()

    # - create optimizer (after sharding)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    device_mesh = None
    if world_size > 1:
        device_mesh = init_device_mesh(
            device, 
            (world_size,), mesh_dim_names=('data_parallel', )
        )

    # some easy to debug dummy data
    input_ids = torch.ones(
        (per_device_train_batch_size, max_seq_length), 
        dtype=torch.long, 
        device=device,
    )
    labels = input_ids

    ave_loss = 0.
    for step in trange(total_train_steps):
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
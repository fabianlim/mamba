import os
import torch
from functools import partial
from tqdm import trange
import time

try:
    from fms_acceleration_moe.utils.scattermoe import ScatterMoE, ScatteredExperts, SCATTERMOE_SPEC_HAS_GATE
    HAS_SCATTERMOE = True
except ImportError:
    HAS_SCATTERMOE = False

from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeConfig, GraniteMoeMoE

from mamba_ssm.models.mixer_seq_simple import _init_weights

def benchmark_kernels(
    d_model: int,
    d_intermediate: int,
    n_expert: int,
    top_k: int,
    batch_size: int = 4,
    seq_len: int = 4096,
    device: str = 'cuda',
    dtype: str = 'bfloat16',
    warmup: int = 10,
    runs: int = 50,
    seed: int = 42,
    check_correctness: bool = True,
):
    torch.manual_seed(seed)

    scattermoe = ScatterMoE(
        hidden_size=d_model,
        hidden_act='silu',
        intermediate_size=d_intermediate,
        num_experts=n_expert,
        has_bias=False, # hardcode this, scattermoe cannot work with bias
        mlp_arch=SCATTERMOE_SPEC_HAS_GATE, # hardcode this, use gated
        top_k=top_k,
        device=device,
        dtype=getattr(torch, dtype),
    )
    # _init_weights(scattermoe, 1, rescale_prenorm_residual=False)
    scattermoe.apply(
        partial(
            _init_weights,
            n_layer=1,
            rescale_prenorm_residual=False
        )
    )

    config = GraniteMoeConfig(
        hidden_size=d_model,
        intermediate_size=d_intermediate,
        hidden_act='silu',
        num_local_experts=n_expert,
        num_experts_per_tok=top_k,
        output_router_logits=True,
    )
    naive = GraniteMoeMoE(config).to(device=device, dtype=getattr(torch, dtype))
    naive.router.layer.weight.data = scattermoe.router.weight.data.clone()
    naive.input_linear.weight.data = torch.cat(
        [
            scattermoe.w1.weight.data.transpose(1,2).clone(),
            scattermoe.w3.weight.data.transpose(1,2).clone(),
        ], dim=1
    )
    naive.output_linear.weight.data = scattermoe.w2.weight.data.transpose(1,2).clone()

    tol = {'atol': 5e-2, 'rtol': 2e-2}

    input_scatter = torch.randn((batch_size, seq_len, d_model), device=device, dtype=getattr(torch, dtype))
    input_scatter.requires_grad = True
    input_naive = input_scatter.clone()
    input_naive.retain_grad()


    def _run_scatter(input):
        scattermoe.router.weight.grad = None
        scattermoe.w1.weight.grad = None
        scattermoe.w2.weight.grad = None
        scattermoe.w3.weight.grad = None
        input.grad = None
        output, logits = scattermoe(input)
        output.norm().backward()
        return output, logits

    def _run_naive(input):
        naive.router.layer.weight.grad = None
        naive.input_linear.weight.grad = None
        naive.output_linear.weight.grad = None
        input.grad = None
        output, logits = naive(input)
        output.norm().backward()
        return output, logits


    # warmup
    try:
        distribution = None
        for i in trange(warmup):
            output_scatter, logits_scatter = _run_scatter(input_scatter)
            output_naive, logits_native = _run_naive(input_naive)

            # count the router distribution
            _, distribution = logits_scatter.topk(2).indices.unique(return_counts=True)
            distribution = distribution.detach().cpu().tolist()

            # do some functional checks 
            if check_correctness and i == 0:
                assert torch.allclose(output_scatter, output_naive, **tol)
                assert torch.allclose(input_scatter.grad, input_naive.grad, **tol)
                assert torch.allclose(logits_scatter.to(torch.float32), logits_native, **tol)

                # this one hard to match because the naive impl casts to float
                # assert torch.allclose(scattermoe.router.weight.grad, naive.router.layer.weight.grad, **tol)
                assert torch.allclose(scattermoe.w2.weight.grad, naive.output_linear.weight.grad.transpose(1, 2), **tol)

        t1_scatter = time.time()
        for i in trange(runs):
            _run_scatter(input_scatter)
        torch.cuda.synchronize()
        t1_scatter = time.time() - t1_scatter

        t1_naive = time.time()
        for i in trange(runs):
            _run_naive(input_naive)
        torch.cuda.synchronize()
        t1_naive = time.time() - t1_naive
        results = {
            'time_scatter': t1_scatter, 'time_naive': t1_naive,
            'distribution': distribution,
        }
    except AssertionError:
        results = {'error': True}
    finally:
        del input_naive, output_naive, logits_native 
        del input_scatter, output_scatter, logits_scatter
        del scattermoe
        del naive
        torch.cuda.empty_cache()

    return {
        'd_model': d_model,
        'd_intermediate': d_intermediate,
        'n_expert': n_expert,
        'top_k': top_k,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'dtype': dtype,
        **results
    }


if __name__ == '__main__':

    from itertools import product

    # import fire
    if HAS_SCATTERMOE:
        for batch_size, n_expert in product(
            [1, 4, 8],
            [2, 4, 8, 16]
        ):
            results = benchmark_kernels(
                4096, 14336, 
                n_expert, 2, batch_size=batch_size, seed=1,
                check_correctness=True
            )
            print (results)

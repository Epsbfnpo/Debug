import json
import time

import torch
from torch.profiler import ProfilerActivity, profile


def unwrap_network(algorithm):
    network = algorithm.network
    if hasattr(network, "module"):
        network = network.module
    return network


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_params_m": total / 1e6,
        "trainable_params_m": trainable / 1e6,
    }


def _profiler_activities(device):
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    return acts


def _summarize_profiler(prof, elapsed_sec):
    total_flops = 0
    for event in prof.key_averages():
        flops = getattr(event, "flops", 0)
        if flops is not None:
            total_flops += flops

    gflops = total_flops / 1e9
    gflops_per_sec = gflops / max(elapsed_sec, 1e-12)

    return {
        "flops": int(total_flops),
        "gflops": float(gflops),
        "elapsed_sec": float(elapsed_sec),
        "gflops_per_sec": float(gflops_per_sec),
    }


def profile_eval_forward(algorithm, input_size, device, branch="fusion", batch_size=1, amp=True):
    """
    Profile one inference forward pass.
    This corresponds to validation/test per-batch computation.
    """
    algorithm.eval()

    x = torch.randn(batch_size, 3, input_size, input_size, device=device)

    # Warm-up.
    with torch.no_grad():
        for _ in range(2):
            if amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = algorithm.predict(x, branch=branch)
            else:
                _ = algorithm.predict(x, branch=branch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=_profiler_activities(device),
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        t0 = time.time()
        with torch.no_grad():
            if amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = algorithm.predict(x, branch=branch)
            else:
                _ = algorithm.predict(x, branch=branch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    out = _summarize_profiler(prof, elapsed)
    out["batch_size"] = batch_size
    out["gflops_per_sample"] = out["gflops"] / max(batch_size, 1)
    return out


def profile_train_update(algorithm, minibatch, device):
    """
    Profile one real algorithm.update() call.

    This should be used inside the real training loop only once, because it performs
    an actual optimizer update.
    """
    algorithm.train()

    image, mask, label, domain = minibatch
    image = image.to(device)
    mask = mask.to(device)
    label = label.to(device).long()
    domain = domain.to(device).long()

    if device.type == "cuda":
        torch.cuda.synchronize()

    with profile(
        activities=_profiler_activities(device),
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        t0 = time.time()
        step_vals = algorithm.update([image, mask, label, domain])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

    out = _summarize_profiler(prof, elapsed)
    out["batch_size"] = int(image.size(0))
    out["gflops_per_sample"] = out["gflops"] / max(image.size(0), 1)
    return step_vals, out


def save_compute_cost_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

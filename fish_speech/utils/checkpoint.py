from torch import nn


def assert_no_meta_tensors(model: nn.Module, source) -> None:
    """Fail when the checkpoint left part of `model` on the meta device.

    Both loaders build on the meta device to skip random initialization, so
    a tensor the checkpoint does not cover has no storage at all. Raising
    here beats handing back a module that faults on the first forward pass.
    """
    unset = [
        name
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if tensor is not None and tensor.is_meta
    ]
    if unset:
        raise RuntimeError(
            f"{source} carries no weights for {len(unset)} tensors: {unset}"
        )

"""CPU compatibility shim for end-to-end smoke tests on machines without a GPU.

The training/model code in this repo hard-codes ``.cuda()`` in many places
(train.py, models/content_extractors.py, models/spin/src/nn/hubert.py, ...).
On a machine with no GPU those calls raise, so a local end-to-end dry run is
impossible without edits everywhere.

This module patches ``torch.Tensor.cuda`` and ``torch.nn.Module.cuda`` to be
no-ops *only when CUDA is unavailable* (or when ``force=True``). Every hard-coded
``.cuda(...)`` then keeps the tensor/module on CPU, and since the checkpoints in
this repo are already loaded with ``map_location='cpu'`` and the speaker encoder
is loaded with ``map_location=config.model.device`` (set to ``cpu`` in the dummy
config), the whole pipeline runs on CPU.

On a real GPU machine this module does nothing, so it is safe to import
unconditionally.

Usage (already wired into train.py):

    import cpu_compat
    cpu_compat.activate_cpu_fallback()   # before any model is built
"""
import logging

import torch

logger = logging.getLogger(__name__)

_ACTIVATED = False


def activate_cpu_fallback(force: bool = False) -> bool:
    """Make ``.cuda()`` a no-op on CPU-only machines.

    Returns True if the fallback was activated, False if a real GPU is used.
    Idempotent: calling it twice has no extra effect.
    """
    global _ACTIVATED

    if _ACTIVATED:
        return True

    if torch.cuda.is_available() and not force:
        return False

    logger.warning(
        "[cpu_compat] CUDA not available -> activating CPU fallback. "
        "All .cuda() calls become no-ops. This is for SMOKE TESTS ONLY; "
        "real training on CPU is impractically slow."
    )

    def _tensor_cuda(self, *args, **kwargs):  # noqa: ANN001
        # Tensor is already on CPU; ignore device index / non_blocking.
        return self

    def _module_cuda(self, *args, **kwargs):  # noqa: ANN001
        return self

    torch.Tensor.cuda = _tensor_cuda
    torch.nn.Module.cuda = _module_cuda

    # LightningModule override .cuda() riêng -> phải patch cả hai nơi
    try:
        from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
        _DeviceDtypeModuleMixin.cuda = lambda self, *a, **k: self
    except Exception:
        pass
    try:
        import pytorch_lightning as pl
        pl.LightningModule.cuda = lambda self, *a, **k: self
    except Exception:
        pass
    # Only ever called under distributed training, which we never enable on CPU,
    # but guard it anyway so an accidental call does not blow up.
    torch.cuda.set_device = lambda *a, **k: None

    _ACTIVATED = True
    return True
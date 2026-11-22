"""The framework adapter: the only part of Caliper LIB that calls torch for the run itself.

The producer runtime and the managed run (``caliper._managed``) are written against this
interface, so that a later framework needs only a new adapter. The interface follows
``design/producer-runtime.md`` ("Framework Adapter Interface") and the "Framework Adapter" section
of the trajectory spec:

- ``export``: a tensor as a DLPack capsule, without a copy.
- ``framework_done`` and ``before_write``: the two sync points of the producer runtime.
- ``hooks``: call back with module outputs, gradients and each optimizer step.
- ``probe``: run the model forward on inputs that Caliper supplies.
- ``state`` and ``restore``: save and load the model and the optimizer, for a fork.

M4 implements ``state`` and ``restore``, and the loss tap of ``caliper.watch(model, optimizer)``
(the wrapper of ``torch.autograd.backward``) lives here because it patches torch. The other
functions are declared with their contracts and raise ``NotImplementedError`` until M2 fills
them. The shipped torch reductions of the data views and the forward and gradient hooks of
``caliper._watch`` stay where they are.
"""
from __future__ import annotations

import functools
import io
import threading

__all__ = ["TorchAdapter", "TORCH"]

_MODEL = "model"
_OPTIMIZER = "optimizer"


class TorchAdapter:
    """The torch adapter of the producer runtime and of the managed run.

    Attributes:
        loss (list): ``[count, tensor]``: how many 0-d tensors ``backward()`` ran on while the
            loss tap is on, and the last one, detached (``None`` before the first). The list is
            shared, so a reader keeps a reference to it and reads both items in one statement.
    """

    def __init__(self):
        self.loss: list = [0, None]
        self._tap_lock = threading.Lock()
        self._tap_users = 0
        self._tap_orig = None

    # -- M2: the producer runtime -----------------------------------------------
    def export(self, tensor):
        """Give ``tensor`` as a DLPack capsule, without a copy.

        On MPS the capsule is ``kDLMetal``, its data is the storage's ``id<MTLBuffer>`` and its
        byte offset is the storage offset times the item size. A tensor that is not contiguous is
        refused.

        Args:
            tensor (torch.Tensor): The tensor that a spec binds.

        Returns:
            PyCapsule: The DLPack capsule of the tensor.

        Raises:
            NotImplementedError: Not available yet.
        """
        raise NotImplementedError("TorchAdapter.export is not available")

    def framework_done(self) -> None:
        """Wait on the CPU until torch's work on every bound tensor is complete.

        On MPS this records a ``torch.mps.Event`` after ``optimizer.step()`` and synchronizes it.

        Raises:
            NotImplementedError: Not available yet.
        """
        raise NotImplementedError("TorchAdapter.framework_done is not available")

    def before_write(self, callback) -> None:
        """Call ``callback`` before torch writes a bound tensor again.

        For parameters this is a step pre-hook of the optimizer.

        Args:
            callback (Callable[[], None]): Called on the training thread.

        Raises:
            NotImplementedError: Not available yet.
        """
        raise NotImplementedError("TorchAdapter.before_write is not available")

    def hooks(self, model, optimizer, on_outputs, on_grads, on_step) -> None:
        """Call back with the outputs and the gradients of modules, and at each optimizer step.

        Args:
            model (torch.nn.Module): The watched model.
            optimizer (torch.optim.Optimizer | None): Its optimizer, or ``None``.
            on_outputs (Callable): Called with a module's name and its output.
            on_grads (Callable): Called with a parameter's name and its gradient.
            on_step (Callable[[], None]): Called before each optimizer step.

        Raises:
            NotImplementedError: Not available yet.
        """
        raise NotImplementedError("TorchAdapter.hooks is not available")

    def probe(self, model, inputs):
        """Run ``model`` forward on ``inputs`` that Caliper supplies, without gradients.

        Args:
            model (torch.nn.Module): The watched model.
            inputs (PyCapsule): A DLPack capsule of the inputs, for example a grid.

        Returns:
            PyCapsule: The outputs as a DLPack capsule.

        Raises:
            NotImplementedError: Not available yet.
        """
        raise NotImplementedError("TorchAdapter.probe is not available")

    # -- State of a fork -----------------------------------------------------
    def state(self, model, optimizer=None, extra=None) -> bytes:
        """Serialize the state of ``model`` and ``optimizer``, with ``extra`` fields, as bytes.

        The bytes are a ``torch.save`` file of one dict: ``model`` (the model's ``state_dict()``),
        ``optimizer`` (the optimizer's ``state_dict()``, or ``None``) and each item of ``extra``.
        Thus ``torch.load(path, weights_only=True)["model"]`` reads the weights of a file that
        holds these bytes.

        Args:
            model (torch.nn.Module): The model.
            optimizer (torch.optim.Optimizer | None): The optimizer, or ``None`` for an
                inference run.
            extra (dict | None): More fields of plain values (numbers, strings, lists, dicts).
                The keys ``model`` and ``optimizer`` are refused.

        Returns:
            bytes: The serialized state.

        Raises:
            ValueError: When ``extra`` has the key ``model`` or ``optimizer``.
        """
        import torch
        out = dict(extra or {})
        if _MODEL in out or _OPTIMIZER in out:
            raise ValueError("extra must not have the keys 'model' and 'optimizer'")
        out[_MODEL] = model.state_dict()
        out[_OPTIMIZER] = None if optimizer is None else optimizer.state_dict()
        buf = io.BytesIO()
        torch.save(out, buf)
        return buf.getvalue()

    def restore(self, data: bytes, model, optimizer=None) -> dict:
        """Load the bytes of [`state`][caliper._adapter.TorchAdapter.state] into ``model`` and
        ``optimizer``, and return the extra fields.

        The file is read with ``weights_only=True`` onto the CPU. ``load_state_dict`` then copies
        the weights to the devices of the model's parameters, and the optimizer moves its state
        to them. The optimizer state of a file without one leaves ``optimizer`` as it is.

        Args:
            data (bytes): The bytes of ``state``, for example the content of a checkpoint file.
            model (torch.nn.Module): The model, with the same parameter names and shapes.
            optimizer (torch.optim.Optimizer | None): The optimizer, or ``None``.

        Returns:
            dict: The extra fields that ``state`` was given.

        Raises:
            ValueError: When the data is not a dict with the key ``model``.
            RuntimeError: When the model does not match the state (from ``load_state_dict``).
        """
        import torch
        got = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(got, dict) or _MODEL not in got:
            raise ValueError("the data is not a state of TorchAdapter.state")
        model.load_state_dict(got.pop(_MODEL))
        opt = got.pop(_OPTIMIZER, None)
        if optimizer is not None and opt is not None:
            optimizer.load_state_dict(opt)
        return got

    # -- the loss tap of watch(model, optimizer) -----------------------------------
    def tap_on(self) -> None:
        """Keep the last 0-d tensor that ``backward()`` runs on in ``loss``; counted per user.

        A wrapper of ``torch.autograd.backward`` (which ``Tensor.backward`` calls) does it. The
        first user puts the wrapper on.
        """
        import torch
        with self._tap_lock:
            self._tap_users += 1
            if self._tap_orig is not None:
                return
            orig = torch.autograd.backward
            Tensor = torch.Tensor
            loss = self.loss

            @functools.wraps(orig)
            def backward(tensors, *args, **kwargs):
                if type(tensors) is Tensor and tensors.ndim == 0:
                    loss[1] = tensors.detach()
                    loss[0] += 1
                return orig(tensors, *args, **kwargs)
            backward._caliper_loss_tap = True
            torch.autograd.backward = backward
            self._tap_orig = orig

    def tap_off(self) -> None:
        """End one user of the loss tap. The last user takes the wrapper off.

        A wrapper that another library put on top of ours stays, and ours below it keeps calling
        through.
        """
        import torch
        with self._tap_lock:
            self._tap_users -= 1
            if self._tap_users > 0 or self._tap_orig is None:
                return
            if getattr(torch.autograd.backward, "_caliper_loss_tap", False):
                torch.autograd.backward = self._tap_orig
            self._tap_orig = None
            self.loss[1] = None


TORCH = TorchAdapter()
"""The torch adapter of this process."""

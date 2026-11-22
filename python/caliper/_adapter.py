"""The framework adapter: the only part of Caliper LIB that calls torch for the run itself.

The producer runtime and the managed run (``caliper._managed``) are written against this
interface, so that a later framework needs only a new adapter. The interface follows
``design/producer-runtime.md`` ("Framework Adapter Interface") and the "Framework Adapter" section
of the trajectory spec:

- ``export``: a tensor as a DLPack capsule, without a copy.
- ``framework_done`` and ``before_write``: the two sync points of the producer runtime.
- ``step_marks``: marks on the framework's queue that time the training steps without a wait, for
  the cadence of the producer runtime.
- ``hooks``: call back with module outputs, gradients and each optimizer step.
- ``probe``: run the model forward on inputs that Caliper supplies.
- ``state`` and ``restore``: save and load the model and the optimizer, for a fork.
- ``points`` and ``ablate``: the intervention points of the model and the ablation of one point
  through a forward hook (``design/gpu-programs.md``, Interventions), in ``caliper._ablate``.

Every function but ``probe`` is built, and ``probe`` raises ``NotImplementedError``. The loss tap of
``caliper.watch(model, optimizer)`` (the wrapper of ``torch.autograd.backward``) lives here because
it patches torch. The shipped torch reductions of the data views and the forward and gradient hooks of
``caliper._watch`` stay where they are.
"""
from __future__ import annotations

import functools
import io
import threading
import time
from collections import deque

__all__ = ["TorchAdapter", "TORCH", "StepMarks"]

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
        self._event = None                  # the torch.mps.Event of framework_done
        self._optimizers: list = []         # the optimizers that hooks() saw
        self._hooks: dict = {}              # id(model) -> the handles that hooks() made

    # -- The producer runtime ------------------------------------------------------
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
            ValueError: When the tensor is not contiguous.
            RuntimeError: When this torch cannot export the tensor's device through DLPack (torch
                2.6.0 cannot export an MPS tensor).
        """
        if not tensor.is_contiguous():
            raise ValueError("the tensor is not contiguous, and the producer runtime never copies a "
                             "tensor")
        try:
            return tensor.detach().__dlpack__()     # detach shares the storage; no copy
        except (BufferError, TypeError) as e:
            raise RuntimeError(f"this torch cannot export the tensor through DLPack: {e}") from e

    def framework_done(self) -> None:
        """Wait on the CPU until torch's work on every bound tensor is complete.

        On MPS this records a ``torch.mps.Event`` after ``optimizer.step()`` and synchronizes it.
        On CUDA it synchronizes the device, and on the CPU it returns at once.
        """
        import torch
        if torch.backends.mps.is_available():
            ev = self._event
            if ev is None:
                ev = self._event = torch.mps.Event()
            ev.record()
            ev.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    def step_marks(self):
        """Give a :class:`StepMarks` over torch's MPS queue, or ``None`` on another device.

        On the CPU and on CUDA the facade times the steps with its own clock and waits with
        :meth:`framework_done`, because the producer runtime runs on MPS only.
        """
        import torch
        if torch.backends.mps.is_available():
            return StepMarks(torch.mps.Event)
        return None

    def before_write(self, callback) -> None:
        """Call ``callback`` before torch writes a bound tensor again.

        For parameters this is a step pre-hook of the optimizer.

        Args:
            callback (Callable[[], None]): Called on the training thread.

        Raises:
            RuntimeError: When :meth:`hooks` has seen no optimizer yet.
        """
        if not self._optimizers:
            raise RuntimeError("before_write needs the optimizer that hooks() was given")
        for opt in self._optimizers:
            h = opt.register_step_pre_hook(lambda *_a, **_k: callback())
            self._hooks.setdefault(id(opt), []).append(h)

    def hooks(self, model, optimizer, on_outputs, on_grads, on_step) -> None:
        """Call back with the outputs and the gradients of modules, and at each optimizer step.

        Args:
            model (torch.nn.Module): The watched model.
            optimizer (torch.optim.Optimizer | None): Its optimizer, or ``None``.
            on_outputs (Callable): Called with a module's name and its output.
            on_grads (Callable): Called with a parameter's name and its gradient.
            on_step (Callable[[], None]): Called before each optimizer step.

        A callback that is ``None`` installs no hook, thus a caller pays only for what it uses.
        The module of ``model`` itself has the name ``""``. :meth:`unhook` removes the hooks.
        """
        made = self._hooks.setdefault(id(model), [])
        if on_outputs is not None:
            for name, mod in model.named_modules():
                made.append(mod.register_forward_hook(
                    lambda _m, _i, out, name=name: on_outputs(name, out)))
        if on_grads is not None:
            for name, p in model.named_parameters():
                if p.requires_grad:
                    made.append(p.register_post_accumulate_grad_hook(
                        lambda t, name=name: on_grads(name, t.grad)))
        if optimizer is not None:
            if all(o is not optimizer for o in self._optimizers):
                self._optimizers.append(optimizer)
            if on_step is not None:
                made.append(optimizer.register_step_pre_hook(lambda *_a, **_k: on_step()))

    def unhook(self, model) -> None:
        """Remove the hooks that :meth:`hooks` put on ``model``.

        Args:
            model (torch.nn.Module): The model that :meth:`hooks` was given.
        """
        for h in self._hooks.pop(id(model), []):
            h.remove()

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

    # -- interventions (design/gpu-programs.md, Interventions) ----------------------
    def points(self, model) -> list:
        """List the intervention points of ``model``.

        Each module of ``model.named_modules()`` is a point with its dotted name. A module with an
        integer ``num_heads``, ``n_heads`` or ``n_head`` and one child ``nn.Linear`` named
        ``out_proj``, ``o_proj``, ``c_proj`` or ``proj``, whose ``in_features`` divides by the head
        count, also gives the points ``<module>#<h>``, one for each head.

        Args:
            model (torch.nn.Module): The watched model.

        Returns:
            list[dict]: One item for each point, in the model's order, with the keys ``point``,
            ``kind`` (``"module"`` or ``"head"``), ``module``, ``head`` (an index or ``None``),
            ``type`` (the module's class name) and ``shape`` (``None`` until a forward pass).
        """
        from ._ablate import points
        return points(model)

    def ablate(self, model, point: str):
        """Set the output of one point of ``model`` to zero, from its next forward pass.

        A module's point zeroes its output with a forward hook. A head's point zeroes that head's
        slice of the input of the module's output projection with a forward pre-hook. The caller
        installs it at a step boundary, between two optimizer steps.

        Args:
            model (torch.nn.Module): The watched model.
            point (str): A point that :meth:`points` lists.

        Returns:
            Callable[[], None]: Removes the ablation; calling it twice does nothing.

        Raises:
            KeyError: When ``model`` has no such point.
        """
        from ._ablate import ablate
        return ablate(model, point)

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


class StepMarks:
    """Marks of the training steps on the framework's queue, read without a wait.

    :meth:`record` puts one mark after the work that the framework has queued so far. On MPS this
    also gives that work to the GPU, because torch's MPS queue holds its work until a sync point
    and the GPU would otherwise idle until then. :meth:`poll` returns each mark that completed since
    the last call as ``(step, lo, hi)``: the mark completed after ``lo`` and before ``hi`` on the
    caller's clock. The queue runs in order, thus a mark completes after every earlier mark.
    :meth:`wait` polls until the mark of one step completed; it is the first sync point of a pass.

    Args:
        event (Callable[[], object]): Makes one event with ``record()`` and ``query()``, such as
            ``torch.mps.Event``.
        size (int): The events in the ring. A mark whose event is needed again before it
            completed is dropped, and the caller gets no time for it.
    """

    SIZE = 256

    def __init__(self, event, size: int = SIZE):
        self._ring = [event() for _ in range(size)]
        self._next = 0
        self._pending: deque = deque()      # (step, event, time of the record), oldest first
        self._seen = float("-inf")          # the last poll that found the oldest mark pending

    def record(self, step: int, now: float) -> None:
        """Put the mark of ``step`` after the queued work; ``now`` is the caller's clock."""
        if len(self._pending) >= len(self._ring) - 1:
            self._pending.popleft()         # its event is recorded again below: no time for it
        ev = self._ring[self._next]
        self._next = (self._next + 1) % len(self._ring)
        ev.record()
        self._pending.append((step, ev, now))

    def poll(self, now: float) -> list:
        """Return ``(step, lo, hi)`` for each mark that completed since the last call."""
        out = []
        while self._pending:
            step, ev, at = self._pending[0]
            if not ev.query():
                self._seen = now
                break
            self._pending.popleft()
            out.append((step, max(at, self._seen), now))
        return out

    def wait(self, step: int, clock=time.perf_counter) -> list:
        """Poll until the mark of ``step`` and every earlier mark completed, and return what
        :meth:`poll` returned on the way. It spins for the first 0.5 ms, then naps 20 µs."""
        out = []
        start = clock()
        while True:
            now = clock()
            out += self.poll(now)
            if not self._pending or self._pending[0][0] > step:
                return out
            time.sleep(0 if now - start < 5e-4 else 2e-5)

    def pending(self) -> int:
        """The marks that did not complete yet."""
        return len(self._pending)

    def clear(self) -> None:
        """Forget the marks that did not complete yet; their times are never returned."""
        self._pending.clear()


TORCH = TorchAdapter()
"""The torch adapter of this process."""

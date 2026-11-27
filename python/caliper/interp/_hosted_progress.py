"""Progress reporting and read-only output parameter helpers for HostedInference."""
from __future__ import annotations

import json
import time

from ._hosted_resources import ALL_PANELS
from ._output import bounded_text


class _HostedProgressMixin:
    """Progress state and read-only output parameter enforcement for hosted runs."""

    def _build_progress(
        self,
        *,
        state: str,
        stage: str,
        label: str,
        done: int | None = None,
        total: int | None = None,
        panels: tuple[str, ...] | dict | None = None,
        error: str | None = None,
    ) -> dict:
        clean_label = bounded_text(str(label), 120)
        clean_error = bounded_text(str(error), 240) if error else None
        active = state not in ("ready", "idle")
        started_at = self._progress_started_at if active else None
        elapsed_s = round(max(0.0, time.time() - self._progress_started_at), 2) if self._progress_started_at else 0.0
        panel_map: dict[str, dict] = {}
        if isinstance(panels, dict):
            panel_map = panels
        elif panels:
            for pid in panels:
                panel_map[pid] = {
                    "state": state,
                    "stage": stage,
                    "label": clean_label,
                    "done": done,
                    "total": total,
                    "error": clean_error,
                }
        return {
            "state": state,
            "stage": stage,
            "label": clean_label,
            "done": done,
            "total": total,
            "started_at": round(started_at, 3) if started_at is not None else None,
            "elapsed_s": elapsed_s,
            "error": clean_error,
            "panels": panel_map,
        }

    def _progress_json(self, payload: dict | None = None) -> str:
        data = self._progress if payload is None else payload
        return json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

    def _set_readonly_param(self, param, value: str, *, raw: bool = False) -> None:
        text = value if raw else bounded_text(value)
        self._readonly_values[param.name] = text
        if param.value != text:
            param.set(text)
        self._readonly_versions[param.name] = param._version

    def _enforce_readonly_outputs(self) -> bool:
        reverted = False
        for param in (self.baseline_text, self.steered_text, self.status, self.progress, self.clamp_status):
            expected = self._readonly_values.get(param.name)
            if expected is None:
                continue
            if param.value != expected:
                param.set(expected)
                reverted = True
            self._readonly_versions[param.name] = param._version
        if reverted:
            self.controls._publish()
        return reverted

    def _active_panel_set(self, primary: tuple[str, ...]) -> tuple[str, ...]:
        ordered = list(primary)
        for pid in ALL_PANELS:
            if pid not in self._panels_ready and pid not in ordered:
                ordered.append(pid)
        return tuple(ordered)

    def _report_progress(
        self,
        *,
        stage: str,
        label: str,
        done: int | None = None,
        total: int | None = None,
        state: str | None = None,
        panels: tuple[str, ...] | dict | None = None,
        error: str | None = None,
        status_text: str | None = None,
        baseline: str | None = None,
        steered: str | None = None,
    ) -> None:
        with self._inference_lock:
            eff_state = state if state is not None else (
                self._progress_state if self._progress_state not in ("ready", "idle") else "running"
            )
            if eff_state in ("loading", "tracing", "running", "starting") and self._progress_started_at is None:
                self._progress_started_at = time.time()
            self._progress_state = eff_state
            if isinstance(panels, tuple):
                self._active_panels = panels
            payload = self._build_progress(
                state=eff_state,
                stage=stage,
                label=label,
                done=done,
                total=total,
                panels=panels,
                error=error,
            )
            self._progress = payload
        if hasattr(self, "progress"):
            self._set_readonly_param(self.progress, self._progress_json(payload), raw=True)
        if hasattr(self, "status"):
            if status_text is not None:
                self._set_readonly_param(self.status, status_text)
            elif eff_state == "ready":
                self._set_readonly_param(self.status, "ready")
            elif eff_state == "error":
                self._set_readonly_param(self.status, error or label)
            else:
                elapsed = payload["elapsed_s"]
                self._set_readonly_param(self.status, f"{payload['label']} ({elapsed:.1f}s)")
        if baseline is not None and hasattr(self, "baseline_text"):
            self._set_readonly_param(self.baseline_text, baseline)
        if steered is not None and hasattr(self, "steered_text"):
            self._set_readonly_param(self.steered_text, steered)
        if getattr(self, "controls", None) is not None:
            self.controls._publish()

    def _on_load_progress(self, *, stage: str, label: str, done: int | None = None, total: int | None = None, **_extra) -> None:
        self._report_progress(
            state="loading",
            stage=stage,
            label=label,
            done=done,
            total=total,
            panels=ALL_PANELS,
        )

    def _on_backend_progress(
        self,
        *,
        stage: str,
        label: str,
        done: int | None = None,
        total: int | None = None,
        baseline: str | None = None,
        steered: str | None = None,
        **_extra,
    ) -> None:
        if stage in ("attribution", "pruning", "publishing"):
            panels = self._active_panel_set(("graph",))
            state = "tracing"
        elif stage in ("output", "intervention", "generation", "sweep"):
            panels = self._active_panel_set(("output",))
            state = "running" if self._progress_state == "ready" else self._progress_state
        elif stage == "examples":
            panels = ("examples",)
            state = "running" if self._progress_state == "ready" else self._progress_state
        else:
            panels = ALL_PANELS
            state = "running"
        self._report_progress(
            state=state,
            stage=stage,
            label=label,
            done=done,
            total=total,
            panels=panels,
            baseline=baseline,
            steered=steered,
        )

    def _on_service_progress(self, *, stage: str, label: str, done: int | None = None, total: int | None = None, **_extra) -> None:
        if stage == "examples":
            if self._progress_state == "ready":
                self._progress_started_at = time.time()
            self._report_progress(
                state="running" if self._progress_state == "ready" else self._progress_state,
                stage="examples",
                label=label,
                done=done,
                total=total,
                panels=("examples",),
            )
        elif stage == "examples_done":
            self._panels_ready.add("examples")
            if not self._in_refresh and self._progress_state != "tracing":
                self._report_progress(state="ready", stage="ready", label="Ready", panels={}, status_text="ready")

    def _set_progress(self, state: str, message: str | None = None) -> None:
        text = state if message is None else message
        if state in ("loading", "tracing", "running", "starting"):
            self._progress_started_at = time.time()
            stage = "attribution" if state == "tracing" else state
            label = "Tracing prompt" if state == "tracing" else (text or state)
            panels = self._active_panel_set(("graph",)) if state == "tracing" else ALL_PANELS
            self._report_progress(
                state=state,
                stage=stage,
                label=label,
                panels=panels,
                status_text=text,
            )
        elif state == "error":
            err_panels = self._active_panels or ALL_PANELS
            self._report_progress(
                state="error",
                stage="error",
                label="Error",
                panels=err_panels,
                error=text,
                status_text=text,
            )
        else:
            self._report_progress(
                state=state,
                stage="ready" if state == "ready" else state,
                label="Ready" if state == "ready" else text,
                panels={},
                status_text=text,
            )
            self._progress_started_at = None

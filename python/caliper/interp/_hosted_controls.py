"""Feature selection, intervention reconciliation, and slider clamp helpers for HostedInference."""
from __future__ import annotations

from copy import deepcopy
import math

from .._controls_spec import intervention_key
from .._intervention_protocol import internal_forward
from ._output import bounded_text
from ._prompt import prompt_inputs


class _HostedControlsMixin:
    """Selection, prompt-length reconciliation, and slider clamp controls."""

    def _owned_record(self):
        if self._owned_clamp is None:
            return None
        iv = self.controls._ivs.get(intervention_key(self._owned_clamp))
        if iv is None or iv.clear or iv.state == "ended" or iv.spec is not self._owned_clamp:
            self._owned_clamp = None
            return None
        return iv

    def _discard_owned_clamp(self) -> None:
        iv = self._owned_record()
        if iv is None:
            return
        self._owned_clamp = None
        iv.clear = True
        self.controls._iv_revision += 1
        self.controls._iv_due = True
        self.controls.counters["interventions_in"] += 1
        self.controls._update_busy()
        self.controls.cond.notify_all()

    def _apply_interventions(self) -> None:
        with self.controls.lock:
            changed, _ready = self.controls._intervention_boundary()
            self._owned_record()
            self.controls._update_busy()
        if changed:
            self.controls._publish()

    def _reconcile_interventions_for_prompt(self, prompt: str) -> bool:
        """Keep active interventions coherent with the new prompt's token count."""
        ids = prompt_inputs(self.backend, self.tokenizer, prompt)["input_ids"]
        seq_len = int(ids.shape[1])
        changed = False
        reset_clamp_param = False

        with self.controls.lock:
            owned = self._owned_record()
            for iv in list(self.controls._ivs.values()):
                if iv.clear or iv.state == "ended":
                    continue
                spec = iv.applied_spec if iv.applied_spec is not None else iv.spec
                if not isinstance(spec, dict):
                    continue
                positions = spec.get("positions")
                if positions is None:
                    continue
                valid_positions = [p for p in positions if isinstance(p, int) and 0 <= p < seq_len]
                if len(valid_positions) == len(positions):
                    continue
                is_owned = owned is not None and iv is owned
                if not valid_positions:
                    if is_owned:
                        self._discard_owned_clamp()
                        reset_clamp_param = True
                    else:
                        iv.clear = True
                        self.controls._iv_revision += 1
                        self.controls._iv_due = True
                        self.controls.counters["interventions_in"] += 1
                        self.controls._update_busy()
                        self.controls.cond.notify_all()
                    changed = True
                else:
                    updated = deepcopy(spec)
                    updated["positions"] = valid_positions
                    try:
                        validated = self.interventions.executor.validate(updated)
                    except ValueError:
                        validated = None
                    if validated is None:
                        if is_owned:
                            self._discard_owned_clamp()
                            reset_clamp_param = True
                        else:
                            iv.clear = True
                            self.controls._iv_revision += 1
                            self.controls._iv_due = True
                            self.controls.counters["interventions_in"] += 1
                    else:
                        old_key = intervention_key(iv.spec)
                        new_key = intervention_key(validated)
                        if new_key != old_key:
                            if is_owned:
                                self._discard_owned_clamp()
                            else:
                                iv.clear = True
                                self.controls._iv_revision += 1
                                self.controls._iv_due = True
                                self.controls.counters["interventions_in"] += 1
                        refused = self.controls._intervene({"set": validated})
                        if refused is None and is_owned:
                            record = self.controls._ivs.get(new_key)
                            if record is not None:
                                self._owned_clamp = record.spec
                        elif refused is not None:
                            if is_owned:
                                self._discard_owned_clamp()
                                reset_clamp_param = True
                            else:
                                iv.clear = True
                                self.controls._iv_revision += 1
                                self.controls._iv_due = True
                                self.controls.counters["interventions_in"] += 1
                    self.controls._update_busy()
                    self.controls.cond.notify_all()
                    changed = True

        if changed:
            self._apply_interventions()
        if reset_clamp_param:
            self.clamp.set(0.0)
            self._clamp_version = self.clamp._version
        if self.selection is not None and self.selection["position"] >= seq_len:
            self.selection = None
        if hasattr(self.interventions, "invalidate_source_cache"):
            self.interventions.invalidate_source_cache()
        elif hasattr(getattr(self.interventions, "executor", None), "invalidate_cache"):
            self.interventions.executor.invalidate_cache()
        return changed

    @staticmethod
    def _pick_important_feature(graph) -> int:
        if graph is None:
            return -1
        features = [(i, n) for i, n in enumerate(graph.nodes) if n.get("kind") == "feature"]
        if not features:
            return -1

        def score(item):
            idx, node = item
            influence = node.get("influence")
            if influence is None:
                influence = node.get("score")
            if influence is None:
                influence = node.get("importance")
            inf_val = (
                abs(float(influence))
                if isinstance(influence, (int, float)) and math.isfinite(influence)
                else 0.0
            )
            act = node.get("activation", 0.0)
            act_val = abs(float(act)) if isinstance(act, (int, float)) and math.isfinite(act) else 0.0
            non_initial_layer = 1 if int(node.get("layer", 0)) > 0 else 0
            return (non_initial_layer, inf_val, act_val, -idx)

        best_idx, _ = max(features, key=score)
        return best_idx

    def _reconcile_selection(self, graph, *, auto_select: bool = True) -> None:
        row = -1
        if graph is not None and self.selection is not None:
            row = next(
                (i for i, n in enumerate(graph.nodes)
                 if n["kind"] == "feature" and all(n[k] == v for k, v in self.selection.items())),
                -1,
            )
        if row == -1 and auto_select and graph is not None:
            row = self._pick_important_feature(graph)
        if row == -1:
            self.selection = None
        else:
            record = graph.nodes[row]
            self.selection = {k: record[k] for k in ("layer", "feature", "position")}
        if self.selected.value != row:
            self.selected.set(row)
        self._selected_version = self.selected._version
        self._selection_graph = graph

    def _refresh_examples(self) -> list:
        if self.service is None:
            return []
        if self.selection is None or self.index is None:
            self.service.publish_empty_examples()
            return []
        return self.service.publish_feature_examples(
            self.selection["layer"],
            self.selection["feature"],
            limit=16,
        )

    def _update_selection(self) -> None:
        graph = self.service.graph
        if graph is not self._selection_graph:
            self._reconcile_selection(graph)
            self._refresh_examples()
        if self.selected._version != self._selected_version:
            node = self.selected.value
            self.selection = None
            self._selected_version = self.selected._version
            if node == -1:
                self._refresh_examples()
                return
            if graph is None or node >= len(graph.nodes):
                raise ValueError("selected_node requires a completed trace")
            record = graph.nodes[node]
            if record["kind"] != "feature":
                raise ValueError("selected_node must identify a feature")
            self.selection = {k: record[k] for k in ("layer", "feature", "position")}
            self._refresh_examples()

    def update_controls(self) -> None:
        self._enforce_readonly_outputs()
        clamp_changed = self.clamp._version != self._clamp_version
        self._clamp_version = self.clamp._version
        try:
            prompt = self.prompt.value
            prompt_inputs(self.backend, self.tokenizer, prompt)
            if prompt != self._last_prompt:
                self._reconcile_interventions_for_prompt(prompt)
                self._last_prompt = prompt
            self.checkpoint_prompt = prompt
            self._update_selection()
            if clamp_changed:
                self._set_clamp()
        finally:
            self._sync_clamp_status(preserve_edit=clamp_changed)

    def _feature_maximum(self, layer: int, feature: int, position: int) -> float:
        maximum = float(self.index.show(layer, feature)["maximum"])
        if maximum > 0 and math.isfinite(maximum):
            return maximum
        graph = self.service.graph
        if graph is not None:
            for node in graph.nodes:
                if (node["kind"] == "feature" and node["layer"] == layer
                        and node["feature"] == feature and node["position"] == position):
                    act = abs(float(node["activation"]))
                    if act > 0 and math.isfinite(act):
                        return act
        raise ValueError("selected feature has no positive activation maximum")

    def _set_clamp(self) -> bool:
        self._check_stop()
        if self.selection is None:
            raise ValueError("clamp requires a selected feature")
        layer, feature, position = (self.selection[k] for k in ("layer", "feature", "position"))
        maximum = self._feature_maximum(layer, feature, position)
        point = f"layers.{layer}.mlp_out" if self.dictionary.kind == "transcoder" else self.dictionary.metadata["write_points"][0]
        spec = self.interventions.executor.validate({
            "kind": "clamp",
            "point": point,
            "feature": feature,
            "positions": [position],
            "value": float(self.clamp.value) * maximum,
            "dictionary": self.interventions.executor.dictionary_id,
        })
        with self.controls.lock:
            owned = self._owned_record()
            if owned is not None and intervention_key(owned.spec) != intervention_key(spec):
                self._discard_owned_clamp()
            refused = self.controls._intervene({"set": spec})
            if refused is not None:
                raise ValueError(refused[1])
            record = self.controls._ivs[intervention_key(spec)]
            self._owned_clamp = record.spec
        self._apply_interventions()
        return True

    def _sync_clamp_status(self, *, preserve_edit: bool = False) -> None:
        reset_input = False
        if self.selection != self._clamp_selection:
            self._clamp_selection = deepcopy(self.selection)
            with self.controls.lock:
                if not preserve_edit and self.clamp._version == self._clamp_version and self.clamp.pending is None:
                    self.clamp._value = 0.0
                    self.clamp._version += 1
                    self._clamp_version = self.clamp._version
                    reset_input = True
            if reset_input:
                self.controls._publish()
        selected = "No feature selected."
        if self.selection is not None:
            selected = (
                f"Selected layer {self.selection['layer']}, feature {self.selection['feature']}, "
                f"position {self.selection['position']}."
            )
        live = "No slider clamp."
        with self.controls.lock:
            owned = self._owned_record()
            if owned is not None and owned.hook is not None and owned.applied_spec is not None:
                s = owned.applied_spec
                live = f"Live slider clamp: {s['point']}, feature {s['feature']}, position {s['positions'][0]}, value {s['value']:g}."
        msg = bounded_text(f"{selected} Clamp input: {self.clamp.value:g} times maximum. {live}")
        self._set_readonly_param(self.clamp_status, msg)

    def apply_clamp(self) -> str:
        self._check_stop()
        with internal_forward():
            self._preset_tuples = None
            self._preset_cell = None
            self.update_controls()
            self._set_clamp()
            self._sync_clamp_status()
            self.update_output(force=True)
        return "The selected feature clamp is applied."

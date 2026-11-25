"""Published Dallas cases from the pinned executable tutorial."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack, nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

import torch

from ._frozen import _trace_features
from ._intervene import (InterventionError, _DictionarySnapshot, _group_memory,
                         _ACTIVE_REPLAYS, _hook_muted, _uses_reference_inputs,
                         _reference_input_scope, _reference_features)
from ._memory import autorelease_scope
from ._output import bounded_text, forward_logits, prompt_inputs
from ._points import _architecture, capture

PROTOCOL = 'published_dallas_v1'

SOURCE_REVISION = '82996f144b3371ae858ea5713590d004c218f0c2'

PROMPTS = {'dallas': 'Fact: the capital of the state containing Dallas is',
 'oakland': 'Fact: the capital of the state containing Oakland is',
 'shanghai': 'Fact: the capital of the country containing Shanghai is',
 'vancouver': 'Fact: the capital of the territory containing Vancouver is'}

GROUPS = {'capital': [(15, 4494, 4), (6, 4662, 4), (4, 7671, 4), (3, 13984, 4), (1, 1000, 4)],
 'state': [(6, 4012, 7), (0, 13727, 7)],
 'Texas': [(20, 15589, 9), (19, 7477, 9), (16, 25, 9), (4, 13154, 9), (14, 2268, 9), (7, 6861, 9)],
 'preposition followed by place name': [(19, 1445, 10), (18, 6101, 10)],
 'capital cities / say a capital city': [(21, 5943, 10),
                                         (17, 7178, 10),
                                         (7, 691, 10),
                                         (16, 4298, 10)],
 'California': [(22, 4367, 10), (21, 2464, 10), (6, 13909, 9), (8, 14641, 9), (14, 12562, 9)],
 'China': [(19, 12274, 9),
           (14, 12274, 9),
           (6, 6811, 9),
           (4, 11570, 9),
           (4, 4257, 9),
           (19, 12274, 10),
           (18, 7639, 10)],
 'British Columbia': [(18, 1025, 10)],
 'Say Austin': [(23, 12237, 10)],
 'Say Sacramento': [(19, 9209, 10)],
 'Say Victoria': [(21, 2236, 10)]}

CASES = {10: {'label': 'Baseline', 'operations': []},
 14: {'label': 'Suppress Say a Capital',
      'operations': [{'group': 'capital cities / say a capital city',
                      'factor': -2,
                      'activation_prompt': 'dallas'}]},
 16: {'label': 'Suppress Capital',
      'operations': [{'group': 'capital', 'factor': -2, 'activation_prompt': 'dallas'}]},
 18: {'label': 'Suppress Texas',
      'operations': [{'group': 'Texas', 'factor': -2, 'activation_prompt': 'dallas'}]},
 20: {'label': 'Suppress State',
      'operations': [{'group': 'state', 'factor': -2, 'activation_prompt': 'dallas'}]},
 24: {'label': 'Replace Texas With California',
      'operations': [{'group': 'Texas', 'factor': -2, 'activation_prompt': 'dallas'},
                     {'group': 'California', 'factor': 2, 'activation_prompt': 'oakland'}]},
 26: {'label': 'Replace Texas With China',
      'operations': [{'group': 'Texas', 'factor': -2, 'activation_prompt': 'dallas'},
                     {'group': 'China', 'factor': 2, 'activation_prompt': 'shanghai'}]},
 29: {'label': 'Replace Texas With British Columbia',
      'operations': [{'group': 'Texas', 'factor': -2, 'activation_prompt': 'dallas'},
                     {'group': 'British Columbia', 'factor': 2, 'activation_prompt': 'vancouver'}]}}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


@contextmanager
def attention_patterns(model, patterns):
    """Replay baseline attention probabilities with current projected values."""
    probes = _architecture(model)
    with ExitStack() as stack:
        for name, pattern in patterns.items():
            probe = probes[name]
            module = model.get_submodule(probe.module)
            if probe.family == 'fixture':
                def replace(_module, _args, _output, pattern=pattern):
                    return pattern
                stack.callback(module.pattern.register_forward_hook(replace).remove)
                continue
            if model.config.model_type != 'gemma2':
                raise ValueError('published attention replay requires Gemma or the CPU fixture')
            values = []
            def read(_module, _args, value, values=values):
                values[:] = [value]
            def replace(_module, args, values=values, pattern=pattern, module=module):
                if len(values) != 1:
                    raise InterventionError('attention replay has no current value projection')
                value = values.pop()
                batch, length, width = value.shape
                head_dim = module.head_dim
                value = value.view(batch, length, width // head_dim, head_dim).transpose(1, 2)
                if pattern.shape[-2:] != (length, length):
                    raise InterventionError('attention replay prompt length differs from preparation')
                heads = pattern.shape[1]
                if heads % value.shape[1]:
                    raise InterventionError('attention replay head dimensions differ')
                value = value.repeat_interleave(heads // value.shape[1], dim=1)
                mixed = (pattern.to(value) @ value).transpose(1, 2).reshape(batch, length, heads * head_dim)
                return (mixed,) + args[1:]
            stack.callback(module.v_proj.register_forward_hook(read).remove)
            stack.callback(module.o_proj.register_forward_pre_hook(replace).remove)
            def report(_module, _args, output, pattern=pattern):
                weights = pattern.expand(output[0].shape[0], -1, -1, -1)
                return (output[0], weights, *output[2:])
            # Semantic capture must observe the pattern that produced the output.
            stack.callback(module.register_forward_hook(report, prepend=True).remove)
            stack.callback(values.clear)
        yield


class PublishedCases:
    """Prepare independent donors and evaluate complete published cases atomically."""

    def __init__(self, model, tokenizer, interventions, *, guard=None, output_directory):
        self.model, self.tokenizer, self.interventions = model, tokenizer, interventions
        self.guard = guard
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.preparation = None
        self.patterns = {}
        self.values = {}
        self.baseline_logits = None
        self.last_result = None
        self.last_attempt = None
        self.sequence = 0
        self.directory = None
        self._identity = None
        self._handles = []
        self._forward_scopes = []

    def _reserve(self, amount):
        if self.guard is not None:
            _DictionarySnapshot._admit(_group_memory(self.model, self.guard), amount, force=True)

    def _model_identity(self):
        from ._dictionary import _HASH_CHUNK_BYTES
        self._reserve(4 * _HASH_CHUNK_BYTES)
        return self.interventions.executor._identity()

    def _write_json(self, name, value):
        target = self.directory / name
        temporary = self.directory / ('.' + uuid.uuid4().hex + '.json')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
                stream.write('\n')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def prepare(self):
        """Capture clean source values before any named action timer starts."""
        if self.preparation is not None:
            return deepcopy(self.preparation)
        service, started = self.interventions, time.perf_counter()
        executor = service.executor
        device = next(self.model.parameters()).device
        probes = _architecture(self.model)
        patterns = [name for name, probe in probes.items() if probe.event == 'pattern']
        reads = executor.dictionary.metadata['read_points']
        original = _uses_reference_inputs(executor.dictionary)
        dtype = next(self.model.parameters()).dtype
        if not executor._default_dictionary_id:
            raise ValueError('published preparation requires an exact dictionary content identity')
        if self.model.config.model_type == 'gemma2' and dtype != torch.bfloat16:
            raise ValueError('published Dallas requires the bfloat16 model and dictionary')
        if self.model.config.model_type == 'gemma2' and self.guard is None:
            raise ValueError('published Gemma requires a bounded memory guard')
        if self.model.config.model_type == 'gemma2' and not original:
            raise ValueError('published Gemma requires qualified original encoder coordinates')
        if any(layer.W_enc.dtype != dtype or layer.W_dec.dtype != dtype for layer in executor.dictionary.layers):
            raise ValueError('published dictionary precision differs from the model')
        with service._ctl.lock, service._operation('preparation'):
            revision = service._ctl._iv_revision
            with service._ctl._outside_lock(), executor.dictionary_admission(), service.suspend(), torch.no_grad():
                identity = self._model_identity()
                tokens, values, saved_patterns = {}, {}, {}
                baseline = None
                nodes = sorted({node for group in GROUPS.values() for node in group})
                for source, prompt in PROMPTS.items():
                    with autorelease_scope(device):
                        inputs = prompt_inputs(self.model, self.tokenizer, prompt)
                        tokens[source] = inputs['input_ids'][0].tolist()
                        length = len(tokens[source])
                        if any(layer >= len(reads) or position >= length or feature >= executor.dictionary.layers[layer].W_enc.shape[0]
                               for layer, feature, position in nodes):
                            raise ValueError('published feature coordinates exceed the source model or prompt')
                        width = self.model.config.hidden_size
                        largest = max(layer.W_enc.shape[0] for layer in executor.dictionary.layers)
                        # Capture buffers, encoder work, attention replay, and full output evidence.
                        self._reserve(length * width * len(reads) * 16 + length * largest * 32
                                      + len(reads) * length * length * 256 + self.model.config.vocab_size * 64)
                        points = (patterns if source == 'dallas' else []) if original else (
                            [*reads, *patterns] if source == 'dallas' else reads)
                        with capture(self.model, points) as captured, (
                            _reference_input_scope(self.model) if original else nullcontext(captured)
                        ) as encoder_inputs:
                            logits = forward_logits(self.model, inputs, guard=self.guard)[0, -1]
                            values[source] = {}
                            for layer in sorted({node[0] for node in nodes}):
                                weights, coordinate = executor.dictionary.layers[layer], encoder_inputs[reads[layer]]
                                features = (_reference_features(weights, coordinate) if original else
                                            _trace_features(weights, coordinate, self.guard))
                                for node in nodes:
                                    if node[0] == layer:
                                        value = features[0, node[2], node[1]]
                                        if not torch.isfinite(value).item():
                                            raise ValueError('published source activation is nonfinite')
                                        values[source][node] = float(value)
                                del features, coordinate, value
                        if source == 'dallas':
                            baseline = logits.detach().clone()
                            saved_patterns = {name: captured[name].detach().clone() for name in patterns}
                        captured.clear()
                if self._model_identity() != identity:
                    raise ValueError('published source model changed during preparation')
            service._check_revision(revision)
        binding = {'protocol': PROTOCOL, 'source_revision': SOURCE_REVISION, 'producer': 'caliper',
                   'model_content_hash': identity[4], 'model_settings': dict(identity[5]),
                   'dictionary_identity': executor.dictionary_id, 'dtype': str(dtype),
                   'reference_input': deepcopy(executor.dictionary.metadata.get('reference_input')),
                   'prompts': {source: {'text': prompt, 'sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                                        'token_ids': tokens[source]} for source, prompt in PROMPTS.items()},
                   'freeze_attention': True, 'freeze_norms': False, 'freeze_mlps': False,
                   'decoder_accumulation': 'sorted_position_feature_index_add', 'softmax_dtype': str(dtype)}
        binding['id'] = _digest(binding)
        binding['seconds'] = time.perf_counter() - started
        self.output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = Path(tempfile.mkdtemp(prefix='published-dallas-', dir=self.output_directory))
        self._write_json('preparation.json', binding)
        self.values, self.patterns, self.baseline_logits = values, saved_patterns, baseline
        self._identity, self.preparation = identity, binding
        self._handles.append(self.model.register_forward_pre_hook(self._enter_forward, with_kwargs=True))
        self._handles.append(self.model.register_forward_hook(self._exit_forward, always_call=True))
        return deepcopy(binding)

    def _enter_forward(self, module, args, kwargs):
        self._forward_scopes.append(None)
        if self.interventions.suspended:
            return
        inputs = args[0] if args else kwargs.get('input_ids')
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            return
        tokens = self.preparation['prompts']['dallas']['token_ids']
        if inputs.shape[1] != len(tokens) or not torch.all(inputs == inputs.new_tensor(tokens)).item():
            return
        executor = self.interventions.executor
        for replay in _ACTIVE_REPLAYS.get():
            if replay.model is self.model:
                executor = replay.executor
        hooks = [hook for hook in executor._hooks if not getattr(hook, 'removed', False) and not _hook_muted(hook)]
        if any(not hasattr(hook, 'spec') for hook in hooks):
            return
        specs = [hook.spec for hook in hooks]
        if self.matching_case(PROMPTS['dallas'], specs) is not None:
            scope = self._attention_scope(executor, inputs.shape)
            scope.__enter__()
            self._forward_scopes[-1] = scope

    @contextmanager
    def _attention_scope(self, executor, shape):
        executor._published_forward_depth += 1
        try:
            original = _uses_reference_inputs(executor.dictionary)
            if original:
                # The arithmetic capture retains one input per layer until scope exit.
                self._reserve(shape[0] * shape[1] * self.model.config.hidden_size * len(executor.dictionary.layers)
                              * next(self.model.parameters()).element_size())
            with (attention_patterns(self.model, self.patterns),
                  _reference_input_scope(self.model) if original else nullcontext()):
                yield
        finally:
            executor._published_forward_depth -= 1

    def _exit_forward(self, module, args, output):
        if self._forward_scopes:
            scope = self._forward_scopes.pop()
            if scope is not None:
                scope.__exit__(None, None, None)

    def close(self):
        while self._handles:
            self._handles.pop().remove()
        while self._forward_scopes:
            scope = self._forward_scopes.pop()
            if scope is not None:
                scope.__exit__(None, None, None)
        self.patterns.clear()
        self.baseline_logits = None

    def case(self, cell):
        """Return exact specifications and Caliper source values for one case."""
        if self.preparation is None:
            raise ValueError('published case preparation is incomplete')
        if type(cell) is not int or cell not in CASES:
            raise ValueError('unknown published case')
        members, specifications = [], []
        dictionary = self.interventions.executor.dictionary
        for operation in CASES[cell]['operations']:
            group, source, factor = (operation[key] for key in ('group', 'activation_prompt', 'factor'))
            for layer, feature, position in GROUPS[group]:
                activation = self.values[source][(layer, feature, position)]
                value = activation * factor
                specifications.append({'kind': 'clamp', 'point': dictionary.metadata['write_points'][layer],
                                       'feature': feature, 'positions': [position], 'value': value,
                                       'dictionary': self.interventions.executor.dictionary_id})
                members.append({'layer': layer, 'feature': feature, 'position': position, 'group': group,
                                'canonical_id': f'feature:{layer}:{feature}:{position}',
                                'activation_prompt': source, 'activation': activation, 'factor': factor, 'value': value})
        return specifications, members

    def matching_case(self, prompt, specifications):
        from .._controls import intervention_key
        if self.preparation is None or prompt != PROMPTS['dallas']:
            return None
        def normalized(specs):
            return {intervention_key(s): {k: v for k, v in s.items()
                                         if k not in ('op', 'from_step', 'until_step')} for s in specs}
        current = normalized(specifications)
        for cell in CASES:
            if current == normalized(self.case(cell)[0]):
                return cell
        return None

    @contextmanager
    def output_scope(self, prompt, specifications):
        if self.matching_case(prompt, specifications) is None:
            yield
        else:
            with attention_patterns(self.model, self.patterns):
                yield

    def run(self, cell, *, prepare=None, check=None):
        """Execute a fresh forward and commit only after all candidate checks pass."""
        self.sequence += 1
        sequence, started = self.sequence, time.perf_counter()
        case_id = f'{PROTOCOL}.cell{cell}'
        evidence = []
        try:
            specs, members = self.case(cell)
            def evaluate():
                executor = self.interventions.executor
                if (executor.dictionary_id != self.preparation['dictionary_identity']
                        or not executor._default_dictionary_id or self._model_identity() != self._identity):
                    raise ValueError('published preparation no longer matches the model')
                sources = {source: prompt_inputs(self.model, self.tokenizer, prompt)
                           for source, prompt in PROMPTS.items()}
                if any(source_inputs['input_ids'][0].tolist() != self.preparation['prompts'][source]['token_ids']
                       for source, source_inputs in sources.items()):
                    raise ValueError('published preparation no longer matches the tokenizer')
                inputs = sources['dallas']
                self._reserve(self.model.config.vocab_size * 64)
                with attention_patterns(self.model, self.patterns):
                    logits = forward_logits(self.model, inputs, guard=self.guard)[0, -1]
                if self._model_identity() != self._identity:
                    raise ValueError('published model changed during the case forward')
                pair = torch.stack((self.baseline_logits, logits))
                probabilities = pair.softmax(-1)
                top_probs, top_ids = probabilities.topk(5, dim=-1)
                result = {'case_id': case_id, 'cell': cell, 'label': CASES[cell]['label'], 'producer': 'caliper',
                          'preparation_id': self.preparation['id'], 'sequence': sequence, 'group_values': members,
                          'argmax_token_ids': probabilities.argmax(-1).tolist(),
                          'top_token_ids': top_ids[:, 0].tolist(),
                          'baseline_top_text': bounded_text(self.tokenizer.decode([int(probabilities[0].argmax())])),
                          'baseline_top_probability': float(probabilities[0].max()),
                          'top_five': [[{'id': token, 'text': bounded_text(self.tokenizer.decode([token])), 'probability': probability}
                                        for token, probability in zip(ids, scores)]
                                       for ids, scores in zip(top_ids.tolist(), top_probs.tolist())],
                          'logits_dtype': str(pair.dtype), 'softmax_dtype': str(probabilities.dtype)}
                import numpy as np
                filename = f'cell{cell}-{sequence}-{uuid.uuid4().hex}.npz'
                target = self.directory / filename
                evidence.append(target)
                # The archive stores every vocabulary entry. Replies contain only its descriptor.
                with target.open('xb') as stream:
                    np.savez(stream, logits=pair.float().cpu().numpy(), probabilities=probabilities.float().cpu().numpy())
                result['evidence'] = {'file': str(target.relative_to(self.output_directory)),
                                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                                      'shape': list(probabilities.shape), 'rows': ['baseline', 'case'],
                                      'storage_dtype': 'float32',
                                      'native_values_preserved': pair.dtype in (torch.bfloat16, torch.float16, torch.float32)}
                result['compute_export_seconds'] = time.perf_counter() - started
                metadata = self.directory / f'cell{cell}-{sequence}.json'
                evidence.append(metadata)
                self._write_json(metadata.name, result)
                result['metadata_file'] = str(metadata.relative_to(self.output_directory))
                return result, pair
            from ._publish import _publication_scope
            with _publication_scope(self.model), autorelease_scope(next(self.model.parameters()).device):
                result, _ = self.interventions.replace_atomic(
                    specs, evaluate, prepare=None if prepare is None else lambda evaluated: prepare(evaluated[1]),
                    check=check)
            result['action_seconds'] = time.perf_counter() - started
            # Only this committed result makes a candidate export authoritative.
            self.last_result = result
            self.last_attempt = {'case_id': case_id, 'sequence': sequence, 'status': 'ok'}
            return deepcopy(result)
        except BaseException:
            for target in evidence:
                target.unlink(missing_ok=True)
            self.last_attempt = {'case_id': case_id, 'sequence': sequence, 'status': 'error'}
            raise

    def status(self, prompt, specifications):
        return {'protocol': PROTOCOL, 'prepared': self.preparation is not None,
                'preparation_id': None if self.preparation is None else self.preparation['id'],
                'preparation_seconds': None if self.preparation is None else self.preparation['seconds'],
                'preparation_file': None if self.directory is None else str((self.directory / 'preparation.json').relative_to(self.output_directory)),
                'active_cell': self.matching_case(prompt, specifications),
                'groups': {name: [{'layer': layer, 'feature': feature, 'position': position}
                                  for layer, feature, position in nodes] for name, nodes in GROUPS.items()},
                'cases': [{'cell': cell, 'action': f'{PROTOCOL}.cell{cell}', **deepcopy(case)} for cell, case in CASES.items()],
                'last_attempt': deepcopy(self.last_attempt), 'result': deepcopy(self.last_result)}

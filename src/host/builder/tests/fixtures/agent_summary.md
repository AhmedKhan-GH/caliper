Done — the plot is live and updating. What I built:

- **Spec** `attention-entropy-per-layer` with six sources, one per layer, bound to the live views `live:entropy.l0` through `live:entropy.l5` from your training session (updated at ~8 Hz).
- **One series panel** with six lines, one per layer, over training steps.

The rendered view shows exactly what I'd expect while training: entropy starting near 3.2 nats for all layers and falling as attention sharpens, with layer 0 settling highest (~1.45) and the deeper layers lower (~1.1). New points will keep appearing as training proceeds.

If you'd like, I can add a companion panel (e.g. the live `attn.entropy` [6, 8] heatmap or the per-head attention matrices from `live:attn`) beside it, or save this as an instrument in the workspace.

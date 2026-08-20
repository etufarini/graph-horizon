---
name: safe-optimization-loop
description: >-
  Iteratively optimize latency or throughput with a measure-change-verify loop
  that preserves the engine's chat output and repository invariants. Use for
  Rust inference hot paths, prefill/decode profiling, or performance tuning
  where correctness, isolated commits, and reproducible evidence are required.
---

# Safe Optimization Loop

Rispondi sempre in italiano. Mantieni comandi e identificatori nella lingua del
codice.

Obbedisci ad `AGENTS.md`. Un candidato che viola struttura approvata, singola
responsabilità o limite produttivo di 200 linee è fallito quanto un test rosso.

Ripeti:

```text
misura -> cambia una cosa -> prova correttezza -> rimisura -> commit o ripristino
```

## Regole

1. La correttezza blocca il commit.
2. Non modificare test, riferimenti o soglie per far passare un candidato.
3. Un'ottimizzazione logicamente indipendente corrisponde a un commit.
4. Mantieni identici modello, prompt, contesto, KV, feature, profilo e hardware
   tra baseline e candidato.
5. Cambi di ordine floating point, layout o precisione non sono presumibilmente
   bit-exact. Esplorali solo se esiste un oracle approvato; altrimenti non
   committarli.
6. Non introdurre dipendenze o astrazioni per un guadagno ipotetico.

## Fase 0 — Baseline

Verifica il tree e la feature target:

```sh
git status --short
cargo test --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features <cpu|vulkan|vulkan-hybrid|metal|metal-hybrid>
```

Usa un artefatto 3B Q4_K_M autenticato da `support/models.tsv`. Q8_0 è soltanto
un caso negativo del gate pubblico e non è un target di parity reale. Registra
SHA, contesto, KV e commit.

Leggi [references/correctness-oracle.md](references/correctness-oracle.md) per
la matrice esatta e
[references/measuring-prefill-decode.md](references/measuring-prefill-decode.md)
per la misura.

## Fase 1 — Profilo

Misura l'API pubblica:

```sh
support/profiling/profile.sh --model "$GRAPH_HORIZON_MODEL" \
  --backend <backend> --context 4096 --kv f16
```

Individua il costo dominante da dati osservabili: prefill, TTFT, decode, memoria
o placement. Consulta
[references/rust-optimization-catalog.md](references/rust-optimization-catalog.md)
solo dopo il profilo.

## Fase 2 — Classificazione

- Bit-exact: elimina lavoro, allocazioni, copie o sincronizzazioni senza
  riordinare l'aritmetica. Può passare alla parity esatta.
- Layout/numerica: cambia raggruppamento, FMA, SIMD reduction, tiling, precisione
  o quantizzazione. Non committare autonomamente senza un oracle numerico
  approvato dalla specifica corrente.
- Architetturale: cambia ownership, threading model, dipendenze o interfacce.
  Escludi dal loop e pianifica separatamente.

## Fase 3 — Modifica

Applica il diff minimo. Non fare cleanup opportunistici. Verifica ogni file Rust
toccato contro `AGENTS.md` e documenta invarianti nei punti di mutazione.

## Fase 4 — Gate

Esegui prima la correttezza:

```sh
cargo fmt --check
cargo test --workspace --no-default-features --features cpu
support/testing/parity-check.sh \
  --models-dir "$GRAPH_HORIZON_MODELS_DIR" \
  --model-id "$GRAPH_HORIZON_MODEL_ID" \
  --backend <backend> --kv f16 \
  --reference-server "$GRAPH_HORIZON_REFERENCE_SERVER"
```

Per un backend ibrido mixed aggiungi `--weights-percent 25 --expect-mode mixed`
e richiedi layer CPU e GPU positivi. Non trasformare E15 Vulkan in fallback CPU.

Solo dopo un gate verde rimisura con lo stesso comando baseline. Un guadagno
deve superare la varianza osservata e non peggiorare metriche o memoria fuori
dal rumore.

## Fase 5 — Decisione

- Tutti i gate verdi e guadagno reale: commit immediato con metrica prima/dopo.
- Divergenza, regressione o guadagno nel rumore: ripristina solo il candidato e
  registrane il motivo.
- Candidato numerico/architetturale: non lasciarlo nel tree; riportalo come
  proposta non applicata.

Ferma il loop quando non restano candidati misurati, l'oracle è instabile o il
budget richiesto è esaurito. Alla fine ripeti matrice test, parity e benchmark,
poi riporta baseline, stato finale e commit mantenuti.

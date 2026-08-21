# Oracle di correttezza

## Gate locale

La base obbligatoria non richiede modelli:

```sh
cargo test --workspace --no-default-features --features cpu
cargo test -p graph_horizon_engine --no-default-features --features vulkan error_matrix
cargo test -p graph_horizon_engine --no-default-features --features vulkan-hybrid error_matrix
```

I test sintetici coprono dimensioni 3B/8B/14B, rifiuto Q8_0, profilo Q4_K_M,
KV f16/int8 e formati backend interni.

## Gate reale

Usa soltanto gli artefatti Q4_K_M con dimensione e SHA fissati in
`support/models.tsv`. Un file capability-compatible ma non corrispondente resta
`compatible/unverified`.

`support/testing/parity-check.sh` invoca i test ignorati con:

- template fisso e prompt ID contro tokenizer di riferimento;
- greedy decoding;
- primi token ID contro CLI di riferimento;
- feature backend e contesto espliciti.

Esempio:

```sh
support/testing/parity-check.sh \
  --models-dir "$GRAPH_HORIZON_MODELS_DIR" \
  --model-id "$GRAPH_HORIZON_MODEL_ID" \
  --backend cpu --kv f16 \
  --reference-server "$GRAPH_HORIZON_REFERENCE_SERVER"
```

Ripeti per f16/int8 quando il candidato tocca la KV. Ripeti per entrambi i
profili quando tocca formati o dispatch. Per Vulkan, E15 è un risultato atteso
se il file non entra; non è autorizzazione a cambiare backend. Una riga hybrid
mixed usa lo stesso Q4_K_M autenticato, contesto esplicito, 25% e richiede layer
positivi su entrambi i lati.

## Determinismo

Mantieni costanti:

- byte e SHA del GGUF;
- prompt e system prompt;
- contesto e `max_tokens`;
- schema KV;
- feature/backend e profilo Cargo;
- riferimento e commit;
- hardware e condizioni termiche.

Non rigenerare un riferimento per accettare un candidato. Una differenza nei
primi ID greedy blocca il commit.

## Candidati numerici

La milestone corrente non conserva un oracle a tolleranza per modifiche
floating-point. Un candidato che cambia ordine, layout o precisione può essere
misurato in un branch temporaneo, ma non viene committato autonomamente. Per
abilitarlo serve una specifica separata con metrica e soglie approvate.

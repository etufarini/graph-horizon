# Misurare prefill e decode

Usa gli esempi pubblici mantenuti, non timer ad hoc.

```sh
support/profiling/profile.sh --model "$GRAPH_HORIZON_MODEL" \
  --backend <backend> --context 4096 --kv f16

cargo run --release --no-default-features --features <backend> \
  --example bench -- "$GRAPH_HORIZON_MODEL" --context 4096 --kv f16 \
  --prompt "Ciao" --max-tokens 32 --warmup 1 --reps 5
```

Registra:

- prompt tokens e prompt tok/s;
- TTFT;
- decode tok/s;
- modalità e layer CPU/GPU;
- breakdown pesi, KV, scratch, fixed, staging, crossing e reserve;
- media/deviazione e numero di ripetizioni.

## Igiene

Usa build release o fast identiche tra baseline e candidato. Esegui warmup,
mantieni la macchina inattiva e controlla throttling. Non confrontare run con
contesto, KV o placement diversi.

Un miglioramento deve superare il rumore della baseline. Controlla anche metriche
non target e memoria: un guadagno decode che peggiora prefill o trasforma il
placement non è automaticamente accettabile.

## Diagnosi

- TTFT alto: concentra il profilo su prompt/prefill.
- Decode lento: aumenta `max_tokens` mantenendo fisso il prompt.
- Mixed lento: confronta crossing e numero layer, senza cambiare split tra
  baseline e candidato.
- Crescita con contesto: confronta f16/int8 separatamente; non mescolare i
  risultati.

Non aggiungere tracing permanente per un'unica misura. Se manca attribuzione,
preferisci un contatore temporaneo ristretto e rimuovilo prima del commit.

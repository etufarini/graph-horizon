<!--
Questo riferimento definisce i gate di correttezza usati dal loop di
ottimizzazione; descrive protocollo e limiti, non risultati di qualifica.
-->

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

- template fisso e prompt ID esatti contro il tokenizer di riferimento;
- sedici token oracle ottenuti dal riferimento e applicati in teacher forcing;
- presenza di ogni token oracle nella top two locale deterministica, con
  registrazione separata degli ID top-1 locali;
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

Non rigenerare un riferimento per accettare un candidato. L'assenza di un token
oracle dalla top two locale blocca la riga; una differenza negli ID top-1
registrati non fallisce da sola questo gate bounded. Se il candidato promette
identità esatta, deve passare anche il gate esatto scelto prima della misura.

## Candidati numerici

Il protocollo corrente conserva un oracle bounded top-two per sedici passi
teacher-forced. Non è una tolleranza universale: un candidato che cambia ordine,
layout o precisione può usarlo solo quando quel criterio misura il rischio
dichiarato. Modifiche KV richiedono inoltre i propri gate di payload, layout e
qualità; ogni altra divergenza numerica richiede metrica e soglia approvate prima
di osservare il risultato.

<!--
Questa mappa assegna un confine di ownership a ogni dominio sorgente; non
definisce comportamento runtime né registra risultati di validazione.
-->

# Ownership dei moduli

Ogni dominio possiede un confine stretto.

- `api/`: tipi pubblici, configurazione, request, ruoli ed eventi.
- `family/mistral/detect.rs`: gate architettura e Q4_K_M prima delle
  allocazioni backend.
- `family/mistral/config.rs`: metadati e dimensioni derivate.
- `family/mistral/version.rs`: valori di release Ministral 3
  Instruct/Reasoning 2512 e System prompt Reasoning fissato; non esegue dispatch.
- `family/mistral/tensors.rs`: nomi, shape e dtype Q4_K_M ammessi.
- `family/mistral/tokenizer/profile.rs`: selezione privata della policy chat
  dai tre nomi Reasoning esatti, senza autenticare il GGUF.
- `family/mistral/tokenizer/reasoning.rs`: encoding dei soli marker del System
  prompt Reasoning posseduto dalla release.
- `family/mistral/tokenizer/`: BPE e pre-tokenizzazione Tekken incorporati nel GGUF.
- `family/mistral/template.rs`: sequenza chat, System implicito/esplicito e
  renderer fisso senza eseguire Jinja.
- `family/mistral/parity.rs`: vettori e criteri della parità Reasoning
  esclusivamente test-only.
- `tests/semantic.rs`: qualificatore Reasoning-only test-only; possiede
  corpus, sampling, scoring, stop telemetry e marker Reasoning, senza policy
  runtime o server.
- `family/mistral/graph/`: ordine denso condiviso da CPU e Vulkan; non decide
  placement.
- `family/mistral/hybrid/`: accounting, selezione del prefisso CPU/suffisso GPU,
  ownership dei due backend e unico attraversamento del residuo.
- `backend/`: contratto tensoriale e implementazioni CPU/Vulkan. Non conosce
  profili pubblici della famiglia.
- `gguf/`: parsing limitato, metadati e indice tensoriale, compreso Q8_0 solo
  per diagnostica e rifiuto; non alloca backend.
- `kv_cache/`: schema f16/int8, layout e lifecycle per richiesta.
- `sampling.rs` e `rng.rs`: scelta token deterministica o campionata.
- `harness/`: misure family-neutral e prove esterne; non è parte del runtime.

Il flusso è:

```text
GGUF -> contratto Mistral -> prompt ID -> piano/backend -> grafo -> decoder -> eventi
```

Invarianti:

- il contratto viene validato prima di qualsiasi allocazione backend;
- il piano hybrid non cambia dopo il load;
- la KV segue il layer;
- un piano misto attraversa il residuo in un solo punto;
- cancellazione ed errori liberano ogni risorsa e non espongono dettagli
  interni.

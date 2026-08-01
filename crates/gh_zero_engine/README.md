<!--
  Contratto tecnico del crate: API, backend, memoria e confini di validazione.
-->

# gh_zero_engine

`gh_zero_engine` è il runtime di inferenza text-to-text per
`general.architecture=mistral3`. La sua API pubblica riceve messaggi chat,
esegue prefill/decode e produce eventi incrementali senza dipendere da console,
HTTP o Web UI.

## Feature backend

Il crate non seleziona un backend di default: il consumer abilita esattamente
uno dei tre profili.

```sh
cargo check -p gh_zero_engine --no-default-features --features cpu
cargo check -p gh_zero_engine --no-default-features --features vulkan
cargo check -p gh_zero_engine --no-default-features --features hybrid
```

- `cpu`: percorso completo e riferimento numerico portabile.
- `vulkan`: modello interamente GPU; memoria insufficiente o device non
  disponibile sono errori senza fallback.
- `hybrid`: compila entrambi e seleziona al load tutto GPU, massimo suffisso GPU
  contiguo con prefisso CPU, oppure tutto CPU.

Il piano hybrid è immutabile dopo il load. Solo il piano misto copia il residuo
CPU→GPU, una volta per passaggio; la KV di ogni layer resta sul suo backend. Il
report contiene modalità, split, conteggi layer e breakdown CPU/GPU di pesi, KV,
scratch, fixed, staging, crossing e reserve.

### Riferimento semantico M3

La validazione M3 applica una policy separata e solo test: usa il medesimo load
hybrid come probe, ma accetta come backend finale soltanto all-GPU oppure
CPU-only. Un probe `mixed` viene distrutto prima di generare token e il modello
viene riaperto CPU-only; questo non modifica il supporto mixed delle sessioni
produttive né aggiunge opzioni a `EngineConfig`.

`Engine::placement()` fornisce il placement finale e il suo breakdown di
memoria pianificata; non espone la VRAM grezza disponibile. Un errore dopo la
selezione finale resta un failure senza retry o fallback. Il comando e il
protocollo operativo sono nella [guida degli script](../../support/README.md).

## Contratto Ministral

Il riconoscimento di architettura, tensori e quantizzazione è capability-based
e non autentica un file tramite nome, directory, byte totali o `general.name`.
Quest'ultimo seleziona soltanto la policy chat: il valore esatto e
case-sensitive `ministral-3B-Reasoning-2512` abilita il profilo privato
Ministral 3 3B Reasoning 2512, mentre ogni altra variante contenente
`Reasoning` viene rifiutata prima dell'allocazione backend. Sono supportati
pubblicamente due profili GGUF:

- `Q8_0`: matrici Q8_0 e ausiliari monodimensionali F32;
- `Q4_K_M`: matrici Q4_K/Q6_K e ausiliari monodimensionali F32.

La configurazione Instruct è dimension-generic. Le righe 3B, 8B e 14B sono
fixture sintetiche realistiche, non whitelist. Il supporto Reasoning è invece
limitato al 3B 2512 Q8_0/Q4_K_M; Reasoning 8B/14B non è supportato. Solo i due
file Reasoning 3B alla revisione fissata nel piano sono verificati come
artefatti reali. I valori fissati per Ministral 3 Instruct/Reasoning 2512
appartengono al modulo privato
[`family/mistral/version.rs`](src/family/mistral/version.rs), che possiede anche
il System prompt Reasoning della release.

Il backend mantiene attivazioni FP16, residuo/logits FP32 e pesi
F16/Q4_K/Q5_K/Q6_K/Q8_0. Questa superficie interna non aggiunge profili
Ministral pubblici.

## Facciata pubblica

La radice espone soltanto:

- engine/config/placement: `Engine`, `EngineConfig`, `BackendMemory`,
  `PlacementReport`;
- chat ed eventi: `Message`, `Role`, `Request`, `SamplingParams`, `Event`,
  `GenerationStats`, `EventSink`, `render_chat_prompt`;
- ispezione Ministral/GGUF: `MistralConfig`, `TekkenTokenizer`,
  `WeightProfile`, `GgufFile`, `GgufValue`, `GgmlType`, `TensorInfo`;
- selezione KV: `KvQuant`;
- `harness::throughput` con `BenchConfig`, `Stat`, `ThroughputReport` e `run`.

Backend, sampling, layout KV, metadati allocativi e grafo restano interni.
L'harness attivo misura esclusivamente lo stream pubblico dell'engine; non
espone profiling di buffer o accessi privati al modello.

## Chat

```rust
use gh_zero_engine::{Engine, EngineConfig, Message, Request, Role, SamplingParams};

let engine = Engine::new(
    std::path::Path::new("/path/model.gguf"),
    EngineConfig::default(),
)?;
let request = Request {
    messages: vec![Message {
        role: Role::User,
        content: "Ciao".into(),
    }],
    sampling: SamplingParams::greedy(),
    max_tokens: 128,
};
engine.generate(request, &mut |event| {
    println!("{event:?}");
    true
});
# Ok::<(), color_eyre::Report>(())
```

Una sequenza valida contiene al massimo un `System` iniziale, poi `User` e
`Assistant` strettamente alternati, e termina con `User`. Il template è fisso:
`tokenizer.chat_template` non viene eseguito. Nel profilo Reasoning, l'assenza
di un `System` iniziale inserisce il System prompt fissato dalla release; un
`System` esplicito, anche vuoto, lo sostituisce. Il contenuto del chiamante non
viene interpretato come token speciale. I marker `[THINK]` e `[/THINK]`
generati dal modello restano testo raw in `TextDelta`, senza un nuovo evento o
canale pubblico.

Gli eventi pubblici sono:

- `TextDelta(String)`;
- `Finished(GenerationStats)`;
- `Error("generation failed")`.

Ogni generazione emette un solo evento terminale. Se il sink restituisce
`false`, l'esecuzione viene cancellata, le risorse sono liberate e non viene
emesso alcun altro evento.

## Memoria e contesto

`EngineConfig` accetta contesto, schema KV `f16|int8`, thread CPU e budget
Vulkan. `EngineConfig.context_tokens = None` applica la policy del motore:
`min(32.768, massimo contesto GGUF)`. `Some(N)` richiede invece esattamente
`N`: ogni valore positivo fino al massimo GGUF è valido se entra nella memoria
del backend e non viene troncato o ridotto. Il limite effettivo è restituito da
`Engine::context_limit` alle superfici locali. In hybrid:

- percentuale pesi Vulkan assente: budget automatico dalla VRAM post-riserva;
- percentuale `0`: CPU-only e nessuna inizializzazione Vulkan;
- Vulkan non disponibile: CPU-only se la RAM basta.

La RAM automatica usa
`floor(MemAvailable × 90 / 100)`, calcolato sul valore valido letto da
`/proc/meminfo`, cioè il 90% intero; swap e `MemFree` sono esclusi. La riserva
VRAM automatica resta il massimo tra 256 MiB e il 5% della VRAM. Errori di
allocation, pipeline, kernel, transfer, readback o decoder dopo la scelta del
piano non provocano retry.

## Verifiche

La suite sintetica copre dimensioni 3B/8B/14B, entrambi i profili pubblici,
attivazioni FP16, residuo/logits FP32, pesi F16/Q4_K/Q5_K/Q6_K/Q8_0, KV
f16/int8 e i tre placement hybrid.

Le prove reali sono ignorate per default e non cercano fallback locali. Le
risorse obbligatorie sono esplicite:

| Variabile | Uso test-only |
|---|---|
| `GH_ZERO_MODEL` | singolo GGUF reale |
| `GH_ZERO_MODEL_Q8_0` | artefatto Q8_0 per la prova tokenizer differenziale |
| `GH_ZERO_MODEL_Q4_K_M` | artefatto Q4_K_M per la stessa prova |
| `GH_ZERO_REFERENCE_CLI` | eseguibile oracle per token greedy |
| `GH_ZERO_REFERENCE_TOKENIZE` | eseguibile oracle per tokenizzazione |
| `GH_ZERO_REFERENCE_PROMPT_IDS` | vettore prompt Reasoning fornito dallo script |
| `GH_ZERO_REFERENCE_COMPLETION_IDS` | 16 ID oracle forniti dallo script |

`GH_ZERO_CONTEXT`, `GH_ZERO_KV` e, nella prova mixed,
`GH_ZERO_VRAM_WEIGHTS_PERCENT` configurano la riga esterna senza individuare
risorse. Le variabili `GH_ZERO_REFERENCE_*_IDS` sono interne allo script di
parità e non sono configurazione runtime stabile.

```sh
GH_ZERO_MODEL="/path/model.gguf" GH_ZERO_CONTEXT=4096 \
  cargo test -p gh_zero_engine --no-default-features --features cpu \
  real_greedy_parity -- --ignored --nocapture
```

Vedi [la guida KV](../../docs/kv-quant-mistral-validation.md) e
[il backend](../../docs/backend.md).

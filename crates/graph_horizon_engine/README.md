<!--
  Questo documento possiede il contratto della libreria, dei backend e della
  memoria; i risultati operativi appartengono ai documenti di validazione.
-->

# graph_horizon_engine

`graph_horizon_engine` è il runtime di inferenza text-to-text per
`general.architecture=mistral3`. La sua API pubblica riceve messaggi chat,
esegue prefill/decode e produce eventi incrementali senza dipendere da console,
HTTP o Web UI.

## Feature backend

Il crate non seleziona un backend di default: il consumer abilita esattamente
uno dei cinque profili.

```sh
cargo check -p graph_horizon_engine --no-default-features --features cpu
cargo check -p graph_horizon_engine --no-default-features --features vulkan
cargo check -p graph_horizon_engine --no-default-features --features vulkan-hybrid
cargo check -p graph_horizon_engine --no-default-features --features metal
cargo check -p graph_horizon_engine --no-default-features --features metal-hybrid
```

- `cpu`: percorso completo e riferimento numerico portabile.
- `vulkan`: modello interamente GPU; memoria insufficiente o device non
  disponibile sono errori senza fallback.
- `vulkan-hybrid`: CPU più Vulkan con piano immutabile all-GPU, mixed o CPU-only.
- `metal`: modello interamente Metal su macOS arm64, senza fallback CPU.
- `metal-hybrid`: CPU più Metal con capacità unificata e modalità all-Metal,
  mixed o CPU-only.

Il piano hybrid è immutabile dopo il load. Solo il piano misto copia il residuo
CPU→GPU, una volta per passaggio; la KV di ogni layer resta sul suo backend. Il
report contiene modalità, split, conteggi layer e breakdown CPU/GPU di pesi, KV,
scratch, fixed, staging, crossing e reserve.

### Qualifica semantica Reasoning

La qualifica semantica corrente è una policy separata e solo test: conserva i
tre pass Instruct revisionati dal piano storico e invoca soltanto i tre profili
Reasoning. Il runner usa il load `vulkan-hybrid` per osservare il placement, ma la
generazione qualificante è valida solo con Vulkan all-GPU; `mixed`, `cpu-only` o
risorse assenti diventano `external-verification` e non attivano fallback.

Il run Reasoning usa KV `f16`, `context=4096`, `max_tokens=4096`,
`temperature=0.7`, `seed=0`, `top_p=1`, `top_k=0`, `min_p=0` e
`repeat_penalty=1`. Il corpus contiene solo i nove casi S01–S04 e S06–S10. Ogni
caso viene tentato una volta; `context`, `max-tokens`, errori engine, marker
Reasoning incompleti e miss semantici sono risultati terminali, non ragioni per
retry, tuning, CPU fallback o oracle.

La riga finale di un Reasoning può essere `qualified`, `not-qualified` o
`external-verification`. Una matrice strutturalmente completa a sei righe è
successo operativo del runner anche quando un modello non supera il gate
semantico. Il runtime continua a emettere testo raw e non possiede questa policy
di assessment.

La matrice revisionata corrente è nel [registro di validazione](../../VALIDATION.md):
i tre Reasoning sono `qualified` nel run Piano 07, mentre i tre Instruct sono
evidenza preservata. Questo qualifica il percorso API Rust configurabile usato
dall'harness. Il server seleziona gli stessi parametri di sampling per un profilo
Reasoning, ma il gate completo resta dell'harness perché fissa anche contesto,
KV, placement e corpus semantico.

`Engine::placement()` fornisce il placement finale e il suo breakdown di
memoria pianificata; non espone la VRAM grezza disponibile. Un errore dopo la
selezione finale resta un failure senza retry o fallback. Il comando e il
protocollo operativo sono nella [guida degli script](../../support/README.md).

## Contratto Ministral

Il riconoscimento di architettura, tensori e quantizzazione è capability-based
e non autentica un file tramite nome, directory, byte totali o `general.name`.
Quest'ultimo seleziona soltanto la policy chat. I valori esatti e case-sensitive
`ministral-3B-Reasoning-2512`, `ministral-8B-Reasoning-2512` e
`ministral-14B-Reasoning-2512` abilitano il profilo privato Reasoning; ogni altro
nome contenente `Reasoning` viene rifiutato prima dell'allocazione backend. La
configurazione Instruct resta dimension-generic per 3B, 8B e 14B.

Il solo profilo GGUF pubblico è `Q4_K_M`: matrici Q4_K/Q6_K e ausiliari
monodimensionali F32. Il parser conserva `GgmlType::Q8_0` per identificare e
diagnosticare un file Q8, ma il gate E04 lo rifiuta prima dell'allocazione. Le
sei righe reali di validazione sono evidenza riproducibile, non una whitelist.
I valori fissati per Ministral 3 Instruct/Reasoning 2512 appartengono al modulo
privato
[`family/mistral/version.rs`](src/family/mistral/version.rs), che possiede anche
il System prompt Reasoning della release.

Il backend mantiene attivazioni FP16, residuo/logits FP32 e formati numerici
interni F16/Q4_K/Q5_K/Q6_K. Questa superficie interna non aggiunge profili
Ministral pubblici.

## Facciata pubblica

La radice espone soltanto:

- engine/config/placement: `Engine`, `EngineConfig`, `BackendMemory`,
  `PlacementReport`;
- chat ed eventi: `Message`, `Role`, `Request`, `SamplingParams`, `Event`,
  `GenerationStats`, `EventSink`, `render_chat_prompt`;
- ispezione Ministral/GGUF: `MistralConfig`, `TekkenTokenizer`,
  `GgufFile`, `GgufValue`, `GgmlType`, `TensorInfo`;
- selezione KV: `KvQuant`;
- `harness::throughput` con `BenchConfig`, `Stat`, `ThroughputReport` e `run`.

Backend, sampling, layout KV, metadati allocativi e grafo restano interni.
L'harness attivo misura esclusivamente lo stream pubblico dell'engine; non
espone profiling di buffer o accessi privati al modello.

## Chat

```rust
use graph_horizon_engine::{Engine, EngineConfig, Message, Request, Role, SamplingParams};

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
device. `EngineConfig.context_tokens = None` applica la policy del motore:
`min(32.768, massimo contesto GGUF)`. `Some(N)` richiede invece esattamente
`N`: ogni valore positivo fino al massimo GGUF è valido se entra nella memoria
del backend e non viene troncato o ridotto. Il limite effettivo è restituito da
`Engine::context_limit` alle superfici locali. Nei profili hybrid:

- percentuale assente: selezione automatica dalla capacità disponibile;
- percentuale `0`: CPU-only e nessuna inizializzazione device;
- percentuale `100`: endpoint device-only (`all-gpu` o `all-metal`).

Vulkan usa RAM e VRAM separate; la RAM automatica è
`floor(MemAvailable × 90 / 100)` su Linux e la riserva VRAM automatica è il
maggiore tra 256 MiB e 5%.
Metal usa memoria unificata: CPU e Metal competono nella stessa capacità
derivata da memoria fisica e working set raccomandato. Errori di
allocation, pipeline, kernel, transfer, readback o decoder dopo la scelta del
piano non provocano retry.

## Verifiche

La suite sintetica copre dimensioni 3B/8B/14B, il gate pubblico Q4_K_M,
attivazioni FP16, residuo/logits FP32, formati numerici interni, KV f16/int8 e i
tre placement hybrid. Test separati verificano che Q8 resti soltanto
diagnosticabile e venga rifiutato.

Le prove reali sono ignorate per default e non cercano fallback locali. Le
risorse obbligatorie sono esplicite:

| Variabile | Uso test-only |
|---|---|
| `GRAPH_HORIZON_MODEL` | singolo GGUF reale |
| `GRAPH_HORIZON_MODEL_Q4_K_M` | artefatto Q4_K_M per la prova tokenizer |
| `GRAPH_HORIZON_REFERENCE_CLI` | eseguibile oracle per token greedy |
| `GRAPH_HORIZON_REFERENCE_TOKENIZE` | eseguibile oracle per tokenizzazione |
| `GRAPH_HORIZON_REFERENCE_PROMPT_IDS` | vettore prompt Reasoning fornito dallo script |
| `GRAPH_HORIZON_REFERENCE_COMPLETION_IDS` | 16 ID oracle forniti dallo script |

`GRAPH_HORIZON_CONTEXT`, `GRAPH_HORIZON_KV` e, nella prova mixed,
`GRAPH_HORIZON_VRAM_WEIGHTS_PERCENT` configurano la riga esterna senza individuare
risorse. Le variabili `GRAPH_HORIZON_REFERENCE_*_IDS` sono interne allo script di
parità e non sono configurazione runtime stabile.

```sh
GRAPH_HORIZON_MODEL="/path/model.gguf" GRAPH_HORIZON_CONTEXT=4096 GRAPH_HORIZON_KV=f16 \
GRAPH_HORIZON_REFERENCE_PROMPT_IDS="..." GRAPH_HORIZON_REFERENCE_COMPLETION_IDS="..." \
  cargo test --release -p graph_horizon_engine --no-default-features --features cpu \
  --test family_agnostic real_selected_runtime_parity_and_lifecycle \
  -- --ignored --nocapture --exact
```

Le interfacce complete della matrice 74-righe e dell'accettazione semantica sono
descritte nella [guida degli script](../../support/README.md); i risultati
revisionati appartengono al [registro di validazione](../../VALIDATION.md).

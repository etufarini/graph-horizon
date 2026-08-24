<!--
  Questa guida possiede le interfacce operative del bootstrap remoto e degli
  script locali e mantiene invarianti le directory modello in sola lettura;
  non definisce policy runtime, modello o prestazionali.
-->

# Supporto operativo

Gli script orchestrano build e verifiche esistenti. Non scaricano modelli, non
modificano GGUF o configurazione utente e non riprovano con backend/contesto
diversi.

| Script | Scopo |
|---|---|
| `../install.sh` | scarica e autentica con SHA-256 l'archivio sorgente `v0.1.1`, poi delega gli argomenti invariati |
| `install.sh` | valida e compila un checkout locale, poi installa uno dei profili espliciti |
| `profiling/profile.sh` | memoria/placement e throughput family-neutral |
| `profiling/validate-kv.sh` | esegue i profili f16/int8 su un modello esplicito; autenticazione e verdetto restano al chiamante |
| `profiling/validate-weights.sh` | autenticazione dei sei Q4_K_M e formati interni sintetici |
| `testing/parity-check.sh` | prompt esatto e top-2 contro oracle fissato |
| `testing/matrix-check.sh` | sei Q8, 60 righe principali e otto endpoint hybrid |
| `testing/semantic-check.sh` | matrice terminale di qualifica semantica Reasoning |
| `testing/run-graph-horizon.sh` | avvio esplicito della console locale |

## Prerequisiti

- Bash, `curl`, `tar`, `mktemp` e `find` per il bootstrap pubblico;
- Rust/Cargo 1.88 o successivo, `uname`, `install` e dipendenze di build della piattaforma;
- Node.js/npm 22.12 o successivo per `install.sh`;
- loader/driver Vulkan per esecuzione Vulkan;
- artefatti GGUF già acquisiti in sola lettura;
- `curl`, `jq`, `stat` e `sha256sum` oppure `shasum -a 256`;
- macOS arm64 e `xcrun metal`/`metallib` per i profili Metal;
- `llama-server` alla revisione fissata `13f2b28b0`.

Il contratto degli artefatti e gli SHA registrati sono nel catalogo
[`models.tsv`](models.tsv); gli esiti revisionati appartengono al
[registro di validazione](../docs/validation.md).
Per una campagna locale, la directory dei modelli va sempre passata
esplicitamente come `--models-dir /path/to/models`; il checkout oracle principale
può trovarsi in `/path/to/llama.cpp`, mentre il worktree disposable previsto è
`target/oracle/llama.cpp-13f2b28b`. L'assenza di una
risorsa esterna produce `external verification: <motivo preciso>`; non
salta test sintetici.

## Installazione

Il bootstrap pubblico è disponibile anonimamente. Scarica l'artefatto sorgente
immutabile `v0.1.1` e il relativo record `.sha256`, verificandolo prima
dell'estrazione:

```sh
curl --fail --location --silent --show-error https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.1/install.sh | bash -s -- --backend cpu
```

`../install.sh` possiede soltanto download HTTPS, validazione dell'archivio e
cleanup della directory temporanea. Il checkout estratto esegue poi lo stesso
installer locale usato direttamente qui:

```sh
support/install.sh --backend metal --profile release
support/install.sh --backend cpu --profile fast --prefix "$PWD/.local"
```

`release|fast` sono profili di compilazione Cargo. `--backend` è obbligatorio e
accetta `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid`; non esiste un
default né un profilo runtime. La matrice accettata, senza fallback, è:

| Piattaforma | Backend |
|---|---|
| macOS arm64 | `cpu`, `vulkan`, `vulkan-hybrid`, `metal`, `metal-hybrid` |
| Linux x86_64 | `cpu`, `vulkan`, `vulkan-hybrid` |

Il profilo predefinito è `release`. Il prefisso deve essere assoluto, diverso
dalla radice e privo di componenti `.` o `..`; `--prefix` prevale su
`GRAPH_HORIZON_INSTALL_PREFIX`, che prevale su `$HOME/.local`. Nessuno dei due
script invoca `sudo`.

`install.sh` verifica anche `curl`, necessario a runtime per i provider di
ricerca pubblici, esegue `npm ci` dal lockfile, poi il solo script `build`, e
infine una build Cargo `--locked`. Gli script npm disponibili sono `dev`,
`check`, `test` e `build`.
Il binario canonico installato è `graph-horizon`; gli asset compilati della Web
UI sono installati in `<prefix>/share/graph-horizon/web` e vengono risolti
rispetto all'eseguibile, indipendentemente dalla directory corrente. Il nome
precedente `gh-zero-engine` è un collegamento di compatibilità allo stesso
artefatto e non può quindi restare fermo a una build precedente.
La policy `allowScripts` autorizza esattamente `esbuild@0.28.1` e nega
`@parcel/watcher@2.5.6`; un nuovo script di dipendenza deve essere classificato
prima di entrare nel lockfile. `GRAPH_HORIZON_INSTALL_PREFIX` imposta il prefisso
quando `--prefix` non è presente.
L'installazione sostituisce `graph-horizon` e il link relativo
`gh-zero-engine`. Se `<prefix>/bin` non è una voce esatta di `PATH`, stampa una
sola istruzione diagnostica e non modifica alcun file shell.

## Verifica

```sh
bash -n install.sh support/install.sh
cargo test -p graph-horizon --no-default-features --features cpu installer_
cargo test -p graph-horizon --no-default-features --features cpu bootstrap_

support/profiling/validate-weights.sh --models-dir "/path/to/models"

support/profiling/validate-kv.sh \
  --model "/path/to/q4.gguf" --backend cpu --context 4096

support/testing/parity-check.sh \
  --models-dir "/path/to/models" --model-id 3b-reasoning \
  --backend cpu --kv f16 \
  --reference-server "/path/llama-server"
```

### Prestazioni iterative

L'unico eseguibile prestazionale iterativo è
[`examples/bench.rs`](../examples/bench.rs). Misura una tupla end-to-end e
pubblica statistiche del flusso API; non autentica artefatti, non confronta
revisioni e non decide se conservare una modifica. `profiling/profile.sh`
fornisce soltanto uno snapshot di memoria, placement e throughput: non è un
comparatore A/B.

Il contratto CLI, l'output e gli errori bounded sono descritti nella
[guida al benchmark](../docs/throughput-bench.md). Selezione della tupla,
confronto, soglie e stati terminali appartengono al
[processo prestazionale](../docs/performance-investigation-process.md).

Una risorsa assente resta `external verification: <motivo preciso>`; non viene
sostituita con altro modello, backend o profilo.

L'interfaccia completa è:

```text
parity-check.sh --models-dir DIR --model-id ID \
  --backend cpu|vulkan|vulkan-hybrid|metal|metal-hybrid --kv f16|int8 \
  --reference-server PATH [--reference-port PORT] \
  [--weights-percent 0..100 --expect-mode all-gpu|all-metal|mixed|cpu-only]
```

Lo script esegue una sola riga, autentica il Q4_K_M catalogato, avvia un solo
oracle CPU `llama-server` su loopback e termina soltanto quel processo. `f16` e
`int8` sono righe distinte; non esistono sostituzioni o retry con altro backend,
modello o contesto. Una risorsa assente o Vulkan indisponibile produce
`external verification: <motivo preciso>`; errori di protocollo, codice o
parità falliscono la riga.

### Qualifica semantica Reasoning

```sh
support/testing/semantic-check.sh --models-dir /home/user/models
```

Il runner costruisce sempre una matrice a sei righe. Le tre righe Instruct sono
preservate dal pass storico e non aprono artefatti né invocano Cargo. Le tre
righe Reasoning sono autenticate una alla volta con byte count e SHA-256 del
catalogo, poi invocate con il test ignorato `real_semantic_acceptance`.

La configurazione Reasoning accettata è esatta: `context=4096`,
`max_tokens=4096`, `temperature=0.7`, `seed=0`, `top_p=1`, `top_k=0`, `min_p=0`,
`repeat_penalty=1`, KV `f16`, feature Cargo `vulkan-hybrid`, backend finale Vulkan
all-GPU. I soli casi reali sono S01–S04 e S06–S10, in quest'ordine. Non esiste
retry, CPU fallback, oracle, tuning o inferenza Instruct.

Per ogni Reasoning tentato lo script inoltra una riga `semantic-config:`, una
`semantic-selection:`, nove righe `semantic-case:`, una `semantic-summary:` e
una `semantic-timing:`. Un placement non all-GPU, un artefatto assente o non
autenticato, o uno strumento richiesto non disponibile produce
`external-verification`. Una generazione tentata che non supera il protocollo o
il gate produce `not-qualified`; un gate completo produce `qualified`.

Le righe finali normalizzate hanno la forma:

```text
qualification: model_id=<id> profile=<instruct|reasoning> evidence=<preserved|current> status=<qualified|not-qualified|external-verification> reason=<motivo> critical=<n/4|not-applicable> semantic=<n/9|not-applicable>
summary: qualified=<n> not_qualified=<n> external_verification=<n> total=6
```

Il codice di uscita è 0 per ogni matrice strutturalmente completa, anche con
`not-qualified` o `external-verification`. Il codice 2 è riservato ad argomenti
o catalogo invalidi prima dell'inferenza. Il run a `temperature=0.7` e `seed=0`
è riproducibile solo a parità di commit, artefatto, backend, parametri e seed;
non è una promessa di determinismo universale tra hardware.

Gli esiti revisionati sono pubblicati nel [registro di validazione](../docs/validation.md) con
il relativo commit. Una campagna precedente non qualifica automaticamente la
sorgente corrente; il runner definisce il protocollo, mentre lo stato della
release `v0.1.0` appartiene al registro.

## Sicurezza

La matrice completa si avvia con:

```sh
support/testing/matrix-check.sh \
  --models-dir /path/to/models \
  --reference-server "$PWD/target/oracle/llama.cpp-13f2b28b/build/bin/llama-server"
```

Tenta 74 righe seriali: sei rifiuti Q8, 60 righe principali e otto endpoint
3B-Instruct. Le righe hybrid principali usano 25%/mixed. Gli endpoint Vulkan e
Metal usano f16/int8 con 100/all-gpu o 100/all-metal e 0/cpu-only. Per ogni KV,
la sequenza completa di 16 `local_ids` dell'endpoint omogeneo deve essere
identica a quella del backend standalone corrispondente. Un controllo o endpoint
non disponibile resta `external verification`, non produce un claim di
uguaglianza e non impedisce le righe indipendenti; una differenza ferma subito la
matrice con failure.

Ogni path modello viene passato come un singolo argomento quotato. Valori enum e
numerici sono validati prima di invocare Cargo o il binario. Gli script non usano
`eval`, non costruiscono comandi da input e non scrivono nelle directory modello.

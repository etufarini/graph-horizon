<!--
  Questa guida possiede le interfacce operative degli script locali e mantiene
  invarianti le directory modello in sola lettura; non definisce policy runtime.
-->

# Supporto operativo

Gli script orchestrano build e verifiche esistenti. Non scaricano modelli, non
modificano GGUF o configurazione utente e non riprovano con backend/contesto
diversi.

| Script | Scopo |
|---|---|
| `install.sh` | build Web UI e binario CPU/Vulkan/hybrid |
| `profiling/profile.sh` | memoria/placement e throughput family-neutral |
| `profiling/validate-kv.sh` | verifica f16/int8 su un Q4_K_M autenticato |
| `profiling/validate-weights.sh` | autenticazione dei sei Q4_K_M e formati interni sintetici |
| `testing/parity-check.sh` | prompt esatto e top-2 Reasoning contro oracle fissato |
| `testing/semantic-check.sh` | matrice terminale di qualifica semantica Reasoning |
| `testing/run-ghzero-engine.sh` | avvio esplicito della console locale |

## Prerequisiti

- Rust/Cargo e dipendenze di build della piattaforma;
- Node/npm per `install.sh`;
- loader/driver Vulkan per esecuzione Vulkan;
- artefatti GGUF già acquisiti in sola lettura;
- `curl`, `jq` e `sha256sum` per la parità Reasoning;
- `llama-server` alla revisione fissata `13f2b28b0`.

Il contratto degli artefatti e gli SHA registrati sono nel catalogo
[`models.tsv`](models.tsv); gli esiti revisionati appartengono al
[registro di validazione](../VALIDATION.md).
L'assenza di una risorsa esterna produce `not verified: <motivo preciso>`; non
salta test sintetici.

## Installazione

```sh
support/install.sh --backend hybrid --profile release
support/install.sh --backend cpu --profile fast --prefix "$PWD/.local"
```

`release|fast` sono profili di compilazione Cargo. Il default backend è hybrid;
non esiste un profilo runtime.

`install.sh` esegue `npm ci` dal lockfile, poi il solo script `build`, e infine
una build Cargo `--locked`. Gli script npm attivi sono `dev`, `check` e `build`.
La policy `allowScripts` autorizza esattamente `esbuild@0.28.1` e nega
`@parcel/watcher@2.5.6`; un nuovo script di dipendenza deve essere classificato
prima di entrare nel lockfile. `GH_ZERO_INSTALL_PREFIX` imposta il prefisso
quando `--prefix` non è presente.

## Verifica

```sh
support/profiling/validate-weights.sh --models-dir "/path/to/models"

support/profiling/validate-kv.sh \
  --model "/path/to/q4.gguf" --backend cpu --context 4096

support/testing/parity-check.sh \
  --models-dir "/path/to/models" --model-id 3b-reasoning \
  --backend cpu --kv f16 \
  --reference-server "/path/llama-server"
```

L'interfaccia completa è:

```text
parity-check.sh --models-dir DIR --model-id ID \
  --backend cpu|vulkan|hybrid --kv f16|int8 \
  --reference-server PATH [--reference-port PORT]
```

Lo script esegue una sola riga, autentica il Q4_K_M catalogato, avvia un solo
server CPU su loopback e termina soltanto quel processo. `f16` e `int8` sono
righe distinte; non esistono sostituzioni o retry con altro backend, modello o
contesto. Una risorsa assente o Vulkan indisponibile produce
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
`repeat_penalty=1`, KV `f16`, feature Cargo `hybrid`, backend finale Vulkan
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

## Sicurezza

Ogni path modello viene passato come un singolo argomento quotato. Valori enum e
numerici sono validati prima di invocare Cargo o il binario. Gli script non usano
`eval`, non costruiscono comandi da input e non scrivono nelle directory modello.

<!--
  Guida operativa degli script locali in sola lettura; documenta le loro
  interfacce, non la policy runtime del prodotto.
-->

# Supporto operativo

Gli script orchestrano build e verifiche esistenti. Non scaricano modelli, non
modificano GGUF o configurazione utente e non riprovano con backend/contesto
diversi.

| Script | Scopo |
|---|---|
| `install.sh` | build Web UI e binario CPU/Vulkan/hybrid |
| `profiling/profile.sh` | memoria/placement e throughput family-neutral |
| `profiling/validate-kv.sh` | matrice f16/int8 su Q8_0 e Q4_K_M |
| `profiling/validate-weights.sh` | size/hash/istogrammi reali e formati interni sintetici |
| `testing/parity-check.sh` | prompt esatto e top-2 Reasoning contro oracle fissato |
| `testing/semantic-check.sh` | accettazione semantica M3 sui sei Q4_K_M autenticati |
| `testing/run-ghzero-engine.sh` | avvio esplicito della console locale |

## Prerequisiti

- Rust/Cargo e dipendenze di build della piattaforma;
- Node/npm per `install.sh`;
- loader/driver Vulkan per esecuzione Vulkan;
- artefatti GGUF già acquisiti in sola lettura;
- `curl`, `jq` e `sha256sum` per la parità Reasoning;
- `llama-server` alla revisione fissata `13f2b28b0`.

Il contratto degli artefatti e gli SHA registrati sono nella
[guida di validazione Ministral](../docs/kv-quant-mistral-validation.md).
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
support/profiling/validate-weights.sh \
  --model-q8 "/path/q8.gguf" --model-q4 "/path/q4.gguf"

support/profiling/validate-kv.sh \
  --model-q8 "/path/q8.gguf" --model-q4 "/path/q4.gguf" \
  --backend cpu --context 4096

support/testing/parity-check.sh \
  --model "/path/Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf" \
  --backend cpu --context 4096 --kv f16 \
  --reference-server "/path/llama-server"
```

L'interfaccia completa è:

```text
parity-check.sh --model PATH --backend cpu|vulkan|hybrid --context 4096 \
  --kv f16|int8 --reference-server PATH [--reference-port PORT] \
  [--vram-weights-percent 25]
```

Lo script esegue una sola riga, verifica size/hash del Q4_K_M o Q8_0
Reasoning 3B approvato, avvia un solo server CPU su loopback e termina soltanto
quel processo. `f16` e `int8` sono righe distinte; non esistono sostituzioni o
retry con altro backend, modello o contesto. Per hybrid è obbligatorio
`--vram-weights-percent 25`. Una risorsa assente o Vulkan indisponibile produce
`external verification: <motivo preciso>`; errori di protocollo, codice o
parità falliscono la riga.

### Accettazione semantica M3

```sh
support/testing/semantic-check.sh --models-dir /home/user/models
```

Il runner autentica esattamente i sei artefatti del catalogo e usa sempre KV
`f16`, contesto 4096, greedy puro e al massimo 256 token per ciascuno dei dodici
casi. Il backend finale è esclusivamente all-GPU oppure CPU-only. Ogni modello
viene caricato inizialmente con il planner hybrid esistente: un placement
completo seleziona il riferimento finale all-GPU con motivo
`full-vram-fit`; Vulkan assente, VRAM completa insufficiente o un placement
CPU selezionano il riferimento CPU-only con motivo `no-full-vram-fit`.

Un probe `mixed` può allocare il modello per osservare il placement, ma non
genera alcun token semantico: viene distrutto e il modello è riaperto CPU-only.
Il backend finale resta invariato per tutti i dodici casi. Qualsiasi errore dopo
la selezione è un failure e non attiva fallback o retry.

Per ogni modello tentato lo script inoltra una riga `semantic-selection:` con
backend e memoria pianificata, dodici righe `semantic-case:`, una
`semantic-summary:` con gate e diagnostica e una `semantic-timing:` con tempi e
confronto prestazionale applicabile. Un artefatto assente, illeggibile o non
autenticato produce `external verification: <motivo preciso>`; l'esatta
condizione di RAM insufficiente produce lo stesso stato esterno. Ogni ID riceve
poi una riga normalizzata `semantic model_id=<id>:` e il run termina con
`summary: pass=<n> external_verification=<n> failure=<n> total=6`.

Il codice di uscita è 0 quando non esistono failure tentati, 1 dopo il riepilogo
se almeno una riga tentata fallisce e 2, prima di caricare modelli, per argomenti
o catalogo invalidi. Tutte le sei righe vengono classificate prima dell'uscita.

## Sicurezza

Ogni path modello viene passato come un singolo argomento quotato. Valori enum e
numerici sono validati prima di invocare Cargo o il binario. Gli script non usano
`eval`, non costruiscono comandi da input e non scrivono nelle directory modello.

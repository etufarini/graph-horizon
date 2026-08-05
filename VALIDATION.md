<!--
Questo registro conserva evidenza revisionata di artefatti, correttezza,
qualifica semantica, Metal e candidati prestazionali rifiutati; non definisce
capability runtime, supporto prodotto o whitelist di modelli.
-->

# Registro di validazione

## Ambito e data

Registro aggiornato il 5 agosto 2026. Conserva la qualifica della specifica
`01-metal-backend` e l'esito revisionato della successiva campagna
prestazionale.

Gli artefatti locali di questa campagna sono in
`/Users/emanuele/Documents/models`; il percorso è solo un input di validazione
e non è un default del runtime.

L'evidenza distingue quattro piani:

- compatibilità tecnica: autenticazione Q4_K_M, rifiuto Q8, backend, KV e
  parità;
- qualifica semantica: solo il corpus e la configurazione esplicitamente
  descritti dal Piano 07;
- indagine prestazionale: misure A/B storiche e relativi verdetti, senza claim
  sull'architettura corrente;
- runtime prodotto: API, CLI, HTTP server e Web UI restano governati dai loro
  contratti, non da questo registro.

## Artefatti Q4_K_M autenticati

| ID | Profilo | File Q4_K_M | Byte | SHA-256 |
|---|---|---|---:|---|
| 3b-instruct | instruct | Ministral-3-3B-Instruct-2512-Q4_K_M.gguf | 2147023008 | 9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8 |
| 3b-reasoning | reasoning | Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf | 2147021472 | 7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc |
| 8b-instruct | instruct | Ministral-3-8B-Instruct-2512-Q4_K_M.gguf | 5198911904 | 33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761 |
| 8b-reasoning | reasoning | Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf | 5198910368 | 894aa3645ef8708a81dbe201c26105ce37c4c741252c89c5a78f81b49ac438c6 |
| 14b-instruct | instruct | Ministral-3-14B-Instruct-2512-Q4_K_M.gguf | 8239593024 | 824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613 |
| 14b-reasoning | reasoning | Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf | 8239591488 | fe08ca2158cd7438211ec6a4e5256d31bc980f016e3f5b635fe91fe6848d461c |

Fonte catalogo: `support/models.tsv`. Gli artefatti sono aperti solo in lettura
dai runner.

## Rifiuti Q8 e parità tecnica

Il Piano 02 conserva il gate Q4_K_M-only: Q8 resta diagnosticabile dal parser ma
non caricabile. L'evidenza storica registra sei rifiuti Q8; nella
rivalidazione più recente gli artefatti Q8 non erano disponibili e sono stati
registrati come verifiche esterne, senza riabilitare il formato.

La matrice finale contiene 70 righe: sei rifiuti Q8, 60 righe Q4_K_M (sei
modelli × cinque profili × due KV) e quattro endpoint 3B-Instruct
Metal-hybrid. Ogni riga usa contesto 4096, prompt token IDs identici all'oracolo
e 16 token teacher-forced; ogni token oracle deve essere nel top-two locale
finito e deterministico. Le righe hybrid principali usano 25%/`mixed`; gli
endpoint usano 100%/`all-metal` e 0%/`cpu-only`, f16 e int8.

Il run finale termina con
`summary: pass=40 external_verification=30 failure=0 total=70`: passano tutte le
12 CPU, 12 Metal, 12 Metal-hybrid mixed e 4 endpoint. Le 24 righe Vulkan sono
esterne perché il device non è presente; i sei Q8 sono esterni perché gli
artefatti non sono presenti. Non sono dichiarati pass. L'oracolo è il worktree
CPU-only/offline `target/oracle/llama.cpp-13f2b28b`, commit
`13f2b28b098623391b1aacfd27995e1c8b7de9a9`, versione `9973 (13f2b28b0)`;
ascolta solo su loopback, senza GPU o KV offload, ed è terminato dal wrapper.

Gli stati terminali della matrice tecnica sono esclusivi: `pass` indica una riga
eseguita che soddisfa tutti i gate; `failure` indica una riga tentata con errore
di protocollo, lifecycle, prompt o numerica; `external_verification` indica un
prerequisito autenticato, device, tool o capacità assente e non conta come pass.
Le tre quantità devono sommare esattamente al totale fissato.

## Prestazioni Metal (nessun gate)

Hardware: MacBook Air `Mac16,13`, Apple M4 10 core, 24 GB, macOS 26.3
(`25D125`), Xcode 26.2, Metal 32023.864, Rust 1.95.0. Artefatto
3B-Instruct Q4_K_M autenticato, contesto 4096, KV f16, prompt
`Quanto fa 17 × 19?`, 32 token, warmup 1, ripetizioni 3:

| Profilo | Placement | TTFT ms | Prompt tok/s | Decode tok/s |
|---|---|---:|---:|---:|
| `metal` | standalone | 5433.02 | 2.58 | 2.13 |
| `metal-hybrid` | 25%, mixed (24 CPU / 2 Metal) | 6292.69 | 2.23 | 1.28 |

Le metriche sono finite e descrittive. Questa modifica non impone velocità o
rapporto minimo.

## Ottimizzazione corrente della proiezione Q4_K Metal — 5 agosto 2026

Ipotesi: il kernel di proiezione ricalcolava scale, minimi e indirizzi Q4_K per
ogni peso. Il candidato `1466f4a` li calcola una volta per sottoblocco senza
cambiare formato, precisione o ordine naturale dell'accumulo FP32. La baseline
è `1e797f8`; il solo profilo modificato è `metal` standalone.

Artefatto `3b-instruct` Q4_K_M autenticato, 2147023008 byte, SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
Build Cargo `release`, MacBook Air Apple M4 10 core con Metal 32023.864,
contesto 4096, KV f16, prompt `Quanto fa 17 × 19?`, greedy, 32 token, un
warm-up e tre ripetizioni:

| Revisione | Prompt tok/s | CV prompt | TTFT ms | CV TTFT | Decode tok/s | CV decode |
|---|---:|---:|---:|---:|---:|---:|
| `1e797f8` | 2,62 | 0,31% | 5341,41 | 0,31% | 2,21 | 0,03% |
| `1466f4a` | 10,74 | 0,74% | 1303,38 | 0,74% | 6,08 | 0,05% |

Il prompt throughput migliora del 309,9%, il decode del 175,1% e il TTFT si
riduce del 75,6%. Tutti i CV sono inferiori al 5%, quindi non è stato usato il
rerun consentito. I 127 test Metal, i 152 test CPU e la parity reale passano;
i 16 token greedy locali coincidono con quelli dell'oracolo fissato
`llama.cpp` `13f2b28b`. Stato terminale: `keep`. L'intera verifica è rimasta
entro il budget di due ore.

### Ciclo successivo sul kernel Q4_K

Il ciclo successivo ha usato `4c591c1` come baseline, lo stesso artefatto e la
stessa tupla. Baseline: prompt 10,75 tok/s (CV 0,62%), TTFT 1302,85 ms
(CV 0,62%) e decode 6,08 tok/s (CV 0,21%). Il target dichiarato era `both`:
prompt e decode dovevano migliorare entrambi almeno del 5% per lo stato
`keep`. Nessun rerun è stato necessario perché ogni CV misurato era inferiore
al 5%.

| Tentativo | Ipotesi e modifica isolata | Prompt | TTFT | Decode | Esito |
|---:|---|---:|---:|---:|---|
| 1 | Accoppiare i sottoblocchi low/high per eliminare divisione e branch | +1,40% | −1,38% | +0,82% | `reject`: obiettivi sotto 3% |
| 2 | Srotolare ×4 il ciclo interno da 32 valori | −1,58% | +1,54% | +0,16% | `reject`: regressione prompt |
| 3 | Calcolare una volta i puntatori base di pesi e attivazioni | +0,74% | −0,77% | +0,82% | `reject`: obiettivi sotto 3% |
| 4 | Leggere le metadate Q4_K come tre parole allineate | non misurato | non misurato | non misurato | `reject`: parity fallita per estrazione byte errata |
| 5 | Correggere le tre letture allineate con maschere a 8 bit | +2,79% | −2,78% | +1,97% | `reject`: obiettivi sotto 3% |
| 6 | Conservare 32 byte quantizzati privati e riusarne il nibble alto | +7,72% | −7,17% | +3,62% | `interesting`: decode sotto 5% |
| 7 | Combinare riuso dei byte e metadate allineate corrette | +8,19% | −7,62% | +4,61% | `interesting`: decode sotto 5% |
| 8 | Aggiungere puntatori base alla variante combinata | +8,65% | −8,02% | +4,93% | `interesting`: decode sotto 5% |
| 9 | Srotolare ×2 i due loop della variante combinata | −5,02% | +5,27% | −2,47% | `reject`: regressione di tutti i controlli |

I tentativi 1–3 e 5–9 hanno superato 127 test Metal e parity 16/16 contro
l'oracolo fissato; il tentativo 4 ha superato i test sintetici ma è stato
fermato dalla parity prima del benchmark. Ogni candidato è stato rimosso: il
tree finale conserva soltanto `1466f4a`. Ulteriore parallelismo della
proiezione richiederebbe una riduzione floating-point con ordine diverso e
quindi un oracle numerico non previsto dalla specifica corrente.

La misura finale dello stesso tree registra prompt 10,88 tok/s (CV 0,27%),
TTFT 1286,75 ms (CV 0,27%) e decode 6,13 tok/s (CV 0,06%); 152 test CPU, 127
test Metal, `cargo fmt --check` e parity reale 16/16 passano.

## Esito della campagna prestazionale — 5 agosto 2026

I valori seguenti sono evidenza storica revisionata. Entrambi i candidati sono
stati rimossi: il runtime corrente non contiene né il batching prefill dinamico
da 32 righe né il completamento greedy fuso. Un rapporto intermedio positivo è
un segnale `interesting`, non un miglioramento verificato e non descrive
l'architettura corrente.

### Candidato prefill dinamico da 32 righe

Revisione candidata:
`cc729eb974431b51eaf067adf3a8de34217951cf`.

| Tentativo | Geomean prefill | Geomean decode | Decisione | Prima regressione |
|---:|---:|---:|---|---|
| 1 | 1.2184276784715833 | 1.0248269438770594 | `repeat required` | `public_decode_tps=0.9246821218888349` |
| 2 | 1.2762628240786342 | 1.0695683954824686 | `revert` — `unstable measurement` | `prefill_tps=0.9174527480250944` |

Il verdetto terminale del secondo tentativo è riportato in tabella; il codice
di produzione è stato rimosso da `46e8cc1`. Il segnale aggregato sul prefill era
interessante, ma le righe instabili e una regressione ripetuta impediscono
qualsiasi claim di miglioramento verificato.

### Candidato decode greedy fuso

Revisione candidata:
`24f3b6fc53ed3b40c1675e8ffaa302f8301d6b04`.

| Tentativo | Geomean prefill | Geomean decode | Decisione | Prima regressione |
|---:|---:|---:|---|---|
| 1 | 1.0899998824625137 | 1.1013495018254948 | `repeat required` | `first_sample_latency=1.089238234438909` |
| 2 | 0.9062425203332453 | 0.9131343535729636 | `revert` — `unstable measurement` | `prefill_tps=0.6349648977217498` |

Il verdetto terminale del secondo tentativo è riportato in tabella; il codice
di produzione è stato rimosso da `d3c27d1`. Il primo tentativo indicava un
segnale positivo ma instabile; il rerun completo ha invertito entrambi gli
aggregati e ha confermato che non è possibile dichiarare un miglioramento.

Fonti: record decisionali versionati in `DECISIONS.md` e valori fissati nella
specifica approvata `plans/03-performance-cleanup/m2.md`. L'evidenza locale
ignorata sotto `target/performance` resta storica e non viene riscritta.

## Configurazione semantica Piano 07

I tre Instruct non sono stati rieseguiti: usano l'evidenza preservata del Piano
05. I tre Reasoning sono stati eseguiti una sola volta con:

- System prompt Reasoning implicito della release;
- solo messaggio `User` del caso;
- `context=4096`, `max_tokens=4096`, KV `f16`;
- `temperature=0.7`, `seed=0`, `top_p=1`, `top_k=0`, `min_p=0`,
  `repeat_penalty=1`;
- backend finale Vulkan all-GPU;
- casi S01, S02, S03, S04, S06, S07, S08, S09 e S10;
- gate `critical=4/4` e `semantic>=8/9`;
- nessun retry, fallback CPU, tuning, oracle o inferenza Instruct.

Questo run qualifica il percorso API Rust configurabile esercitato
dall'harness. Non qualifica il server HTTP produttivo, che resta greedy.

## Matrice finale

| ID | Compatibilità tecnica | Qualifica semantica | Evidenza | Stato | Motivo | Critical | Semantic |
|---|---|---|---|---|---|---|---|
| 3b-instruct | compatibile | Piano 05 preservato | preserved | qualified | plan-05-pass | 4/4 | 8/9 |
| 3b-reasoning | compatibile | Piano 07 corrente | current | qualified | semantic-gate-pass | 4/4 | 9/9 |
| 8b-instruct | compatibile | Piano 05 preservato | preserved | qualified | plan-05-pass | 4/4 | 8/9 |
| 8b-reasoning | compatibile | Piano 07 corrente | current | qualified | semantic-gate-pass | 4/4 | 9/9 |
| 14b-instruct | compatibile | Piano 05 preservato | preserved | qualified | plan-05-pass | 4/4 | 9/9 |
| 14b-reasoning | compatibile | Piano 07 corrente | current | qualified | semantic-gate-pass | 4/4 | 9/9 |

Evidenza normalizzata, copiata byte-per-byte dal log M1:

```text
qualification: model_id=3b-instruct profile=instruct evidence=preserved status=qualified reason=plan-05-pass critical=4/4 semantic=8/9
qualification: model_id=3b-reasoning profile=reasoning evidence=current status=qualified reason=semantic-gate-pass critical=4/4 semantic=9/9
qualification: model_id=8b-instruct profile=instruct evidence=preserved status=qualified reason=plan-05-pass critical=4/4 semantic=8/9
qualification: model_id=8b-reasoning profile=reasoning evidence=current status=qualified reason=semantic-gate-pass critical=4/4 semantic=9/9
qualification: model_id=14b-instruct profile=instruct evidence=preserved status=qualified reason=plan-05-pass critical=4/4 semantic=9/9
qualification: model_id=14b-reasoning profile=reasoning evidence=current status=qualified reason=semantic-gate-pass critical=4/4 semantic=9/9
summary: qualified=6 not_qualified=0 external_verification=0 total=6
```

## Interpretazione e limiti

La compatibilità tecnica non implica automaticamente qualifica semantica: sono
colonne e claim separati. In questa esecuzione tutti e tre i Reasoning correnti
hanno superato il gate Piano 07, ma il risultato vale solo per la configurazione
elencata sopra.

`temperature=0.7` con `seed=0` rende riproducibile il percorso RNG di GH Zero a
parità di commit, artefatto, backend e parametri; non promette identità
universale fra hardware o implementazioni. Il Piano 07 non testa contesto 256k,
non testa un System prompt custom esplicito e non pubblica testo raw del
ragionamento.

## Riferimenti

- `DECISIONS.md`;
- Piani 02, 03, 04, 05, 06 e 07 in `plans/`;
- `support/models.tsv`;
- `target/ministral-q4-validation/semantic-reasoning-qualification.log`;
- `target/m3-matrix.log` (evidenza locale non versionata).

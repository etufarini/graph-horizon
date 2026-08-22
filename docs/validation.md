<!--
Questo registro pubblica soltanto identità degli artefatti, evidenza revisionata
e stato di qualifica. Le procedure appartengono a support/README.md e docs/;
log grezzi e decisioni sperimentali restano nella cronologia Git.
-->

# Registro di validazione

## Stato corrente

Graph Horizon `v0.1.0` non è ancora stato rilasciato: non esistono il tag, la
GitHub Release o gli artefatti pubblici. La campagna del 19 agosto 2026 qualifica
il solo runtime `d1bf18f034fd44df5b8e81931e7feea32edeb47f`; le modifiche runtime
successive impediscono di applicarne automaticamente gli esiti al commit finale
ancora da scegliere.

Il software dichiara versione Cargo/frontend `0.1.0`, ma resta pre-release. Il
presente audit parte da `main`/`origin/main` `6b46331` del 22 agosto 2026. La
correzione engine più recente qualificata è `e7edc83`, confluita in `main` con
PR #41 (`24eac82`); i commit successivi cambiano presentazione, documentazione e
commenti, non il percorso numerico oggetto di quella campagna. Questo consente
di mantenere l'evidenza tecnica revisionata, ma non sostituisce la campagna
finale sull'unico commit che verrà taggato.

La compatibilità tecnica, la correttezza numerica, la qualità semantica e le
prestazioni sono claim distinti. Un file caricabile non è per questo
qualificato; un risultato storico non qualifica una sorgente successiva.

## Artefatti autenticati

Il catalogo machine-readable autoritativo è
[`support/models.tsv`](../support/models.tsv). I modelli sono input esterni in sola
lettura e non vengono distribuiti o scaricati dal progetto.

| ID | Profilo | File Q4_K_M | Byte | SHA-256 |
|---|---|---|---:|---|
| 3b-instruct | instruct | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 2147023008 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| 3b-reasoning | reasoning | `Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf` | 2147021472 | `7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc` |
| 8b-instruct | instruct | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` | 5198911904 | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| 8b-reasoning | reasoning | `Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf` | 5198910368 | `894aa3645ef8708a81dbe201c26105ce37c4c741252c89c5a78f81b49ac438c6` |
| 14b-instruct | instruct | `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` | 8239593024 | `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613` |
| 14b-reasoning | reasoning | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 8239591488 | `fe08ca2158cd7438211ec6a4e5256d31bc980f016e3f5b635fe91fe6848d461c` |

Il runtime pubblico accetta il profilo Q4_K_M. I nomi Q8_0 catalogati sono casi
negativi e devono essere rifiutati prima dell’allocazione del backend. Il
catalogo autentica l’evidenza, ma non è una whitelist letta dal runtime.

## Evidenza backend corrente

Questa tabella descrive lo stato tecnico revisionato, non il contratto della
futura release. Le misure, le tuple fisiche e i limiti sono riassunti in
[`current-optimization-state.md`](current-optimization-state.md).

| Profilo | Evidenza revisionata | Stato tecnico |
|---|---|---|
| CPU | suite sintetica e matrice reale post-repair sui sei artefatti/f16-int8; nessuna promessa prestazionale | REFERENCE |
| Vulkan | suite, oracle numerici e matrici reali NVIDIA RTX 3060 / AMD RX 6750 XT | PRODUCTION |
| Vulkan-hybrid | all-GPU NVIDIA qualificato; matrice AMD mixed/CPU/all-GPU post-repair completa, da ripetere sul commit finale | QUALIFIED |
| Metal | suite, oracle, teacher row e misure su Apple M4/macOS 26.3 | QUALIFIED |
| Metal-hybrid | suite e percorso mixed sullo stesso host; claim limitato alla tupla documentata | QUALIFIED |

Le etichette indicano maturità del percorso corrente. Non trasformano queste
righe in qualifica v0.1.0 e non si estendono a hardware non misurato.

## Evidenza post-cleanup integrata

Sul runtime corretto `e7edc83`, la matrice AMD disponibile ha prodotto
`pass=40 external_verification=34 failure=0 total=74`: tutte le 36 righe CPU,
Vulkan standalone e Vulkan-hybrid mixed sui sei Q4_K_M e sui due KV, più quattro
endpoint, sono passate. Le 34 righe esterne erano i sei Q8_0 assenti e le 28
righe Metal/Metal-hybrid non eseguibili sull'host Linux; non sono failure e non
sono PASS. Lo stesso runtime ha prodotto
`qualified=6 not_qualified=0 external_verification=0` nel gate semantico.

L'evidenza Apple M4, raccolta in una campagna separata, qualifica Metal e
Metal-hybrid nella tupla dichiarata, ma non trasforma le righe esterne del
report AMD in esecuzioni su quello SHA. Analogamente, RTX 3060 e RX 6750 XT non
qualificano genericamente ogni GPU NVIDIA o AMD. I dettagli e le identità sono in
[`amd-deep-clean-regression-repair.md`](amd-deep-clean-regression-repair.md) e
[`current-optimization-state.md`](current-optimization-state.md).

## Campagna preliminare v0.1.0

La campagna del 19 agosto 2026 usava Vulkan-hybrid all-GPU su Linux x86_64,
RTX 3060 12 GiB, driver 595.84, KV F16, contesto 4096 e llama.cpp
`13f2b28b098623391b1aacfd27995e1c8b7de9a9`. Non ci sono stati retry.

| Modello | Generazione semantica | Teacher-forced | Esito storico |
|---|---|---|---|
| 3B Instruct | 8/9 preservato | 16/16 top-1 | QUALIFIED |
| 3B Reasoning | 8/9, 9/9, 9/9 | 16/16 due volte | QUALIFIED |
| 8B Instruct | 8/9 preservato | 16/16 top-1 | QUALIFIED |
| 8B Reasoning | 9/9, 9/9, 8/9; byte divergenti | 16/16 due volte | NOT SUPPORTED |
| 14B Instruct | 9/9 preservato | 16/16 top-1 | QUALIFIED |
| 14B Reasoning | 9/9, 9/9, 9/9 | 16/16 due volte | QUALIFIED |

Per 8B Reasoning, le lunghezze S08 furono 330, 883 e 3.324 token. Il gate
predefinito richiedeva conteggi e byte identici tra tre processi freschi; la
parità teacher-forced non sostituiva quel requisito. L’esito resta `NOT
SUPPORTED` per quella campagna.

Il servizio preliminare superò tre richieste SSE sequenziali nello stesso
processo, arresto pulito, errori bounded, versione installata e smoke da archivio
locale. Scheduling concorrente, backend runtime, Q8, MoE, tool, multimodalità e
un canale Reasoning separato erano fuori contratto.

## Gate della release finale

La release può essere dichiarata qualificata soltanto dopo avere:

1. scelto e registrato un commit finale pulito;
2. rieseguito sullo stesso commit i gate applicabili di build, lint, frontend,
   runtime, parity, semantica, serving, failure path e documentazione;
3. creato il tag annotato immutabile `v0.1.0` su quel commit;
4. generato dal tag `graph-horizon-0.1.0.tar.gz` e il relativo `.sha256`;
5. verificato installazione pulita, versione e inferenza dall’archivio;
6. pubblicato tag e asset solo con autorizzazione esplicita e verificato il
   bootstrap anonimo se il repository è pubblico.

I dettagli della campagna preliminare sono in
[`release-qualification.md`](release-qualification.md). Le interfacce operative
sono in [`support/README.md`](../support/README.md); i processi riutilizzabili sono
nel [processo di validazione modello](model-validation-process.md), nel
[processo oracle](oracle-validation-process.md) e nella
[validazione KV](kv-quant-mistral-validation.md).

## Aggiornamento del registro

Una nuova evidenza sostituisce una riga soltanto quando registra revisione,
identità dell’artefatto, configurazione, criterio e stato terminale. Una risorsa
assente resta `external verification`; una failure non viene nascosta da un
benchmark favorevole. Il tag pubblicato non deve mai essere spostato.

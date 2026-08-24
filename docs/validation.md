<!--
Questo registro pubblica soltanto identità degli artefatti, evidenza revisionata
e stato di qualifica. Le procedure appartengono a support/README.md e docs/;
log grezzi e decisioni sperimentali restano nella cronologia Git.
-->

# Registro di validazione

## Stato corrente

Graph Horizon `v0.1.1` è stato rilasciato dal tag annotato e immutabile
`v0.1.1` come correzione di packaging della Web UI. La GitHub Release contiene
`graph-horizon-0.1.1.tar.gz` e il relativo record SHA-256. Il tag `v0.1.0` e i
suoi asset restano immutati e possiedono la qualifica numerica storica.

Le versioni Cargo e frontend correnti sono `0.1.1`; l'identità della patch è
esclusivamente il commit risolto da `v0.1.1^{commit}`. La campagna finale
v0.1.0 sostituisce, per il proprio contratto, le campagne preliminari sui runtime
`d1bf18f034fd44df5b8e81931e7feea32edeb47f` ed
`e7edc8315d397f6eb34c5efb91a9ae20b9b59bc4`; l'evidenza precedente resta
storica e valida soltanto per le revisioni dichiarate.

La compatibilità tecnica, la correttezza numerica, la qualità semantica e le
prestazioni sono claim distinti. Un file caricabile non è per questo
qualificato; un risultato storico non qualifica una sorgente successiva.

## Identità canonica della release

Questo file è il registro canonico dello stato di qualifica, ma non duplica
identità che possono divergere. Il commit sorgente della patch si ricava con
`git rev-parse v0.1.1^{commit}`: il tag annotato immutabile è il riferimento
autoritativo e punta al commit che contiene questo registro. Nessun hash
intermedio costituisce l'identità della release.

L'artefatto corrente è `graph-horizon-0.1.1.tar.gz`, generato con `git archive`
dal tag. Il solo valore SHA-256 autoritativo è il record affiancato
`graph-horizon-0.1.1.tar.gz.sha256`; il digest non viene copiato in questo file.
Archivio, record e annotazione riportano lo stesso nome/versione, mentre
l'header Git dell'archivio risolve allo stesso commit del tag.

`v0.1.0`, `graph-horizon-0.1.0.tar.gz` e il relativo `.sha256` restano
l'identità autoritativa della campagna numerica descritta più avanti. La patch
non sposta il tag né sostituisce quegli asset.

## Correzione di packaging v0.1.1

La patch modifica soltanto installazione, risoluzione degli asset Web, test e
documentazione/versioni. Il crate motore non cambia rispetto a `v0.1.0`, salvo
la versione del manifest; nessun risultato numerico o semantico storico viene
riattribuito alla nuova revisione.

Il difetto era riproducibile sul prodotto installato: `v0.1.0` compilava la Web
UI ma copiava nel prefisso soltanto il binario, che cercava
`web/frontend/dist` nella directory corrente. `v0.1.1` installa esclusivamente
file regolari sotto `<prefix>/share/graph-horizon/web`, rifiuta symlink e
risolve gli asset rispetto all'eseguibile.

Ambiente del 23 agosto 2026: Linux x86_64, AMD Radeon RX 6750 XT disponibile ma
smoke di packaging eseguito sul backend CPU, Rust/Cargo 1.97.1 più toolchain
minima 1.88.0, Node.js 24.18.0 e npm 11.16.0.

| Gate della patch v0.1.1 | Esito |
|---|---|
| formato Rust, sintassi shell e diff | PASS |
| Rust 1.88 e check CPU/Vulkan/Vulkan-hybrid | PASS |
| CPU Clippy `-D warnings` e suite completa | PASS: app 145, engine 163, integration 5+12 |
| frontend | PASS: 119 test, 0 errori/warning Svelte, build, 0 vulnerabilità |
| bootstrap/installer, inclusi asset annidati e rifiuto symlink | PASS |
| archivio/tag/checksum e installazione CPU in prefisso pulito | PASS |
| prodotto installato fuori checkout: versione, Web HTTP e inferenza SSE 3B | PASS |
| bootstrap pubblico anonimo dopo pubblicazione | PASS |

Il modello 3B Instruct usato nello smoke corrispondeva a byte count e SHA-256
del catalogo. La richiesta Web ha restituito HTTP 200 per UI, contesto e runtime
e ha completato `OK.` con statistiche e frame terminale `done`. Questo verifica
il confine di packaging e prodotto; non è una nuova campagna di parità o qualità.

## Ricerca pubblica non rilasciata

Questa evidenza appartiene alla working tree del 25 agosto 2026, non al tag
immutabile `v0.1.1`. Ambiente: Linux x86_64 7.0.0-30, Rust 1.95.0 con verifica
aggiuntiva su Rust 1.88.0, Node.js 25.9.0, npm 11.12.1 e curl 8.18.0.

La ricerca Web DuckDuckGo Lite e la ricerca Google News RSS sono state eseguite
contro Internet senza chiavi. Tre round consecutivi per adapter sono passati;
il run finale con le asserzioni complete ha prodotto `3 passed`: risultati reali
non vuoti, massimo cinque URL HTTP(S), titoli ed estratti non vuoti, publisher
presenti e risultati Web/News datati tutti provvisti di timestamp dentro
l'intervallo locale esatto richiesto. I test restano `#[ignore]` nella suite
ordinaria perché dipendono da servizi e rete esterni.

```sh
cargo test --locked --no-default-features --features cpu -p graph-horizon
cargo test --locked --no-default-features --features cpu -p graph-horizon \
  live_ -- --ignored
cd web/frontend && npm test && npm run check && npm run build
```

| Gate ricerca/Web UI | Esito |
|---|---|
| suite Rust ordinaria | PASS: 166, con 3 live ignorati |
| provider reali Web con date, News e News con date | PASS: 3/3 per tre round consecutivi |
| errori deterministici: 429, timeout, oversize, feed/JSON invalido, zero risultati | PASS |
| frontend | PASS: 132 test, 0 errori/warning Svelte, build |
| Clippy CPU `-D warnings` e check Rust 1.88 | PASS |
| installer CPU `fast` in prefisso pulito, binario e asset installati | PASS |
| dipendenze dirette | `scraper` 0.27.0 ISC; `roxmltree` 0.20.0 e `httpdate` 1.0.3 MIT/Apache-2.0 |

La qualità strutturale delle citazioni è coperta: il contesto ordina identificatori
`[S1]`, mantiene gli URL fuori dal testo affidato al modello, conserva provenienza
separata e marca come `Cited` solo gli identificatori presenti nella risposta
visibile. La working tree non contiene un GGUF, quindi la valutazione semantica
di una risposta realmente generata con queste fonti resta **external
verification**; non viene trasformata in PASS usando un modello diverso o una
risposta simulata. Inoltre gli estratti appartengono ai motori di ricerca, non
alle pagine originali: una citazione ben formata non prova da sola ogni claim.

## Contratto minimo Rust

Graph Horizon supporta Rust e Cargo 1.88 o successivi. Rust 1.85 è il minimo per
edition 2024, ma il grafo bloccato contiene dipendenze che dichiarano 1.88;
quindi una versione inferiore renderebbe falso il contratto corrente. Entrambi i
crate ereditano `rust-version = "1.88"` dal workspace e l'installer rifiuta prima
della build una toolchain precedente o non interpretabile.

La verifica riproducibile del limite è:

```sh
cargo +1.88.0 test --locked --workspace --all-targets \
  --no-default-features --features cpu
cargo +1.88.0 check --locked --workspace --all-targets \
  --no-default-features --features vulkan
cargo +1.88.0 check --locked --workspace --all-targets \
  --no-default-features --features vulkan-hybrid
```

I profili Metal richiedono macOS arm64 e restano una verifica esterna su questo
host Linux; il grafo Cargo bloccato, inclusi i pacchetti Metal opzionali, non
dichiara una versione Rust superiore a 1.88.

## Campagna finale v0.1.0

La determinazione seguente appartiene soltanto al commit risolto dal tag locale
annotato `v0.1.0`. La campagna è stata ripetuta dopo l'aggiornamento di questo
registro; se il tag è assente o punta altrove, questa sezione non qualifica
alcuna revisione.

Ambiente del 22 agosto 2026: Linux x86_64 `7.0.0-30-generic`, Intel Core
i5-9600K, NVIDIA RTX 3060 12 GiB, driver 595.84/Vulkan 1.4.329, Rust/Cargo
1.95.0, toolchain minima Rust/Cargo 1.88.0, Node.js 24.15.0, npm 11.12.1 e GCC
15.2.0. I sei Q4_K_M corrispondevano esattamente al catalogo; l'oracle era
llama.cpp `13f2b28b0`.

| Gate finale sul commit taggato | Esito |
|---|---|
| formato Rust, sintassi shell, diff e working tree | PASS |
| Rust 1.88: suite CPU; check Vulkan e Vulkan-hybrid | PASS |
| Clippy `-D warnings`, suite complete e build release CPU | PASS: app 143, engine 163, integration 5+12 |
| Clippy `-D warnings`, suite completa e build release Vulkan | PASS: app 143, engine 162, integration 5+12 |
| Clippy `-D warnings`, suite completa e build release Vulkan-hybrid | PASS: app 143, engine 233, integration 6+12 |
| frontend | PASS: 119 test, 0 errori/warning Svelte, build, 0 vulnerabilità |
| harness `support_scripts` ripetuto | PASS: 20 esecuzioni mirate, nessun `ETXTBSY` |
| autenticazione Q4_K_M | PASS: 6/6 |
| matrice reale/oracle | 37 PASS, 37 external verification, 0 failure, totale 74 |
| semantica terminale | 6 qualified, 0 not-qualified, 0 external verification |

Le 37 righe esterne della matrice sono sei Q8_0 assenti, 28 righe Metal non
eseguibili su Linux e tre righe Vulkan-hybrid mixed 14B senza memoria
sufficiente su questa macchina. Nessuna è riportata come PASS. Le righe CPU e
Vulkan disponibili, le righe mixed 3B/8B, una riga 14B Reasoning mixed e i
quattro endpoint Vulkan-hybrid 3B sono passati. Le tre righe Reasoning correnti
hanno ottenuto 4/4 casi critici, 9/9 semantici e 9/9 marker completi; le tre
righe Instruct mantengono l'evidenza preservata dichiarata dal protocollo.

La qualifica include inoltre: tag e header Git dell'archivio risolti allo stesso
commit; checksum affiancato verificato; installazione CPU in prefisso pulito
eseguita dall'archivio estratto; versione, CLI/Web UI e inferenza smoke
verificate sul binario installato. Il record `.sha256` pubblicato con la GitHub
Release coincide con il digest dell'archivio remoto.

Il 23 agosto 2026, dopo la pubblicazione del repository, il comando documentato
ha scaricato anonimamente bootstrap, archivio e checksum, verificato SHA-256,
compilato Web UI e backend CPU in profilo release, installato in un prefisso
pulito e restituito `graph-horizon 0.1.0`. Questo smoke non ha scaricato modelli
né ripetuto l'inferenza già verificata dall'archivio durante la campagna finale.

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

Questa tabella descrive lo stato tecnico revisionato oltre il contratto puntuale
della release v0.1.0. Le misure, le tuple fisiche e i limiti sono riassunti in
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

## Invarianti della release finale

La qualifica sopra resta valida soltanto mantenendo questi invarianti:

1. il tag annotato risolve al commit pulito che contiene questo registro;
2. sullo stesso commit sono stati rieseguiti i gate applicabili di build, lint, frontend,
   runtime, parity, semantica, superfici di prodotto, failure path e documentazione;
3. il tag annotato immutabile `v0.1.0` non viene spostato;
4. archivio `graph-horizon-0.1.0.tar.gz` e relativo `.sha256` derivano dal tag;
5. checksum, header Git, installazione pulita, versione e inferenza sono verificati
   dall'archivio, non dalla working tree;
6. tag remoto e asset pubblicati coincidono con l'identità qualificata; il
   bootstrap anonimo verificato continua a risolvere quegli asset immutabili.

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
benchmark favorevole. Il tag immutabile non deve mai essere spostato.

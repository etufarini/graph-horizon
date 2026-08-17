<!--
Questo registro pubblica la fotografia revisionata della validazione corrente:
identita degli artefatti, matrici terminali, limiti e fonti riproducibili. Non
contiene log grezzi, cronologia prestazionale o capability definite dal runtime.
-->

# Registro di validazione

## Ambito

Questo documento e il registro revisionato richiamato dai contratti pubblici
del repository. La compatibilita tecnica, la qualifica semantica e le
prestazioni sono claim distinti: il superamento di uno non implica gli altri.

La presente versione compatta ripubblica l'ultima evidenza revisionata del
registro rimosso il 6 agosto 2026. Il ripristino non costituisce una nuova
campagna. Le successive verifiche numeriche Vulkan sono indicate separatamente
e non sostituiscono la matrice tecnica o la qualifica semantica.

Il registro e una fotografia, non un diario: una nuova campagna sostituisce gli
esiti superati. Log grezzi e output dipendenti dalla macchina restano locali;
la storia delle decisioni rimane nei commit e nei report di fase.

## Artefatti autenticati

Il catalogo autoritativo e [`support/models.tsv`](support/models.tsv). I modelli
sono input esterni in sola lettura e non sono distribuiti con il repository.

| ID | Profilo | File Q4_K_M | Byte | SHA-256 |
|---|---|---|---:|---|
| 3b-instruct | instruct | `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf` | 2147023008 | `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8` |
| 3b-reasoning | reasoning | `Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf` | 2147021472 | `7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc` |
| 8b-instruct | instruct | `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` | 5198911904 | `33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761` |
| 8b-reasoning | reasoning | `Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf` | 5198910368 | `894aa3645ef8708a81dbe201c26105ce37c4c741252c89c5a78f81b49ac438c6` |
| 14b-instruct | instruct | `Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` | 8239593024 | `824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613` |
| 14b-reasoning | reasoning | `Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf` | 8239591488 | `fe08ca2158cd7438211ec6a4e5256d31bc980f016e3f5b635fe91fe6848d461c` |

Il runtime pubblico accetta questi profili Q4_K_M. I nomi Q8_0 catalogati sono
casi negativi: devono essere rifiutati prima dell'allocazione del backend.
L'elenco revisionato non e una whitelist letta dal runtime.

## Matrice tecnica a 74 righe

Il protocollo corrente e definito in
[`support/README.md`](support/README.md) e viene eseguito da
[`support/testing/matrix-check.sh`](support/testing/matrix-check.sh). Comprende
sei rifiuti Q8, 60 righe principali e otto endpoint 3B-Instruct. Ogni riga ha
uno solo stato terminale: `pass`, `external_verification` o `failure`.

L'ultima matrice Linux revisionata del 6 agosto 2026 aveva disponibile soltanto
il 3B-Reasoning Q4_K_M e il relativo Q8. Il rifiuto Q8 e passato; gli altri
artefatti erano assenti e il binario oracle esponeva un identificatore di
revisione troppo corto per il gate fissato. Nessun endpoint omogeneo era quindi
qualificabile in quella sessione.

```text
summary: pass=1 external_verification=73 failure=0 total=74
```

Il completamento macOS della stessa campagna, su Apple M4 con i sei Q4_K_M e
l'oracolo pinned disponibili, ha passato le 12 righe CPU, le 12 Metal, le 12
Metal-hybrid mixed e i quattro endpoint Metal. Vulkan e i rifiuti Q8 privi di
artefatto sono rimasti esterni:

```text
summary: pass=40 external_verification=34 failure=0 total=74
```

`external_verification` non e un pass e non autorizza sostituzioni di modello,
backend, KV, placement od oracle.

## Qualifica semantica

I tre Instruct conservano l'evidenza revisionata del Piano 05. I tre Reasoning
sono l'evidenza corrente del Piano 07, eseguita una sola volta con System prompt
Reasoning implicito, `context=4096`, `max_tokens=4096`, KV `f16`, Vulkan
all-GPU, `temperature=0.7`, `seed=0`, `top_p=1`, `top_k=0`, `min_p=0` e
`repeat_penalty=1`.

Il corpus Reasoning comprende S01-S04 e S06-S10. Il gate richiede
`critical=4/4` e almeno `semantic=8/9`; non ammette retry, tuning o fallback
CPU. Questa qualifica riguarda il percorso API Rust configurabile del runner,
non promette determinismo fra hardware e non qualifica contesti lunghi o System
prompt personalizzati.

```text
qualification: model_id=3b-instruct profile=instruct evidence=preserved status=qualified reason=plan-05-pass critical=4/4 semantic=8/9
qualification: model_id=3b-reasoning profile=reasoning evidence=current status=qualified reason=semantic-gate-pass critical=4/4 semantic=9/9
qualification: model_id=8b-instruct profile=instruct evidence=preserved status=qualified reason=plan-05-pass critical=4/4 semantic=8/9
qualification: model_id=8b-reasoning profile=reasoning evidence=current status=qualified reason=semantic-gate-pass critical=4/4 semantic=9/9
qualification: model_id=14b-instruct profile=instruct evidence=preserved status=qualified reason=plan-05-pass critical=4/4 semantic=9/9
qualification: model_id=14b-reasoning profile=reasoning evidence=current status=qualified reason=semantic-gate-pass critical=4/4 semantic=9/9
summary: qualified=6 not_qualified=0 external_verification=0 total=6
```

## Verifiche Vulkan successive

Il bring-up 8B e le ottimizzazioni Vulkan successive hanno aggiunto controlli
mirati di correttezza, senza rieseguire una campagna semantica completa:

- [Phase 22](docs/vulkan-second-model-bringup-phase22.md): parita CPU/Vulkan
  teacher-forced, oracle esterno e fallback del secondo modello;
- [Phase 23](docs/vulkan-model2-largek-matmul-phase23.md): gate numerico e
  non-regressione per la proiezione down large-K;
- [Phase 24](docs/vulkan-model2-gate-up-phase24.md): oracle Q4_K, parita
  teacher-forced e fallback per gate/up.

Questi risultati qualificano soltanto i percorsi e le tuple dichiarati nei
rispettivi report. Non cambiano gli stati delle due matrici precedenti.

## Aggiornamento del registro

Una nuova evidenza puo sostituire una riga soltanto se registra revisione,
identita dell'artefatto, configurazione, criterio e stato terminale. I risultati
incompleti restano `external_verification`; una failure non viene nascosta da un
benchmark favorevole. Dettagli operativi e criteri appartengono a:

- [processo di validazione modello](docs/model-validation-process.md);
- [processo oracle](docs/oracle-validation-process.md);
- [validazione KV](docs/kv-quant-mistral-validation.md).

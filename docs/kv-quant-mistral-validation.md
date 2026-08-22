<!--
Questo documento definisce il confronto ripetibile tra le KV cache f16 e int8
per Ministral; descrive protocollo ed evidenza richiesta, non risultati impliciti.
-->

# Validazione KV Ministral

## Scopo

La validazione confronta `f16` e `int8` mantenendo invariati artefatto,
backend, placement, prompt e contesto `4096`. Dimostra separatamente che ogni
schema rispetta il layout previsto, completa l'esecuzione e rimane entro il
criterio numerico scelto prima della prova.

Il riconoscimento di un formato GGUF non costituisce evidenza KV. Un risultato
vale soltanto per l'artefatto autenticato e per la configurazione registrata.

## Prerequisiti

- modello Ministral Q4_K_M leggibile e autenticato tramite dimensione e SHA-256;
- revisione Git e profilo Cargo registrati;
- uno dei profili `cpu`, `vulkan`, `vulkan-hybrid`, `metal`,
  `metal-hybrid` compilabile;
- contesto `4096` disponibile senza cambiare placement;
- hardware e driver richiesti dal backend scelto;
- criterio numerico e soglia definiti prima di osservare l'esito.

Una risorsa esterna assente produce `external-verification`; non autorizza un
altro modello, backend, contesto o schema KV.

## Matrice

Ogni riga fissa un artefatto e un backend, quindi esegue esattamente due casi:

| Caso | Contesto | KV |
|---|---:|---|
| baseline | 4096 | `f16` |
| candidato | 4096 | `int8` |

La matrice di profilo richiesta rende esplicite entrambe le righe KV:

| Profilo | KV richieste |
|---|---|
| `cpu` | f16, int8 |
| `vulkan` | f16, int8 |
| `vulkan-hybrid` | f16, int8 |
| `metal` | f16, int8 |
| `metal-hybrid` | f16, int8 |

Per un profilo hybrid, percentuale e riserva restano identiche tra i due casi. Il report
di caricamento deve registrare almeno `cpu_layers` e `gpu_layers`; una differenza
di placement invalida il confronto invece di diventare un risultato numerico.

## Procedura

1. Autenticare il file contro il catalogo senza modificarlo.
2. Registrare revisione, hardware, backend, contesto, prompt e limiti di output.
3. Eseguire i test sintetici di layout, quantizzazione e aritmetica della KV.
4. Avviare una sola volta la riga `f16` e conservarne output e placement.
5. Avviare una sola volta la riga `int8` con la sola KV modificata.
6. Confrontare completamento, token e metrica numerica prestabilita.
7. Assegnare alla riga uno stato terminale con una motivazione concreta.

Lo script operativo è:

```sh
support/profiling/validate-kv.sh \
  --model "/path/to/model.gguf" --backend cpu --context 4096
```

La stessa interfaccia accetta tutti i cinque profili. Lo script non effettua retry
e non sostituisce automaticamente il backend quando una riga non è eseguibile.
Esegue soltanto i due processi di profiling, prima `f16` e poi `int8`, su path,
backend e contesto espliciti. Non autentica il file contro `models.tsv`, non
confronta output o placement, non applica una soglia numerica e non assegna uno
stato terminale: questi passi restano responsabilità della procedura e del
report revisionato.

## Evidenza richiesta

Per ciascun caso si registrano:

- identificatore, dimensione e SHA-256 dell'artefatto;
- revisione, feature Cargo, backend e hardware;
- schema KV e contesto;
- `cpu_layers`, `gpu_layers` e memoria allocata;
- esito di caricamento e generazione;
- metrica, soglia e verdetto numerico;
- eventuale limite esterno che ha impedito l'esecuzione.

I test sintetici provano layout e conversioni locali, ma non sostituiscono una
prova con l'artefatto. Allo stesso modo, una generazione completata non prova da
sola che la perdita introdotta da `int8` rispetti la soglia.

## Stati terminali

- `verified`: entrambi i casi sono stati eseguiti e il criterio è superato;
- `failed`: configurazione, esecuzione o criterio numerico non sono validi;
- `external-verification`: una risorsa necessaria non era disponibile.

Ogni riga riceve un solo stato. Gli esiti revisionati appartengono a
[`docs/validation.md`](validation.md); questo documento conserva soltanto il
protocollo e non dichiara superate prove che non sono state eseguite.

<!--
This report records the bounded Vulkan-to-Metal long-context kernel review,
including reproducible measurements and retained or rejected decisions. It
defines no runtime API or general hardware support contract.
-->

# Indagine long-context Metal

## Ambito e mappatura

L'indagine parte dal branch Vulkan `perf/decode-long-context-stall`, confluito
in `main` a `f5807ed`, e valuta soltanto le modifiche trasferibili al kernel
Metal. Le modifiche web, runtime e di profiling Vulkan non sono kernel Metal.

| Modifica Vulkan | Stato Metal | Decisione |
|---|---|---|
| tile Q4_K prefill da 32 token | già presente in `metal_matmul_batched` | nessuna duplicazione |
| attention con più subgroup | applicabile al decode F16 long-context | mantenuta con 4 SIMD-group |
| riuso K/V tra query prefill | già espresso dal percorso tiled Metal | nessuna seconda variante |
| timestamp/query profiler Vulkan | dipende da command e query pool Vulkan | non trasferito nel kernel |
| candidati Vulkan revertiti | nessun beneficio finale dimostrato | non trasferiti |

## Tuple

- baseline: `f5807edf3fc2a265a013bcb32e54c21b8d977d21`;
- candidato wide prefill+decode: `a205802`, revertito da `388c9b4`;
- candidato finale decode-only: `7b5dd61`;
- modello: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2.147.023.008 byte;
- SHA-256: `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`;
- hardware: Apple M4, GPU 10 core, memoria unificata 24 GB;
- backend Metal standalone, KV F16, contesto allocato 32.768, greedy;
- prompt sintetici: ripetizione di `a ` calibrata a 2.042 e 8.192 token;
- benchmark `release`: una warm-up e tre repliche, salvo il controllo finale 8K
  esplicitamente indicato come singola replica senza warm-up.

## Baseline

| Prompt | Prompt tok/s | TTFT ms | CV TTFT | Decode tok/s | CV decode |
|---:|---:|---:|---:|---:|---:|
| 2.042 | 125,86 | 16.224,93 | 0,29% | 18,67 | 0,86% |
| 8.192 | 88,71 | 92.856,54 | 8,87% | 6,57 | 2,67% |

Il punto 8K è termicamente rumoroso sul MacBook Air fanless; il punto 2K è la
misura A/B primaria e 8K serve a individuare regressioni long-context ampie.

## Candidato rifiutato: wide prefill e decode

Il primo adattamento ha raddoppiato da 2 a 4 i SIMD-group del percorso
segmentato, portando i segmenti temporali da 8 a 16 sia in prefill sia in decode.
La online-softmax è rimasta causale e il test Metal contro il percorso seriale è
passato.

| Prompt | Prompt tok/s | Delta | TTFT ms | Delta | Decode tok/s | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2.042 | 128,03 | +1,72% | 15.949,71 | −1,70% | 25,55 | +36,85% |
| 8.192 | 71,35 | −19,57% | 114.855,31 | +23,69% | 8,96 | +36,38% |

Il parallelismo aggiuntivo è utile al decode ma perde occupancy/efficienza nel
prefill lungo. Il candidato è stato rifiutato e revertito senza riscrivere la
cronologia.

## Candidato mantenuto: wide decode F16

Il percorso finale seleziona 4 SIMD-group soltanto quando `rows == 1`, KV è F16,
la head dimension è 128, la SIMD width è 32 e la history ha almeno 1.024 token.
Prefill, INT8, mixed placement e forme non qualificate mantengono i percorsi
precedenti. La selezione aggiunge un solo mode uniforme allo stesso kernel.

| Prompt | Prompt tok/s | Delta | TTFT ms | Delta | Decode tok/s | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2.042 | 126,18 | +0,25% | 16.183,67 | −0,25% | 24,98 | +33,80% |

Il controllo finale 8K, una replica senza warm-up, ha prodotto 97,71 prompt
tok/s, TTFT 83.838,78 ms e 9,50 decode tok/s. Non è usato per rivendicare un
guadagno prefill contro la baseline a tre repliche; dimostra che la regressione
ampia del wide prefill non è presente nello stato finale.

## Correttezza e stato finale

- 29/29 test Metal passano, incluso l'oracle numerico che confronta attention
  segmentata e seriale a contesto 1.024 per F16 e INT8;
- parità esterna F16: prompt IDs identici e 16/16 token greedy identici
  all'oracle llama.cpp `13f2b28b`;
- parità esterna INT8: prompt IDs identici e 16/16 token greedy identici;
- `cargo fmt --check` e `git diff --check` passano;
- il matmul prefill resta invariato perché possiede già il tile trasferibile;
- il solo cambiamento produttivo mantenuto è il wide decode F16 long-context.

Il maggiore collo di bottiglia restante è il prefill attention: aumentare i
SIMD-group nello stesso workgroup non migliora il regime 8K. Un eventuale passo
successivo richiede un algoritmo tiled/multi-workgroup distinto, non un'altra
variante locale della stessa geometria.

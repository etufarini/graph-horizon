<!--
Questa pagina conserva risultati, tuple, decisioni e revisioni dell'indagine
Vulkan sul solo prefill long-context. Non definisce supporto o API runtime.
-->

# Indagine prefill long-context Vulkan

## Ambito e tuple

Target esclusivo: prefill. Il decode è stato usato solo come controllo di
regressione. Nessuna percentuale decode è stata usata per scegliere i candidati.

- baseline: `42b4e4073a8d26389d94b8e78b0588f3af5067b0`;
- profiler dettagliato: `0ea4472`;
- stato prestazionale ripristinato: `87a3909` più questo report;
- modello: `Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2.147.023.008 byte;
- SHA-256: `9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`;
- forma: 26 layer, hidden 3072, Q 4096, K/V 1024, FFN 9216,
  32 query head, 8 KV head, head dimension 128;
- backend: Vulkan puro, tutte le weight su RX 6750 XT 12 GiB, KV F16;
- contesto allocato 32.768, greedy, due token richiesti;
- prompt sintetico calibrato a 128, 512, 2.048, 8.192, 16.384 e 28.000
  token effettivi;
- una warm-up separata e tre ripetizioni misurate per la baseline pubblica;
- build Cargo `release`, feature diagnostica `vulkan-profile`;
- RX 6750 XT, RADV Mesa 26.0.3, Vulkan 1.4.335, Ryzen 5 5500,
  Rust 1.97.1.

Il wall prefill è il TTFT pubblico: include la prima riduzione/sampling, inferiore
a 1 ms, e quindi costituisce un upper bound praticamente identico al wall prefill.
I timestamp GPU riportati sotto contengono invece soltanto la fase prefill.

## Baseline

| Token | Wall prefill / TTFT (ms) | Prompt tok/s | ms/token | CV |
|---:|---:|---:|---:|---:|
| 128 | 935,71 | 136,80 | 7,310 | 0,07% |
| 512 | 3.812,81 | 134,28 | 7,447 | 0,23% |
| 2.048 | 16.662,53 | 122,91 | 8,136 | 0,14% |
| 8.192 | 84.921,56 | 96,47 | 10,366 | 0,06% |
| 16.384 | 222.110,16 | 73,77 | 13,557 | 0,16% |
| 28.000 | 561.274,91 | 49,89 | 20,046 | 0,19% |

Warm-up, caricamento del modello e compilazione pipeline non fanno parte delle
tre ripetizioni pubbliche. La VRAM di picco dopo il caricamento è 6,94–6,98 GB.
Durante i run la GPU è al 95–99%, 2,42–2,65 GHz, fino a 157 W, 69 °C edge e
92 °C junction. Non è stato osservato throttling.

## Scaling

Il rapporto nell'ultima colonna è rispetto alla riga precedente; il fattore N è
indicato per evitare di chiamare impropriamente `T(4N)/T(N)` un raddoppio.

| N | T(N) ms | T/N ms | T/N² ms | Fattore N | Rapporto T |
|---:|---:|---:|---:|---:|---:|
| 128 | 935,71 | 7,310234 | 0,057111206 | — | — |
| 512 | 3.812,81 | 7,446895 | 0,014544716 | 4× | 4,0748× |
| 2.048 | 16.662,53 | 8,136001 | 0,003972657 | 4× | 4,3701× |
| 8.192 | 84.921,56 | 10,366401 | 0,001265430 | 4× | 5,0966× |
| 16.384 | 222.110,16 | 13,556528 | 0,000827425 | 2× | 2,6155× |
| 28.000 | 561.274,91 | 20,045533 | 0,000715912 | 1,709× | 2,5270× |

Scaling delle famiglie GPU, per prompt:

| N | Attention ms | Attention/N² | MLP ms | MLP/N | Proiezioni ms | Proiezioni/N | KV write ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6,83 | 0,0004168 | 624,93 | 4,882 | 268,46 | 2,097 | 0,092 |
| 512 | 82,36 | 0,0003142 | 2.505,37 | 4,893 | 1.071,98 | 2,094 | 0,334 |
| 2.048 | 1.737,50 | 0,0004143 | 10.015,30 | 4,890 | 4.287,33 | 2,093 | 1,320 |
| 8.192 | 25.927,38 | 0,0003863 | 39.614,18 | 4,836 | 16.944,54 | 2,068 | 5,214 |
| 16.384 | 102.962,82 | 0,0003836 | 79.728,98 | 4,866 | 34.175,07 | 2,086 | 12,778 |
| 28.000 | 352.272,63 | 0,0004493 | 139.521,14 | 4,983 | 59.506,25 | 2,125 | 24,440 |

MLP, proiezioni, KV write, command recording e dispatch crescono
sostanzialmente in modo lineare. L'attention cresce quasi quadraticamente:
8K→16K fa 3,97× mentre N raddoppia. È il termine che degrada il throughput.

## Top 10 kernel del solo prefill

Le dimensioni sono quelle di un chunk da 32 token:

- `ATTN`: Q `[32,32,128]`, K/V cache `[history,8,128]`, output
  `[32,32,128]`, workgroup `[32,32]`, local size 512;
- `DOWN`: A `[32,9216]`, W `[3072,9216]`, Y `[32,3072]`, WG `[48,1]`;
- `GATE`/`UP`: A `[32,3072]`, W `[9216,3072]`, Y `[32,9216]`, WG `[144,1]`;
- `OUT`: A `[32,4096]`, W `[3072,4096]`, Y `[32,3072]`, WG `[48,1]`;
- `Q`: A `[32,3072]`, W `[4096,3072]`, Y `[32,4096]`, WG `[64,1]`;
- `V`: A `[32,3072]`, W `[1024,3072]`, Y `[32,1024]`, WG `[16,1]`;
- `LOGITS`: A `[1,3072]`, W `[131072,3072]`, Y `[131072]`, WG `[2048,1]`;
- `NORM`: `[32,3072]` → `[32,3072]`, WG massimo `[32,1]`;
- `ROPE`: una riga `[32 o 8,128]` → stessa forma, WG massimo `[32,1]`;
- `ELEM`: residual/Silu su `[32,3072 o 9216]`, WG massimo `[4608,1]`.

Q/K e gate/up sono coppie indipendenti senza barrier intermedia e possono
sovrapporsi. Il timestamp della prima può includere il lavoro concorrente della
seconda; la somma di famiglia è affidabile e additiva, il singolo split va letto
come attribuzione operativa, non come isolamento matematico.

### 2K

| # | Operazione / shader | GPU ms | GPU prefill | Invocazioni | Media ms | Forma |
|---:|---|---:|---:|---:|---:|---|
| 1 | MLP down / `matmul_q6k_batch_f16` | 4.707,35 | 28,51% | 1.664 | 2,828935 | DOWN |
| 2 | MLP gate / `matmul_q4k_batch_f16` | 3.329,14 | 20,16% | 1.664 | 2,000685 | GATE |
| 3 | output projection / `matmul_q4k_batch_f16` | 2.131,33 | 12,91% | 1.664 | 1,280847 | OUT |
| 4 | MLP up / `matmul_q4k_batch_f16` | 1.971,97 | 11,94% | 1.664 | 1,185076 | UP |
| 5 | Q projection / `matmul_q4k_batch_f16` | 1.939,48 | 11,74% | 1.664 | 1,165551 | Q |
| 6 | `attention_prefill_wide` | 1.737,23 | 10,52% | 1.664 | 1,044008 | ATTN |
| 7 | `logits_q6k` | 277,29 | 1,68% | 64 | 4,332691 | LOGITS |
| 8 | V projection / `matmul_q6k_batch_f16` | 219,92 | 1,33% | 1.664 | 0,132160 | V |
| 9 | `rmsnorm_x` | 28,25 | 0,17% | 3.392 | 0,008329 | NORM |
| 10 | `rope` | 26,60 | 0,16% | 106.496 | 0,000250 | ROPE |

### 8K

| # | Operazione / shader | GPU ms | GPU prefill | Invocazioni | Media ms | Forma |
|---:|---|---:|---:|---:|---:|---|
| 1 | `attention_prefill_wide` | 25.942,66 | 30,75% | 6.656 | 3,897636 | ATTN |
| 2 | MLP down / `matmul_q6k_batch_f16` | 18.620,11 | 22,07% | 6.656 | 2,797492 | DOWN |
| 3 | MLP gate / `matmul_q4k_batch_f16` | 13.140,86 | 15,58% | 6.656 | 1,974287 | GATE |
| 4 | output projection / `matmul_q4k_batch_f16` | 8.395,55 | 9,95% | 6.656 | 1,261350 | OUT |
| 5 | MLP up / `matmul_q4k_batch_f16` | 7.858,53 | 9,32% | 6.656 | 1,180668 | UP |
| 6 | Q projection / `matmul_q4k_batch_f16` | 7.681,57 | 9,11% | 6.656 | 1,154081 | Q |
| 7 | `logits_q6k` | 1.084,71 | 1,29% | 256 | 4,237135 | LOGITS |
| 8 | V projection / `matmul_q6k_batch_f16` | 885,74 | 1,05% | 6.656 | 0,133074 | V |
| 9 | `rmsnorm_x` | 103,90 | 0,12% | 13.568 | 0,007658 | NORM |
| 10 | `rope` | 97,36 | 0,12% | 425.984 | 0,000229 | ROPE |

### 16K

| # | Operazione / shader | GPU ms | GPU prefill | Invocazioni | Media ms | Forma |
|---:|---|---:|---:|---:|---:|---|
| 1 | `attention_prefill_wide` | 103.236,76 | 46,67% | 13.312 | 7,755165 | ATTN |
| 2 | MLP down / `matmul_q6k_batch_f16` | 37.649,17 | 17,02% | 13.312 | 2,828213 | DOWN |
| 3 | MLP gate / `matmul_q4k_batch_f16` | 26.338,66 | 11,91% | 13.312 | 1,978565 | GATE |
| 4 | output projection / `matmul_q4k_batch_f16` | 16.815,22 | 7,60% | 13.312 | 1,263162 | OUT |
| 5 | MLP up / `matmul_q4k_batch_f16` | 15.738,11 | 7,12% | 13.312 | 1,182249 | UP |
| 6 | Q projection / `matmul_q4k_batch_f16` | 15.616,10 | 7,06% | 13.312 | 1,173084 | Q |
| 7 | `logits_q6k` | 2.488,69 | 1,13% | 512 | 4,860721 | LOGITS |
| 8 | V projection / `matmul_q6k_batch_f16` | 1.780,44 | 0,80% | 13.312 | 0,133747 | V |
| 9 | `rmsnorm_x` | 214,94 | 0,10% | 27.136 | 0,007921 | NORM |
| 10 | `rope` | 188,40 | 0,09% | 851.968 | 0,000221 | ROPE |

### 28K

| # | Operazione / shader | GPU ms | GPU prefill | Invocazioni | Media ms | Forma |
|---:|---|---:|---:|---:|---:|---|
| 1 | `attention_prefill_wide` | 351.270,94 | 62,95% | 22.750 | 15,440481 | ATTN |
| 2 | MLP down / `matmul_q6k_batch_f16` | 66.123,48 | 11,85% | 22.750 | 2,906527 | DOWN |
| 3 | MLP gate / `matmul_q4k_batch_f16` | 46.213,76 | 8,28% | 22.750 | 2,031374 | GATE |
| 4 | output projection / `matmul_q4k_batch_f16` | 29.242,15 | 5,24% | 22.750 | 1,285369 | OUT |
| 5 | MLP up / `matmul_q4k_batch_f16` | 27.289,54 | 4,89% | 22.750 | 1,199540 | UP |
| 6 | Q projection / `matmul_q4k_batch_f16` | 27.277,77 | 4,89% | 22.750 | 1,199023 | Q |
| 7 | `logits_q6k` | 4.707,32 | 0,84% | 875 | 5,379799 | LOGITS |
| 8 | V projection / `matmul_q6k_batch_f16` | 3.019,16 | 0,54% | 22.750 | 0,132710 | V |
| 9 | `rmsnorm_x` | 413,28 | 0,07% | 46.375 | 0,008912 | NORM |
| 10 | residual + Silu | 390,88 | 0,07% | 68.250 | 0,005727 | ELEM |

## Attribuzione del wall time

Il profiler attribuisce 99,27–99,73% del GPU time. A 28K la ripartizione del
wall è:

| Componente | ms | % wall | Metodo |
|---|---:|---:|---|
| QK | ~161.041 | ~28,69% | stima da ablation causale 8K |
| online softmax | ~6.358 | ~1,13% | stima da ablation causale 8K |
| V read + AV + combine | ~184.874 | ~32,94% | stima da ablation causale 8K |
| QKV projection | 30.278,72 | 5,39% | timestamp GPU |
| output projection | 29.227,54 | 5,21% | timestamp GPU |
| MLP gate/up/down | 139.521,14 | 24,86% | timestamp GPU |
| norm + RoPE | 743,29 | 0,13% | timestamp GPU |
| KV write | 24,44 | 0,004% | timestamp GPU |
| embedding/elementwise/logits | 5.223,95 | 0,93% | timestamp GPU |
| barrier/inter-kernel GPU residual | 1.530,31 | 0,27% | total meno kernel |
| CPU command recording | 1.970,10 | 0,35% | clock CPU |
| altro TTFT | ~325,28 | ~0,06% | residuo wall esclusivo |

La scomposizione interna dell'attention è stata misurata direttamente a 8K:

| Ablation | Attention GPU ms | Quota fused |
|---|---:|---:|
| QK-only | 11.859,64 | 45,72% |
| QK + online softmax, senza V | 12.327,84 | 47,52% |
| softmax incrementale | 468,20 | 1,81% |
| V read + AV + combine incrementale | 13.614,83 | 52,48% |

Le quote 28K sono estrapolazioni esplicitamente etichettate, non timestamp
indipendenti: il kernel è fused. QK, softmax e AV eseguono tutti una iterazione
per coppia causale, quindi condividono lo scaling quadratico osservato.

## CPU, GPU e command path

Il fence wait include l'esecuzione GPU e non va sommato ad essa. `Wait−GPU` è
la parte esclusiva del wait.

| N | Wall ms | GPU prefill ms | CPU record ms | Submit ms | Fence wait ms | Wait−GPU ms |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 935,71 | 929,88 | 8,86 | 0,15 | 930,44 | 0,56 |
| 512 | 3.812,81 | 3.778,52 | 33,77 | 0,68 | 3.780,74 | 2,22 |
| 2.048 | 16.662,53 | 16.516,48 | 134,99 | 2,70 | 16.525,45 | 8,97 |
| 8.192 | 84.921,56 | 84.311,89 | 548,45 | 10,83 | 84.347,61 | 35,72 |
| 16.384 | 222.110,16 | 220.863,54 | 1.133,65 | 22,03 | 220.935,14 | 71,59 |
| 28.000 | 561.274,91 | 558.822,00 | 1.970,10 | 39,11 | 558.940,43 | 118,43 |

Il tempo push-descriptor è 10,88 / 42,75 / 90,55 / 162,43 ms a
2K/8K/16K/28K: 0,029% del wall a 28K ed è incluso nel record. Non ci sono
`vkQueueWaitIdle` o `vkDeviceWaitIdle` nel percorso caldo, né submit/wait per
layer: un command, submit e fence per chunk da 32.

| N | Command/submission | Dispatch | Dispatch/token | Barrier |
|---:|---:|---:|---:|---:|
| 128 | 4 | 8.248 | 64,44 | 4.608 |
| 512 | 16 | 32.992 | 64,44 | 18.432 |
| 2.048 | 64 | 131.968 | 64,44 | 73.728 |
| 8.192 | 256 | 527.872 | 64,44 | 294.912 |
| 16.384 | 512 | 1.055.744 | 64,44 | 589.824 |
| 28.000 | 875 | 1.804.250 | 64,44 | 1.008.000 |

Le barrier sono compute-shader → compute-shader, shader-write →
shader-read/write. Il loro upper bound è il residuo GPU 0,27–0,65%; non sono il
bottleneck nonostante il conteggio elevato.

## Attention e materializzazione

`NxN materialization: NO`.

- score buffer a 2K/8K/16K/28K: 0 byte;
- scritture/letture score: 0/0;
- pass separati QK/mask/softmax/AV: 0;
- traffico DRAM da score intermedi: 0 byte;
- barrier tra QK, softmax e AV: 0.

`attention_prefill_wide` esegue QK, online softmax e AV in un solo shader. Ogni
workgroup gestisce una coppia `(query head, query row)` e conserva max, somma e
accumulatore nei registri/shared memory. La causa quadratica non è un buffer NxN,
ma la scansione causale ripetuta della history per ogni query/head.

## KV del solo prefill

| Operazione KV | Costo/attività |
|---|---|
| write F16 | 5,21 ms a 8K; 12,78 ms a 16K; 24,44 ms a 28K |
| copy separata | assente |
| transpose | assente |
| reformat | assente |
| resize durante il prompt | assente |
| allocazione contenuto KV | 2 buffer per richiesta; inclusa sotto |
| totale gestione KV | 0,004–0,008% del prefill long-context |

I read K/V interni al calcolo attention sono conteggiati in QK/AV, non in
“KV management”. Le 13 allocazioni per richiesta (2 KV + 11 scratch) richiedono
3.493.068.800 byte virtuali complessivi ma soltanto 0,379 ms nel campione; non
crescono con N. Non avvengono map/unmap o copie staging nel prefill caldo.

## Traffico e limiti hardware

Per ogni coppia causale lo shader legge logicamente 512 byte (K+V F16) ed esegue
circa 512 FLOP QK+AV. L'intensità è quindi ~1 FLOP/byte.

| N | Traffico logico attention | Tempo | GB/s logici | TFLOP/s effettivi |
|---:|---:|---:|---:|---:|
| 2.048 | 0,894 TB | 1,737 s | 514,5 | 0,514 |
| 8.192 | 14,295 TB | 25,943 s | 551,0 | 0,551 |
| 16.384 | 57,178 TB | 103,237 s | 553,9 | 0,554 |
| 28.000 | 166,992 TB | 351,271 s | 475,4 | 0,475 |

Il dato è “logico” e può superare i ~432 GB/s fisici grazie alle cache. Dimostra
comunque un kernel bandwidth/cache-reuse-bound, non SFU/compute-bound.

Classificazione delle cinque famiglie più costose a 28K:

| Famiglia | Throughput indicativo | Classificazione |
|---|---:|---|
| attention fused | 475 GB/s logici, 0,475 TFLOP/s | bandwidth/cache-reuse |
| MLP down Q6_K | ~0,62 TFLOP/s | compute/dequant/occupancy |
| MLP gate Q4_K | ~0,89 TFLOP/s | compute/dequant, sovrapposta con up |
| output projection Q4_K | ~0,63 TFLOP/s | compute/dequant |
| Q projection Q4_K | ~0,67 TFLOP/s | compute/dequant, sovrapposta con K |

## Profilo per layer

| N | Min layer ms | Mediana layer ms | Max layer ms |
|---:|---:|---:|---:|
| 2.048 | 607,37 | 625,21 | 634,77 |
| 8.192 | 3.090,96 | 3.211,00 | 3.236,89 |
| 16.384 | 8.104,81 | 8.385,15 | 8.518,75 |
| 28.000 | 19.622,35 | 21.317,04 | 21.899,04 |

Non ci sono layer anomali, resize periodici o copie localizzate. La dispersione
28K segue soprattutto differenze Q4_K/Q6_K e scheduling, non outlier strutturali.

## Esperimenti e decisioni

| Revisione | Ipotesi | Quota/atteso | Risultato 8K | Correttezza | Decisione / revert |
|---|---|---|---|---|---|
| `aa13b6d` | ablation QK-only | isola QK | attention 11,860 s | output diagnostico | evidenza; `7bc8ca5` |
| `5f0fce1` | QK+softmax senza V | separa softmax/AV | attention 12,328 s | output diagnostico | evidenza; `ef7b1dc` |
| `5ade6a8` | due query/WG, un load K/V | −~50% traffico, +20–35% E2E | 96,47→71,56 tok/s, −25,8%; attention +104% | F16/INT8 max_err 0 | reject; `0c3d98e` |
| `65e0763` | 1024 thread, 16 subgroup | +6–16% E2E long | 96,47→86,44 tok/s, −10,4%; attention +38,1% | F16/INT8 max_err 0 | reject; `8387a78` |
| `2543896` | wave32 su WG 512 | più subgroup senza più LDS | 96,47→98,81 tok/s, +2,43%; attention −7,77% | F16/INT8 max_err 0 | interesting ma Amdahl 28K ≤5,14%; `6d1090a` |

Il query-pair reuse dimezza i load ma raddoppia registri/shared per query e dimezza
i workgroup: l'occupancy collassa. Il variant 1024 usa 33,3 KiB LDS e limita la
residenza. Wave32 conserva LDS ma porta da 2 a 4 componenti per lane; il guadagno
locale è reale ma troppo piccolo per giustificare feature/device/pipeline logic.

Tre strategie strutturali distinte non hanno prodotto un beneficio proporzionato,
soddisfacendo la condizione di arresto. Non sono state provate tile maggiori o
taglie intermedie perché sarebbero varianti degli stessi meccanismi già falsificati.

## Risultato finale

Nessuna patch prestazionale è stata mantenuta. Shader, dispatch e pipeline finali
sono identici alla baseline. Il controllo finale 8K è 96,34 tok/s, 85.031,08 ms,
attention 25.947,33 ms: entro 0,14% dalla baseline a singola traccia.

| Prompt | Baseline tok/s | Finale tok/s | Delta | Baseline TTFT | Finale TTFT | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 122,91 | 122,91 | 0% per identità codice | 16.662,53 ms | 16.662,53 ms | 0% |
| 8K | 96,47 | 96,47 | 0% per identità codice | 84.921,56 ms | 84.921,56 ms | 0% |
| 16K | 73,77 | 73,77 | 0% per identità codice | 222.110,16 ms | 222.110,16 ms | 0% |
| 28K | 49,89 | 49,89 | 0% per identità codice | 561.274,91 ms | 561.274,91 ms | 0% |

Decode regression finale: 0% per identità degli shader/dispatch; il controllo
8K singolo è 29,99 tok/s contro 30,35–30,65 nei campioni baseline, entro il rumore
di una singola rep e senza modifica del percorso decode.

## Sintesi

```text
Prefill root cause:
scansione causale ripetuta della history nel kernel attention fused

Evidence:
attention 10,5% a 2K, 30,8% a 8K, 46,7% a 16K, 63,0% a 28K;
T_attention/N² quasi costante; GPU ≈ wall

Dominant PREFILL component:
attention_prefill_wide = 351,27 s / 62,95% GPU a 28K

Scaling:
attention ~N²; MLP/proiezioni/KV/command ~N

CPU vs GPU:
558,02 s GPU su 560,43 s wall nella traccia 28K

NxN materialization:
NO

KV cost during PREFILL:
KV write/management 0,004–0,008%; i read causali sono lavoro attention

Highest-impact experiment:
query-pair KV reuse, potenziale teorico alto ma −25,8% E2E per occupancy

Fixes kept:
solo profiler prefill dettagliato feature-gated

Fixes rejected:
query-pair reuse; 1024-thread; wave32 (segnale +2,43%, complessità non proporzionata)

Prefill improvement:
0% finale

Decode regression:
0% per identità codice

Remaining PREFILL bottleneck:
QK + V/AV bandwidth/cache-reuse nel kernel fused

Next highest-impact structural optimization:
un algoritmo multi-workgroup/tiled che riusi K/V mantenendo molti workgroup
residenti, con partial state in scratch e riduzione separata; richiede nuova
orchestrazione, buffer temporaneo e dispatch, quindi è fuori dal perimetro dei
tre prototipi locali conclusi e va progettato come percorso prefill dedicato
```

## Follow-up split-KV multi-workgroup — RTX 3060, 2026-08-10

Questa indagine ha implementato il percorso multi-workgroup proposto sopra e lo
ha poi rimosso perché non raggiunge la soglia di accettazione del 10% end-to-end.
I risultati non sono direttamente confrontabili con quelli RX 6750 XT delle
sezioni precedenti: macchina, driver e baseline sono diversi.

### Riproducibilità e baseline

- branch locale: `perf/vulkan-prefill-split-kv`;
- commit di partenza e stato finale del codice: `43f2da5`;
- GPU: NVIDIA GeForce RTX 3060, driver 595.84;
- modello e SHA-256: gli stessi dichiarati in apertura;
- backend Vulkan, KV F16, contesto 32.768, greedy, due token richiesti;
- prompt sintetico: `" a"` ripetuto fino a ottenere esattamente N token dopo
  l'applicazione del chat template;
- build `release`, feature `vulkan-profile`;
- per ogni punto pubblico: una warm-up più tre ripetizioni misurate;
- i contatori profiler aggregano i quattro run e sono stati divisi per quattro;
  wall, tok/s e CV escludono la warm-up.

La baseline è stata acquisita sul worktree pulito prima della prima modifica di
produzione. Tutti i CV del prompt sono inferiori allo 0,4%.

| Prompt | Tok/s | Wall prefill (ms) | GPU prefill (ms) | Attention (ms) | Attention/GPU | CV |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 74,61 | 27.447,79 | 27.173,44 | 3.105,04 | 11,43% | 0,33% |
| 8K | 54,93 | 149.138,63 | 148.280,05 | 51.206,35 | 34,53% | 0,04% |
| 16K | 40,24 | 407.152,88 | 405.613,55 | 209.462,67 | 51,64% | 0,04% |
| 28K | 28,89 | 969.170,17 | 966.629,22 | 627.180,43 | 64,88% | 0,02% |

### Prototipo

Il prototipo usava due dispatch solo nel prefill:

1. un workgroup per `(query, head, split)` calcolava online-softmax stabile e
   scriveva `(m, l, O)` per una partizione causale contigua della history;
2. un workgroup per `(query, head)` fondeva gli stati con la correzione di scala
   `exp(m_i - m)` e produceva l'output finale.

Non veniva materializzata una matrice N×N. Il buffer era fisso, allocato insieme
al backend, con capacità massima di 16 split: 8.519.680 byte. La configurazione
selezionata a 8 split ne usava 4.259.840 per batch. Il percorso decode e le sue
pipeline non venivano modificati. Gli split supportati dal prototipo erano
1 (baseline), 2, 4, 8 e 16.

### Sweep e candidati scartati

Prima selezione a 8K, singolo run per candidato, workgroup da 256 thread:

| Variante | Split | Tok/s | Wall (ms) | Partial (ms) | Reduce (ms) | Attention totale (ms) |
|---|---:|---:|---:|---:|---:|---:|
| split-KV | 2 | 56,09 | 146.053,43 | 48.853,41 | 49,41 | 48.902,82 |
| split-KV | 4 | 56,03 | 146.219,93 | 48.322,39 | 76,84 | 48.399,23 |
| split-KV | 8 | 55,94 | 146.446,67 | 48.347,83 | 131,72 | 48.479,55 |
| GQA, 2 query head per KV scan | 4 | 49,49 | 165.532,40 | 68.417,08 | 76,69 | 68.493,77 |
| GQA, 2 query head per KV scan | 8 | 49,24 | 166.381,75 | 68.366,86 | 130,57 | 68.497,44 |
| GQA, 2 query head per KV scan | 16 | 49,03 | 167.064,60 | 68.941,70 | 231,90 | 69.173,60 |

La curva split 2/4/8 è sostanzialmente piatta: già il kernel originale emette
1.024 workgroup per dispatch, quindi la sola moltiplicazione dei workgroup non
risolve il limite di banda/cache. La variante GQA riusava K/V per due query head,
ma raddoppiava Q e accumulatori vivi; la regressione piatta al crescere degli
split attribuisce il costo a pressione registri/residenza, non a parallelismo
insufficiente. Non è stata raccolta una misura hardware diretta di occupancy:
Nsight Compute non è integrato nel benchmark; questa attribuzione è quindi
un'inferenza dai tempi e dal live state, non un contatore dichiarato.

Ridurre il workgroup da 256 a 128 thread e usare 8 split è stato il miglior
candidato. A 16K il suo attention era 195.986 ms contro 195.569 ms del candidato
256-thread/4-split: entro lo 0,22%, ulteriore evidenza del plateau di banda.

### Risultato ripetuto del miglior candidato

| Prompt | Baseline tok/s | Split tok/s | Delta tok/s | Baseline wall (ms) | Split wall (ms) | Delta wall | Speedup attention |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 74,61 | 74,68 | +0,09% | 27.447,79 | 27.422,56 | +0,09% | 1,012× |
| 8K | 54,93 | 55,99 | +1,93% | 149.138,63 | 146.324,17 | +1,89% | 1,057× |
| 16K | 40,24 | 41,66 | +3,53% | 407.152,88 | 393.242,01 | +3,42% | 1,069× |
| 28K | 28,89 | 30,34 | +5,02% | 969.170,17 | 922.904,32 | +4,77% | 1,078× |

Tempi GPU del candidato selezionato:

| Prompt | GPU prefill (ms) | Partial (ms) | Reduce (ms) | Buffer R+W cumulativo |
|---:|---:|---:|---:|---:|
| 2K | 27.149,64 | 3.033,93 | 33,34 | 14,18 GB |
| 8K | 145.483,83 | 48.308,01 | 131,53 | 56,71 GB |
| 16K | 391.727,55 | 195.723,96 | 261,69 | 113,41 GB |
| 28K | 920.217,34 | 581.455,85 | 450,57 | 193,82 GB |

A 28K la riduzione costa soltanto lo 0,049% del tempo GPU e il traffico scratch
R+W è circa lo 0,116% dei 166,99 TB logici letti dalla KV. Buffer e merge non
sono quindi il limite. Il modello di Amdahl, usando il 64,88% baseline attribuito
all'attention e lo speedup locale 1,078×, predice circa 1,048× end-to-end; il
rapporto wall misurato è 1,050×. Il guadagno è quasi interamente spiegato dallo
speedup locale e resta circa metà della soglia minima richiesta.

### Correttezza e decode

Il test focalizzato, esteso durante l'esperimento e poi rimosso con il candidato,
era:

```text
cargo test -p graph_horizon_engine --locked --no-default-features \
  --features vulkan \
  'backend::vulkan::cpu_vulkan_parity::vulkan_prefill_attention_matches_sequential_decode' \
  -- --nocapture
```

Ha confrontato split 2/4/8 con una reference CPU sequenziale su base 0/N=3 e
base 33/N=32, contesto 65, head dimension 128 e GQA 2:1. Copriva partizioni
vuote, ultime partizioni parziali, causal mask, KV F16 e il percorso INT8 già
esistente. Sia l'output attention sia i logits FP32 proiettati hanno prodotto
`max_abs=0`, `mean_abs=0` e `mean_relative=0` per tutti i casi.

La parity esterna pinned non è stata dichiarata superata: il `llama-server`
installato è alla revisione `9bebfcb4b`, mentre il repository richiede
`13f2b28b`; lo script ha correttamente riportato revisione non supportata.

Il decode non era nel grafo modificato. Nel candidato ripetuto misurava 31,81,
19,75, 13,10 e 8,88 tok/s rispettivamente a 2K/8K/16K/28K, con CV massimo 1,23%;
la baseline era 31,96, 19,65, 13,01 e 8,87 tok/s. Non emerge una regressione.

### Decisione

Il risultato è una falsificazione forte sul target RTX 3060:

- split-KV reale, online-softmax stabile e merge separato implementati;
- tre configurazioni significative split 2/4/8 misurate;
- workgroup 128/256 e GQA reuse split 4/8/16 esplorati;
- costi di merge, scratch, occupancy indiretta e speedup locale attribuiti;
- migliore risultato +5,02% tok/s e +4,77% wall a 28K, sotto il 10%;
- nessuna patch shader/runtime mantenuta; il codice finale coincide con la
  baseline e questo report è l'unica modifica.

Il bottleneck residuo maggiore è la scansione causale ripetuta di K/V: lo split
aumenta il parallelismo ma non riduce i byte della history. La prossima strada
strutturale a più alto impatto è un tile FlashAttention-style di più query che
carichi K/V una volta e controlli esplicitamente accumulatori/registri per non
ripetere il collasso di residenza osservato nel prototipo GQA. Richiede una
diversa mappatura subgroup/query, non un ulteriore valore di split.

### Verifica finale del rollback

`cargo fmt --all -- --check` e il test focalizzato CPU/Vulkan sono passati. La
suite Vulkan ha completato 139 unit test, con due test autenticati ignorati. Le
suite residue, escludendo il solo contratto documentale non eseguibile, hanno
completato 4 test family-agnostic e 12 test semantici, con un test autenticato
ignorato in ciascuna suite. `docs_contract` non è eseguibile sul commit di
partenza perché `VALIDATION.md`, che il test apre obbligatoriamente, non è
presente né nel worktree né nel tree Git di `43f2da5`; non è una regressione del
prototipo. L'audit Git finale non mostra differenze fuori da questo documento.

## Follow-up multi-query tiled — modello iniziale, RTX 3060, 2026-08-10

Questa sezione registra il modello quantitativo costruito prima della prima
modifica al kernel. Il codice di produzione è ancora quello di `43f2da5`; la
baseline misurata nella sezione split-KV è quindi riutilizzabile senza ripetere
il profiling generale già concluso. Il nuovo branch pulito è
`perf/vulkan-prefill-multi-query-tiled`.

### Invarianti, modifica minima e rischio

- invariante: ogni output `(layer, token, query head)` conserva causalità, GQA e
  online softmax stabile senza materializzare una matrice N×N;
- modifica minima: un kernel F16 prefill dedicato e il solo dispatch necessario
  a selezionarlo; decode e attention INT8 restano percorsi distinti invariati;
- rischio principale: il risparmio di load globali viene perso riducendo i warp
  che partizionano la history o aumentando registri, shared memory e barrier.

Struttura prevista prima del prototipo, con stima delle righe produttive:

```text
crates/graph_horizon_engine/
├── build.rs (~175 righe produttive, orchestrazione shader)
└── src/backend/vulkan/
    ├── kernels/attention/mod.rs (~185 righe produttive, dispatch attention)
    ├── exec/profile/category.rs (~160 righe produttive, classificazione profiler)
    ├── pipeline/mod.rs (~185 righe produttive, registry e capability gate)
    ├── pipeline/kernel.rs (~125 righe produttive, ABI pipeline)
    └── shaders/attention/
        ├── attention_prefill.comp (115 righe produttive, baseline categoria K)
        └── attention_prefill_tiled.comp (~190 righe produttive, categoria K)
```

Il nuovo shader è una categoria K: una sola operazione numerica attention, senza
I/O host, ownership di risorse o dispatch. Le quattro modifiche Rust restano file di
orchestrazione sotto 200 righe produttive; se il prototipo richiede una seconda
architettura shader, questa entra nello stesso dominio `shaders/attention/` e
non modifica la struttura dell'orchestrazione.

### Modello dei byte della baseline

La forma reale è 26 layer, 32 query head, 8 KV head, head dimension 128 e K/V
F16. Per una query logica `(layer, token, query head)`, ogni posizione causale
legge 256 byte K e 256 byte V. Per N token:

```text
coppie causali              = N × (N + 1) / 2
byte K+V                    = 26 × 32 × 512 × coppie causali
byte Q+output               = 26 × 32 × 512 × N
FLOP QK+AV                  ≈ byte K+V
intensità aritmetica K/V    ≈ 1 FLOP/byte
```

`Byte/query` indica una query logica di un singolo layer/head; `byte/token`
include tutti i 26 layer e i 32 query head. I byte sono logici: le cache possono
servire load ripetuti, quindi la banda effettiva non è un contatore DRAM.

| Prompt | K letti | V letti | K+V | Byte/query | Byte/token | Byte attention, Q/O inclusi | Attention baseline | Banda effettiva |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0,446895 TB | 0,446895 TB | 0,893789 TB | 0,525 MB | 0,436 GB | 0,894662 TB | 3.105,04 ms | 287,85 GB/s |
| 8K | 7,147698 TB | 7,147698 TB | 14,295396 TB | 2,097 MB | 1,745 GB | 14,298886 TB | 51.206,35 ms | 279,17 GB/s |
| 16K | 28,589047 TB | 28,589047 TB | 57,178094 TB | 4,195 MB | 3,490 GB | 57,185074 TB | 209.462,67 ms | 272,98 GB/s |
| 28K | 83,495846 TB | 83,495846 TB | 166,991692 TB | 7,168 MB | 5,964 GB | 167,003619 TB | 627.180,43 ms | 266,26 GB/s |

### Minimo plausibile con riuso temporale

Il modello raggruppa query adiacenti e carica fino alla posizione causale
dell'ultima query del tile. Include quindi il piccolo over-read mascherato della
diagonale, ma assume che ogni K/V tile venga caricato globalmente una sola volta
per query head. Non include ancora riuso tra query head GQA: quello è una
dimensione separata, con massimo teorico ulteriore 4× per questo modello.

| Prompt | Q_TILE=1 | Q_TILE=2 | Riduzione | Q_TILE=4 | Riduzione | Q_TILE=8 | Riduzione |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0,893789 TB | 0,447113 TB | 49,976% | 0,223775 TB | 74,963% | 0,112105 TB | 87,457% |
| 8K | 14,295396 TB | 7,148570 TB | 49,994% | 3,575158 TB | 74,991% | 1,788451 TB | 87,489% |
| 16K | 57,178094 TB | 28,590792 TB | 49,997% | 14,297141 TB | 74,995% | 7,150315 TB | 87,495% |
| 28K | 166,991692 TB | 83,498828 TB | 49,998% | 41,752396 TB | 74,997% | 20,879180 TB | 87,497% |

L'intensità K/V teorica cresce rispettivamente a circa 1×, 2×, 4× e 8×
FLOP/byte. Il primo sweep non sarà combinatorio: `Q_TILE=2/4`, `KV_TILE=32/64`
e workgroup da 256/512 thread, scartando `KV_TILE=128` finché il costo di almeno
32 KiB di shared tile non sia giustificato. Il controllo `Q_TILE=1` distingue il
costo del tiling dal beneficio del reuse; `Q_TILE=8` viene valutato soltanto se
registri/shared e residenza dei candidati piccoli lo rendono plausibile.

### Budget delle risorse e strategie

La RTX 3060 espone subgroup 32, massimo 1.024 invocation/workgroup e 49.152 byte
shared/workgroup. La baseline wide dichiara 512 thread e circa 16.640 byte shared;
ogni invocation dichiara fino a 8 componenti Q e 8 accumulatori FP32, anche se
con subgroup 32 ne usa quattro per head dimension 128. I registri compilati e gli
spill non sono esposti da Vulkan; saranno riportati da strumenti shader/vendor
se disponibili, altrimenti tramite conteggio comparativo dello stato privato,
shared dichiarata e residenza inferita da configurazioni isolate.

Le architetture da separare sperimentalmente sono:

1. K e V residenti insieme in shared, più subgroup per query che partizionano la
   history e fondono stati online-softmax parziali;
2. tile shared riutilizzata in due fasi K poi V, score di tile in shared e output
   dimension distribuita tra subgroup, per ridurre shared e stato privato;
3. solo dopo un candidato temporale efficiente, riuso GQA limitato combinato con
   un query tile piccolo, mantenendo costante il numero totale di query state.

Il gate del prototipo resta triplo: riduzione reale/proxy dei byte, tempo
attention inferiore e costo di risorse/residenza quantificato.

### MQ-A1 — K/V simultanei in shared, subgroup history-partitioned

Prima architettura: `Q_TILE=2`, `KV_TILE=32`, 512 thread, subgroup 32 e otto
subgroup per query. K e V F16 occupano insieme 16.384 byte shared; stati parziali
e accumulatori portano il dichiarato totale a 33.024 byte/workgroup, contro
16.640 byte della baseline wide. Ogni invocation conserva le stesse due array
private FP32 da otto elementi della baseline (`Q` e output accumulator), quindi
non duplica lo stato per query nell'invocation.

Il modello del codice prova una sola lettura globale K e V per tile/query-pair:
il traffico stimato 8K scende da 14,295396 a 7,148570 TB, −49,994%, e
l'intensità teorica sale da 1 a 2 FLOP/byte. Non sono disponibili contatori DRAM,
cache o occupancy: Nsight Systems dichiara la RTX 3060 non supportata per GPU
metrics e Nsight Compute non profila Vulkan. La proxy compilata SPIR-V passa da
124 a 151 istruzioni statiche (+21,8%), conserva due sole variabili-array private
e passa da una barrier statica a tre; le prime due barrier vengono eseguite per
ogni tile da 32 chiavi. Spill e registri ISA restano non misurabili.

Misura 8K, un run senza warm-up dedicata perché il candidato è regressivo:

| ID | Architettura | Q_TILE | KV_TILE | WG/subgroup | GQA reuse | Shared/WG | Registri/proxy | Occupancy/proxy | K/V stimati | Banda su byte stimati | Attention | Speedup attention | Tok/s | Wall | Decisione |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| MQ-A1 | K+V shared, 8 subgroup history/query | 2 | 32 | 512/32 | 1 | 33.024 B | 2 array private × 8 FP32; 151 istruzioni SPIR-V | shared 1,98×; warp/WG invariati | 7,148570 TB | 123,14 GB/s | 58.051,43 ms | 0,882× | 52,72 | 155.392,95 ms | reject |

La baseline comparabile è 51.206,35 ms attention, 54,93 tok/s e 149.138,63 ms
wall. MQ-A1 peggiora attention dell'11,79% e tok/s del 4,02%; il modello di
Amdahl predice 0,956× end-to-end e il wall misura 0,960×. Il tempo GPU non
attention resta 96,53 s contro 97,07 s baseline, quindi la regressione è locale
al kernel. Il dimezzamento dei byte è reale per costruzione, ma 33 KiB shared,
barrier ogni 32 chiavi e minore parallelismo history per query riducono la banda
utile abbastanza da perdere il beneficio. Il prossimo candidato separa le fasi
K e V, dimezza la tile shared, materializza soltanto 32 score/query in shared e
distribuisce le dimensioni output tra subgroup.

### MQ-B1 — tile phased K→V, accumulatore distribuito

Seconda architettura, a parità di `Q_TILE=2`, `KV_TILE=32` e WG 512. La tile K
produce 32 score/query in shared, viene sovrascritta dalla tile V e quattro
subgroup/query possiedono dimensioni output disgiunte. Lo shader non fonde più
stati history parziali: mantiene `(max, sum, alpha)` per query in shared e un solo
accumulatore FP32 nelle invocation che possiedono output.

| ID | Architettura | Q_TILE | KV_TILE | WG/subgroup | GQA reuse | Shared/WG | Registri/proxy | Occupancy/proxy | K/V stimati | Banda su byte stimati | Attention | Speedup attention | Tok/s | Wall | Decisione |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| MQ-B1 | phased K→V, output distribuito | 2 | 32 | 512/32 | 1 | 8.984 B | nessuna array Function SPIR-V; 1 accumulatore FP32 | shared −73% vs A; warp/WG invariati | 7,148570 TB | 154,71 GB/s | 46.204,90 ms | 1,108× | 57,20 | 143.213,30 ms | keep per sweep |

MQ-B1 riduce attention del 9,77% rispetto alla baseline e del 20,41% rispetto ad
MQ-A1. Il tok/s migliora del 4,13% e il wall del 4,14%; Amdahl predice 1,035× e
il wall misura 1,041×. La proxy SPIR-V ha 161 istruzioni statiche, cinque
subgroup operation, nessuna variabile Function e cinque barrier statiche: una
iniziale e quattro eseguite per tile. Il minor live state/shared vince quindi sul
costo di sincronizzazione, ma il risultato 8K non supera ancora lo split-KV.
`Q_TILE=4` è ora plausibile: mantiene 512 thread, assegna quattro subgroup sia ai
key sia alle 128 dimensioni di ogni query e porta il traffico previsto a −75%.

### MQ-B2 — quattro query, tile phased K→V

`Q_TILE=4`, `KV_TILE=32`, WG 512: quattro subgroup per query partizionano i key
e coprono esattamente le 128 dimensioni output su subgroup 32. Il totale shared
sale soltanto a 9.776 byte e la proxy SPIR-V resta identica a B1: 161 istruzioni,
cinque subgroup operation, nessuna variabile Function e quattro barrier/tile.

| ID | Architettura | Q_TILE | KV_TILE | WG/subgroup | GQA reuse | Shared/WG | Registri/proxy | Occupancy/proxy | K/V stimati | Banda su byte stimati | Attention | Speedup attention | Tok/s | Wall | Decisione |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| MQ-B2 | phased K→V, output distribuito | 4 | 32 | 512/32 | 1 | 9.776 B | nessuna array Function; 1 accumulatore FP32 | shared −41% vs baseline; 16 warp/WG | 3,575158 TB | 91,81 GB/s | 38.940,93 ms | 1,315× | 60,31 | 135.821,35 ms | keep per sweep |

Rispetto alla baseline 8K, MQ-B2 riduce attention del 23,95%, aumenta tok/s del
9,79% e riduce wall dell'8,93%. Amdahl predice 1,090× end-to-end e il wall misura
1,098×. La banda calcolata sui byte globali stimati scende perché il kernel
esegue lo stesso lavoro aritmetico su un quarto dei byte; rapportata al lavoro
logico originale è 367,10 GB/s. Il risultato prova insieme riduzione effettiva
per costruzione, maggiore intensità (4 FLOP/byte) e beneficio attention. Prima
del benchmark long-context viene provato `KV_TILE=64`: stesso traffico asintotico,
metà barrier, ma 18.480 byte shared e loop score/V doppi.

### MQ-B3 — tile phased K→V da 64

`Q_TILE=4`, `KV_TILE=64`, WG 512. La trasformazione softmax assegna score
strided alle 32 lane, mantenendo una sola riduzione subgroup anche quando la tile
supera la subgroup width. La proxy SPIR-V è 158 istruzioni, cinque subgroup
operation, nessuna variabile Function e quattro barrier/tile.

| ID | Architettura | Q_TILE | KV_TILE | WG/subgroup | GQA reuse | Shared/WG | Registri/proxy | Occupancy/proxy | K/V stimati | Banda su byte stimati | Attention | Speedup attention | Tok/s | Wall | Decisione |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| MQ-B3 | phased K→V, output distribuito | 4 | 64 | 512/32 | 1 | 18.480 B | nessuna array Function; 1 accumulatore FP32 | shared 1,11× baseline; 16 warp/WG | 3,575158 TB | 96,07 GB/s | 37.214,79 ms | 1,376× | 61,09 | 134.093,80 ms | keep per sweep |

MQ-B3 migliora ulteriormente B2: −4,43% attention locale. Rispetto alla baseline
porta +11,21% tok/s, −10,09% wall e 1,376× attention; Amdahl predice 1,104× e il
wall misura 1,112×. Il target minimo è quindi già superato a 8K in una singola
misura. `KV_TILE=128` richiede 35.888 byte shared, ancora sotto il limite 49.152,
e dimezza nuovamente la densità di barrier; viene provato prima dei run ripetuti
per verificare il crossover tra sincronizzazione e residenza/shared footprint.

### MQ-B4 — tile phased K→V da 128

| ID | Architettura | Q_TILE | KV_TILE | WG/subgroup | GQA reuse | Shared/WG | Registri/proxy | Occupancy/proxy | K/V stimati | Banda su byte stimati | Attention | Speedup attention | Tok/s | Wall | Decisione |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| MQ-B4 | phased K→V, output distribuito | 4 | 128 | 512/32 | 1 | 35.888 B | nessuna array Function; 1 accumulatore FP32 | shared 2,16× baseline; 16 warp/WG | 3,575158 TB | 91,26 GB/s | 39.176,15 ms | 1,307× | 60,14 | 136.226,68 ms | reject |

MQ-B4 resta migliore della baseline, ma perde il 5,27% attention rispetto a B3
e scende a +9,48% tok/s. La proxy SPIR-V è invariata a 158 istruzioni e lo stato
privato non cresce: la differenza controllata è la tile shared raddoppiata e la
densità di barrier dimezzata. Il crossover attribuisce quindi la regressione al
costo di residenza/shared/cache della tile da 35 KiB. `KV_TILE=64` viene
ristabilita come configurazione migliore senza riscrivere la storia Git.

### MQ-C1 — otto query, due segmenti output per invocation

Terza strategia di ownership: `Q_TILE=8`, `KV_TILE=64`, WG 512. Due subgroup
partizionano i key per query; ogni lane mantiene due accumulatori FP32 per
coprire due segmenti da 64 dimensioni complessive. La shared è 20.576 byte. La
proxy SPIR-V sale a 178 istruzioni e una array Function da due FP32, senza
evidenza di spill misurabile; le barrier restano quattro/tile.

| ID | Architettura | Q_TILE | KV_TILE | WG/subgroup | GQA reuse | Shared/WG | Registri/proxy | Occupancy/proxy | K/V stimati | Banda su byte stimati | Attention | Speedup attention | Tok/s | Wall | Decisione |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| MQ-C1 | phased K→V, 2 segmenti/lane | 8 | 64 | 512/32 | 1 | 20.576 B | array privata 2 FP32; 178 istruzioni | shared 1,24× baseline; 16 warp/WG | 1,788451 TB | 54,16 GB/s | 33.019,47 ms | 1,551× | 63,12 | 129.781,51 ms | keep per sweep |

MQ-C1 porta +14,91% tok/s, −12,98% wall e 1,551× attention a 8K. Amdahl
predice 1,140× e il wall misura 1,149×. Rapportata al lavoro logico originale,
la banda effettiva è 432,94 GB/s; sui byte globali stimati è 54,16 GB/s perché
l'intensità sale a circa 8 FLOP/byte. Il beneficio aggiuntivo rispetto a B3 prova
che due accumulatori/lane non causano ancora collasso di risorse. Il successivo
esperimento mantiene otto query state ma sostituisce metà del reuse temporale
con reuse GQA (`4 posizioni × 2 query head`) per isolare la seconda dimensione.

### MQ-D1 — quattro posizioni × due query head GQA

Stesso `KV_TILE=64`, WG 512, otto query state, due accumulatori/lane, 20.576
byte shared e riduzione teorica 8× di C1. Il grid passa da `(32 head, 4 tile)` a
`(16 coppie head, 8 tile)`; il percorso è valido soltanto per gruppi GQA pari.

| ID | Architettura | Q_TILE | KV_TILE | WG/subgroup | GQA reuse | Shared/WG | Registri/proxy | Occupancy/proxy | K/V stimati | Banda su byte stimati | Attention | Speedup attention | Tok/s | Wall | Decisione |
|---|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| MQ-D1 | phased K→V, temporal × GQA | 4 | 64 | 512/32 | 2 | 20.576 B | array privata 2 FP32; 184 istruzioni | identica a C1 | 1,788451 TB | 53,86 GB/s | 33.206,81 ms | 1,542× | 63,00 | 130.036,96 ms | reject |

MQ-D1 è 0,57% più lento di C1 in attention e 0,19% più lento in tok/s: entro la
scala del rumore di un singolo run, ma senza alcun segnale positivo che compensi
mapping e capability gate aggiuntivi. A parità di byte logici e risorse, il
reuse temporale puro è più generale e marginalmente migliore. Questo spiega
anche il precedente semplice GQA packing: sostituire reuse temporale con reuse
tra head non riduce ulteriormente i byte e la cache può già servire i quattro
query head contigui associati allo stesso KV head. C1 viene ripristinato.

### Controllo 2K e correttezza estesa di MQ-C1

Il controllo singolo 2K produce 78,26 tok/s e 26.168,18 ms wall contro baseline
74,61 tok/s e 27.447,79 ms: +4,89% tok/s e −4,66% wall. Attention scende da
3.105,04 a 2.041,64 ms, speedup 1,521×. Il traffico stimato è 0,112105 TB,
−87,46%; banda sui byte stimati 54,91 GB/s e sul lavoro logico 437,78 GB/s.
Amdahl predice 1,041× end-to-end, il wall misura 1,049×. Non emerge quindi una
regressione a 2K e non serve una threshold long-context su questa GPU.

Il test focalizzato ora usa head dimension 128 e GQA 2:1, e per F16/INT8 copre:

- base/N `0/3`, `33/32`, `65/9`;
- tile K/V completa e diagonale, query tile completa e incompleta;
- base e lunghezze non multiple di 64/8;
- output attention e logits sintetici proiettati;
- F16 contro reference CPU indipendente e prefill contro decode sequenziale;
- INT8 contro decode sequenziale sullo stesso KV quantizzato.

Tutti i confronti prefill/decode F16 e INT8 hanno `max_abs=mean_abs=mean_relative=0`.
Contro CPU F16, il massimo `max_abs` attention è `2,94e-5`, il massimo
`mean_abs` è `5,52e-6` e il massimo `mean_relative` è `1,66e-4`. Per i logits il
massimo `max_abs` è `3,37e-6`, il massimo `mean_abs` è `6,22e-7`; il massimo
`mean_relative` è `7,66e-3` per logits prossimi allo zero.

## Risultato finale multi-query tiled — 2026-08-11

La configurazione mantenuta è MQ-C1: prefill F16 dedicato, `Q_TILE=8`,
`KV_TILE=64`, WG 512, subgroup 32 sulla RTX 3060, nessun reuse GQA esplicito e
due segmenti output FP32 per invocation. Il dispatch mantiene il fallback
esistente quando il device non supporta 512 invocation, 20.576 byte shared o un
subgroup 16/32/64. Decode e attention INT8 non sono modificati.

### Benchmark ripetuto

Stesso protocollo della baseline split-KV: prompt esatti, contesto 32.768, KV
F16, due token, una warm-up e tre ripetizioni pubbliche. I timestamp profiler
aggregano quattro run e sono divisi per quattro nelle tabelle.

| Prompt | Baseline tok/s | MQ-C1 tok/s | Delta tok/s | Baseline wall | MQ-C1 wall | Delta wall | CV MQ-C1 | Baseline attention | MQ-C1 attention | Speedup attention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 74,61 | 77,69 | +4,13% | 27.447,79 ms | 26.362,16 ms | −3,96% | 0,14% | 3.105,04 ms | 2.052,82 ms | 1,513× |
| 8K | 54,93 | 62,67 | +14,09% | 149.138,63 ms | 130.714,95 ms | −12,35% | 0,03% | 51.206,35 ms | 33.166,26 ms | 1,544× |
| 16K | 40,24 | 49,51 | +23,04% | 407.152,88 ms | 330.955,33 ms | −18,72% | 0,02% | 209.462,67 ms | 134.651,88 ms | 1,556× |
| 28K | 28,89 | 38,30 | +32,57% | 969.170,17 ms | 731.075,26 ms | −24,57% | 0,05% | 627.180,43 ms | 395.863,27 ms | 1,584× |

GPU prefill medio del candidato: 26.085,42 / 129.836,14 / 328.949,96 /
728.987,36 ms a 2K/8K/16K/28K. La quota attention passa da 11,43/34,53/51,64/
64,88% baseline a 7,87/25,54/40,93/54,30%.

### Traffico, banda e intensità

Il caricamento cooperativo indicizza ogni elemento K e V una sola volta per tile
di otto query; i subgroup consumano soltanto `sh_tile`. Il byte model include
l'over-read causale dell'ultima query del tile e coincide quindi con i load
globali emessi per costruzione. Sono stime verificate dal layout/shader, non
contatori DRAM. Nsight Systems non supporta GPU metrics su questa RTX 3060 e
Nsight Compute non profila Vulkan; hit rate L1/L2 e byte fisici restano
esplicitamente non misurabili.

| Prompt | K+V baseline | K+V MQ-C1 stimati | Riduzione | Banda baseline su byte logici | Banda MQ-C1 su byte stimati | Banda su lavoro logico originale | Intensità MQ-C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 0,893789 TB | 0,112105 TB | 87,457% | 287,85 GB/s | 54,61 GB/s | 435,40 GB/s | 7,973 FLOP/B |
| 8K | 14,295396 TB | 1,788451 TB | 87,489% | 279,17 GB/s | 53,92 GB/s | 431,02 GB/s | 7,993 FLOP/B |
| 16K | 57,178094 TB | 7,150315 TB | 87,495% | 272,98 GB/s | 53,10 GB/s | 424,64 GB/s | 7,997 FLOP/B |
| 28K | 166,991692 TB | 20,879180 TB | 87,497% | 266,26 GB/s | 52,74 GB/s | 421,84 GB/s | 7,998 FLOP/B |

A 28K K e V sono 10,439590 TB ciascuno nel candidato, contro 83,495846 TB
ciascuno baseline. L'intensità cresce da circa 1 a circa 8 FLOP/byte. La banda
calcolata sui byte nuovi diminuisce perché QK/AV, exp, shared traffic e barrier
restano lavoro reale; la banda rapportata al lavoro logico mostra invece che il
kernel mantiene oltre 421 GB/s equivalenti mentre evita sette scansioni su otto.

### Risorse e occupancy

| Risorsa/proxy | Baseline wide | MQ-C1 |
|---|---:|---:|
| Workgroup / subgroup target | 512 / 32 | 512 / 32 |
| Warp per workgroup RTX 3060 | 16 | 16 |
| Shared dichiarata/workgroup | 16.640 B | 20.576 B |
| Stato privato SPIR-V | Q[8] + acc[8] FP32 | acc[2] FP32; Q in shared |
| Istruzioni SPIR-V statiche | 124 | 178 |
| Barrier | 1 finale | 1 iniziale + 4/tile |
| Registri ISA / invocation | non misurabile | non misurabile |
| Spill / local memory | non misurabile | non misurabile |
| Workgroup/warp residenti per SM | non misurabile | non misurabile |

Non viene dichiarata una occupancy hardware inventata. L'evidenza comparativa è:
stesso WG/warp count, shared +23,7%, array privata ridotta da sedici a due FP32,
nessuna ulteriore variabile Function nel disassemblato oltre `acc[2]`, e sweep
controllato. A parità di architettura, 35.888 byte shared di KV_TILE 128 peggiorano
attention del 5,27%; 20.576 byte con due accumulatori migliorano invece B3.

### Amdahl e decode

| Prompt | Quota attention baseline | Speedup attention | Speedup E2E previsto | Speedup wall misurato |
|---:|---:|---:|---:|---:|
| 2K | 11,43% | 1,513× | 1,040× | 1,041× |
| 8K | 34,53% | 1,544× | 1,138× | 1,141× |
| 16K | 51,64% | 1,556× | 1,226× | 1,230× |
| 28K | 64,88% | 1,584× | 1,315× | 1,326× |

Decode baseline/candidato in tok/s: 31,96/31,86 (−0,31%) a 2K,
19,65/19,83 (+0,92%) a 8K, 13,01/13,16 (+1,15%) a 16K e 8,87/8,91
(+0,45%) a 28K. Non emerge regressione; il relativo shader/dispatch è invariato.

### Decisione e report richiesto

```text
Root cause:
repeated bandwidth/cache-limited K/V scans

Baseline 28K:
wall: 969.170,17 ms
tok/s: 28,89
attention time: 627.180,43 ms
attention %: 64,88%

Original K/V traffic:
K 83,495846 TB + V 83,495846 TB = 166,991692 TB logici

Best multi-query candidate:
Q_TILE: 8
KV_TILE: 64
subgroup mapping: 2 subgroup/query; key partition + 2 output segment/lane
register strategy: Q shared, 2 accumulatori FP32 privati/lane
shared memory: 20.576 B/WG
GQA reuse: 1 (solo reuse temporale esplicito)

Candidate K/V traffic:
K 10,439590 TB + V 10,439590 TB = 20,879180 TB stimati

Traffic reduction:
87,497%

Effective bandwidth:
baseline: 266,26 GB/s sui byte logici
candidate: 52,74 GB/s sui byte stimati; 421,84 GB/s sul lavoro logico

Register/occupancy:
baseline: Q[8]+acc[8], 16.640 B shared, 16 warp/WG
candidate: acc[2], 20.576 B shared, 16 warp/WG
registri ISA, spill e resident warp hardware non misurabili

Attention speedup:
1,584× a 28K

Prefill speedup:
+32,57% tok/s; wall −24,57%; 1,326× wall speedup a 28K

2K: 77,69 tok/s, +4,13%, CV 0,14%
8K: 62,67 tok/s, +14,09%, CV 0,03%
16K: 49,51 tok/s, +23,04%, CV 0,02%
28K: 38,30 tok/s, +32,57%, CV 0,05%

Decode regression:
nessuna; delta da −0,31% a +1,15%

Correctness:
F16 CPU max_abs attention 2,94e-5; logits 3,37e-6;
prefill/decode F16 e INT8 max/mean/relative = 0 sui casi focalizzati

Strategies rejected:
MQ-A1 K+V simultanei shared: byte −50% ma attention 0,882×;
MQ-B4 KV_TILE 128: −5,27% attention vs KV_TILE 64;
MQ-D1 temporal×GQA: +0,57% attention time vs reuse temporale puro

Remaining bottleneck:
attention ancora 54,30% del GPU prefill; nuovo limite combinato di QK/AV,
shared traffic, exp e quattro barrier per tile, non più otto scansioni globali

Next architectural candidate:
solo per un target ulteriore: persistent/larger query block che riusi K/V tra
più tile da otto senza mantenere simultaneamente tutti gli accumulatori
```

Il percorso è mantenuto: supera il precedente split-KV (+5,02%) e il target
minimo +10% a 28K con complessità locale, nessuna nuova allocazione, dipendenza,
API pubblica o modifica decode. La complessità concettuale aumenta di un kernel
prefill categoria K e di un capability-gated dispatch; è proporzionata alla
riduzione misurata e resta separata dall'attention INT8/decode.

### Verifica finale

- `cargo fmt --all -- --check`: pass;
- Clippy `--all-targets -D warnings`, feature `vulkan-profile`: pass;
- suite library Vulkan: 140 pass, 2 test autenticati ignorati;
- suite library `vulkan-profile`: 144 pass, 2 test autenticati ignorati;
- suite semantica: 12 pass, 1 test autenticato ignorato;
- family-agnostic: 4 test eseguibili pass, 1 autenticato ignorato;
- build release del benchmark Vulkan: pass;
- test CPU/Vulkan focalizzato incluso nelle due suite: pass per tutti i casi
  F16/INT8 e boundary descritti sopra;
- capability gate subgroup 16/32/64: testato; subgroup 8 rifiutato e fallback al
  kernel esistente;
- formattazione e `git diff --check`: pass.

Il solo `docs_contract` family-agnostic resta non eseguibile: apre
obbligatoriamente `VALIDATION.md`, assente sia dal worktree sia dal tree Git del
commit baseline `43f2da5`. È lo stesso limite documentato prima dell'esperimento
e non è causato dal kernel. Lo stato finale conserva MQ-C1 e il report; tutti i
candidati rigettati restano soltanto nella storia Git e nelle tabelle.

## Follow-up MQ-C1 — attribuzione e latenza AV, 2026-08-11

La nuova baseline richiesta è `9c1527f`. Prima di modificare il kernel è stato
ripetuto un run pulito 8K, stessa tupla RTX 3060/F16/contesto 32.768 e un solo
campione senza warm-up: 62,71 tok/s, 130.624,45 ms wall e 33.132,90 ms attention.
Il timestamp attention differisce dello 0,10% dai 33.166,26 ms ripetuti di MQ-C1,
quindi il profilo precedente è riproducibile. Il profiler attribuisce il 99,92%
del GPU prefill; Nsight Systems conferma che questa RTX 3060 non espone GPU
metrics e Nsight Compute non accetta il percorso Vulkan.

### Risorse hardware e metodo causale

Una query temporanea `VK_KHR_pipeline_executable_properties`, rimossa dallo
stato finale, ha restituito per MQ-C1: subgroup 32, 38 registri/thread, stack
0 B e 20.576 B shared/workgroup. La GPU espone 65.536 registri, 102.400 B shared
e 1.536 thread per SM. Tre workgroup da 512 richiedono 58.368 registri prima
dell'arrotondamento di allocazione e 61.728 B shared: la baseline raggiunge
quindi 48 warp/SM, il 100% del limite thread, senza spill. Il candidato finale
usa 39 registri, stack 0 e la stessa shared, restando alla stessa residenza.

Le ablation usano 2K per contenere il costo, con lo stesso shader, griglia,
traffico globale e numero di dispatch salvo la singola variabile dichiarata.
Il timestamp baseline fresco è 2.049,345 ms. Sono diagnostiche non numericamente
corrette e sono state rimosse; in particolare l'hot-set non è un nuovo candidato
di riuso K/V.

| Ablation | Attention | Delta | Interpretazione |
|---|---:|---:|---|
| baseline `KV_TILE=64` | 2.049,345 ms | — | 38 registri, 20.576 B shared, 3 WG/SM |
| K/V hot-set, istruzioni invariate | 2.029,116 ms | −0,99% | upper bound global/cache stall |
| `KV_TILE=32`, occupancy invariata | 2.151,849 ms | +5,00% | densità barrier circa doppia |
| `KV_TILE=128`, 2 WG/SM | 2.153,376 ms | +5,08% | costo della residenza 66,7% nonostante metà barrier |
| un prodotto QK/lane su quattro | 1.484,418 ms | −27,57% | QK completo estrapolato 36,75% |
| una key AV per tile su 64 | 1.502,271 ms | −26,70% | AV completo estrapolato 27,12% |
| softmax lineare dipendente dagli score | 2.038,738 ms | −0,52% | upper bound delle `exp` |

A 8K il kernel richiede 14,2954 TFLOP QK+AV e, contando in modo conservativo
ogni lettura score per lane prima del broadcast shared, 35,7385 TB shared. Le
bande osservate sono soltanto 0,431 TFLOP/s e 1,079 TB/s: 3,4% del picco FP32
e 16,7% del picco shared teorico di circa 6,45 TB/s. Gli accessi Q/K/V sono
contigui per lane e gli score sono broadcast; non emerge un limite di bandwidth
shared. Il traffico K/V stimato è 1,788 TB, 54 GB/s contro 360 GB/s globali
teorici, coerente con l'hot-set quasi piatto.

L'attribuzione sottrattiva assegna il 100% del tempo 2K, superando il gate 95%:

| Limite | Quota GPU | Evidenza |
|---|---:|---|
| compute/issue QK + AV + SFU | 64,39% | ablation estrapolate 36,75% + 27,12% + 0,52% |
| sincronizzazione/serializzazione interna | 34,62% | barrier 5,00% + residuo phased/subgroup 29,62% |
| global bandwidth/cache | 0,99% | hot-set a istruzioni e occupancy costanti |
| shared-memory bandwidth | 0% come limite attivo | richiesta conservativa 16,7% del picco, accessi senza conflitti |
| register pressure/occupancy | 0% come limite attivo | 38 registri, stack 0, 3 WG/SM e 48 warp/SM |

Il residuo interno è assegnato per esclusione dopo le ablation: comprende
riduzioni subgroup, subgroup inattivi durante softmax, controllo/indirizzamento
tile e attese fra le fasi K, softmax, V e AV. Non è presentato come contatore di
stall hardware. Lo sweep 32/64 ne misura direttamente 5 punti percentuali come
barrier; il resto è serializzazione phased non separabile con i contatori
disponibili. Il maggiore upper bound isolato è QK, 36,75%.

### Candidati e risultato mantenuto

Il primo candidato ha diviso il dot QK in due catene pari/dispari. Il driver ha
mantenuto 38 registri e la stessa binary size; a 2K ha ottenuto soltanto −0,89%
attention. È stato rimosso: il compilatore schedulava già il dot corto.

Il candidato mantenuto `b268e8b` applica lo stesso principio al vero tratto
limitante seriale: per ogni dimensione AV accumula key pari e dispari in due
somme indipendenti da 32 FMA, poi le fonde una volta nello stato online. Layout,
byte K/V, shared, barrier, griglia e riuso temporale Q_TILE=8 restano invariati;
non viene introdotto riuso K/V tra head o tile. Il driver misura 39 registri,
stack 0 e 20.576 B shared, quindi la residenza resta 3 WG/SM.

| Prompt | Baseline attention | Candidato attention | Delta | Baseline wall | Candidato wall | Delta wall | Tok/s baseline/candidato |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 2.049,345 ms | 1.921,570 ms | −6,23% | 26.184,54 ms | 26.033,82 ms | −0,58% | 78,21 / 78,67 |
| 8K | 33.132,900 ms | 31.158,092 ms | −5,96% | 130.624,45 ms | 128.202,56 ms | −1,85% | 62,71 / 63,90 |
| 28K | 395.863,270 ms | 373.977,473 ms | −5,53% | 731.075,26 ms | 709.476,42 ms | −2,95% | 38,30 / 39,47 |

I punti 2K/8K sono baseline e candidato singoli freschi; la baseline 28K è il
run MQ-C1 ripetuto già registrato sopra e il candidato è un singolo run. Il
beneficio attention è monotono e stabile fra 5,53% e 6,23%. A 28K il prompt
migliora del 3,05%; decode è 8,94 contro 8,91 tok/s baseline (+0,34%), quindi
non emerge regressione. Il test focalizzato conserva identità esatta F16/INT8
contro decode sequenziale; contro CPU F16 restano i massimi baseline:
`2,94e-5` attention e `3,37e-6` logits.

La conferma pubblica 8K completa, una warm-up e tre repliche, produce 63,50
tok/s, CV 0,05%, wall 129.010,83 ms e 125.277,727 ms attention aggregati sui
quattro run, cioè 31.319,432 ms/run. Contro la baseline MQ-C1 ripetuta
`9c1527f` (62,67 tok/s, 130.714,95 ms wall, 33.166,26 ms attention) sono
rispettivamente +1,32%, −1,30% e −5,57%. Decode è 19,72 contro 19,83 tok/s,
−0,55% e ampiamente entro il controllo 5%.

Il collo residuo resta la combinazione compute/issue e serializzazione phased,
non global/shared bandwidth né occupancy. L'ottimizzazione AV usa il maggiore
upper bound rimasto dopo il candidato QK falsificato e riduce la catena critica
senza ampliare architettura, API, allocazioni o dipendenze.

### Verifica finale follow-up

- `cargo fmt --all -- --check`: pass;
- Clippy `--all-targets -D warnings`, feature `vulkan-profile`: pass;
- suite library `vulkan-profile`: 144 pass, 2 test autenticati ignorati;
- suite semantica: 12 pass, 1 test autenticato ignorato;
- family-agnostic: 4 test eseguibili pass, 1 autenticato ignorato;
- test CPU/Vulkan focalizzato: pass, inclusi F16/INT8 e boundary tiled;
- build release del benchmark Vulkan profile: pass;
- `git diff --check`: pass.

`docs_contract` resta escluso per lo stesso `VALIDATION.md` assente dalla
baseline già documentato sopra; non è toccato da questo follow-up.

## Refinement compute/serialization — 2026-08-11

La baseline di questa fase è il commit pulito `2f7527e`; il risultato finale è
`0b76273`. La tupla resta RTX 3060, driver 595.84, modello e SHA-256 dichiarati
in apertura, Vulkan puro, KV F16, contesto 32.768, prompt `" a"` calibrato a N
token e due token richiesti. La baseline fresca usa un campione per punto; il
risultato finale usa una warm-up e tre repliche misurate. I timestamp finali del
profiler aggregano quattro run e nelle tabelle sono divisi per quattro.

### Nuova baseline end-to-end e gate

| Prompt | GPU prefill | Attention | MLP | Proiezioni | Norm + RoPE | Altro | Attention/GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 26.353,698 ms | 1.987,565 ms | 16.114,869 ms | 7.655,844 ms | 257,029 ms | 338,391 ms | 7,54% |
| 8K | 129.135,892 ms | 31.594,332 ms | 64.552,314 ms | 30.604,937 ms | 1.038,517 ms | 1.345,792 ms | 24,47% |
| 16K | 323.619,555 ms | 127.715,823 ms | 129.524,453 ms | 61.589,954 ms | 2.079,264 ms | 2.710,061 ms | 39,46% |
| 28K | 715.051,649 ms | 376.913,813 ms | 223.207,666 ms | 106.728,252 ms | 3.551,964 ms | 4.649,954 ms | 52,71% |

Attention non è il maggiore componente a 2K/8K e a 16K è appena sotto MLP;
resta però il maggiore componente singolo a 28K. Il gate ha quindi autorizzato
solo candidati attention semplici con un upper bound end-to-end credibile. Dopo
le due patch mantenute, l'insieme dei sette matmul batched supera attention anche
a 28K: 50,99% contro 47,74%. MLP è già dominante da 2K a 16K.

### Attribuzione intra-kernel e dependency graph

Il kernel contiene cinque siti SPIR-V `OpControlBarrier`: uno dopo
l'inizializzazione di Q/stato e quattro per ogni KV tile, dopo K staging, QK,
V staging e AV. Tutti e quattro i barrier nel loop sono workgroup-wide: i dati
sono prodotti da invocation diverse da quelle che li consumano e lo stesso tile
shared viene sovrascritto nella fase successiva. Subgroup scope non preserva
queste dipendenze. L'ablation `KV_TILE=32` già acquisita assegna 5,00% della
baseline ai barrier espliciti, pari al 6,13% del kernel finale perché il numero
dei barrier non cambia mentre il kernel si accorcia.

Per un full tile sulla RTX 3060:

| Fase | Operazioni / KV tile | Catena dipendente | Lavoro indipendente / esito |
|---|---|---|---|
| K staging | 16 load F16 per invocation | indirizzo per iterazione nella baseline | dimensione lane invariata; tre divisioni hot complessive eliminate fra Q/K/V |
| QK | 32 key per subgroup, 4 prodotti FP32/lane per dot | 4 add/mul più subgroup reduction e scale | dot di key diverse; lo split pari/dispari QK precedente diede solo −0,89% |
| max | 2 score/lane più `subgroupMax` | 2 max e una reduction | 32 lane e otto query indipendenti |
| exp/sum | 2 score exp/lane più un alpha exp/lane | 2 add per lane, `subgroupAdd`, un update `(m,l)` | otto softmax subgroup indipendenti; due stati online diedero solo −0,56% |
| V staging | 16 load F16 per invocation | come K nella baseline | si sovrappone al softmax, poi barrier workgroup |
| AV baseline | 64 FMA per dimensione, 2 dimensioni/invocation | 2 catene da 32 FMA | due segmenti output indipendenti |
| AV finale | stesso lavoro matematico | 4 catene da 16 FMA con merge bilanciato | quattro catene coprono la latenza FMA senza cambiare traffico |
| output | 2 divisioni e store F16/invocation | normalizzazione finale | costo O(N), sotto la risoluzione rispetto al loop O(N²) |

La stima di lavoro per lo scheduler è una FMA FP32 dipendente ogni circa quattro
cicli e una warp instruction emettibile per ciclo/scheduler: servono quindi
circa quattro catene indipendenti per coprire la latenza. Non è un contatore ISA
del driver, ma predice il risultato dello sweep: una catena era la baseline
precedente a `b268e8b`, due catene hanno dato −5,53–6,23% attention, quattro
catene danno un ulteriore −4,37–4,69% senza spill e con un solo registro in più.

L'interleaving già sicuro è V staging in parallelo alla trasformazione softmax
dei subgroup eletti. AV deve invece attendere pesi e V, mentre QK del tile
successivo richiede di sovrascrivere la stessa `sh_tile`: anticiparlo esige un
secondo fragment shared o ownership differente. Con tutti i barrier limitati al
6,13% locale e shared già esclusa come collo, il suo upper bound non giustifica
il buffering e il controllo aggiuntivi. Per lo stesso motivo non è stata riaperta
la pipeline K/V simultanea già falsificata.

L'attribuzione sottrattiva finale usa lo stesso metodo causale della baseline.
Non è un contatore hardware di stall; il driver non espone tali contatori sul
percorso Vulkan. La quota residua `dependency/control` resta volutamente
combinata perché subgroup reduction, lane inattive, controllo e phase wait non
sono separabili con gli strumenti disponibili.

| Limite finale | Quota attention | Evidenza |
|---|---:|---|
| QK compute/issue | 45,05% | ablation QK baseline, byte e lavoro invariati |
| AV compute/issue | 28,37% | ablation AV corretta per il guadagno AV4 |
| trascendentali | 0,64% | ablation lineare delle `exp`, costo assoluto invariato |
| dependency serialization + control residui | 18,60% | residuo dopo le ablation e la rimozione degli indici costosi |
| sincronizzazione esplicita | 6,13% | sweep barrier-density 32/64 |
| global/cache | 1,21% | hot-set baseline, byte globali invariati |

Il profilo sintetico delle `exp` conta una chiamata per score e un alpha
ridondante per lane del subgroup softmax: 1,515 / 1,504 / 1,502 / 1,501 chiamate
per coppia causale a 2K/8K/16K/28K, rispettivamente 2,645 / 41,988 / 167,729 /
489,601 miliardi di invocation shader. Nonostante il numero elevato, sostituire
tutte le `exp` aveva ridotto attention soltanto dello 0,52%; eliminare solo gli
alpha ridondanti ha quindi un upper bound locale inferiore a 0,2%.

### Instruction-level findings

Shaderc ottimizzato è usato come rappresentazione statica comparabile; la build
runtime continua a lasciare l'ottimizzazione finale al driver. Il driver espone
statistiche eseguibili ma non testo ISA NVIDIA né contatori di stall Vulkan.

| SPIR-V statico ottimizzato | Baseline | Finale |
|---|---:|---:|
| byte binary | 9.436 | 10.116 |
| siti instruction nel body | 457 | 492 |
| `OpUDiv` | 7 | 4 |
| `OpIMul` | 31 | 30 |
| `OpISub` | 8 | 5 |
| shift + mask | 0 | 6 |
| FP add / mul | 7 / 6 | 11 / 8 |
| conversioni F16→FP32 | 5 | 7 |
| subgroup add / max / elect | 2 / 1 / 2 | invariati |
| barrier | 5 | 5 |

Il totale statico cresce per le quattro catene AV e non predice da solo il
tempo: la modifica vincente elimina tre divisioni dinamiche nei loop Q/K/V e
aggiunge ILP FP32. SPIR-V esprime mul/add separati; l'eventuale fusione FMA è una
decisione del driver. Le statistiche driver finali sono subgroup 32, 40
registri/thread, stack 0, 20.576 B shared/workgroup e binary 16.640 B. Contro i
39 registri della baseline mantenuta, tre workgroup da 512 richiedono 61.440
registri e 61.728 B shared: restano 48 warp/SM e 100% occupancy thread.

Q, K e V restano F16 in storage; QK, max, argomento exp, somma softmax, AV e
normalizzazione restano FP32. Ridurli a F16 allungherebbe il percorso numerico e
non attacca il limite misurato. FP16 dot/cooperative matrix sono esposti dal
device, ma interessano soprattutto QK, il cui candidato ILP precedente è stato
falsificato e il cui layout ridotto non giustifica un nuovo percorso hardware.

### Esperimenti e Amdahl a due livelli

| ID | Target | Ipotesi | Delta locale | Delta totale | Esito |
|---|---|---|---:|---:|---|
| IDX128 | address/issue | specializzare head dimension e usare shift/mask | −15,12% 2K; −14,14% 8K attention | −1,65% 2K; −3,57% 8K wall | keep `650579a` |
| AV4 | dependency AV | quattro catene nascondono la latenza FMA | −4,69% 2K; −4,37% 8K incrementale | −0,76% 2K; −1,17% 8K wall | keep `0b76273` |
| ADDR-IND | addressing | cache pointer incrementale | −0,59% attention 2K | −0,28% wall | reject, entro rumore/complessità |
| SOFT2 | online softmax | due stati alternati spezzano la recurrence | −0,56% attention 2K | non attribuibile oltre rumore | reject, +10,7% SPIR-V e barriera finale |
| QK2 precedente | QK dependency | due catene nel dot | −0,89% attention | insignificante | reject |
| BAR-SCOPE | barrier | restringere scope | upper bound 5,00% attention per tutti i barrier | ≤2,64% a 28K se eliminati tutti | reject semantico; producer/consumer workgroup-wide |

Per IDX128, Amdahl kernel prevede circa 14–15% perché rimuove tre divisioni hot;
la misura è 14,14–15,12%. A 8K, `24,47% × 14,14% = 3,46%` GPU prefill,
coerente con −3,57% wall. Per AV4, AV valeva 27,12% attention e il guadagno
incrementale di 4,37% attention equivale a circa 1,19× sulla sola porzione AV;
al livello prefill vale circa 0,9% GPU a 8K e circa 2% al vecchio mix 28K.

Riepilogo delle due catene richieste: AV baseline di fase è due accumulatori da
32 FMA, AV best è quattro da 16 con merge bilanciato; online-softmax baseline è
un solo stato stabile `(m, l, o)` aggiornato una volta per tile. Il best provato
a due stati alternati misura appena −0,56% attention, aggiunge 10,7% di SPIR-V e
una merge/barriera finale: il best mantenuto resta quindi lo stato singolo.

### Risultato finale ripetuto

| Prompt | Attention baseline | Attention finale | Delta attention | Wall baseline | Wall finale | Delta wall | Tok/s baseline/finale | Delta tok/s | CV finale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 1.987,565 ms | 1.621,397 ms | −18,42% | 26.593,88 ms | 26.157,68 ms | −1,64% | 77,01 / 78,29 | +1,66% | 0,04% |
| 8K | 31.594,332 ms | 26.102,069 ms | −17,38% | 129.940,00 ms | 124.677,93 ms | −4,05% | 63,04 / 65,71 | +4,24% | 0,03% |
| 16K | 127.715,823 ms | 104.551,929 ms | −18,14% | 325.186,73 ms | 302.106,96 ms | −7,10% | 50,38 / 54,23 | +7,64% | 0,01% |
| 28K | 376.913,813 ms | 307.360,813 ms | −18,45% | 717.678,20 ms | 646.633,07 ms | −9,90% | 39,01 / 43,30 | +11,00% | 0,01% |

Breakdown GPU finale:

| Prompt | GPU prefill | Attention | MLP | Proiezioni | Norm + RoPE | Altro |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 25.887,074 ms | 6,26% | 62,09% | 29,37% | 0,99% | 1,29% |
| 8K | 123.807,547 ms | 21,08% | 52,25% | 24,74% | 0,84% | 1,09% |
| 16K | 300.431,986 ms | 34,80% | 43,12% | 20,49% | 0,69% | 0,90% |
| 28K | 643.828,076 ms | 47,74% | 34,52% | 16,47% | 0,55% | 0,72% |

Decode finale è 31,41 / 19,68 / 13,01 / 8,87 tok/s a 2K/8K/16K/28K,
rispettivamente −0,41% / +0,51% / +0,46% / +0,23% contro i controlli baseline:
nessuna regressione oltre la soglia 5%.

Il test focalizzato conserva identità esatta F16/INT8 fra prefill e decode
sequenziale. Contro CPU F16, i massimi restano `2,94e-5` attention e `3,37e-6`
logits; i massimi mean absolute sono `5,52e-6` e `6,22e-7`, e i massimi mean
relative `1,66e-4` e `7,66e-3` per logits prossimi allo zero.

### Stop condition e prossimo componente

Attention resta il maggiore kernel singolo soltanto a 28K, ma non è più il
maggiore candidato pratico end-to-end. MLP domina fino a 16K e tutti i matmul
batched insieme valgono 50,99% anche a 28K. Le restanti leve attention hanno
upper bound misurati piccoli: tutte le exp 0,64% finale, tutti i barrier 6,13%
ma workgroup-wide, multi-state softmax 0,56%, address induction 0,59%, QK split
0,89%. Percorsi mixed/native o una pipeline K/V più complessa hanno quindi un
rapporto guadagno/complessità peggiore del matmul batched.

La maggiore catena residua è QK (45,05% attention), ma il candidato semplice è
già falsificato; il maggiore componente successivo da profilare è il matmul
batched condiviso da MLP e proiezioni, iniziando da MLP gate/up/down. Questa
conclusione soddisfa il gate di abbandono del micro-tuning attention.

### Verifica finale refinement

- `cargo fmt --all -- --check`: pass;
- `git diff --check`: pass;
- Clippy `--all-targets -D warnings`, feature `vulkan-profile`: pass;
- suite library `vulkan-profile`: 144 pass, 2 test autenticati ignorati;
- suite semantica Vulkan: 12 pass, 1 test autenticato ignorato;
- family-agnostic: 4 test eseguibili pass, 1 autenticato ignorato;
- build release del benchmark `vulkan-profile`: pass.

`docs_contract` resta escluso per il `VALIDATION.md` assente già registrato nella
baseline; nessuna modifica di questa fase interessa quel contratto.

## Follow-up MLP e matmul batched — RTX 3060, 2026-08-11

Questa fase parte dal risultato attention `937e630` su un worktree pulito e dal
branch dedicato `perf/vulkan-prefill-mlp`. La tupla resta RTX 3060, driver
595.84, Vulkan puro, contesto 32.768, KV F16 e lo stesso modello 3B/SHA-256
dichiarato sopra. Il candidato mantenuto è `d150a8b`. Attention non è stata
modificata e compare soltanto come controllo di regressione.

### Baseline e scomposizione

Un nuovo run pulito, prima di ogni modifica produttiva, ha confermato la
baseline pubblica precedente. I tempi seguenti sono un singolo campione
diagnostico; la tabella finale usa invece una warm-up e tre ripetizioni per
entrambi gli stati.

| Prompt | Wall / TTFT | Tok/s | GPU prefill | Attention | Q | K | V | Output | Gate | Up | Down |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 26.179,41 | 78,23 | 25.941,864 | 1.622,435 | 2.761,351 | 1.156,087 | 1.177,712 | 2.533,106 | 5.230,623 | 5.270,245 | 5.594,160 |
| 8K | 124.247,09 | 65,93 | 123.436,659 | 26.005,422 | 11.013,840 | 4.652,656 | 4.704,056 | 10.192,506 | 21.011,054 | 21.103,188 | 22.377,475 |
| 16K | 301.792,60 | 54,29 | 300.219,932 | 104.629,313 | 22.084,942 | 9.294,223 | 9.400,139 | 20.689,502 | 42.259,213 | 42.266,612 | 44.833,115 |
| 28K | 646.346,73 | 43,32 | 643.699,316 | 307.257,846 | 37.847,519 | 15.918,701 | 16.086,594 | 36.181,669 | 72.854,456 | 72.632,335 | 76.734,136 |

A 28K i sette matmul valgono 328.255,410 ms, cioè 50,99% del GPU
prefill; attention vale 47,73%. Gate/up/down da soli valgono 222.220,927 ms
(34,52%). Norm e RoPE valgono 3.548,593 ms; embedding, logits, KV write,
elementwise e residuo GPU completano il profilo, attribuito al 99,94%.

Il profiler iniziale aggregava `silu_mul` e residual. La sola modifica
diagnostica che li separa misura a 2K 18,133 ms di attivazione e 13,633 ms di
residual; l'attivazione è appena 0,07% del GPU prefill. Il costo cresce
linearmente e raggiunge circa 0,25 s a 28K, confermato dal candidato finale
(253,212 ms). Questo separa esplicitamente gate, up, attivazione e down senza
attribuire al SiLU il costo dei tre GEMM.

### Inventario dei matmul reali

Ogni layer riceve chunk FP16 di `M=32`; a 28K sono 875 chunk e 22.750
invocazioni per operazione. Input e output sono FP16, l'accumulatore è FP32. Le
weight sono row-major in super-block quantizzati, non materializzate in un
buffer dequantizzato: Q4_K usa 144 byte per 256 pesi, Q6_K 210 byte.

| Ordine 28K | Operazione | M×N×K per invocazione | Quantizzazione nei 26 layer | GPU ms | TFLOP/s effettivi | GB/s logici minimi |
|---:|---|---|---|---:|---:|---:|
| 1 | MLP down | 32×3072×9216 | 13 Q4_K + 13 Q6_K | 76.734,136 | 0,537 | 6,04 |
| 2 | MLP gate | 32×9216×3072 | 26 Q4_K | 72.854,456 | 0,566 | 5,22 |
| 3 | MLP up | 32×9216×3072 | 26 Q4_K | 72.632,335 | 0,568 | 5,23 |
| 4 | Q projection | 32×4096×3072 | 26 Q4_K | 37.847,519 | 0,484 | 4,53 |
| 5 | output projection | 32×3072×4096 | 26 Q4_K | 36.181,669 | 0,506 | 4,74 |
| 6 | V projection | 32×1024×3072 | 13 Q4_K + 13 Q6_K | 16.086,594 | 0,285 | 3,45 |
| 7 | K projection | 32×1024×3072 | 26 Q4_K | 15.918,701 | 0,288 | 2,90 |

Gate/up/down eseguono ciascuno 1,812 GFLOP per invocazione; Q/output 0,805
GFLOP, K/V 0,201 GFLOP. Per gate/up il minimo logico per invocazione è 15,925
MiB di weight più 0,786 MiB di input/output; down usa in media 19,575 MiB di
weight più 0,786 MiB di I/O. L'intensità gate è circa 108 FLOP/byte, ben oltre
il crossover roofline teorico della RTX 3060, ma raggiunge soltanto il 4–5% del
picco FP32. Anche i 2,9–6,0 GB/s logici sono lontani dalla banda DRAM. Il limite
non è quindi la banda globale pura: il kernel generico spende issue in FMA
scalari, unpack/dequant Q4/Q6, indirizzamento, shared-memory e barrier.

Il kernel generico usa tile `M=32, N=64, K=32`, workgroup 256 e dequantizza
ogni weight Q4_K una volta in shared per il batch. I layout sono già contigui
per riga quantizzata; cambiare layout o materializzare pesi FP16 aumenterebbe
memoria e traffico senza attaccare il limite misurato.

### Intermedi MLP e gate/up fusion

Un tensore intermedio `[prompt,9216]` FP16 occupa 36 / 144 / 288 / 492,188
MiB a 2K/8K/16K/28K. Non è tutto residente contemporaneamente per layer: gli
scratch sono riusati, ma il traffico logico corrente è otto volte la sua
dimensione per layer:

1. gate e up scrivono due tensori (`2S`);
2. SiLU legge entrambi, scrive l'attivazione e la copia volatile necessaria a
   preservare l'esatto arrotondamento, poi rilegge e riscrive l'output (`5S`
   complessivi nella fase);
3. down rilegge l'attivazione (`S`), oltre alle letture già incluse sopra.

Il totale è `8S`, 107,348 GB sui 26 layer a 28K. Una fusione gate+up+SiLU
potrebbe eliminare al massimo `6S`, 80,511 GB, più 4,473 GB di seconda lettura
dell'input condiviso. Tuttavia il kernel SiLU misurato costa soltanto ~0,25 s;
il residuo GPU che include barrier e dispatch è 0,36 s su 1,804 milioni di
dispatch. Anche attribuendo in modo irrealisticamente favorevole tutti questi
costi alla fusione, l'upper bound è sotto 0,4 s: meno dello 0,06% del GPU
prefill e dello 0,2% del tempo MLP a 28K.

La fusione parziale non elimina il down né la dequantizzazione dominante; una
fusione producer→down risparmierebbe soltanto gli ultimi `2S` (26,837 GB) e
richiederebbe di mantenere o ricomputare un tile FFN molto ampio. Per questo non
ha superato il gate Amdahl per un esperimento produttivo: il rischio di pressione
registri, shader più complesso e semantica FP16 diversa è maggiore del beneficio
massimo misurabile. L'intervento è stato concentrato sul GEMM stesso.

### Esperimenti cooperative-matrix

La RTX 3060 espone `VK_KHR_cooperative_matrix`, subgroup 32 e la forma
16×16×16 FP16×FP16→FP32. Tutti i candidati mantengono accumulo FP32 e store
FP16. L'uscita K=4096 è stata misurata separatamente, non inclusa per analogia.

| Candidato 2K | Routing | Wall | Delta wall | Risultato locale |
|---|---|---:|---:|---|
| generico | tutti i matmul generici | 26.005,45 ms | — | baseline A/B del ciclo |
| MMA-1-all | un subgroup per tile, include output K=4096 | 22.240,42 ms | −14,48% | MLP ~−20,7%, ma output +27%; reject routing |
| MMA-1-select | un subgroup, solo K=3072/9216 | 21.605,51 ms | −16,92% | MLP ~−20,7%, proiezioni ~−15,3% |
| MMA-2-select | due subgroup condividono B, K=3072/9216 | 16.065,68 ms | −38,22% | MLP −45,7%, proiezioni −34,6%; keep |

Il candidato mantenuto assegna allo stesso workgroup due tile M16 adiacenti.
I due subgroup caricano A separatamente ma condividono un tile B dequantizzato:
tile effettivo `M=32, N=16, K=16`, workgroup 64 e griglia
`ceil(M/32) × ceil(N/16)`. Usa 3.584 byte shared/workgroup: 1.024 byte A,
512 byte B e 2.048 byte C. I tail M/N sono azzerati e gli store sono guardati;
K è multiplo del super-block Q4_K e non ha tail.

Le statistiche eseguibili NVIDIA riportano subgroup 32, 34 registri/thread,
stack 0, 3.584 byte shared e binary 33.920 byte. Con 2.176 registri e 3,5 KiB
per workgroup, registri e shared non limitano la residenza; il limite di 16
workgroup/SM dà 1.024 thread, 32 warp, cioè 66,7% dell'occupancy thread
modellata. Non sono contatori di stall: Nsight non espone GPU metrics su questa
scheda tramite il percorso disponibile.

Il riuso fra i due subgroup dimezza l'unpack rispetto al candidato MMA a un
subgroup, che doveva dequantizzare la stessa B due volte per le 32 righe. Non
riduce invece il carico teorico sotto il kernel generico, che già carica ogni
weight una volta per batch: il guadagno viene dalla sostituzione delle FMA
scalari con MMA native e dal riuso necessario a rendere efficiente quel tiling.

Il routing è volutamente stretto: Q4_K, device NVIDIA `0x10de`, subgroup 32,
forma esatta 16×16×16 e `K=3072` o `K=9216`. Q, K, gate e up usano il nuovo
kernel; i 13 down Q4_K lo usano, mentre i 13 down Q6_K e tutte le V Q6_K
mantengono il kernel generico. Output K=4096 resta generico per la regressione
misurata. Device, forma o formato non supportati fanno fallback silenzioso.
`GRAPH_HORIZON_PREFILL_COOPMAT=0` disabilita il percorso per A/B e recovery;
il decode non attraversa questo routing.

### Correttezza

Il test CPU-oracle del matmul Q4_K usa `K=3072`, `M=37`, `N=70`, quindi
esercita il percorso mantenuto e i tail su entrambi gli assi. Ha misurato
`max_abs=4,442383`, `max_relative=4,7589e-4`, `mean_abs=1,962477` e
`mean_relative=1,9103e-4`; il grande errore assoluto riflette i valori sintetici
di riferimento, mentre tutti gli elementi rispettano `abs <= 0,05` oppure
`rel <= 0,5%`. Il test Q6_K CPU-oracle continua a passare sul fallback.

La suite library `vulkan-profile` passa 145 test con due test autenticati
ignorati. Il profiler passa i tre test di attribuzione e un run reale 2K con KV
INT8 completa il prefill senza valori non finiti: 18.861,40 ms, 108,58 tok/s,
18.641,161 ms GPU; il routing MMA è visibile su Q/K/gate/up e il decode resta
sui kernel originali. La tupla prestazionale pubblica resta F16.

La parità full-model con il server esterno pinned non è stata dichiarata pass:
il repository richiede llama.cpp `13f2b28b`, mentre il solo eseguibile locale è
`9bebfcb4b`. Il controllo si ferma correttamente come verifica esterna non
disponibile. Le verifiche locali coprono direttamente gli output intermedi
FP16 del matmul contro CPU, formati Q4/Q6, tail, fallback, KV F16/INT8 e prompt
reali lunghi; nessun dato del server non pinned è stato usato come oracle.

### Risultato finale ripetuto

Una warm-up separata e tre ripetizioni misurate producono CV sotto 0,5% su ogni
prompt:

| Prompt | Wall baseline | Wall finale | Delta wall | Tok/s baseline/finale | Delta tok/s | CV finale | Decode baseline/finale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 26.157,68 | 16.194,26 | −38,09% | 78,29 / 126,47 | +61,54% | 0,39% | 31,41 / 31,71 |
| 8K | 124.677,93 | 85.002,95 | −31,82% | 65,71 / 96,37 | +46,66% | 0,32% | 19,68 / 19,42 |
| 16K | 302.106,96 | 222.786,06 | −26,26% | 54,23 / 73,54 | +35,61% | 0,04% | 13,01 / 12,97 |
| 28K | 646.633,07 | 511.075,56 | −20,96% | 43,30 / 54,79 | +26,54% | 0,02% | 8,87 / 8,83 |

Breakdown GPU finale, media delle quattro esecuzioni profiler incluse warm-up e
ripetizioni:

| Prompt | GPU prefill | Attention | Q/K/V/output | Gate/up/act/down | Norm+RoPE | Altro |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 15.934,967 | 1.619,477 | 4.980,917 | 8.755,185 | 262,090 | 317,298 |
| 8K | 84.078,532 | 26.178,396 | 20.144,279 | 35.415,905 | 1.055,497 | 1.284,455 |
| 16K | 221.168,461 | 104.825,171 | 40.591,226 | 71.076,177 | 2.112,058 | 2.563,829 |
| 28K | 508.409,440 | 308.205,287 | 70.235,308 | 121.977,558 | 3.601,082 | 4.390,205 |

A 28K attention è ora 60,62% del GPU prefill. MLP è 23,99%, proiezioni
13,81%, norm+RoPE 0,70% e altro 0,88%. Attention cambia soltanto di +0,27%
rispetto alla baseline ripetuta, entro la variabilità di un componente non
modificato. Il decode cambia da +0,96% a −1,32% secondo il prompt, molto sotto
la soglia di regressione 5%.

I sette matmul scendono da 328.255,410 ms nella baseline pulita a 191.959,654
ms nel finale 28K, −41,52% locale. Amdahl prevede `50,995% × 41,52% = 21,17%`
di riduzione GPU; la misura ripetuta è −21,03% GPU e −20,96% wall. La
coincidenza chiude l'attribuzione: il guadagno proviene dal matmul, non da
attention, CPU recording o rumore.

### Stop condition

La condizione di arresto è soddisfatta: MLP e proiezioni insieme scendono dal
50,99% al 37,80%, mentre attention sale al 60,62% ed è nuovamente il maggiore
collo di bottiglia. Ulteriori fusioni MLP hanno un upper bound inferiore allo
0,06% e Q6_K richiederebbe un nuovo kernel/rapporto numerico non giustificato
dal target già raggiunto. Il maggiore limite residuo è quindi attention, già
caratterizzato nelle sezioni precedenti; questa fase non lo riapre.

### Verifica finale MLP

- `cargo fmt --all -- --check`: pass;
- `git diff --check`: pass;
- Clippy `--all-targets -D warnings`, feature `vulkan-profile`: pass;
- suite library `vulkan-profile`: 145 pass, 2 test autenticati ignorati;
- test focalizzati matmul `vulkan-hybrid`: 3 pass;
- family-agnostic: 4 pass, 1 autenticato ignorato, `docs_contract` fallisce
  soltanto per il `VALIDATION.md` assente già presente nella baseline.

Il probe temporaneo delle statistiche pipeline è stato rimosso; la build finale
non stampa né richiede `VK_KHR_pipeline_executable_properties`.

## Fase 2 attention: QK cooperative matrix — 2026-08-11

Questa fase parte dal risultato matmul preservato. La baseline fresca è
`b381843`, discendente di `d150a8b` e del report `6158af5`; il candidato
mantenuto è `4fe17c9` sul branch `perf/vulkan-prefill-attention-phase2`.
L'unica variabile intenzionale è il produttore QK del prefill attention F16.
Decode, attention INT8, layout KV, online softmax, AV e matmul restano invariati.

La tupla è RTX 3060 12 GiB, driver NVIDIA 595.84, Vulkan device 1.4.329,
`release` con `vulkan-profile`, Vulkan puro, KV F16, contesto 32.768, greedy,
due token richiesti, una warm-up e tre repliche. Il modello è
`Ministral-3-3B-Instruct-2512-Q4_K_M.gguf`, 2.147.023.008 byte, SHA-256
`9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8`.
Il prompt `" a"` è calibrato a N token renderizzati: N−3 ripetizioni per
2.048, 8.192, 16.384 e 28.000. Un lancio esplorativo a 28.672 è stato fermato
appena rilevata la tupla errata e non entra in alcuna misura.

### Baseline fresca e gate

I timestamp profiler seguenti sono medie delle quattro esecuzioni, inclusa la
warm-up. La premessa è confermata: a 28K attention vale 308.379,017 ms, il
60,64% del GPU prefill, mentre i sette matmul valgono 191.948,848 ms, 37,74%.

| Prompt | Wall | Tok/s | CV | GPU prefill | Attention | Quota attention | Decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 16.199,21 ms | 126,43 | 0,34% | 15.939,626 ms | 1.621,662 ms | 10,17% | 31,72 |
| 8K | 84.899,22 ms | 96,49 | 0,12% | 83.996,087 ms | 26.144,495 ms | 31,13% | 19,59 |
| 16K | 222.708,16 ms | 73,57 | 0,03% | 221.073,370 ms | 104.760,976 ms | 47,39% | 12,99 |
| 28K | 511.300,16 ms | 54,76 | 0,02% | 508.562,285 ms | 308.379,017 ms | 60,64% | 8,83 |

Breakdown richiesto della baseline corrente:

| Prompt | Attention GPU | Matmul GPU | Other GPU | Attention / Matmul / Other |
|---:|---:|---:|---:|---:|
| 2K | 1.621,662 ms | 13.719,298 ms | 598,667 ms | 10,17 / 86,07 / 3,76% |
| 8K | 26.051,592 ms | 55.159,646 ms | 2.396,111 ms | 31,16 / 65,97 / 2,87% |
| 16K | 104.760,976 ms | 111.497,963 ms | 4.814,432 ms | 47,39 / 50,43 / 2,18% |
| 28K | 308.379,017 ms | 191.948,848 ms | 8.234,420 ms | 60,64 / 37,74 / 1,62% |

La riga 8K completa una riga categoria troncata nell'output ripetuto con un
controllo scalare singolo separato, stesso artefatto e tupla: total 83.607,349
ms, attention 26.051,592 ms e sette matmul 55.159,646 ms. Le altre righe sono
le medie fresche ripetute. `Other` include residuo profiler e tutte le categorie
diverse da attention e dai sette matmul.

Il gate Amdahl autorizza quindi una modifica QK: nella precedente attribuzione
del kernel scalare QK era il primo limite, 45,05% attention. AV era 28,37%,
dependency/control 18,60%, barrier espliciti 6,13%, global/cache 1,21% e
trascendentali 0,64%. I vecchi QK scalar-ILP, softmax a due stati, barrier scope
e doppio buffering restano falsificati dalle misure già riportate e non sono
stati ripetuti senza una nuova ragione.

### Candidato QK nativo

La RTX 3060 espone solo la forma subgroup 16×16×16 FP16×FP16→FP32 utile. Il
nuovo kernel mantiene `Q_TILE=8`, `KV_TILE=64`, WG 512 e la fusione QK →
softmax → AV senza score matrix globale. Quattro subgroup producono quattro
tile score 16×16; otto passi K coprono `head_dim=128`. Le righe Q 8–15 sono
azzerate perché la forma 8×16 non è disponibile. K è già `[key][dimension]` e
viene letto direttamente come operando B column-major: nessun transpose e
nessun cambio del cache layout.

Q/K/V restano FP16 in storage; cooperative accumulation, score, max, exp,
somma softmax, AV e normalizzazione restano FP32. `sh_score` e `sh_q` crescono
da 8 a 16 righe per accogliere lo store cooperativo, portando la shared da
20.576 a 24.672 byte/workgroup. Non esistono nuove allocazioni, dipendenze o
API. Il routing richiede head dimension 128, tiled attention disponibile,
24.672 byte shared e forma esatta 16×16×16; altrimenti seleziona lo shader
scalare precedente, rimasto byte-identico.

Il prototipo QK-only a 8K ha superato il gate prima dell'integrazione:

| QK-only 8K | Attention | Wall | Registri | Shared | Esito |
|---|---:|---:|---:|---:|---|
| scalare | 20.282,956 ms | 78.571,43 ms | 33 | 20.576 B | baseline diagnostica |
| cooperative | 6.507,163 ms | 64.474,62 ms | 36 | 24.672 B | 3,117×, keep |

La prima esecuzione QK-only con riduttore non-finito sommava intenzionalmente
score mascherati `-inf`; è stata scartata. Il riduttore corretto cambia il tempo
scalare soltanto dello 0,14%. L'integrazione completa passa i primi screen 8K e
28K con −47,43% e −47,01% attention e quindi soddisfa il gate locale senza
stack di modifiche ulteriori.

### Attribuzione aggiornata del kernel

Le ablation 8K sono singole, sullo stesso eseguibile e sulla stessa tupla del
screen completo da 13.744,580 ms. Conservano staging, store, barrier e lavoro a
valle salvo la classe nominata. `no-QK` scrive score uniformi mantenendo Q/K
staging; `no-AV` mantiene V staging e update online; `no-softmax` mantiene loop,
mask e riduzione della cardinalità ma rimuove max/exp/stato online.

| Ablation 8K | Attention | Risparmio dal completo | Quota candidato |
|---|---:|---:|---:|
| completo cooperative | 13.744,580 ms | — | — |
| no QK datapath | 10.193,586 ms | 3.550,994 ms | 25,84% |
| no AV load/FMA | 9.326,118 ms | 4.418,462 ms | 32,15% |
| no max/exp/online state | 13.497,071 ms | 247,509 ms | 1,80% |

Barrier e hot-set provengono dalle ablation immediatamente precedenti: numero
dei barrier, dependency graph e byte globali non cambiano. Trasferendo il loro
costo assoluto, non la vecchia percentuale, l'attribuzione aggiornata è:

| Classe instruction/dependency | Quota attention | Evidenza |
|---|---:|---|
| AV shared-load + FP32 FMA issue | 32,15% | ablation candidata diretta |
| dependency/control/shared staging residui | 26,25% | residuo dopo tutte le classi misurate |
| cooperative QK load/MMA/store | 25,84% | ablation candidata diretta |
| barrier workgroup producer→consumer | 11,66% | 1.602,658 ms assoluti, grafo e 5 siti invariati |
| global/cache K+V | 2,30% | 316,348 ms assoluti, byte/layout invariati |
| max/exp/reduction/stato softmax | 1,80% | ablation candidata diretta |

Le prime quattro classi spiegano il 95,90% del kernel. Il maggiore limite
aritmetico residuo è AV, non QK. La classe dependency/control resta combinata:
senza stall counter Vulkan non è corretto attribuirla artificialmente a branch,
shared latency o attese di phase issue individuali.

La classe dinamica dominante è quindi AV: load shared dello score e di V,
conversione V FP16→FP32 e quattro catene da 16 multiply/add FP32. La catena
aritmetica dipendente più lunga è una catena AV da 16 aggiornamenti; QK ha otto
MMA dipendenti sullo stesso accumulatore, mentre softmax concatena max → `m_new`
→ exp → sum → update `(m,l)` una volta per tile. Il principale issue limit è la
combinazione AV scalar-FP32/shared-load e phase wait workgroup, non le exp.

Il fast path causale preesistente evita il mask branch quando tutto il tile è
visibile. Ponderato per score, copre 96,94 / 99,23 / 99,62 / 99,78% a
2K/8K/16K/28K. Un nuovo candidato causal-full ha quindi upper bound inferiore
allo 0,22% del lavoro score a 28K ed è stato respinto prima di una patch
produttiva. Diagonal tile, history parziale e tail restano nel percorso guardato.

### SPIR-V, risorse e occupancy

Shaderc `-O`, target Vulkan 1.3, è il confronto statico. È un proxy compilatore,
non ISA NVIDIA. Il driver espone statistiche pipeline ma restituisce zero
internal representation: non sono disponibili testo SASS né stall counter.

| Sito SPIR-V ottimizzato | Scalare | Cooperative |
|---|---:|---:|
| binary | 10.116 B | 10.232 B |
| righe assembly | 635 | 624 |
| branch / branch condizionali | 49 / 30 | 47 / 28 |
| load / store | 39 / 14 | 37 / 14 |
| add / mul FP32 | 11 / 8 | 10 / 7 |
| subgroup elect/add/max | 2 / 2 / 1 | 1 / 1 / 1 |
| cooperative load/MMA/store | 0 / 0 / 0 | 2 / 1 / 1 |
| barrier | 5 | 5 |
| `OpUDiv` / shift+mask | 4 / 6 | 4 / 6 |

| Statistica driver | Scalare | Cooperative |
|---|---:|---:|
| subgroup | 32 | 32 |
| registri/thread | 40 | 40 |
| shared/workgroup | 20.576 B | 24.672 B |
| stack | 0 | 0 |
| binary driver | 16.640 B | 13.568 B |

Tre workgroup richiedono 61.440 registri, 74.016 byte shared e 1.536 thread per
SM, entro 65.536 registri, 102.400 byte shared e 1.536 thread della GPU. Restano
quindi 48 warp/SM e 100% occupancy thread modellata, uguale allo scalare. Non ci
sono spill; la velocità viene dal datapath MMA, non da maggiore occupancy. Il
probe temporaneo `VK_KHR_pipeline_executable_properties` è stato rimosso dal
checkpoint produttivo.

### Operation model e roofline

Per ogni coppia causale e testa, QK e AV eseguono ciascuno 256 FLOP; insieme
512 FLOP. Sui 26 layer e 32 query head il lavoro matematico QK+AV è 0,894 /
14,295 / 57,178 / 166,992 TFLOP, cioè 0,436 / 1,745 / 3,490 / 5,964 GFLOP per
token a 2K/8K/16K/28K. Il candidato esegue inoltre le otto righe QK padded; i
valori restano deliberatamente il lavoro matematico utile, non le MMA emesse.

Per un workgroup e un full KV tile il lavoro richiesto è 65.536 FMA QK e
65.536 FMA AV, cioè 131.072 FLOP per fase. La forma cooperative emette 131.072
FMA QK perché calcola 16 anziché 8 righe: il doppio del lavoro QK utile, ma su
unità tensor. Softmax esegue 512 scale/max, 512 exp score, 256 exp `alpha`, otto
`subgroupMax`, otto `subgroupAdd` e otto merge online. AV converte dinamicamente
65.536 valori V FP16→FP32. Lo store finale normalizza e converte 1.024 output
FP32→FP16. K e V caricano ciascuno 8.192 elementi globali F16 e riusano la
stessa tile shared; quattro barrier per KV tile proteggono K, score, V e AV,
oltre al barrier di inizializzazione per dispatch.

Il lavoro matematicamente inevitabile è QK+AV sui pair causali, max/exp/sum e
la normalizzazione stabile. L'overhead implementativo è il padding QK 2×, gli
exp `alpha` per tutte le lane, gli indici/branch, gli store score shared e i
barrier producer→consumer. Le misure mostrano però che rimuovere tutto il blocco
softmax vale soltanto 1,80%; il padding MMA resta vantaggioso 3,117× sul QK
scalare e i barrier sono necessari all'ownership corrente.

| Prompt | TFLOP/s utili scalare | TFLOP/s utili cooperative |
|---:|---:|---:|
| 2K | 0,551 | 1,079 |
| 8K | 0,547 | 1,036 |
| 16K | 0,546 | 1,021 |
| 28K | 0,542 | 1,018 |

Il tiling precedente mantiene 20,879 TB logici K+V a 28K, 64 byte per coppia
Q/K causale e intensità utile 7,998 FLOP/B. Sul tempo candidato sono 127,34 GB/s
logici, contro 67,71 GB/s scalari. Come ceiling plausibile, il clock massimo
configurato 2,13 GHz dà circa 15,3 TFLOP/s FP32 e 61,1 TFLOP/s tensor dense;
con metà lavoro QK tensor e metà AV FP32 il ceiling armonico è ~24,4 TFLOP/s.
La memoria GDDR6 nominale dà 360 GB/s e quindi ~2,88 TFLOP/s al rapporto 8
FLOP/B. Il candidato raggiunge circa il 35% del roof di banda e il 4% del roof
compute misto. Insieme all'hot-set 2,30% e a dependency+barrier 37,91%, questo
indica un limite prevalente di latency/issue/shared synchronization, non
saturazione pura di DRAM o unità aritmetiche.

### Risultato ripetuto e Amdahl

| Prompt | Wall baseline/finale | Delta wall | Tok/s baseline/finale | Delta tok/s | CV finale | Attention baseline/finale | Delta attention | Decode baseline/finale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 16.199,21 / 15.425,36 ms | −4,78% | 126,43 / 132,77 | +5,01% | 0,31% | 1.621,662 / 828,575 ms | −48,91% | 31,72 / 31,62 |
| 8K | 84.899,22 / 72.239,37 ms | −14,91% | 96,49 / 113,40 | +17,53% | 0,01% | 26.144,495 / 13.803,053 ms | −47,20% | 19,59 / 19,61 |
| 16K | 222.708,16 / 173.792,83 ms | −21,96% | 73,57 / 94,27 | +28,14% | 0,04% | 104.760,976 / 55.996,002 ms | −46,55% | 12,99 / 13,00 |
| 28K | 511.300,16 / 366.031,89 ms | −28,41% | 54,76 / 76,50 | +39,70% | 0,04% | 308.379,017 / 163.967,008 ms | −46,83% | 8,83 / 8,82 |

| Prompt | GPU baseline/finale | Delta GPU | Amdahl previsto | Quota attention finale |
|---:|---:|---:|---:|---:|
| 2K | 15.939,626 / 15.174,317 ms | −4,80% | −4,98% | 5,46% |
| 8K | 83.996,087 / 71.413,757 ms | −14,98% | −14,69% | 19,33% |
| 16K | 221.073,370 / 172.008,057 ms | −22,19% | −22,06% | 32,55% |
| 28K | 508.562,285 / 363.095,028 ms | −28,60% | −28,40% | 45,16% |

L'accordo Amdahl chiude l'attribuzione. Il decode varia da −0,32% a +0,10%,
molto sotto il limite 5%. Sullo screen 28K i sette matmul invariati valgono
190.154,878 ms, 52,56% GPU, contro attention 45,17%: a livello sistema il
collo maggiore torna all'insieme matmul; `other` vale 8.222,364 ms, 2,27%.
Dentro attention AV è la prima classe.

### Registro delle ipotesi ordinate

Prima dell'integrazione, il QK-only misurato prevedeva un risparmio di
13.775,793 ms rispetto ai 26.144,495 ms attention 8K, upper bound −52,69%
attention. Con le quote baseline ciò prevedeva −16,40% GPU a 8K e −31,95% a
28K; il full kernel misura −47,20/−46,83% attention e −14,98/−28,60% GPU.

| ID | Target e frazione | Ipotesi / cambio ISA | Registri / occupancy | 8K / 28K locale e totale | Correttezza | Decisione |
|---|---|---|---|---|---|---|
| QK-CM | QK, 45,05% scalare | loop FMA+reduction → 2 load + 1 MMA + 1 store cooperative | 40→40, 100%→100% | attention −47,20/−46,83%; GPU −14,98/−28,60% | F16/INT8/tail pass | keep `4fe17c9` |
| CAUSAL-FULL | mask diagonale, 0,225% score a 28K | fast path è già presente; nessun cambio | invariati | upper bound <0,23% attention, <0,11% GPU | semantica invariata | reject prima della patch |
| REDUCE | softmax completo, 1,80% candidato | shuffle/manual tree al posto di subgroup max/add | non misurati | upper bound <0,81% GPU 28K anche eliminando tutto | non eseguito | reject per Amdahl |
| PIPE-MINI | dependency+barrier, 37,91% | prefetch frammento successivo durante compute | richiede nuovo live/shared state | removable fraction non dimostrata senza stall counter | non eseguito | defer: attention non è più target n.1 |
| SPEC-2 | address/control residuo | fixed GQA e ulteriore unroll | rischio code-size; 4 `UDiv` statici | mapping GQA è fuori dal loop KV; precedente address induction −0,59% attention | path corrente pass | reject per operation count |
| AV-CM | AV, 32,15% candidato | probabilities×V su MMA/native packed | forma richiede 16 righe e input F16 | ceiling ipotetico ~−21% attention, ~−9,7% GPU 28K | richiederebbe nuovo gate precisione | defer: conversione/materializzazione non giustificata nello scope corrente |
| AV-VEC | AV, 32,15% candidato | vec2/vec4/packed FMA | SPIR-V resta scalar FP32; nessuna primitive provata | beneficio non dimostrato | non eseguito | defer insieme ad AV-CM |
| EXP2 | softmax, 1,80% candidato | `exp2(x·log2(e))` | unknown | upper bound <0,81% GPU 28K | richiederebbe nuovo gate numerico | reject per Amdahl |

La pipeline mini e AV native non sono dichiarate falsificate: sono direzioni
future con rischio o informazione insufficiente. La condizione di arresto usata
è la n.1 della missione, non il semplice successo architetturale: dopo QK-CM i
matmul valgono 52,56% e attention 45,17% del GPU prefill 28K. Il prossimo target
end-to-end evidence-led è quindi il matmul Q6_K batched (soprattutto down, poi
V); se una fase futura riapre attention, AV-CM è l'ipotesi con upper bound più
alto, ma deve preservare probabilità/accumulo FP32 o dimostrare separatamente il
proprio errore.

### Correttezza, decisione e stop

Il test focalizzato confronta prefill e decode sequenziale per `(base,n)`
`(0,3)`, `(33,32)` e `(65,9)`, coprendo full tile, diagonale e tail. F16 e INT8
hanno `max_abs=mean_abs=mean_relative=0` contro il percorso Vulkan precedente.
Contro CPU F16, attention ha massimo `max_abs=2,938509e-5`, massimo
`mean_abs=5,523212e-6` e massimo relativo `1,659979e-4`; logits ha massimo
`max_abs=3,368943e-6`, `mean_abs=6,223882e-7` e relativo `7,655985e-3` solo
vicino a zero. I quattro prompt full-model finiscono senza non-finiti. L'oracle
esterno pinned resta non disponibile per la revisione locale già documentata e
non viene sostituito con un oracle diverso.

Il candidato viene mantenuto: supera ampiamente il target minimo −10%
attention su tutte le lunghezze, produce un guadagno wall visibile e stabile,
mantiene fallback e precisione, e aggiunge un solo kernel categoria K più
routing/capability locale. Non si prosegue oltre il target: AV e matmul sono ora
i limiti maggiori e richiederebbero una nuova ipotesi, mentre causal-full è già
coperto al 99,78%. Risultati finali del ciclo:

- `cargo fmt --all -- --check`: pass;
- `git diff --check`: pass;
- Clippy `--all-targets -D warnings`, `vulkan-profile`: pass;
- suite library `vulkan-profile`: 146 pass, 2 test autenticati ignorati;
- semantic: 12 pass, 1 test autenticato ignorato;
- family-agnostic: 4 pass, 1 autenticato ignorato; `docs_contract` conserva il
  fallimento baseline per `VALIDATION.md` assente;
- nessun probe temporaneo, shader ablation o output di profiling resta nel tree.

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
    ├── pipeline/mod.rs (~170 righe produttive, registry e capability gate)
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

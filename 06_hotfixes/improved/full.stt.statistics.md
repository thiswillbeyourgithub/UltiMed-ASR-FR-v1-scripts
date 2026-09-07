# CER statistics

- Generated: 2026-07-31 14:00:17
- Input: `improved/full.stt.jsonl`
- CER field: `cer`
- Model: `whisper-1` (n=599740 rows)
- Bad-clip threshold (CER >=): 0.15

## Coverage

How much of the manifest has a numeric CER so far. On a still-running job this is below 100%; the statistics below are computed over the scored rows only.

| Category | Scored | Total | Coverage |
|---|--:|--:|--:|
| dictionary | 534424 | 534424 | 100.0% |
| drugs | 21901 | 21901 | 100.0% |
| parhaf | 43415 | 43415 | 100.0% |
| **Overall** | **599740** | **599740** | **100.0%** |

## Duration ceiling

`At max` counts the clips within one 12.5 Hz frame (0.08 s) of the longest one in the category. More than a couple means the audio was generated against a token cap, not to the end of its text: those clips are cut off mid-sentence and no regeneration seed can fix them, so the source text has to be re-chunked.

| Category | Longest clip | At max | Share | Verdict |
|---|--:|--:|--:|---|
| dictionary | 127.84 s | 1 | 0.0% | no ceiling |
| drugs | 27.92 s | 1 | 0.0% | no ceiling |
| parhaf | 159.20 s | 1 | 0.0% | no ceiling |
| **Overall** | 159.20 s | 1 | 0.0% | no ceiling |

## CER summary

| Category | N | Mean | Median | Std | Min | Max | CER>=0.15 |
|---|--:|--:|--:|--:|--:|--:|--:|
| dictionary | 534424 | 0.0127 | 0.0094 | 0.0173 | 0.0000 | 1.0392 | 0.0% (144) |
| drugs | 21901 | 0.0160 | 0.0115 | 0.0179 | 0.0000 | 0.9085 | 0.0% (6) |
| parhaf | 43415 | 0.0143 | 0.0097 | 0.0210 | 0.0000 | 0.9718 | 0.1% (27) |
| **Overall** | **599740** | **0.0129** | **0.0095** | **0.0176** | **0.0000** | **1.0392** | **0.0% (177)** |

## Triple-check coverage of high-CER clips

Of the clips at CER >= 0.15, how many carry `n_stt_check` >= 3, i.e. have already been re-transcribed at every recheck temperature and stayed bad. "Not yet" clips are high-CER but still short of 3 readings (awaiting a recheck, or the rechecks were disabled). A clip a recheck rescued is no longer high-CER and is not counted here.

| Category | High-CER (bad) | Triple-checked | Not yet | Coverage |
|---|--:|--:|--:|--:|
| dictionary | 144 | 82 | 62 | 56.9% |
| drugs | 6 | 3 | 3 | 50.0% |
| parhaf | 27 | 27 | 0 | 100.0% |
| **Overall** | **177** | **112** | **65** | **63.3%** |

## CER quantiles (step 0.1)

| Category | N | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dictionary | 534424 | 0.0000 | 0.0000 | 0.0032 | 0.0045 | 0.0071 | 0.0094 | 0.0121 | 0.0155 | 0.0201 | 0.0280 | 1.0392 |
| drugs | 21901 | 0.0000 | 0.0000 | 0.0043 | 0.0056 | 0.0088 | 0.0115 | 0.0149 | 0.0193 | 0.0253 | 0.0365 | 0.9085 |
| parhaf | 43415 | 0.0000 | 0.0000 | 0.0034 | 0.0055 | 0.0075 | 0.0097 | 0.0123 | 0.0155 | 0.0203 | 0.0300 | 0.9718 |
| **Overall** | **599740** | **0.0000** | **0.0000** | **0.0032** | **0.0046** | **0.0072** | **0.0095** | **0.0122** | **0.0156** | **0.0203** | **0.0284** | **1.0392** |

## Tail CER (last ~30s of clips longer than that)

`cer_tail` scores the END of a clip on its own, because a whole-clip CER averages a late defect away: a chunk truncated at the TTS output cap keeps most of its text and lands around 0.13, under the gate. A clip counts as bad when EITHER its CER reaches 0.15 OR its tail CER reaches 0.3 (looser: the tail window is ~5x shorter, so ~5x noisier). Clips at or under 30s carry no tail score and are not counted here.

| Category | N | Mean | Median | Std | Min | Max |
|---|--:|--:|--:|--:|--:|--:|
| dictionary | 1779 | 0.0208 | 0.0121 | 0.0422 | 0.0000 | 0.8364 |
| drugs | 0 | - | - | - | - | - |
| parhaf | 21509 | 0.0237 | 0.0141 | 0.0357 | 0.0000 | 0.7882 |
| **Overall** | **23288** | **0.0235** | **0.0141** | **0.0362** | **0.0000** | **0.8364** |

| Category | Long clips scored | Flagged by the tail ONLY | Bad on both | Tail-only rate |
|---|--:|--:|--:|--:|
| dictionary | 1779 | 0 | 4 | 0.0% |
| drugs | 0 | 0 | 0 | - |
| parhaf | 21509 | 2 | 14 | 0.0% |
| **Overall** | **23288** | **2** | **18** | **0.0%** |

### Tail CER quantiles (step 0.1)

| Category | N | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dictionary | 1779 | 0.0000 | 0.0037 | 0.0046 | 0.0080 | 0.0100 | 0.0121 | 0.0161 | 0.0201 | 0.0261 | 0.0382 | 0.8364 |
| drugs | 0 | - | - | - | - | - | - | - | - | - | - | - |
| parhaf | 21509 | 0.0000 | 0.0020 | 0.0060 | 0.0080 | 0.0120 | 0.0141 | 0.0181 | 0.0221 | 0.0301 | 0.0482 | 0.7882 |
| **Overall** | **23288** | **0.0000** | **0.0021** | **0.0060** | **0.0080** | **0.0105** | **0.0141** | **0.0181** | **0.0221** | **0.0301** | **0.0482** | **0.8364** |

## Audio pace (seconds per source character)

Clip duration divided by the number of UNNORMALIZED characters of the text it was synthesized from (`asr_training_source`, raw, whitespace and punctuation included). The TTS voice speaks at a near-constant rate, so information density per second should be near-constant too: clips far from the median pace are suspicious. A HIGH value means the audio is much longer than its text (a stall, a repetition, trailing silence), a LOW one means it is too short (a truncated or skipped reading). This is model-independent, so it covers every row carrying a duration and a source text, transcribed or not.

| Category | N | Mean | Median | Std | Min | Max |
|---|--:|--:|--:|--:|--:|--:|
| dictionary | 534424 | 0.06091 | 0.06039 | 0.00500 | 0.04429 | 0.56070 |
| drugs | 21901 | 0.06106 | 0.06038 | 0.00581 | 0.04450 | 0.12850 |
| parhaf | 43415 | 0.05991 | 0.05896 | 0.00635 | 0.04369 | 0.19521 |
| **Overall** | **599740** | **0.06084** | **0.06030** | **0.00515** | **0.04369** | **0.56070** |

### Audio pace quantiles (step 0.1)

The same seconds-per-character value at each quantile, so the suspicious tails are readable: compare P0 / P10 and P90 / P100 against the median.

| Category | N | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dictionary | 534424 | 0.04429 | 0.05512 | 0.05680 | 0.05809 | 0.05925 | 0.06039 | 0.06159 | 0.06295 | 0.06467 | 0.06729 | 0.56070 |
| drugs | 21901 | 0.04450 | 0.05439 | 0.05627 | 0.05776 | 0.05911 | 0.06038 | 0.06174 | 0.06332 | 0.06533 | 0.06857 | 0.12850 |
| parhaf | 43415 | 0.04369 | 0.05339 | 0.05514 | 0.05648 | 0.05770 | 0.05896 | 0.06031 | 0.06183 | 0.06387 | 0.06713 | 0.19521 |
| **Overall** | **599740** | **0.04369** | **0.05495** | **0.05666** | **0.05797** | **0.05914** | **0.06030** | **0.06152** | **0.06290** | **0.06465** | **0.06733** | **0.56070** |

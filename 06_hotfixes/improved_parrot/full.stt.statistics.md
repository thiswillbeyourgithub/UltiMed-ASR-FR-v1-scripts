# CER statistics

- Generated: 2026-07-31 14:00:17
- Input: `improved_parrot/full.stt.jsonl`
- CER field: `cer`
- Model: `whisper-1` (n=1549 rows)
- Bad-clip threshold (CER >=): 0.15

## Coverage

How much of the manifest has a numeric CER so far. On a still-running job this is below 100%; the statistics below are computed over the scored rows only.

| Category | Scored | Total | Coverage |
|---|--:|--:|--:|
| parrot | 1549 | 1549 | 100.0% |
| **Overall** | **1549** | **1549** | **100.0%** |

## Duration ceiling

`At max` counts the clips within one 12.5 Hz frame (0.08 s) of the longest one in the category. More than a couple means the audio was generated against a token cap, not to the end of its text: those clips are cut off mid-sentence and no regeneration seed can fix them, so the source text has to be re-chunked.

| Category | Longest clip | At max | Share | Verdict |
|---|--:|--:|--:|---|
| parrot | 63.28 s | 1 | 0.1% | no ceiling |
| **Overall** | 63.28 s | 1 | 0.1% | no ceiling |

## CER summary

| Category | N | Mean | Median | Std | Min | Max | CER>=0.15 |
|---|--:|--:|--:|--:|--:|--:|--:|
| parrot | 1549 | 0.0207 | 0.0144 | 0.0244 | 0.0000 | 0.3949 | 0.3% (4) |
| **Overall** | **1549** | **0.0207** | **0.0144** | **0.0244** | **0.0000** | **0.3949** | **0.3% (4)** |

## Triple-check coverage of high-CER clips

Of the clips at CER >= 0.15, how many carry `n_stt_check` >= 3, i.e. have already been re-transcribed at every recheck temperature and stayed bad. "Not yet" clips are high-CER but still short of 3 readings (awaiting a recheck, or the rechecks were disabled). A clip a recheck rescued is no longer high-CER and is not counted here.

| Category | High-CER (bad) | Triple-checked | Not yet | Coverage |
|---|--:|--:|--:|--:|
| parrot | 4 | 4 | 0 | 100.0% |
| **Overall** | **4** | **4** | **0** | **100.0%** |

## CER quantiles (step 0.1)

| Category | N | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| parrot | 1549 | 0.0000 | 0.0024 | 0.0057 | 0.0085 | 0.0115 | 0.0144 | 0.0179 | 0.0215 | 0.0283 | 0.0449 | 0.3949 |
| **Overall** | **1549** | **0.0000** | **0.0024** | **0.0057** | **0.0085** | **0.0115** | **0.0144** | **0.0179** | **0.0215** | **0.0283** | **0.0449** | **0.3949** |

## Tail CER (last ~30s of clips longer than that)

`cer_tail` scores the END of a clip on its own, because a whole-clip CER averages a late defect away: a chunk truncated at the TTS output cap keeps most of its text and lands around 0.13, under the gate. A clip counts as bad when EITHER its CER reaches 0.15 OR its tail CER reaches 0.3 (looser: the tail window is ~5x shorter, so ~5x noisier). Clips at or under 30s carry no tail score and are not counted here.

| Category | N | Mean | Median | Std | Min | Max |
|---|--:|--:|--:|--:|--:|--:|
| parrot | 353 | 0.0356 | 0.0201 | 0.0503 | 0.0000 | 0.4578 |
| **Overall** | **353** | **0.0356** | **0.0201** | **0.0503** | **0.0000** | **0.4578** |

| Category | Long clips scored | Flagged by the tail ONLY | Bad on both | Tail-only rate |
|---|--:|--:|--:|--:|
| parrot | 353 | 0 | 2 | 0.0% |
| **Overall** | **353** | **0** | **2** | **0.0%** |

### Tail CER quantiles (step 0.1)

| Category | N | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| parrot | 353 | 0.0000 | 0.0060 | 0.0100 | 0.0141 | 0.0181 | 0.0201 | 0.0245 | 0.0301 | 0.0457 | 0.0743 | 0.4578 |
| **Overall** | **353** | **0.0000** | **0.0060** | **0.0100** | **0.0141** | **0.0181** | **0.0201** | **0.0245** | **0.0301** | **0.0457** | **0.0743** | **0.4578** |

## Audio pace (seconds per source character)

Clip duration divided by the number of UNNORMALIZED characters of the text it was synthesized from (`asr_training_source`, raw, whitespace and punctuation included). The TTS voice speaks at a near-constant rate, so information density per second should be near-constant too: clips far from the median pace are suspicious. A HIGH value means the audio is much longer than its text (a stall, a repetition, trailing silence), a LOW one means it is too short (a truncated or skipped reading). This is model-independent, so it covers every row carrying a duration and a source text, transcribed or not.

| Category | N | Mean | Median | Std | Min | Max |
|---|--:|--:|--:|--:|--:|--:|
| parrot | 1549 | 0.06142 | 0.05995 | 0.01063 | 0.04661 | 0.38822 |
| **Overall** | **1549** | **0.06142** | **0.05995** | **0.01063** | **0.04661** | **0.38822** |

### Audio pace quantiles (step 0.1)

The same seconds-per-character value at each quantile, so the suspicious tails are readable: compare P0 / P10 and P90 / P100 against the median.

| Category | N | P0 | P10 | P20 | P30 | P40 | P50 | P60 | P70 | P80 | P90 | P100 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| parrot | 1549 | 0.04661 | 0.05414 | 0.05605 | 0.05731 | 0.05845 | 0.05995 | 0.06147 | 0.06304 | 0.06574 | 0.07006 | 0.38822 |
| **Overall** | **1549** | **0.04661** | **0.05414** | **0.05605** | **0.05731** | **0.05845** | **0.05995** | **0.06147** | **0.06304** | **0.06574** | **0.07006** | **0.38822** |

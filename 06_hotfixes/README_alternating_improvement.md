# Alternating single-GPU improvement (STT and TTS cannot share VRAM)

`01_recursive_improvement.py` normally runs STT and TTS at once (`--mode both`).
When the two servers cannot fit on the GPU together, split the work into alternating
passes with `--mode stt` / `--mode tts`, driven unattended by `driver.sh`.

This setup was written with Claude Code.

## Why a split is needed

Scoring a regenerated candidate requires transcribing it, so `--mode both`'s repair
loop needs both servers up at the same time. The split breaks that coupling:

- `--mode tts` (only the TTS server up): for every clip a prior `stt` pass flagged
  bad, synthesize all `--n-improv` candidates and persist each as a
  `<clip>.draft<seed>.<fmt>` sidecar. No scoring.
- `--mode stt` (only the STT server up): transcribe and score the originals (flagging
  bad ones), and score the candidates a prior `tts` pass left, promoting the best draw
  (or exhausting the clip) and deleting the candidate files.

State lives in each row's `improvement.status` (`pending_tts` -> `pending_stt` ->
`improved` / `exhausted`), so every pass resumes where the last stopped. The only thing
lost versus `--mode both` is per-clip early stop: the whole candidate set is generated
before any is scored. That is the right trade when minimizing server swaps dominates.

The `stt` pass only picks from a **complete** candidate set. The pick is final (the clip
becomes `improved` / `exhausted` and is never revisited), so judging 3 candidates when
`--n-improv` asks for 5 would lock in a draw that may not be the best available: such a
clip goes back to `pending_tts` for the `tts` pass to make the set in full. "Complete"
means the `tts` pass delivered every draw it was asked for, not one file per draw: a
draw it skipped as a duplicate or as frame-cap truncated is skipped deterministically
(the seed of each attempt is fixed), so demanding a file for it would bounce the clip
between the two passes forever. In practice this triggers when `--n-improv` is raised
between runs, or when a sidecar has been deleted.

## How the alternation converges

`driver.sh` loops: switch to the STT server, run `--mode stt`; switch to the TTS
server, run `--mode tts`; repeat. The `--mode stt` pass exits **10** when no clip
anywhere still needs either pass, which is the driver's stop signal. A typical run is
`stt` (find bad clips) -> `tts` (draft) -> `stt` (score + promote) -> exit 10.

It is fine to start an alternating run on a dataset with **no scoring work left**: that
is what a run stopped (or killed) right after an `stt` pass leaves behind, every flagged
clip waiting on a `tts` pass. The `stt` pass then exits **11** ("this mode has run dry",
see below), which the alternating driver reads as "the work is on the other side" and
follows with the `tts` pass rather than stopping. It gives up on the dataset only when
both passes run dry in the same round, i.e. the leftovers are stuck (a flagged clip whose
audio has since gone missing, say), since swapping the two servers again would not move
them. Same stop for the less obvious shape of that dead end: two rounds in a row where
the `tts` pass has nothing to draft while the `stt` pass keeps reporting work done and
never converges, which means it is re-running rows it can never resolve.

### Running one half on its own

`MODE=stt` (or `MODE=tts`) runs only that pass and never brings the other server up,
for when the other half is down, unconfigured, or mid-rebuild. Scoring is useful on
its own: bad clips are flagged `pending_tts` and simply wait for a later run, so an
STT-only night loses nothing.

```bash
sudo -E MODE=stt CATEGORY="dictionary drugs" ./driver.sh
```

A single-mode loop stops on exit **11**, "this mode has run dry": the pass completed
nothing, so repeating it changes nothing, but clips are still waiting on the other
mode (which is why it is not 10). The alternating driver does see 11 and does not stop
on it: it just runs the other pass (see the paragraph above).

## One-time setup

**What CrispASR is.** A private, unpublished project of mine: a high-performance
self-hosted Whisper server (several GPU replicas behind a Caddy load balancer,
OpenAI-compatible `/v1/audio/transcriptions`), with the TTS service bolted into the
same compose project so both can take turns on the one GPU. Nothing in this stage
needs *that* server specifically. Any OpenAI-compatible transcription endpoint works,
and if your STT and TTS do not have to share a GPU you can skip `switch_server.sh`
and the alternating driver entirely and just run the passes. The references to it
throughout stage 05 and 06 are there because it is what actually built the dataset.

Both servers are services in the same CrispASR `docker-compose.yml`:

- STT: `crispasr` (whisper replicas) + `crispasr-lb` (Caddy, host `:8002`)
- TTS: `voxtral-tts` (host `:8003`)

1. **Config** (nothing is hardcoded, so no path or username is committed). Either keep
   these in your shell env (and launch with `sudo -E`), or put them in a local
   `06_hotfixes/.env` file (`KEY=value` lines) that `driver.sh` sources:

   ```sh
   CRISPASR_COMPOSE=/abs/path/to/CrispASR/docker-compose.yml   # TODO: your path
   WDOC_WHISPER_ENDPOINT=...   # STT base URL (you already set these)
   WDOC_WHISPER_API_KEY=...    # token -> gitignore .env
   WDOC_WHISPER_MODEL=...      # must match how the dataset was transcribed
   ```

2. **Make the scripts executable:** `chmod +x switch_server.sh driver.sh`.

That is the whole setup: **no sudoers edit**. `driver.sh` runs as root (one password at
launch) so it can flip docker without a per-command sudo, and drops back to your normal
user (`$SUDO_USER`, or `TARGET_USER=<name>`) via `runuser` for each Python pass, so
every output / backup / draft file stays user-owned (matching the UID the containers
write as). It also sets the GPU power limit once at start (`nvidia-smi -pl`,
`GPU_POWER_LIMIT` default 350W).

## Run it

```sh
# from 06_hotfixes/, ideally inside tmux so it survives a disconnect:
sudo -E ./driver.sh
```

`-E` carries your session env (`WDOC_WHISPER_*`, `CRISPASR_COMPOSE`) into the root
process. Alternatively, put those vars in `06_hotfixes/.env` (sourced by the driver) and
just run `sudo ./driver.sh`. `.env` holds a token, so it must be gitignored.

Override any default via env, e.g. `N_IMPROV=8 CER_THRESHOLD=0.25 sudo -E ./driver.sh`.
The defaults mirror the manual runs (`--tts-voice fr_female`, `--stt-parallel 10`,
`--tts-parallel 2`, `--shuffle`, output dir `improved/`, bad-clip threshold
`CER_THRESHOLD=0.08`, tail threshold `TAIL_CER_THRESHOLD=0.12`, minimum gain to replace a
clip `MIN_IMPROVEMENT=0.02`).

### What differs between the draws of one redo

Two things, and only two. Draw N (from 0) gets seed `--start-seed + N` (43, 44, 45, ...)
and classifier-free guidance `CFG_ALPHA_START + CFG_ALPHA_STEP * N`, which the driver
defaults to 1, 1.5, 2, 2.5, 3 across five draws. The spread is the point: a clip the TTS
derailed on is unlikely to be rescued by four more draws at the setting that derailed
it, and since the winner is picked on CER, a draw that guidance pushed too hard simply
loses. It stays near the server's tuned value (it boots at 1.3) instead of sweeping far
out, so the draws land where the model is actually usable. `--speed-jitter` is NOT a
third lever (it is off by default and should stay there): vllm-omni applies `speed` as a
phase vocoder over the finished waveform, so it
time-stretches identical audio instead of drawing it differently.

Both knobs are sent as flat request fields, and both of them require the **forked**
vllm-omni that the CrispASR `voxtral-tts` image builds from
(`CrispASR/voxtral/vllm-omni`). Against a stock server they are silently inert: upstream
has no `cfg_alpha` field at all and drops it (pydantic `extra='ignore'`), and its `seed`
only reaches a sampler that Voxtral's stage 0 feeds forced logits, so the audio's
randomness (the flow-matching Gaussian) comes off the global torch RNG regardless. That
was the state of this stage for a while: five "identical" requests whose audio differed
only by RNG drift, no error anywhere. `tests/test_tts_request_knobs.py` pins the request
body so it cannot regress quietly. One server-side requirement comes with it: stage 0's
deploy YAML must keep a default `cfg_alpha` under `default_sampling_params.extra_args`,
or vllm-omni sets `has_sampling_extra_args=False` and drops every per-request extra.

The clips regenerated during that inert period had already been recorded with the knobs
the client ASKED for (`cfg_alpha` 1.3 to 1.7, `seed` 43 to 47), which is not what made
the audio. `oneoff_fix_preforked_knobs.py` is the one-shot migration that corrects them:
for every win drawn before the fork went live (`--cutoff`, which sits between the last
promotion of a pre-fork draw and the first post-fork one, NOT at the image build, because
the tts pass persists candidates that a later stt pass promotes) it sets `cfg_alpha` to
the server's startup value
(`--server-cfg-alpha`, 1.3 from `VOXTRAL_CFG_ALPHA`), sets `seed` to `null` (the draw was
unseeded, and since every draw of one redo went out with identical effective parameters,
which one won carries no information), and stamps `knobs_effective: false` on the audit
record so it still says why. It touches the audit file, the row's `cfg_alpha` column and
the row's `improvement` marker; it is a dry run unless given `--apply`, refuses to run
while a pass is alive, keeps a `.prefork.bak` copy, re-reads what it wrote, and is
idempotent. It has been run over 289 wins (264 pre-fork draws, then the 25 candidates the
interrupted run had already synthesized and the restarted driver only scored afterwards);
it should not be needed again.

### Two thresholds: whole clip and tail

A clip is flagged for regeneration when EITHER gate fires:

| Gate | Env | Default | Applies to |
|------|-----|---------|-----------|
| whole-clip CER | `CER_THRESHOLD` | 0.08 | every scored clip |
| tail CER | `TAIL_CER_THRESHOLD` | 0.12 | clips longer than 30 s only |

Both defaults are a SWEEP DEPTH, not a claim about where good audio ends. The corpus is
taken to convergence at one pair (no clip left above either gate), then the gates are
lowered and it is swept again, worst first. It has converged at 0.15 / 0.20 (505 clips
regenerated, 500 of them kept); 0.08 / 0.12 is the next sweep, ~1,550 clips and ~10.5 h of
audio, of which the first ~395 are what 0.10 / 0.15 alone would have queued.

The tail CER scores only the last ~30 seconds (the last `TAIL_CHARS` normalized characters
of each side, extrapolated from the corpus median pace of seconds per character). A long
clip that loops, derails or is cut short at the end can still pass the whole-clip gate
because its correct beginning outweighs the broken ending, and the tail gate is what
catches it. Its threshold is looser than the whole-clip one because the window boundary
cuts mid-sentence: the window is ~500 characters, the size of a whole PARHAF chunk rather
than a fifth of one. Measured over 23,639 long clips of the finished corpus, the tail
distribution is P95 0.060, P99 0.161, P99.5 0.191: 0.20 sat just past P99.5, and the
current 0.12 sits around P98 (390 long clips).

The two gates are NOT scaled versions of each other, and moving one is not a reason to
move the other. Counting the clips the tail gate catches on its own (whole-clip CER under
the gate, tail over it) and detecting the TTS derailments among them mechanically (a
25-character window the transcript repeats three or more times):

| tail band | clips caught by the tail gate alone | of them looping |
|---|---|---|
| 0.20 to 0.25 | 56 | 18 |
| 0.25 to 0.30 | 24 | 5 |
| over 0.30 | 2 | 0 |

That table was measured when the gate sat at 0.20, and it answers the question of whether
to RAISE it: raising does not buy precision, it discards the band where the real
derailments are. 0.25 would drop 18 of the 23 confirmed loops, and 0.30 turns the gate off
(2 clips, none of them looping). Set `TAIL_CER_THRESHOLD=-1` to disable it entirely.

Going the other way is a different trade. The whole-clip band below 0.15 is genuinely
mixed: sampling the diffs, ~43% of clips at 0.10-0.12 and ~57% at 0.12-0.15 carry a
dropped or invented run of words, but the rest is French agreement homophony
(`adenopathies axillaires reactionnelles` heard as the singular, same audio), Whisper's
spelling of rare medical terms, and notation neither side folds yet. So a lower gate
spends GPU time on clips that were never broken, and it also degrades the PICK: the loop
promotes the draw whose transcript best matches the label, so below the noise floor it
selects whichever draw best fits Whisper's spelling habits rather than the best audio.

`MIN_IMPROVEMENT` is what makes a low gate safe. A draw that clears both gates is always
promoted, but a draw that only comes closer has to beat the original by that margin
(0.02) to replace it. Without it (the script's own default is 0.0) any gain at all wins,
which at a gate of 0.08 means swapping audio on a 0.002 CER difference, i.e. on nothing.
Clips whose best draw does not clear that bar are marked `exhausted` and keep their
original audio, so a sweep that finds nothing real costs time and no quality.

### What the CER ignores on purpose

The audio is what is on trial, but the two sides of the comparison write it down
differently: the label spells out what a French dictation spells out, and Whisper
abbreviates whatever it hears. Before scoring, both sides are therefore folded to one
written form per spoken thing (`normalize_for_scoring`), on top of the older case /
accent / ligature / punctuation folding:

| Class | Same thing, both spellings | Canonical |
|---|---|---|
| units | `500 mg` = `500 milligrammes`, `4,2 mmol/L` = `4,2 millimoles par litre` | spoken |
| units, squared / cubed | `750 cm3` = `750 centimetres cubes`, `2 mm2` = `2 millimetres carres` | spoken |
| unit inflection | `par kilogramme` = `par kilogrammes` (every word, either side) | the table's form |
| rate tails | `15 milligrammes par kg` = `... par kilogramme` = `... par kilo`, `15 mg/kg` | spoken |
| `G/L` vs `g/L` | `189 G/L` = `189 giga par litre`, kept apart from `grammes par litre` | spoken, case-sensitive |
| titles | `M. Dupont` = `Monsieur Dupont`, `Dr` = `docteur` | spoken |
| percent | `96 %` = `96 pour cent` | one word |
| numbers | `trois` = `3`, `quatre-vingt-seize` = `96`, `stade IV` = `stade quatre` | digits |
| decimals | `0,5` = `zero virgule cinq` | `0 5`, separator dropped |
| thousands / leading zeros | `11 500` = `11500`, `05 janvier` = `5 janvier` | bare digits |
| ordinals | `premier` = `1er` = `1re`, `troisieme` = `3eme` = `3e`, plural too (`quatriemes` = `4emes`) | `<n>e` |
| spoken symbols | `1+` = `1 plus`, `T2*` = `T2 etoile` | spoken |
| ratios | `3/0` = `3 barre 0` = `3 0`, `120/80` = `120 sur 80` | `<n> <n>`, separator dropped |
| day offsets | `J2` = `J+2` = `J plus 2` | `j plus <n>` |
| compact times | `2h05` = `2 heures 5` = `2 heures 5 min` = `2 heures 5 minutes` | `2 heures 5` |

The unit table is not a second copy: it is `utils/voxtral_normalize.py`'s `UNIT_SPOKEN`,
read the other way round (that module spells the unit out for the TTS, the scorer folds
Whisper's abbreviation back onto it), plus the bare units voxtral pronounces correctly
and so does not list. Numbers canonicalize to digits because that direction needs only a
reader, not a French number speller.

Measured on 12,000 real scored clips when the first table landed: mean CER 0.031 to
0.013, clips over the then-0.15 gate 63 to 11, clips over the tail gate 15 to 1. A third
of clips score slightly WORSE, by a median of 0.0001: folding shortens the reference, so
the errors that are real weigh a little more, which is the point.

The later rows of the table were picked off a cost ranking rather than guessed: every
surviving diff over 60,000 scored clips, each pair weighted by the characters CER charges
for it. The top of that list was not exotic, it was `par kilogramme` against `par
kilogrammes` (7,128 characters), `cm3` against `centimetres cubes` (6,585), the decimal
comma (5,422), the accented `zero` that could never match the accent-folded text it was
compared against (3,832), and `G/L` read as grammes (3,718).

Every rule was then replayed over a 30,000-clip sample, comparing the stored CER with the
re-derived one and printing the clips that got WORSE, which is how the number separators
ended up being DROPPED rather than spelled. Spelling the decimal out (`3,0` to `3 virgule
0`) looked right and was the single worst rule in the batch: surgical reports are full of
suture gauges the label writes `3 0` and Whisper writes `3,0` or `3/0`, and only deletion
puts all three spellings on one string. The same replay demoted `J+7` to its short form
(Whisper mishears the letter as `G` often enough that the long form charged 7 characters
for a 1-character error) and put the letter guard on the unit rules (`M1M2`, the
metatarsals, was being read as square metres). Final effect: mean CER 0.01337 to 0.01272,
clips over the 0.10 gate 110 to 92, and the clips that score worse do so by a median of
0.00012, only 13 in 30,000 by more than 0.01. A fold that fires on one side but not the
other is worse than no fold at all, so a new rule is worth the corpus replay every time.

The rate tail came later, off the same kind of ranking run over the clips no redraw could
fix: the unit rules are anchored on a NUMBER, so `15 mg/kg` folds whole while the form the
two sides actually disagree on (`15 milligrammes par kg` against `par kilogramme`) only
folds at the head. `kg` alone was the most frequent surviving diff in those clips, and
`par kg` appears in 8,015 clips of the corpus. Folding a unit that sits after `par` is
restricted to multi-letter symbols on purpose: `par l` is the elided article far more often
than it is litres, and the punctuation strip turns `par l'aorte` into `par l aorte`.

That rule is also the one that most needed the corpus replay, and needed it over the WHOLE
corpus rather than a sample: at 30,000 clips it looked perfect (214 better, 0 worse), and
at 601,289 it was not. Two of its entries had to go:

- `kilo` fires only after `par`, never after a number. Number-anchored it ate the head of
  `87,5 kilo-unites par litre` (the IgE assay unit, kU/L) and stretched Whisper's
  mishearing of `3 culots globulaires` as `3 kilos` into a longer error.
- bare `mm` is barred entirely. It earns 4 clips in 600k (a rate tail is a dose
  denominator, not a length, and `par mm3` has its own entry) and it collides with a real
  one: Whisper writes the M-M-RVAXPRO vaccine as `MM-VAX-PRO`.

Final effect over all 601,289 scored clips: 8,707 better, 5 worse, 174 clips crossing under
the 0.08 gate, tails 384 better and 6 worse. The 5 survivors share one benign shape, a real
STT error sitting inside a rate the label states twice and the transcript once, so folding
both sides makes the difference between them 7 characters longer.

### Count charged characters, not clips whose CER moved

The last batch of rules (six imaging / physiology units, the clock time, plural ordinals)
was measured the usual way first, and the usual way said to throw it out: replayed over the
whole corpus it left 1432 clips with a lower CER and **1688 with a higher one**. Reading the
regressions clip by clip showed why that number is not what it looks like. CER is edit
distance over the length of the normalized reference, and a fold moves BOTH. Folding
`dixiemes` to `10e` shortens the reference by five characters, so every clip carrying an
unrelated error elsewhere scores a higher CER on exactly the same errors, and a fold that
lengthens the reference (`ms` to `millisecondes`) quietly flatters itself the same way.

So the question a replay must ask is not "did CER move" but **"did the fold stop the clip
being charged for something that was never an error"**, which is the raw edit distance on
the normalized strings, plus how many clips cross the gate. Measured that way, over the
601,289 scored clips:

| rule | charged less | charged more | net chars | crosses under 0.08 | crosses over |
|------|--------------|--------------|-----------|--------------------|--------------|
| the six units | 714 | 6 | -6,395 | 14 | 0 |
| plural ordinals | 141 | 2 | -1,533 | 3 | 1 |
| clock time | 570 | 50 | -4,900 | 6 | 0 |
| all three | 1,423 | 58 | -12,828 | 23 | 1 |

Same corpus, same rules, opposite conclusion: 1688 clips "worse" is 58 clips actually
charged more. All six unit regressions are one family, the label writing the symbol while
Whisper mangles it (`662 keV` heard as `kV`, `465 ms` as `mSq`, `0,2 milligray` as `mY`),
together about 25 characters against 6,395 saved, so they stay.

One rejected candidate is worth recording, because it looked obligatory: the label often
writes the `et` (`1 heure et 10 minutes`) where Whisper writes `1h10`, and the clock rule
does not match across it, so it strips the word from one side only. Adding an `et` rule
does fix those clips, and over the corpus it buys 35 and breaks 24 for no gate movement at
all. Not worth the rule. `1 heures 10 minutes` in a label whose transcript has no clock
form at all is the same shape and is the bulk of the clock rule's own 50.

What is deliberately NOT folded: French plural and gender endings (`antalgiques` vs
`antalgique`, `suivie` vs `suivi`). They are inaudible and they do show up in the diffs,
but folding them would also hide a TTS that dropped a word, and at 1-2 characters they
cost far less than the classes above. Bare `m` only counts as a title in front of a real
word, so a spelled-out `L.M.B.R.1` is not read as `L. Monsieur B. R. 1`.

Changing any of this changes every stored CER, so follow it with a `RESCORE_ONLY=1` pass
(see below): it re-derives the scores from the transcripts already on disk, without
transcribing anything again.

### The TTS frame cap (a "stopped for length" guard)

`/v1/audio/speech` returns raw audio with no `finish_reason`, so a generation that ran out
of tokens instead of reaching the end of the text still answers HTTP 200, with a clip that
just stops mid-sentence. The only observable is a duration pinned at the backend's frame
cap, so any draw landing within half a second of `MAX_AUDIO_SECONDS` (default 327.68 s, the
served 4096 frames at 12.5 Hz) is discarded rather than scored or promoted, and the clip
keeps its original. When every draw for a clip hits the cap the log says so explicitly: no
seed can fix it, the source text is simply too long to be spoken in full, and the fix is
re-chunking that text. Keep `MAX_AUDIO_SECONDS` in step with the server's actual
`VOXTRAL_MAX_TOKENS` (163.84 for the packaged 2048 frames, 327.68 for 4096);
`MAX_AUDIO_SECONDS=0` disables the check.

Note this deliberately does NOT re-flag the old clips stuck at 163.84 s, which were
generated back when the server capped at 2048 frames: they are truncated, but under the
current cap they are simply bad audio, and the CER / tail gates already catch them and can
now regenerate them in full. Stage 05 applies the same guard when it generates a clip in
the first place
(`--max-audio-seconds`), where a truncated clip is not written at all so a resume retries it.

### After a scoring-rule change: `RESCORE=1`

A clip that already carries a transcript is normally judged on its **stored** CER, which
was computed by whatever scoring rules were in force at the time. Change the scoring
normalization (fold ligatures, fold accents, ...) and those stored numbers go stale, so a
clip that is bad under the new rules can sit there looking fine and never be queued.

`RESCORE=1` re-derives every score from the transcript already on the row, then judges it
under the current rules:

```bash
sudo -E RESCORE=1 MODE=stt CATEGORY="dictionary drugs" ./driver.sh
```

It costs **no STT call**: the pass reuses stored transcripts (only never-transcribed clips
go to the server), so this is a file read plus a CER computation per row. Clips whose audio
file is missing are left alone, so a good stored transcript is never replaced by nothing.

It is a one-time migration, and the driver treats it as one: only the **first `stt` pass of
each dataset** gets the flag. Repeating it would recompute already-current scores, and a
pass that re-examines every row always reports work done, which would stop a single-mode
`MODE=stt` loop from ever reaching its "ran dry" exit 11. `MODE=tts` runs no `stt` pass, so
`RESCORE` does not apply there and the driver says so.

Every row the pass reports carries a marker saying which half it took, so a run that is
quietly transcribing when you expected it to reuse stored text is visible at a glance:

```
000123_0001_pneumopathie  RESCORED cer=0.041                     <- stored transcript, no call
000124_0000_bronchiolite           cer=0.062 stt=1.8s rtf=0.11   <- transcribed now
```

and the end-of-run summary counts both (`N rescored from stored transcripts (no STT call),
M transcribed`), plus how many of the rescored rows now read bad under the current rules.

**It also re-judges the regeneration queue**, which is the half that is easy to forget: a
row already flagged `pending_tts` carries the `orig_cer` a candidate will later have to
beat, and a normal `stt` pass hands those rows straight to the `tts` pass without looking
at them, so nothing else ever revisits them. Under a rescore they are re-derived from the
stored transcript: one that now reads clean drops its marker and leaves the queue (no
regeneration spent on a clip that was never broken), one that stays bad keeps it with a
refreshed score, so the comparison a candidate has to win is against today's number and not
a stale inflated one that any draw would beat. The summary reports it:

```
full.jsonl: regeneration queue re-judged, 812 clip(s) left it (they read clean now, no
regeneration needed), 305 stayed with a refreshed score
```

Rows already **resolved** (`improved` / `exhausted`) are the exception: they keep the score
they were resolved with, since nothing will regenerate them again. Re-deriving the
`improved` ones means calling the script directly with `--force` (the driver does not
expose it as an env var); for the `exhausted` ones, see the next section.

### Giving the clips that gave up another go: `RETRY_EXHAUSTED=1`

`exhausted` means the clip was flagged bad and **no draw was good enough**, so its ORIGINAL
audio is what ships. Those are the known-weak clips of the release, and the marker is
final: nothing revisits them, which also freezes their score under the rules of the run
that gave up. A scoring change therefore cannot reach the rows most likely to benefit from
it, and neither can a new seed or `cfg_alpha` ladder.

`RETRY_EXHAUSTED=1` reopens exactly those rows, as if they carried no marker:

```bash
# Re-judge them offline first (no GPU, no server call), then redraw for what is left.
sudo -E RESCORE_ONLY=1 RETRY_EXHAUSTED=1 ./driver.sh
sudo -E START_SEED=48 ./driver.sh
```

`START_SEED` is not optional here. A clip's attempt count restarts when it is requeued,
and draw N is seed `START_SEED + N`, so a retry at the default 43 asks the server for the
exact five draws it already rejected: same seed, same `cfg_alpha`, same audio. Move it past
the range the previous run used (43..47 for a five-draw redo) and the retry actually draws
something new.

The first pass re-derives each one's score from its stored transcript. A clip that now
reads clean **loses its marker** (nothing left to fix, no regeneration spent), and one that
still reads bad goes back to `pending_tts` with a refreshed `orig_cer`, which is the number
a new candidate then has to beat. The second command is a plain alternating run: it drafts
for that queue and judges the results, ending in `improved` or a fresh `exhausted`.

It is deliberately narrower than `--force`: the clips that passed and the ones already
improved are left alone, so a retry costs the weak tail (173 clips after the 0.08 / 0.12
sweep) instead of re-reading the corpus. Like `RESCORE` it rides the **first `stt` pass of
each dataset only**, and here that is not just about saving work: the retry ends by writing
a fresh `exhausted` marker for whatever still fails, so a flag left on for every pass would
reopen its own output and the alternating loop would never converge.

### Rescoring on its own first: `RESCORE_ONLY=1`

`RESCORE=1` mixes the two: it rescores what has a transcript and transcribes what does not,
in one long pass. To separate them, and see the damage a rule change did before spending
any GPU time on it:

```bash
sudo -E RESCORE_ONLY=1 CATEGORY="dictionary drugs" ./driver.sh
```

That pass reads stored transcripts, re-derives their scores and stops. It flags nothing:
clips that now read bad are reported (`N clip(s) now read bad under the current rules`) and
left for the next normal `stt` pass, which triple-checks them before queueing.
Never-transcribed clips are left untouched for a normal run too. It makes **no server call
at all**, which is why the driver skips the docker switch entirely: the GPU is left as it
is, so this can run while the TTS server is loaded.

It cannot loop (re-examining the same rows always counts as work done, so the loop would
never end): the driver runs exactly **one** pass per dataset, then moves to the next and
exits. `MODE=tts` is rejected outright, since a `tts` pass scores nothing.

### Several datasets in a row

`INPUT` is a **space-separated list of manifests**, processed one after the other: the
alternating loop runs to convergence on the first, then starts over on the next. The
default chains both manifests the release ships:

| # | Manifest | Output dir | Audio root |
|---|---|---|---|
| 1 | `../99_hf_release/data/NeMO_files/full.jsonl` (dictionary + drugs + PARHAF) | `improved/` | `.../NeMO_files` |
| 2 | `../99_hf_release/data/NeMO_files/PARROT/full.jsonl` (1,549 radiology clips) | `improved_parrot/` | `.../NeMO_files/PARROT` |

PARROT lives in its own manifest because it ships under its own licence (CC BY-NC-SA 4.0,
evaluation-only) as a separate HF subset, so it is never merged into the main corpus.

Both files are named `full.jsonl` and the pass names its output after the manifest stem
(`<stem>.stt.jsonl`), so the first dataset keeps `${OUTPUT}` unchanged (an in-flight run is
untouched) and every later one gets `${OUTPUT}_<parent directory, lowercased>`. The driver
refuses to start if two manifests would still land on the same output file, and each
dataset's audio root defaults to the manifest's own directory (set `AUDIO_ROOT` to pin one
root for all of them). Pass a single path in `INPUT` to work on just one dataset.

Note that `CATEGORY` applies to every dataset in the list: with `CATEGORY=parhaf` the
PARROT pass converges immediately, since none of its rows match.

### When the manifest is rebuilt under a running output

The `.stt.jsonl` is a resume file, not a second source of truth: each pass rebuilds it
from the manifest it is given, so a row the new manifest no longer lists is simply gone
from the next write. Re-chunking a source (which renumbers the clips) therefore drops the
old rows on its own, with no cleanup step.

A row resumes on its audio path **and** its text. Chunk filenames are only unique within
one chunking, so a re-chunk can reuse a filename for different text; the stored
transcription, CER and improvement state then describe audio that no longer exists.
Inheriting them would score the new clip against the old label, and an inherited
`exhausted` status would freeze it for good, so a changed text drops the stored work and
the pass logs how many rows that hit.

Two things do NOT follow the manifest, by design: the `.improved.jsonl` audit keeps every
recorded win (it tracks audio that really was overwritten, whatever the manifest says
today), and a `<clip>.flac.bak` stays next to its clip as the pristine original. Both
survive a re-chunk on purpose.

### Alternative: driver as your user + NOPASSWD switch

If you would rather not run the driver as root, run it as yourself and grant a narrow
NOPASSWD entry for just `switch_server.sh` (so each server flip needs no password):
see `switch_server.sudoers.example` (two TODOs: your username and the absolute
`switch_server.sh` path), install with `sudo visudo -f /etc/sudoers.d/crispasr-switch`.
You would then change the driver's `"${SWITCH}" ...` call to `sudo "${SWITCH}" ...` and
drop the run-as-root / `runuser` wrapper.

## Listening to what the numbers flagged: `03_collect_suspicious.py`

Every gate here is a proxy. A high CER usually means bad audio, but it can also mean
Whisper mishearing a rare drug name, and a clip nobody ever plays can sit in the release
either way. This script gathers the audio behind each reason into one local folder, with
the text beside it:

```bash
uv run 03_collect_suspicious.py --dry-run          # how many, how big, copies nothing
uv run 03_collect_suspicious.py                    # fill ./suspicious_audio/
uv run 03_collect_suspicious.py --reason exhausted --limit 50
uv run 03_collect_suspicious.py --match 'facteur V'   # hear one normalization rule
```

The reasons are the pipeline's own: `worst-cer`, `worst-tail`, `pending-tts`,
`exhausted` (regeneration gave up, so this audio ships as it is: the pile that most
deserves ears), `ceiling` (durations piled at the top of the distribution, the shape a TTS
that stopped on its token limit leaves behind, using the release statistics' own
`utils/nemo_manifest.duration_ceiling`), `stt-error`, and `match` (a plain listening
sample, not a defect). A clip is copied once, under the first reason that claimed it, and
named `<reason>_<cer>_<category>_<clip>.flac` so `ls` sorts by badness. `index.md` lists
each one with what it should say, what the TTS was given, and what Whisper heard;
`index.jsonl` is the same, machine-readable. `--symlink` skips the disk cost (but then the
audio drive must stay mounted to play them).

**Rescore before you listen.** The scores come from the jsonl as it stands, so a stale
list is mostly false positives: after the unit / title / number folding landed, 855 of the
1,152 clips over the 0.15 gate scored clean, and 32 of the 35 tail flags did too. Run
`RESCORE_ONLY=1` first, then collect, or you will spend an evening listening to clips that
were never broken.

## Manual equivalents

```sh
sudo ./switch_server.sh stt        # bring up whisper + LB, wait until healthy
uv run 01_recursive_improvement.py --mode stt  ...   # score + promote; exits 10 when done
sudo ./switch_server.sh tts        # bring up Voxtral, wait until healthy
uv run 01_recursive_improvement.py --mode tts  ...   # generate candidates
```

## TODOs to fill in

- `CRISPASR_COMPOSE`: absolute path to your CrispASR `docker-compose.yml` (env or `.env`).
- `.env`: gitignore it (it holds `WDOC_WHISPER_API_KEY`).
- `CRISPASR_STT_SCALE` (optional, default 6) if you want a different whisper replica count.
- `GPU_POWER_LIMIT` (optional, default 350) watts for `nvidia-smi -pl` at start.
- Only if you use the alternative driver-as-user path: fill the two TODOs in
  `switch_server.sudoers.example`.

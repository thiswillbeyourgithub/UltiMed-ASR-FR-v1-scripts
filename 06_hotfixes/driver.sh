#!/usr/bin/env bash
# Unattended driver for the alternating single-GPU improvement flow. Loops:
#
#   switch to STT server -> run --mode stt (transcribe + score + promote candidates)
#   switch to TTS server -> run --mode tts (generate candidates for flagged clips)
#
# repeating until the --mode stt pass reports the dataset has converged (exit code 10:
# no clip anywhere still needs either pass), then moving on to the next manifest in INPUT
# (by default the main corpus first, then the separately licensed PARROT manifest).
#
# NO sudoers edit needed: launch this ONCE with sudo so it can flip docker without a
# per-command password, and it drops back to your normal user for each Python pass (so
# every output / backup / draft file stays user-owned, matching the UID the TTS/STT
# containers write as). Run under tmux / a systemd unit so it survives a disconnect; a
# single long-lived loop (not cron) so two passes can never overlap on the GPU.
#
#     sudo -E ./driver.sh
#
# -E carries your session env (WDOC_WHISPER_*, CRISPASR_COMPOSE) into the root process;
# alternatively (or in addition), put those in 06_hotfixes/.env (KEY=value lines), which
# is sourced below, and run plain `sudo ./driver.sh`. .env holds a token: gitignore it.
#
# Required env:
#   CRISPASR_COMPOSE       your CrispASR docker-compose.yml
#   WDOC_WHISPER_ENDPOINT  STT base URL (/v1/audio/transcriptions is appended)
#   WDOC_WHISPER_API_KEY   STT bearer token
#   WDOC_WHISPER_MODEL     STT model name (match how the dataset was transcribed so
#                          resume reuses the stored transcripts)
# Optional overrides (defaults match the manual runs):
#   TARGET_USER (defaults to whoever ran sudo) OUTPUT AUDIO_ROOT N_IMPROV
#   CER_THRESHOLD STT_PARALLEL TTS_PARALLEL TTS_VOICE GPU_POWER_LIMIT
#   CFG_ALPHA_START / CFG_ALPHA_STEP  TTS guidance ladder across the draws of one redo:
#                       draw N uses START + STEP * N, so the defaults (1.0 / 0.5) walk
#                       1, 1.5, 2, 2.5, 3 over five draws. A negative START omits the
#                       knob and leaves the server's startup value on every draw.
#   TAIL_CER_THRESHOLD  CER gate on the last ~30s of clips longer than 30s (default 0.12,
#                       looser than CER_THRESHOLD because the tail is a short window;
#                       negative disables the tail gate)
#   MIN_IMPROVEMENT     how much better a draw must be to replace the original when it
#                       does not clear the gates (default 0.02, see the note below)
#   RESCORE=1           re-derive every stored CER under the CURRENT scoring rules before
#                       deciding what is bad, for after a normalization change (costs no
#                       STT call: stored transcripts are reused). One-time migration, so
#                       only the FIRST stt pass of each dataset gets it, see below.
#   RESCORE_ONLY=1      do ONLY that migration and stop: one pass per dataset that reads
#                       stored transcripts, re-derives their scores and flags what now
#                       reads bad, without transcribing anything. Never-transcribed clips
#                       are left for a normal pass. Makes no server call at all, so the
#                       GPU is left exactly as it is (no docker switch) and it is safe to
#                       run while the TTS server is loaded. Implies RESCORE.
#   RETRY_EXHAUSTED=1   give the clips a previous run gave up on (`exhausted`) another
#                       chance: they are re-judged, cleared if they now read clean, and
#                       put back in the tts queue if they do not. Also one-shot, and for
#                       a stronger reason than RESCORE: see the block below. The clips
#                       that passed and the ones already improved are left alone, unlike
#                       --force. Pair it with START_SEED past the previous run's range,
#                       or the redraws reproduce the draws that already lost.
#   START_SEED          seed of the first draw of a redo, +1 per draw (default 43, so a
#                       five-draw redo uses 43..47). The only reason to move it is a
#                       retry over clips that were already drawn for.
#   MAX_AUDIO_SECONDS   TTS frame cap in seconds (default 327.68 = the served 4096 frames
#                       at 12.5 Hz). A draw landing on the cap stopped for LENGTH, not at
#                       the end of the text, so it is discarded instead of promoted. Must
#                       match the server's VOXTRAL_MAX_TOKENS; 0 disables the check.
#   INPUT     space-separated list of NeMo manifests, processed one after the other
#             (each to convergence). Defaults to the main corpus manifest followed by
#             the PARROT one, which is a separate file because PARROT ships under its
#             own licence (CC BY-NC-SA 4.0, eval-only) as its own HF subset. The first
#             manifest writes to ${OUTPUT}, each later one to ${OUTPUT}_<parent dir>
#             (e.g. improved_parrot), since both are named full.jsonl and the output
#             file is named after the manifest stem. Set INPUT to a single path to work
#             on just one dataset.
#   MODE      alternate|stt|tts (default alternate). `alternate` is the normal flow
#             above. `stt` runs ONLY the scoring passes and never touches the TTS
#             server, which is what you want when the TTS side is down or being
#             reconfigured: clips found bad are flagged pending_tts and left for a
#             later alternate run, and the loop stops once scoring has run dry
#             (pass exit 11) instead of spinning on work only a tts pass can clear.
#             `tts` is the mirror image. Example, an STT-only night over the two
#             finished subsets:
#                 sudo -E MODE=stt CATEGORY="dictionary drugs" ./driver.sh
#   CATEGORY  space-separated list of categories to restrict work to (e.g.
#             CATEGORY=parhaf, or CATEGORY="parhaf parrot"). Rows in other
#             categories are passed through unchanged and convergence is judged
#             over the selected categories only, so the loop stops when just
#             those subsets are done. Empty (default) = the whole dataset.
#
# This file was written with Claude Code.
set -uo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "driver.sh runs as root so it can flip docker without per-command sudo." >&2
  echo "launch it with:  sudo -E ./driver.sh   (from 06_hotfixes/)" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"   # so the default ../99_hf_release/... relative paths resolve as they do manually
SWITCH="${SWITCH_SERVER:-${HERE}/switch_server.sh}"

# Local config: load 06_hotfixes/.env if present (KEY=value lines, e.g. CRISPASR_COMPOSE
# and the WDOC_WHISPER_* paths/token). Auto-exported (set -a) so child processes
# (switch_server.sh) inherit them too. This lets you run plain `sudo ./driver.sh`
# without -E. .env holds a token, so it MUST be gitignored (see README).
if [[ -f "${HERE}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${HERE}/.env"
  set +a
fi

# Non-root user that runs the Python passes. Defaults to whoever ran sudo, so nothing
# is hardcoded; override with TARGET_USER=<name> for a dedicated account.
TARGET_USER="${TARGET_USER:-${SUDO_USER:-}}"
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == root ]]; then
  echo "driver.sh: cannot tell which non-root user to run the Python as." >&2
  echo "launch via 'sudo -E ./driver.sh' (sets SUDO_USER) or export TARGET_USER=<you>." >&2
  exit 1
fi

: "${CRISPASR_COMPOSE:?set CRISPASR_COMPOSE (sudo -E to carry it, or put it in improve.env)}"
: "${WDOC_WHISPER_ENDPOINT:?set WDOC_WHISPER_ENDPOINT}"

TARGET_HOME="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
# Resolve uv. It lives on ${TARGET_USER}'s PATH (~/.local/bin), not root's, and a
# NON-interactive login shell may not source it (under zsh the PATH line is in
# ~/.zshrc, which `runuser -l` does not read), so probe the home dir directly first.
# Override by setting UV_BIN in .env.
resolve_uv() {
  local c
  if [[ -n "${UV_BIN:-}" && -x "${UV_BIN}" ]]; then printf '%s\n' "${UV_BIN}"; return 0; fi
  for c in "${TARGET_HOME}/.local/bin/uv" "${TARGET_HOME}/.cargo/bin/uv" \
           /usr/local/bin/uv /usr/bin/uv; do
    [[ -x "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  done
  # Last resort: ask the user's login shell (works if uv is on a login-sourced PATH).
  c="$(runuser -l "${TARGET_USER}" -c 'command -v uv' 2>/dev/null || true)"
  [[ -n "${c}" ]] && { printf '%s\n' "${c}"; return 0; }
  return 1
}
UV_BIN="$(resolve_uv || true)"
if [[ -z "${UV_BIN}" ]]; then
  echo "driver.sh: could not find 'uv'. As ${TARGET_USER}, run 'command -v uv' and put" >&2
  echo "  UV_BIN=<that absolute path> in 06_hotfixes/.env" >&2
  exit 1
fi
# PATH forwarded to the passes so uv (and anything it shells out to) resolves.
USER_PATH="${TARGET_HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
echo "driver.sh: uv=${UV_BIN}, running passes as ${TARGET_USER}"

# Raise the GPU power limit once at start (root-only; best-effort, never fatal).
GPU_POWER_LIMIT="${GPU_POWER_LIMIT:-350}"
nvidia-smi -pl "${GPU_POWER_LIMIT}" \
  || echo "driver: could not set GPU power limit to ${GPU_POWER_LIMIT}W (continuing)" >&2

# INPUT may list SEVERAL manifests (space separated). They are processed one after the
# other: the alternating loop runs to convergence on the first, then moves to the next.
# The default chains the main corpus (dictionary + drugs + PARHAF) and then PARROT, which
# lives in its own manifest because it ships under a different licence (CC BY-NC-SA 4.0,
# eval-only) as a separate HF subset.
INPUT="${INPUT:-../99_hf_release/data/NeMO_files/full.jsonl ../99_hf_release/data/NeMO_files/PARROT/full.jsonl}"
OUTPUT="${OUTPUT:-improved}"
# Empty (the default) = derive each manifest's audio root from its own directory, which is
# what every manifest here needs (their audio_filepath values are relative to themselves:
# ../PARHAF/x.flac from NeMO_files, ../../PARROT/x.flac from NeMO_files/PARROT). Set it to
# pin ONE root for every manifest instead.
AUDIO_ROOT="${AUDIO_ROOT:-}"
N_IMPROV="${N_IMPROV:-5}"
# Sweep depth, not a permanent verdict. The corpus converged at 0.15 / 0.20 (no clip
# left above either gate), so these are the SECOND sweep: they queue the next ~1550
# clips, about 10.5 h of audio, which is roughly one night of alternating passes. Raise
# them back if you only want the worst clips.
CER_THRESHOLD="${CER_THRESHOLD:-0.08}"
TAIL_CER_THRESHOLD="${TAIL_CER_THRESHOLD:-0.12}"
# How much better a draw must be to REPLACE the original when it does not clear the
# gates. It matters at a low gate: a clip flagged at 0.09 whose best draw comes in at
# 0.088 has not been fixed, it has been fitted to the judge, and the CER of a clean clip
# moves by that much on Whisper's own spelling choices. 0 (the script's default) accepts
# any gain at all, which is the right call only when the gate is high enough that
# everything flagged is genuinely broken.
MIN_IMPROVEMENT="${MIN_IMPROVEMENT:-0.02}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-327.68}"
STT_PARALLEL="${STT_PARALLEL:-10}"
TTS_PARALLEL="${TTS_PARALLEL:-2}"
TTS_VOICE="${TTS_VOICE:-fr_female}"
# Classifier-free guidance per draw: draw N gets START + STEP * N (N from 0), so the
# defaults walk 1, 1.5, 2, 2.5, 3 across the five draws of a redo. The point is spread:
# a clip the TTS derailed on at its tuned setting is unlikely to be rescued by another
# draw at the same setting, and the winner is picked on CER anyway, so a draw that
# guidance pushed too hard simply loses. The range stays near the tuned value (the
# server boots at 1.3) rather than reaching for 5, on the theory that a wide sweep
# spends most of its draws where the model is not usable. Needs the forked vllm-omni
# (upstream drops the field silently); the pass sends it flat, see tts_synthesize.
CFG_ALPHA_START="${CFG_ALPHA_START:-1.0}"
CFG_ALPHA_STEP="${CFG_ALPHA_STEP:-0.5}"
# The other half of what makes one draw differ from the next: draw N uses seed START + N.
# A clip's attempt count restarts when it is requeued, so a retry over already-drawn clips
# (RETRY_EXHAUSTED=1) with this left at its default asks the server for the SAME five
# draws it already rejected. Move it past the range the previous run used (43..47 for a
# five-draw redo) or the retry only spends GPU time reproducing known losers.
START_SEED="${START_SEED:-43}"

# Which passes to run. `alternate` is the stt <-> tts flow; `stt` / `tts` run that one
# pass in a loop and never bring the other server up, so a half-configured GPU (or a
# TTS backend mid-rebuild) does not block the half that works.
MODE="${MODE:-alternate}"
case "${MODE}" in
  alternate|stt|tts) ;;
  *) echo "driver.sh: MODE must be alternate, stt or tts (got '${MODE}')" >&2; exit 1 ;;
esac
if [[ "${MODE}" != alternate ]]; then
  echo "driver: ${MODE}-only mode, the ${MODE} server is the only one this run touches"
fi

# RESCORE=1 re-derives every stored score from the transcript already on the row, so a
# clip whose CER was computed under older normalization rules is judged by the current
# ones. It is a one-time migration and is applied to the FIRST stt pass of each dataset
# only, for two reasons: the second pass would recompute scores that are already current,
# and a pass that re-examines every row always reports work done, which would keep a
# single-mode (MODE=stt) loop from ever reaching its "ran dry" stop (exit 11).
case "${RESCORE:-0}" in
  1|true|TRUE|yes|YES|y|Y) RESCORE_ON=1 ;;
  0|""|false|FALSE|no|NO|n|N) RESCORE_ON=0 ;;
  *) echo "driver.sh: RESCORE must be 0 or 1 (got '${RESCORE}')" >&2; exit 1 ;;
esac
if (( RESCORE_ON )); then
  echo "driver: RESCORE=1, the first stt pass of each dataset re-derives every stored score"
  if [[ "${MODE}" == tts ]]; then
    echo "driver: ...but MODE=tts runs no stt pass, so RESCORE will not apply" >&2
  fi
fi

# RESCORE_ONLY=1 does that migration and NOTHING else: one offline pass per dataset, then
# the next dataset. It cannot loop (every pass would re-examine the same rows and always
# report work done), and it makes no server call at all, so the docker switch is skipped
# and the pass can run while the TTS server is up.
case "${RESCORE_ONLY:-0}" in
  1|true|TRUE|yes|YES|y|Y) RESCORE_ONLY_ON=1 ;;
  0|""|false|FALSE|no|NO|n|N) RESCORE_ONLY_ON=0 ;;
  *) echo "driver.sh: RESCORE_ONLY must be 0 or 1 (got '${RESCORE_ONLY}')" >&2; exit 1 ;;
esac
NO_SWITCH="${RESCORE_ONLY_ON}"
if (( RESCORE_ONLY_ON )); then
  if [[ "${MODE}" == tts ]]; then
    echo "driver.sh: RESCORE_ONLY=1 scores stored transcripts, MODE=tts scores nothing." >&2
    exit 1
  fi
  RESCORE_ON=1
  echo "driver: RESCORE_ONLY=1, one offline scoring pass per dataset then done"
  echo "driver: no STT/TTS call is made, the GPU and the running container are left alone"
fi

# RETRY_EXHAUSTED=1 reopens the clips a previous run gave up on (`exhausted`: flagged
# bad, no draw good enough). They are the only resolved rows worth revisiting, since they
# still ship their ORIGINAL, known-weak audio, and nothing else would ever look at them
# again. Like RESCORE it rides on the FIRST stt pass of each dataset only, and for a
# sharper reason than saving work: the retry cycle ends in a fresh `exhausted` marker for
# whatever still fails, so a flag left on for every pass would reopen its own output and
# the alternating loop would never converge.
#
# Pairs naturally with RESCORE_ONLY=1: the offline pass re-judges those clips under the
# current rules, drops the marker on the ones that now read clean, and puts the rest back
# in the tts queue for a plain `./driver.sh` run to redraw.
case "${RETRY_EXHAUSTED:-0}" in
  1|true|TRUE|yes|YES|y|Y) RETRY_EXHAUSTED_ON=1 ;;
  0|""|false|FALSE|no|NO|n|N) RETRY_EXHAUSTED_ON=0 ;;
  *) echo "driver.sh: RETRY_EXHAUSTED must be 0 or 1 (got '${RETRY_EXHAUSTED}')" >&2; exit 1 ;;
esac
if (( RETRY_EXHAUSTED_ON )); then
  echo "driver: RETRY_EXHAUSTED=1, the first stt pass of each dataset reopens the clips that gave up"
  if [[ "${MODE}" == tts ]]; then
    echo "driver: ...but MODE=tts runs no stt pass, so RETRY_EXHAUSTED will not apply" >&2
  fi
fi

# Optional category restriction (space-separated, e.g. CATEGORY="parhaf parrot").
# Expanded once into repeated --category flags so both passes share the same filter;
# empty (default) processes the whole dataset. Unquoted expansion below is intentional
# word-splitting to turn the list into separate flags.
CATEGORY="${CATEGORY:-}"
CATEGORY_ARGS=()
for _cat in ${CATEGORY}; do
  CATEGORY_ARGS+=(--category "${_cat}")
done
if (( ${#CATEGORY_ARGS[@]} )); then
  echo "driver: restricting work to categories: ${CATEGORY}"
fi

# Expand the INPUT list into one (manifest, output dir, audio root) triple per dataset.
# The output FILE is named after the manifest stem (<stem>.stt.jsonl), so two manifests
# both called full.jsonl would fight over one resume file: the first dataset keeps
# ${OUTPUT} as-is (so an in-flight run is untouched) and every later one gets
# ${OUTPUT}_<its parent directory, lowercased>, e.g. improved_parrot. Unquoted expansion
# below is intentional word-splitting.
MANIFESTS=(); OUT_DIRS=(); AUDIO_ROOTS=()
for _m in ${INPUT}; do
  if [[ ! -f "${_m}" ]]; then
    echo "driver.sh: manifest not found: ${_m} (run from 06_hotfixes/, or fix INPUT)" >&2
    exit 1
  fi
  MANIFESTS+=("${_m}")
  if (( ${#MANIFESTS[@]} == 1 )); then
    OUT_DIRS+=("${OUTPUT}")
  else
    _label="$(basename "$(dirname "${_m}")" | tr 'A-Z' 'a-z')"
    OUT_DIRS+=("${OUTPUT}_${_label}")
  fi
  AUDIO_ROOTS+=("${AUDIO_ROOT:-$(dirname "${_m}")}")
done

# Guard: two datasets writing the same <output>/<stem>.stt.jsonl would corrupt each
# other's resume state, so refuse up front instead of discovering it mid-run.
for _i in "${!MANIFESTS[@]}"; do
  for _j in "${!MANIFESTS[@]}"; do
    (( _j <= _i )) && continue
    _si="$(basename "${MANIFESTS[$_i]}" .jsonl)"; _sj="$(basename "${MANIFESTS[$_j]}" .jsonl)"
    if [[ "${OUT_DIRS[$_i]}/${_si}" == "${OUT_DIRS[$_j]}/${_sj}" ]]; then
      echo "driver.sh: ${MANIFESTS[$_i]} and ${MANIFESTS[$_j]} both write" >&2
      echo "  ${OUT_DIRS[$_i]}/${_si}.stt.jsonl -- give one of them a different OUTPUT." >&2
      exit 1
    fi
  done
done

echo "driver: ${#MANIFESTS[@]} dataset(s) queued:"
for _i in "${!MANIFESTS[@]}"; do
  echo "  $((_i + 1)). ${MANIFESTS[$_i]} -> ${OUT_DIRS[$_i]}/ (audio root ${AUDIO_ROOTS[$_i]})"
done

run_pass() {  # $1 = stt|tts ; runs on the CUR_* dataset; returns the pass's exit code
  local mode="$1"
  if (( NO_SWITCH )); then
    # This pass talks to no server, so flipping the GPU would only cost a container restart.
    echo "=== $(date '+%F %T') ${mode} pass needs no server, leaving docker as it is ==="
  else
    echo "=== $(date '+%F %T') switching to ${mode} server (root) ==="
    if ! "${SWITCH}" "${mode}" "${CRISPASR_COMPOSE}"; then
      echo "!!! switch to ${mode} failed" >&2
      return 70
    fi
  fi
  echo "=== $(date '+%F %T') running --mode ${mode} pass as ${TARGET_USER} ==="
  # --rescore only ever rides on an stt pass (a tts pass scores nothing), and only on the
  # first one of this dataset: consumed here so a retry loop does not repeat it.
  local extra=()
  if [[ "${mode}" == stt ]] && (( RESCORE_PENDING )); then
    if (( RESCORE_ONLY_ON )); then
      extra+=(--rescore-only)
      echo "driver: offline pass, stored transcripts rescored, nothing transcribed (--rescore-only)"
    else
      extra+=(--rescore)
      echo "driver: re-deriving every stored score under the current rules (--rescore)"
    fi
    RESCORE_PENDING=0
  fi
  # Same shape, same reason, plus one of its own: reopening the exhausted clips on every
  # pass would reopen the markers this very retry writes (see the RETRY_EXHAUSTED block).
  if [[ "${mode}" == stt ]] && (( RETRY_EXHAUSTED_PENDING )); then
    extra+=(--retry-exhausted)
    echo "driver: reopening the clips a previous run gave up on (--retry-exhausted)"
    RETRY_EXHAUSTED_PENDING=0
  fi
  # Drop to the normal user for the Python (files stay user-owned). runuser without -l
  # keeps CWD (=HERE), so the relative --input/--audio-root resolve as they do manually;
  # we forward the STT config explicitly and call uv by absolute path.
  # `nice -n 40` puts the pass, and everything it spawns, at 19: the lowest priority.
  # This is an unattended job that runs for days and must never be the reason an
  # interactive session stutters, and it costs nothing, a pass waits on the GPU server
  # far more than on the CPU. 40 rather than 19 because nice takes an INCREMENT, not a
  # level: sudo can hand the driver a negative priority (-5 observed here), where +19
  # would land at 14. The result is clamped to 19, so an oversized step pins the floor
  # wherever it starts from. Placed after `env` so it resolves on USER_PATH.
  runuser -u "${TARGET_USER}" -- \
    env HOME="${TARGET_HOME}" \
        PATH="${USER_PATH}" \
        WDOC_WHISPER_ENDPOINT="${WDOC_WHISPER_ENDPOINT}" \
        WDOC_WHISPER_API_KEY="${WDOC_WHISPER_API_KEY:-}" \
        WDOC_WHISPER_MODEL="${WDOC_WHISPER_MODEL:-}" \
        nice -n 40 \
        "${UV_BIN}" run "${HERE}/01_recursive_improvement.py" \
        --mode "${mode}" \
        --input "${CUR_INPUT}" --output "${CUR_OUTPUT}" --audio-root "${CUR_AUDIO_ROOT}" \
        --stt-endpoint "${WDOC_WHISPER_ENDPOINT%/}/v1/audio/transcriptions" \
        --stt-api-token "${WDOC_WHISPER_API_KEY:-}" --stt-model "${WDOC_WHISPER_MODEL:-}" \
        --tts-voice "${TTS_VOICE}" \
        --cfg-alpha-start "${CFG_ALPHA_START}" --cfg-alpha-step "${CFG_ALPHA_STEP}" \
        --start-seed "${START_SEED}" \
        --n-improv "${N_IMPROV}" --cer-threshold "${CER_THRESHOLD}" \
        --tail-cer-threshold "${TAIL_CER_THRESHOLD}" \
        --min-improvement "${MIN_IMPROVEMENT}" \
        --max-audio-seconds "${MAX_AUDIO_SECONDS}" \
        --stt-parallel "${STT_PARALLEL}" --tts-parallel "${TTS_PARALLEL}" \
        ${CATEGORY_ARGS[@]+"${CATEGORY_ARGS[@]}"} \
        ${extra[@]+"${extra[@]}"} \
        --shuffle
}

# One dataset at a time, each run to convergence before the next one starts, so the GPU
# only ever hosts the server the current pass needs.
for i in "${!MANIFESTS[@]}"; do
  CUR_INPUT="${MANIFESTS[$i]}"
  CUR_OUTPUT="${OUT_DIRS[$i]}"
  CUR_AUDIO_ROOT="${AUDIO_ROOTS[$i]}"
  # Per dataset: each manifest carries its own stored scores, so each one needs the
  # rescore pass once.
  RESCORE_PENDING="${RESCORE_ON}"
  # Likewise: each manifest holds its own exhausted clips, so each gets one reopening.
  RETRY_EXHAUSTED_PENDING="${RETRY_EXHAUSTED_ON}"
  # Rounds in a row whose tts pass found nothing to draft. Bounded below, see the loop.
  DRY_TTS_ROUNDS=0
  echo "=== $(date '+%F %T') dataset $((i + 1))/${#MANIFESTS[@]}: ${CUR_INPUT} -> ${CUR_OUTPUT}/ ==="
  # RESCORE_ONLY is a migration, not a loop: one pass, then the next dataset. Looping would
  # never end, since re-examining the same stored transcripts always counts as work done.
  if (( RESCORE_ONLY_ON )); then
    run_pass stt
    rc=$?
    if (( rc != 0 && rc != 10 && rc != 11 )); then
      echo "!!! rescore pass exited ${rc} on ${CUR_INPUT}; aborting driver" >&2
      exit "${rc}"
    fi
    echo "=== $(date '+%F %T') rescored ${CUR_INPUT}; nothing was transcribed. ==="
    continue
  fi
  while :; do
    # Single-mode runs loop on that one pass; the alternating run leads with stt.
    if [[ "${MODE}" == tts ]]; then FIRST=tts; else FIRST=stt; fi
    run_pass "${FIRST}"
    rc=$?
    if (( rc == 10 )); then
      echo "=== $(date '+%F %T') converged: no improvement work left on ${CUR_INPUT}. ==="
      break
    fi
    # 11 = this mode has nothing left to do, but the OTHER one still has work.
    # In a single-mode run that is the stop condition. In an alternating run it is
    # NOT: it is exactly the state a run left behind when it stopped (or was killed)
    # after an stt pass, so every flagged clip is waiting on a tts pass and there is
    # nothing for stt to do until that pass has run. Breaking here would skip the
    # dataset entirely, so we carry on to the tts pass instead.
    FIRST_DRY=0
    if (( rc == 11 )); then
      if [[ "${MODE}" != alternate ]]; then
        echo "=== $(date '+%F %T') ${FIRST} pass ran dry on ${CUR_INPUT}; the rest needs the other pass. ==="
        break
      fi
      FIRST_DRY=1
      echo "=== $(date '+%F %T') stt pass had nothing to do on ${CUR_INPUT}; its clips are waiting on the tts pass. ==="
    elif (( rc != 0 )); then
      echo "!!! ${FIRST} pass exited ${rc} on ${CUR_INPUT}; aborting driver" >&2
      exit "${rc}"
    fi
    [[ "${MODE}" == alternate ]] || continue
    run_pass tts
    rc=$?
    # 11 from the tts pass is "no clip is waiting for candidates", not a failure:
    # the stt pass owns convergence (exit 10), so let the loop go back to it.
    if (( rc != 0 && rc != 11 )); then
      echo "!!! tts pass exited ${rc} on ${CUR_INPUT}; aborting driver" >&2
      exit "${rc}"
    fi
    if (( rc == 11 )); then DRY_TTS_ROUNDS=$((DRY_TTS_ROUNDS + 1)); else DRY_TTS_ROUNDS=0; fi
    # Neither pass could do anything, yet stt did not report convergence: whatever is
    # left is stuck (e.g. flagged clips whose audio has since gone missing). Stop
    # rather than swap the two servers forever over work nobody can clear.
    if (( FIRST_DRY && rc == 11 )); then
      echo "=== $(date '+%F %T') both passes ran dry on ${CUR_INPUT} without converging;" \
           "stopping, the leftovers need a look (see the pending counts above). ==="
      break
    fi
    # Same dead end, one step less obvious: a healthy round either converges (stt exit
    # 10) or leaves clips for the tts pass to draft, so an stt pass that keeps reporting
    # work done while the tts pass keeps finding nothing to draft is re-running rows it
    # can never resolve (clips whose audio has gone missing, say). Bound the alternation
    # instead of swapping the two servers over them forever.
    if (( DRY_TTS_ROUNDS >= 2 )); then
      echo "=== $(date '+%F %T') the tts pass found nothing to draft in ${DRY_TTS_ROUNDS} rounds" \
           "while ${CUR_INPUT} still did not converge: the rest is stuck (missing audio?)," \
           "stopping this dataset. ==="
      break
    fi
  done
done
echo "=== $(date '+%F %T') every dataset converged. Done. ==="

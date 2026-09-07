#!/usr/bin/env bash
# Bring exactly ONE of the STT / TTS server stacks up (freeing the GPU held by the
# other) for the alternating single-GPU improvement flow driven by driver.sh. Both
# stacks live in the same CrispASR docker-compose project, so "switching" is just
# stopping one set of services and starting the other, then blocking until the
# started stack is healthy (using the compose healthchecks) so the caller can launch
# the Python improvement pass immediately.
#
# Runs as ROOT and touches only docker (never the long Python pass). Two ways in:
#   * driver.sh runs as root (sudo -E ./driver.sh) and calls this directly, no sudoers.
#   * standalone / driver-as-user: invoke via a narrow NOPASSWD sudoers entry (see
#     switch_server.sudoers.example).
# The compose file path comes from the environment / an argument so no local absolute
# path (which would embed a username) is committed.
#
# Usage:  switch_server.sh {stt|tts} [/path/to/docker-compose.yml]     # already root
#         sudo switch_server.sh {stt|tts} [/path/to/docker-compose.yml] # standalone
#         (or export CRISPASR_COMPOSE and omit the 2nd argument)
#
# This file was written with Claude Code.
set -euo pipefail

MODE="${1:-}"
# TODO: point CRISPASR_COMPOSE (or the 2nd arg) at your CrispASR docker-compose.yml.
COMPOSE="${CRISPASR_COMPOSE:-${2:-}}"
if [[ -z "${COMPOSE}" || ! -f "${COMPOSE}" ]]; then
  echo "switch_server: set CRISPASR_COMPOSE (or pass the compose path as arg 2); got '${COMPOSE}'" >&2
  exit 64
fi

DC=(docker compose -f "${COMPOSE}")
# How many whisper replicas the STT stack runs, and how long to wait for healthy.
STT_SCALE="${CRISPASR_STT_SCALE:-6}"
HEALTH_TIMEOUT="${SWITCH_HEALTH_TIMEOUT:-900}"

wait_healthy() {  # args: container ids / names to poll until docker reports "healthy"
  local deadline=$(( SECONDS + HEALTH_TIMEOUT )) c status
  for c in "$@"; do
    while :; do
      status="$(docker inspect -f '{{.State.Health.Status}}' "$c" 2>/dev/null || echo missing)"
      [[ "${status}" == healthy ]] && break
      if (( SECONDS > deadline )); then
        echo "switch_server: '${c}' still '${status}' after ${HEALTH_TIMEOUT}s, giving up" >&2
        exit 69
      fi
      sleep 5
    done
  done
}

case "${MODE}" in
  stt)
    # Free the GPU held by any TTS stack, then bring up the whisper replicas + Caddy LB
    # (the exact command used manually: up --scale crispasr=N crispasr-lb -d).
    "${DC[@]}" stop voxtral-tts crispasr-tts >/dev/null 2>&1 || true
    "${DC[@]}" up "--scale" "crispasr=${STT_SCALE}" crispasr-lb -d
    # crispasr-lb (Caddy) has no healthcheck and is up instantly; gate on the whisper
    # replicas, which DO have one (healthy == model loaded and /health answering).
    mapfile -t ids < <("${DC[@]}" ps -q crispasr)
    if (( ${#ids[@]} == 0 )); then
      echo "switch_server: no crispasr replicas started" >&2
      exit 69
    fi
    wait_healthy "${ids[@]}"
    ;;
  tts)
    # Free the GPU held by the STT stack, then bring up Voxtral TTS. Plain `up -d` (no
    # --build / --force-recreate): the image is stable, so reuse it and just start it.
    "${DC[@]}" stop crispasr crispasr-lb crispasr-tts >/dev/null 2>&1 || true
    "${DC[@]}" up -d voxtral-tts
    wait_healthy voxtral-tts
    ;;
  *)
    echo "usage: sudo switch_server.sh {stt|tts} [compose.yml]   (or export CRISPASR_COMPOSE)" >&2
    exit 64
    ;;
esac
echo "switch_server: ${MODE} stack up and healthy"

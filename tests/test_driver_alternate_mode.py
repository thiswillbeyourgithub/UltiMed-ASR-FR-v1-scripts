#!/usr/bin/env python
"""`06_hotfixes/driver.sh` must keep alternating when one half has nothing to do.

The alternating loop leads with the `stt` pass, and that pass legitimately reports
"I have run dry" (exit 11) whenever a previous run stopped right after flagging clips
`pending_tts`: there is nothing left to score until a `tts` pass has drafted
candidates. Treating that as a stop condition (as the driver used to) skips the whole
dataset and the flagged clips are never regenerated, which is exactly the state
`improved_parrot/full.stt.jsonl` was left in (4 pending_tts, 0 pending_stt).

This runs the REAL driver.sh with the passes stubbed out: a fake `uv` records the
`--mode` it was called with and exits with the next code from a plan file, so a whole
alternation is a plan like `11, 0, 10`. driver.sh insists on being root (it flips
docker), so the test re-execs itself under `unshare -r`, which maps the current user to
uid 0 without granting anything; `runuser` and `nvidia-smi` are stubbed on PATH. If
unprivileged user namespaces are disabled, the test SKIPS loudly rather than lying.

    python tests/test_driver_alternate_mode.py

This file was written with Claude Code.
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "06_hotfixes" / "driver.sh"

RUNUSER_STUB = """#!/usr/bin/env bash
# Stand-in for `runuser -u USER -- env ... uv run ...`: drop everything up to the
# `--` separator and exec the rest, so the stubbed uv runs in this same shell.
while [[ $# -gt 0 && "$1" != "--" ]]; do shift; done
shift
exec "$@"
"""

NVIDIA_SMI_STUB = "#!/usr/bin/env bash\nexit 0\n"

UV_STUB = """#!/usr/bin/env bash
# Stand-in for the improvement pass: record the --mode it was asked for and the
# scheduling priority it inherited, then exit with the next code from the plan. Running
# past the end of the plan means the driver looped more times than the test expected, so
# answer 99 (an abort) instead of spinning forever.
mode=""
retry=0
while [[ $# -gt 0 ]]; do
  [[ "$1" == --mode ]] && mode="$2"
  [[ "$1" == --retry-exhausted ]] && retry=1
  shift
done
echo "${mode}|$(nice)|${retry}" >> "%(log)s"
n=$(wc -l < "%(log)s")
code=$(sed -n "${n}p" "%(plan)s")
exit "${code:-99}"
"""

SWITCH_STUB = """#!/usr/bin/env bash
echo "switch $1" >> "%(log)s"
exit 0
"""


def _write_exec(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def run_driver(tmp: Path, plan: list[int], env_extra: dict[str, str],
               n_manifests: int = 1,
               record: list | None = None) -> tuple[int, list[str], list[str], str]:
    """Run driver.sh once with the passes stubbed. Returns (exit code, the --mode of
    each pass in order, the server switches in order, combined output). `record`, if
    given, also receives one [mode, niceness, retry-exhausted] entry per pass, for the
    cases that care about which flags a pass was handed."""
    work = Path(tempfile.mkdtemp(dir=tmp))
    # Run a COPY of driver.sh: it sources `<its dir>/.env`, and the real one holds the
    # user's STT token and would override the test's env.
    driver = work / "driver.sh"
    shutil.copy(DRIVER, driver)

    pass_log, switch_log = work / "passes.log", work / "switches.log"
    pass_log.touch()
    switch_log.touch()
    (work / "plan.txt").write_text("".join(f"{c}\n" for c in plan), encoding="utf-8")

    stub_bin = work / "bin"
    stub_bin.mkdir()
    _write_exec(stub_bin / "runuser", RUNUSER_STUB)
    _write_exec(stub_bin / "nvidia-smi", NVIDIA_SMI_STUB)
    uv = _write_exec(work / "uv-stub", UV_STUB % {"log": pass_log, "plan": work / "plan.txt"})
    switch = _write_exec(work / "switch-stub", SWITCH_STUB % {"log": switch_log})

    # driver.sh only checks the manifests exist (and names their outputs after their
    # parent dir), so empty files in distinct dirs are enough.
    manifests = []
    for i in range(n_manifests):
        d = work / f"ds{i}"
        d.mkdir()
        (d / "full.jsonl").touch()
        manifests.append(str(d / "full.jsonl"))
    (work / "compose.yml").touch()

    env = dict(os.environ)
    env.update({
        "PATH": f"{stub_bin}:{env['PATH']}",
        "TARGET_USER": getpass.getuser(),
        "UV_BIN": str(uv),
        "SWITCH_SERVER": str(switch),
        "CRISPASR_COMPOSE": str(work / "compose.yml"),
        "WDOC_WHISPER_ENDPOINT": "http://127.0.0.1:1",
        "INPUT": " ".join(manifests),
        "OUTPUT": str(work / "out"),
    })
    env.update(env_extra)

    cmd = ["bash", str(driver)]
    if os.geteuid() != 0:
        cmd = ["unshare", "-r", *cmd]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    entries = [e.split("|") for e in pass_log.read_text(encoding="utf-8").split()]
    modes = [e[0] for e in entries]
    switches = switch_log.read_text(encoding="utf-8").split()[1::2]  # "switch stt" pairs
    # Invariant of every case below, not of one of them: this is a multi-day unattended
    # job and no pass may ever compete with an interactive session for the CPU.
    niceness = [e[1] for e in entries]
    assert all(n == "19" for n in niceness), (
        f"every pass must run at nice 19, got {niceness}\n{proc.stdout}{proc.stderr}")
    if record is not None:
        record.extend(entries)
    return proc.returncode, modes, switches, proc.stdout + proc.stderr


def test_alternate_survives_a_dry_stt_start(tmp: Path) -> None:
    """The regression: nothing to score, everything waiting on tts. The stt pass exits
    11 and the driver must run the tts pass anyway, not give up on the dataset."""
    rc, modes, switches, out = run_driver(tmp, [11, 0, 10], {"MODE": "alternate"})
    assert modes == ["stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    assert switches == ["stt", "tts", "stt"], f"servers switched: {switches}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_alternate_survives_a_dry_stt_start: OK")


def test_alternate_stops_when_both_halves_are_dry(tmp: Path) -> None:
    """...but it must not spin: if neither pass can do anything and stt never reported
    convergence, the leftovers are stuck, so stop after the round."""
    rc, modes, _switches, out = run_driver(tmp, [11, 11], {"MODE": "alternate"})
    assert modes == ["stt", "tts"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "both passes ran dry" in out, out
    print("test_alternate_stops_when_both_halves_are_dry: OK")


def test_alternate_tolerates_a_dry_tts_pass(tmp: Path) -> None:
    """A tts pass with no clip to draft for exits 11 too. That is not a failure: the
    stt pass owns convergence, so the loop goes back to it."""
    rc, modes, _switches, out = run_driver(tmp, [0, 11, 10], {"MODE": "alternate"})
    assert modes == ["stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_alternate_tolerates_a_dry_tts_pass: OK")


def test_alternate_gives_up_on_a_stuck_dataset(tmp: Path) -> None:
    """The other dead end: the stt pass keeps reporting work done (it re-runs rows it
    can never resolve, e.g. clips whose audio is gone) while the tts pass has nothing
    to draft. Bounded at two such rounds, or the driver swaps servers forever. Note the
    stub exits 99 past the end of the plan, so a loop that did not stop aborts here."""
    rc, modes, _switches, out = run_driver(tmp, [0, 11, 0, 11], {"MODE": "alternate"})
    assert modes == ["stt", "tts", "stt", "tts"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "found nothing to draft in 2 rounds" in out, out
    print("test_alternate_gives_up_on_a_stuck_dataset: OK")


def test_alternate_still_aborts_on_a_real_failure(tmp: Path) -> None:
    """Only 10 and 11 are outcomes; any other non-zero code still aborts the driver."""
    rc, modes, _switches, out = run_driver(tmp, [7], {"MODE": "alternate"})
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    assert rc == 7, f"driver exited {rc}, expected the pass's 7\n{out}"
    print("test_alternate_still_aborts_on_a_real_failure: OK")


def test_single_mode_still_stops_when_dry(tmp: Path) -> None:
    """Unchanged for MODE=stt: 11 is the stop condition there, since no tts pass is
    coming to clear the rest."""
    rc, modes, _switches, out = run_driver(tmp, [11], {"MODE": "stt"})
    assert modes == ["stt"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    assert "ran dry" in out, out
    print("test_single_mode_still_stops_when_dry: OK")


def test_dry_start_on_the_second_dataset(tmp: Path) -> None:
    """The live shape: the main corpus converges, then PARROT starts with only
    pending_tts clips. The second dataset must still be worked, not skipped."""
    rc, modes, _switches, out = run_driver(tmp, [10, 11, 0, 10], {"MODE": "alternate"},
                                           n_manifests=2)
    assert modes == ["stt", "stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_dry_start_on_the_second_dataset: OK")


def test_retry_exhausted_is_one_shot_per_dataset(tmp: Path) -> None:
    """RETRY_EXHAUSTED=1 reopens the clips that gave up, and must do so exactly ONCE per
    dataset: the retry ends by writing a fresh `exhausted` marker for whatever still
    fails, so a flag left on for every stt pass would reopen its own output and the
    alternating loop would never converge. Each dataset gets its own reopening, though,
    since each manifest holds its own."""
    entries: list = []
    rc, modes, _switches, out = run_driver(
        tmp, [0, 0, 10, 0, 0, 10], {"MODE": "alternate", "RETRY_EXHAUSTED": "1"},
        n_manifests=2, record=entries)
    assert modes == ["stt", "tts", "stt", "stt", "tts", "stt"], f"passes ran: {modes}\n{out}"
    retried = [e[2] for e in entries]
    assert retried == ["1", "0", "0", "1", "0", "0"], (
        "--retry-exhausted should ride the first stt pass of each dataset only, got "
        f"{list(zip(modes, retried))}\n{out}")
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_retry_exhausted_is_one_shot_per_dataset: OK")


def test_retry_exhausted_defaults_off(tmp: Path) -> None:
    """Nothing changes for a plain run: the resolved clips stay resolved."""
    entries: list = []
    rc, _modes, _switches, out = run_driver(tmp, [10], {"MODE": "alternate"}, record=entries)
    assert [e[2] for e in entries] == ["0"], f"unasked-for retry\n{out}"
    assert rc == 0, f"driver exited {rc}\n{out}"
    print("test_retry_exhausted_defaults_off: OK")


def main() -> None:
    if shutil.which("unshare") is None and os.geteuid() != 0:
        print("SKIP test_driver_alternate_mode: no `unshare`, cannot fake the root check")
        return
    if os.geteuid() != 0:
        probe = subprocess.run(["unshare", "-r", "true"], capture_output=True)
        if probe.returncode != 0:
            print("SKIP test_driver_alternate_mode: unprivileged user namespaces are "
                  "disabled, cannot fake driver.sh's root check")
            return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_alternate_survives_a_dry_stt_start(tmp)
        test_alternate_stops_when_both_halves_are_dry(tmp)
        test_alternate_tolerates_a_dry_tts_pass(tmp)
        test_alternate_gives_up_on_a_stuck_dataset(tmp)
        test_alternate_still_aborts_on_a_real_failure(tmp)
        test_single_mode_still_stops_when_dry(tmp)
        test_dry_start_on_the_second_dataset(tmp)
        test_retry_exhausted_is_one_shot_per_dataset(tmp)
        test_retry_exhausted_defaults_off(tmp)
    print("test_driver_alternate_mode: OK")


if __name__ == "__main__":
    sys.exit(main())

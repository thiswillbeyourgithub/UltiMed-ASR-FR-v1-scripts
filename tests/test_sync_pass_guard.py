"""Stdlib test for 99_hf_release/03_sync_hotfix_results.py's stage-06 pass guard.

The guard refuses to sync while a stage-06 pass is writing the file it reads. It used to
substring-match /proc cmdlines, so anything that merely NAMED the pass counted as one: a
shell, a grep, or a `pgrep -f` wait loop, which matches its own pattern and therefore
blocked the sync forever with nothing running.

Run: python tests/test_sync_pass_guard.py
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_sync_module():
    """Import the uv script without its (irrelevant here) click / loguru dependencies."""
    click = types.ModuleType("click")

    def passthrough(*_args, **_kwargs):
        return lambda fn: fn

    click.command = click.option = click.argument = passthrough
    click.Path = lambda *a, **k: None
    loguru = types.ModuleType("loguru")
    loguru.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda *a, **k: None, success=lambda *a, **k: None)
    sys.modules.setdefault("click", click)
    sys.modules.setdefault("loguru", loguru)

    import importlib.util
    path = ROOT / "99_hf_release" / "03_sync_hotfix_results.py"
    spec = importlib.util.spec_from_file_location("sync_hotfix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_pass_cmdline(mod):
    real = [
        # uv run from inside the stage folder, and with a full path
        "uv\0run\x0001_recursive_improvement.py\0--mode\0stt",
        "/usr/bin/python\0/home/x/06_hotfixes/01_recursive_improvement.py\0--mode\0tts",
        "python\0./01_recursive_improvement.py",
    ]
    for cmdline in real:
        assert mod.is_pass_cmdline(cmdline), f"should count as a live pass: {cmdline!r}"

    fake = [
        # The wait loop that used to block the sync forever, matching its own pattern.
        '/bin/zsh\0-c\0until ! pgrep -f "01_recursive_improvement.py --mode stt"; '
        'do sleep 30; done',
        'pgrep\0-f\x0001_recursive_improvement.py --mode stt --rescore-only',
        "grep\0-n\0status\x0001_recursive_improvement.py.bak",
        # A different script whose name merely ends with the same suffix.
        "python\0my_01_recursive_improvement.py",
        "",
    ]
    for cmdline in fake:
        assert not mod.is_pass_cmdline(cmdline), f"should NOT count as a pass: {cmdline!r}"


def main():
    mod = load_sync_module()
    test_is_pass_cmdline(mod)
    print("ALL PASSED")


if __name__ == "__main__":
    main()

"""Check portable CLI paths and the entry module's supported delegation API."""

import importlib
from pathlib import Path
import sys
import warnings

import pytest


@pytest.fixture
def entrypoint():
    """Load the local entry module without leaking its warning policy to tests."""
    root = Path(__file__).resolve().parents[1]
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, str(root))
        with warnings.catch_warnings():
            yield importlib.import_module("spec_reduction_parallel")
    finally:
        sys.path[:] = old_path


@pytest.mark.parametrize("absolute", [False, True])
def test_cli_preserves_observation_and_config_paths(entrypoint, tmp_path, monkeypatch, absolute):
    """Accept relative and absolute inputs from an unrelated working directory."""
    observations = tmp_path / "observation data"
    config = tmp_path / "instrument settings.yaml"
    observations.mkdir()
    config.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    arguments = [str(observations), str(config)] if absolute else [observations.name, config.name]
    received = []
    expected = {"status": "completed"}

    def record_run(obspath, configpath):
        """Check path identity at the boundary without reducing observations."""
        received.append((obspath, configpath))
        assert Path(obspath).resolve() == observations
        assert Path(configpath).resolve() == config
        return expected

    monkeypatch.setattr(entrypoint, "main", record_run)
    assert entrypoint.cli(arguments) is expected
    assert received == [tuple(arguments)]


def test_cli_requires_both_paths(entrypoint, monkeypatch):
    """Reject incomplete input before any reduction starts."""
    def unexpected_run(*args):
        """Fail if argument validation accidentally starts the controller."""
        pytest.fail("Reduction started with incomplete command-line input")

    monkeypatch.setattr(entrypoint, "main", unexpected_run)
    with pytest.raises(SystemExit) as error:
        entrypoint.cli(["observations"])
    assert error.value.code == 2


@pytest.mark.parametrize("name", ["make_group_masters_worker", "process_single_frame_worker"])
def test_worker_aliases_forward_tasks(entrypoint, monkeypatch, name):
    """Retain worker call signatures and structured return values."""
    import speclibparallel
    task = {"task_id": "example"}
    expected = {"status": "success"}

    def worker(received):
        """Return a known result only for the original task object."""
        assert received is task
        return expected

    monkeypatch.setattr(speclibparallel, name, worker)
    assert getattr(entrypoint, name)(task) is expected


def test_main_forwards_path_objects(entrypoint, monkeypatch, tmp_path):
    """Preserve the programmatic main API alongside the corrected CLI."""
    import speclibparallel
    observations, config = tmp_path / "observations", tmp_path / "config.yaml"
    expected = {"status": "completed"}

    def controller(obspath, configpath):
        """Check that direct callers keep their original path objects."""
        assert obspath is observations and configpath is config
        return expected

    monkeypatch.setattr(speclibparallel, "main", controller)
    assert entrypoint.main(observations, config) is expected

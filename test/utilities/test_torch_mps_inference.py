from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import torch

from pyslam.utilities.torch import serialized_mps_inference


def test_mps_inference_is_serialized_and_synchronized(monkeypatch):
    active_runs = 0
    max_active_runs = 0
    synchronize_calls = 0
    state_lock = Lock()

    def synchronize(device=None):
        nonlocal synchronize_calls
        synchronize_calls += 1

    def run_inference():
        nonlocal active_runs, max_active_runs
        with serialized_mps_inference(torch.device("mps")):
            with state_lock:
                active_runs += 1
                max_active_runs = max(max_active_runs, active_runs)
            time.sleep(0.02)
            with state_lock:
                active_runs -= 1

    monkeypatch.setattr(torch.mps, "synchronize", synchronize)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: run_inference(), range(2)))

    assert max_active_runs == 1
    assert synchronize_calls == 2


def test_cpu_inference_does_not_use_mps_synchronization(monkeypatch):
    def unexpected_synchronize(device=None):
        raise AssertionError("CPU inference must not synchronize MPS")

    monkeypatch.setattr(torch.mps, "synchronize", unexpected_synchronize)
    with serialized_mps_inference(torch.device("cpu")):
        pass

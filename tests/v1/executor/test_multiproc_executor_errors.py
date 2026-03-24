# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import signal
from types import SimpleNamespace

import pytest

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import (UnreadyWorkerProcHandle,
                                                 WorkerProc)


def test_wait_for_ready_surfaces_child_startup_traceback():
    ctx = multiprocessing.get_context("spawn")
    ready_reader, ready_writer = ctx.Pipe(duplex=False)

    ready_writer.send({
        "status": WorkerProc.FAILURE_STR,
        "error": "ValueError: boom",
        "traceback": "Traceback (most recent call last):\nValueError: boom",
    })
    ready_writer.close()

    handle = UnreadyWorkerProcHandle(
        proc=SimpleNamespace(exitcode=1),
        rank=1,
        ready_pipe=ready_reader,
    )

    with pytest.raises(RuntimeError, match="rank 1: ValueError: boom") as exc:
        WorkerProc.wait_for_ready([handle])

    assert "Worker traceback:" in str(exc.value)
    assert "ValueError: boom" in str(exc.value)


def test_wait_for_ready_reports_worker_exit_signal():
    ctx = multiprocessing.get_context("spawn")
    ready_reader, ready_writer = ctx.Pipe(duplex=False)
    ready_writer.close()

    handle = UnreadyWorkerProcHandle(
        proc=SimpleNamespace(
            pid=4321,
            exitcode=-signal.SIGABRT,
            join=lambda timeout=None: None,
        ),
        rank=1,
        ready_pipe=ready_reader,
    )

    with pytest.raises(
            RuntimeError,
            match=r"rank 1 exited before signaling READY.*pid=4321.*signal=SIGABRT",
    ):
        WorkerProc.wait_for_ready([handle])


def test_worker_main_reports_startup_exception_to_parent(monkeypatch):
    worker_main = WorkerProc.worker_main
    messages = []

    class FakeReader:
        def close(self):
            pass

    class FakeWriter:
        def send(self, msg):
            messages.append(msg)

        def close(self):
            pass

    class FailingWorkerProc:
        READY_STR = "READY"
        FAILURE_STR = "FAILURE"

        def __init__(self, *args, **kwargs):
            raise ValueError("boom")

    monkeypatch.setattr(multiproc_executor, "WorkerProc", FailingWorkerProc)

    worker_main(
        vllm_config=object(),
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:12345",
        input_shm_handle=object(),
        ready_pipe=(FakeReader(), FakeWriter()),
        death_pipe=None,
    )

    assert len(messages) == 1
    message = messages[0]
    assert message["status"] == "FAILURE"
    assert message["error"] == "ValueError: boom"
    assert "Traceback (most recent call last):" in message["traceback"]
    assert "ValueError: boom" in message["traceback"]


def test_worker_main_reports_system_exit_to_parent(monkeypatch):
    worker_main = WorkerProc.worker_main
    messages = []

    class FakeReader:
        def close(self):
            pass

    class FakeWriter:
        def send(self, msg):
            messages.append(msg)

        def close(self):
            pass

    class FailingWorkerProc:
        READY_STR = "READY"
        FAILURE_STR = "FAILURE"

        def __init__(self, *args, **kwargs):
            raise SystemExit(1)

    monkeypatch.setattr(multiproc_executor, "WorkerProc", FailingWorkerProc)

    with pytest.raises(SystemExit):
        worker_main(
            vllm_config=object(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:12345",
            input_shm_handle=object(),
            ready_pipe=(FakeReader(), FakeWriter()),
            death_pipe=None,
        )

    assert len(messages) == 1
    message = messages[0]
    assert message["status"] == "FAILURE"
    assert message["error"] == "SystemExit: 1"
    assert "Traceback (most recent call last):" in message["traceback"]
    assert "SystemExit: 1" in message["traceback"]

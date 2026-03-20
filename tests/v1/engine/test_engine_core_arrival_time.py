# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCore, EngineCoreProc


class BlockingFuture:

    def __init__(self, started: threading.Event, release: threading.Event,
                 result: Any):
        self._started = started
        self._release = release
        self._result = result

    def result(self):
        self._started.set()
        assert self._release.wait(timeout=1)
        return self._result


def _make_scheduler_output():
    return SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"req-0": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[],
                                              num_computed_tokens=[]),
    )


def _make_core_proc():
    core = EngineCoreProc.__new__(EngineCoreProc)
    core.input_queue = queue.Queue()
    core.output_queue = queue.Queue()
    core._scheduler_state_lock = threading.Lock()
    core._allow_immediate_add_requests = threading.Event()
    core._profile_events = []
    core._pending_batch_event = None
    core.batch_id = 0
    core.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=0))
    core.add_request = MagicMock()
    return core


def _make_step_core(execute_model, update_from_output=None):
    core = _make_core_proc()
    scheduler_output = _make_scheduler_output()

    scheduler = MagicMock()
    scheduler.batch_id = 0
    scheduler.has_requests.return_value = True
    scheduler.schedule.return_value = scheduler_output
    scheduler.get_rejected_requests.return_value = []
    scheduler.get_exec_plan.return_value = None
    scheduler.get_load_statistics.return_value = {
        "num_free_blocks": 0,
        "n_waitings": 0,
        "n_running": 0,
    }
    scheduler.update_from_output.side_effect = (
        update_from_output if update_from_output is not None else
        (lambda *_args, **_kwargs: {}))

    core.scheduler = scheduler
    core.model_executor = SimpleNamespace(execute_model=execute_model)
    core.batch_queue = None
    return core, scheduler_output


def _run_in_thread(target):
    errors = []

    def runner():
        try:
            target()
        except BaseException as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, errors


def test_dispatch_input_request_queues_add_outside_execution_window():
    core = _make_core_proc()
    request = (object(), 0, 0, 0)

    core._dispatch_input_request(EngineCoreRequestType.ADD, request)

    assert not core.add_request.called
    assert core.input_queue.get_nowait() == (EngineCoreRequestType.ADD,
                                             request)


def test_dispatch_input_request_queues_non_add_requests():
    core = _make_core_proc()
    abort_request = ["req-0"]
    core._allow_immediate_add_requests.set()

    core._dispatch_input_request(EngineCoreRequestType.ABORT, abort_request)

    assert not core.add_request.called
    assert core.input_queue.get_nowait() == (EngineCoreRequestType.ABORT,
                                             abort_request)


def test_add_request_is_immediate_while_step_executes_model():
    execution_started = threading.Event()
    release_execution = threading.Event()
    add_called = threading.Event()

    def execute_model(_scheduler_output):
        execution_started.set()
        assert release_execution.wait(timeout=1)
        return object()

    core, _ = _make_step_core(execute_model)
    core.add_request = MagicMock(
        side_effect=lambda req, wave=0: add_called.set())

    thread, errors = _run_in_thread(core.step)
    assert execution_started.wait(timeout=1)

    request = (object(), 0, 0, 0)
    core._dispatch_input_request(EngineCoreRequestType.ADD, request)

    assert add_called.wait(timeout=1)
    assert core.input_queue.empty()

    release_execution.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not errors


def test_add_request_is_queued_after_execution_window_closes():
    execution_started = threading.Event()
    release_execution = threading.Event()
    update_started = threading.Event()
    release_update = threading.Event()
    add_called = threading.Event()

    def execute_model(_scheduler_output):
        execution_started.set()
        assert release_execution.wait(timeout=1)
        return object()

    def update_from_output(*_args, **_kwargs):
        update_started.set()
        assert release_update.wait(timeout=1)
        return {}

    core, _ = _make_step_core(execute_model, update_from_output)
    core.add_request = MagicMock(
        side_effect=lambda req, wave=0: add_called.set())

    thread, errors = _run_in_thread(core.step)
    assert execution_started.wait(timeout=1)

    release_execution.set()
    assert update_started.wait(timeout=1)

    request = (object(), 0, 0, 0)
    core._dispatch_input_request(EngineCoreRequestType.ADD, request)

    assert not add_called.wait(timeout=0.1)
    assert core.input_queue.get_nowait() == (EngineCoreRequestType.ADD,
                                             request)

    release_update.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not errors


def test_add_request_is_immediate_while_waiting_for_batch_queue_result():
    future_started = threading.Event()
    release_future = threading.Event()
    add_called = threading.Event()

    core = _make_core_proc()
    scheduler_output = _make_scheduler_output()
    future = BlockingFuture(future_started, release_future, object())

    scheduler = MagicMock()
    scheduler.schedule.return_value = _make_scheduler_output()
    scheduler.update_from_output.return_value = {}
    core.scheduler = scheduler
    core.model_executor = SimpleNamespace()
    core.batch_queue = queue.Queue(maxsize=1)
    core.batch_queue.put_nowait((future, scheduler_output))
    core.add_request = MagicMock(
        side_effect=lambda req, wave=0: add_called.set())

    thread, errors = _run_in_thread(core.step_with_batch_queue)
    assert future_started.wait(timeout=1)

    request = (object(), 0, 0, 0)
    core._dispatch_input_request(EngineCoreRequestType.ADD, request)

    assert add_called.wait(timeout=1)
    assert core.input_queue.empty()

    release_future.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not errors
    scheduler.schedule.assert_not_called()


def test_add_request_ack_is_sent_for_immediate_add():
    core = _make_core_proc()
    request = (object(), 0, 7, 11)

    core._allow_immediate_add_requests.set()
    core.add_request = MagicMock(return_value=False)

    core._dispatch_input_request(EngineCoreRequestType.ADD, request)

    client_index, outputs = core.output_queue.get_nowait()
    assert client_index == 7
    assert outputs.utility_output is not None
    assert outputs.utility_output.call_id == 11
    assert outputs.utility_output.result is not None
    assert outputs.utility_output.result.result is False


def test_get_load_statistics_uses_safe_scheduler_defaults():
    core = _make_core_proc()
    core.scheduler = SimpleNamespace(
        waiting_attainable=[1, 2],
        waiting_kv_xfer=[3],
        running=[4],
        kv_cache_manager=SimpleNamespace(get_num_free_blocks=lambda: 11),
    )

    assert core.get_load_statistics() == {
        "num_free_blocks": 11,
        "n_waitings": 3,
        "n_running": 1,
    }
    assert core._make_engine_state_snapshot() == {
        "exec_plan": None,
        "load_stats": {
            "num_free_blocks": 11,
            "n_waitings": 3,
            "n_running": 1,
        },
    }


def test_prepare_scheduler_batch_context_sets_atfc_planner_metadata():
    core = _make_core_proc()
    core.engine_index = 4
    atfc_planner = SimpleNamespace(tag=None, batch_id=None)
    core.scheduler = SimpleNamespace(batch_id=0, atfc_planner=atfc_planner)

    core._prepare_scheduler_batch_context()

    assert core.batch_id == 1
    assert core.scheduler.batch_id == 1
    assert atfc_planner.tag == "4_1"
    assert atfc_planner.batch_id == 1


def test_finalize_pending_batch_event_uses_next_schedule_interval():
    core = _make_core_proc()
    core._pending_batch_event = {
        "event_type": "batch",
        "batch_id": 3,
        "_schedule_timestamp": 10.0,
        "req_ids": ["req-0"],
        "num_computed_tokens": [4],
        "num_scheduled_tokens": {
            "req-0": 1,
        },
        "scheduling_overhead": 0.1,
    }

    core._finalize_pending_batch_event(12.5)

    assert core._pending_batch_event is None
    assert len(core._profile_events) == 1
    event = core._profile_events[0]
    assert event["timestamp"] == 12.5
    assert event["elapsed"] == 2.5
    assert event["between_batch_time"] == 2.5
    assert "_schedule_timestamp" not in event


def test_get_estimated_batch_time_uses_exec_plan_first_batch_time():
    core = _make_core_proc()
    scheduler_output = _make_scheduler_output()

    estimated_time = core._get_estimated_batch_time(
        {
            "exec_plan": {
                "batch_times": [11.75],
            },
        },
        scheduler_output,
        schedule_timestamp=10.5,
    )

    assert estimated_time == 1.25


def test_get_estimated_batch_time_falls_back_to_perf_model():
    core = _make_core_proc()
    get_batch_time = MagicMock(return_value=1.75)
    core.scheduler = SimpleNamespace(
        perf_model=SimpleNamespace(get_batch_time=get_batch_time))
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="req-new", num_computed_tokens=5),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-cached"],
            num_computed_tokens=[7],
        ),
        num_scheduled_tokens={
            "req-new": 3,
            "req-cached": 1,
        },
    )

    estimated_time = core._get_estimated_batch_time(
        {
            "exec_plan": None,
        },
        scheduler_output,
        schedule_timestamp=10.0,
    )

    assert estimated_time == 1.75
    get_batch_time.assert_called_once_with([(5, 3), (7, 1)])


def test_step_defers_batch_event_until_next_schedule_point():
    core, scheduler_output = _make_step_core(lambda _scheduler_output: object())
    scheduler_output.scheduled_new_reqs = [
        SimpleNamespace(req_id="req-0", num_computed_tokens=5),
    ]
    core.scheduler.get_exec_plan.return_value = SimpleNamespace(
        req_plans={"req-0": [(6, 0)]},
        batch_times=[100.0],
        num_free_blocks=7,
    )

    outputs, scheduled = core.step()

    assert outputs == {}
    assert scheduled is True
    assert core._pending_batch_event is not None
    assert not any(event["event_type"] == "batch"
                   for event in core._profile_events)

    core.scheduler.has_requests.return_value = False
    outputs, scheduled = core.step()

    assert outputs == {}
    assert scheduled is False
    assert core._pending_batch_event is None

    batch_events = [
        event for event in core._profile_events
        if event["event_type"] == "batch"
    ]
    assert len(batch_events) == 1
    batch_event = batch_events[0]
    assert batch_event["req_ids"] == ["req-0"]
    assert batch_event["num_computed_tokens"] == [5]
    assert batch_event["num_scheduled_tokens"] == {"req-0": 1}
    assert batch_event["estimated_time"] >= 0.0
    assert batch_event["publish_overhead"] >= 0.0
    assert batch_event["output_processing_elapsed"] >= 0.0
    assert batch_event["between_batch_time"] == batch_event["elapsed"]
    assert batch_event["rejected_reqs"] == []
    assert batch_event["extra_args"]["to_schedule"] == 0.0
    assert batch_event["extra_args"]["to_launch"] >= 0.0
    assert batch_event["extra_args"]["to_finish"] >= 0.0
    assert batch_event["extra_args"]["to_est_finish"] >= \
        batch_event["extra_args"]["to_launch"]


def test_add_request_records_arrival_and_add_request_events():
    core = _make_core_proc()
    scheduler = MagicMock()
    scheduler.get_kv_connector.return_value = True
    scheduler.add_request.return_value = True
    core.scheduler = scheduler

    request = SimpleNamespace(
        request_id="req-0",
        pooling_params=None,
        kv_transfer_params={
            "do_remote_decode": True,
            "do_remote_prefill": False,
        },
        sampling_params=SimpleNamespace(extra_args={
            "zero_load_ttft": 1.5,
            "profit": 2.0,
            "kv_transfer_params": {
                "arrival_time": 9.0,
                "do_remote_decode": True,
            },
        }),
        num_prompt_tokens=4,
        max_tokens=8,
        arrival_time=3.0,
        prefill_ddl=7.0,
        client_index=0,
    )

    with patch("vllm.v1.engine.core.time.time",
               side_effect=[5.0, 5.1, 5.2, 5.2]):
        admitted = EngineCore.add_request(core, request)

    assert admitted is True
    assert [event["event_type"] for event in core._profile_events] == [
        "arrival",
        "add_request",
    ]

    arrival_event, add_request_event = core._profile_events
    assert arrival_event["request_id"] == "req-0"
    assert arrival_event["prefill_ddl"] == 7.0
    assert arrival_event["prefill_only"] is True
    assert arrival_event["decode_only"] is False
    assert arrival_event["extra_args"] == {
        "batch_id": 0,
    }

    assert add_request_event["request_id"] == "req-0"
    assert add_request_event["extra_args"]["admitted"] is True
    assert add_request_event["extra_args"]["elapsed"] >= 0.0
    assert add_request_event["extra_args"]["kv_transfer_params"] == {
        "arrival_time": 9.0,
        "do_remote_decode": True,
    }
    assert add_request_event["extra_args"]["kv_ready_time"] == 3.8


def test_step_publishes_engine_state_snapshot_before_model_execution():
    execution_started = threading.Event()
    release_execution = threading.Event()

    def execute_model(_scheduler_output):
        execution_started.set()
        assert release_execution.wait(timeout=1)
        return object()

    core, _ = _make_step_core(execute_model)
    core.scheduler.get_exec_plan.return_value = SimpleNamespace(
        req_plans={"req-0": [(1, 0)]},
        batch_times=[2.5],
        num_free_blocks=7,
    )
    core.scheduler.get_load_statistics.return_value = {
        "num_free_blocks": 9,
        "n_waitings": 2,
        "n_running": 1,
    }

    thread, errors = _run_in_thread(core.step)
    assert execution_started.wait(timeout=1)

    client_index, outputs = core.output_queue.get_nowait()
    assert client_index == 0
    assert outputs.engine_state_snapshot == {
        "exec_plan": {
            "req_plans": {
                "req-0": [(1, 0)],
            },
            "batch_times": [2.5],
            "num_free_blocks": 7,
        },
        "load_stats": {
            "num_free_blocks": 9,
            "n_waitings": 2,
            "n_running": 1,
        },
    }

    release_execution.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not errors


def test_step_with_batch_queue_publishes_engine_state_snapshot_on_schedule():
    core = _make_core_proc()
    scheduler_output = _make_scheduler_output()

    scheduler = MagicMock()
    scheduler.schedule.return_value = scheduler_output
    scheduler.get_exec_plan.return_value = SimpleNamespace(
        req_plans={"req-0": [(2, 0)]},
        batch_times=[1.25],
        num_free_blocks=5,
    )
    scheduler.get_load_statistics.return_value = {
        "num_free_blocks": 6,
        "n_waitings": 1,
        "n_running": 2,
    }
    core.scheduler = scheduler
    core.model_executor = SimpleNamespace(execute_model=MagicMock(
        return_value=object()))
    core.batch_queue = queue.Queue(maxsize=1)

    outputs, scheduled_batch = core.step_with_batch_queue()

    assert outputs is None
    assert scheduled_batch is True
    client_index, published = core.output_queue.get_nowait()
    assert client_index == 0
    assert published.engine_state_snapshot == {
        "exec_plan": {
            "req_plans": {
                "req-0": [(2, 0)],
            },
            "batch_times": [1.25],
            "num_free_blocks": 5,
        },
        "load_stats": {
            "num_free_blocks": 6,
            "n_waitings": 1,
            "n_running": 2,
        },
    }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from SLOsServe.router.execplan_bus import ExecPlan
from vllm import SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient


def _make_engine_core_request(request_id: str = "req-0") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        mm_kwargs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _make_request_output(request_id: str = "req-0") -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="done",
                token_ids=[4],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        finished=True,
    )


def _make_async_llm() -> AsyncLLM:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.log_requests = False
    llm.log_stats = False
    llm.output_handler = None
    llm.logger_manager = None
    llm._engine_state_bus = None
    llm._engine_state_device_id = 0
    llm.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        add_request_with_admission_async=AsyncMock(),
        shutdown=MagicMock(),
    )
    llm.output_processor = SimpleNamespace(add_request=MagicMock())
    llm.processor = SimpleNamespace(process_inputs=MagicMock())
    llm._run_output_handler = MagicMock()
    return llm


def _make_output_handler_llm() -> AsyncLLM:
    llm = AsyncLLM.__new__(AsyncLLM)
    llm.log_requests = False
    llm.log_stats = False
    llm.output_handler = None
    llm.logger_manager = None
    llm._engine_state_bus = None
    llm._engine_state_device_id = 0
    llm.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        get_output_async=AsyncMock(),
        abort_requests_async=AsyncMock(),
        shutdown=MagicMock(),
    )
    llm.output_processor = SimpleNamespace(
        process_outputs=MagicMock(return_value=SimpleNamespace(
            request_outputs=[],
            reqs_to_abort=[],
        )),
        propagate_error=MagicMock(),
    )
    return llm


@pytest.mark.asyncio
async def test_async_mp_client_add_request_with_admission_waits_for_ack():
    client = AsyncMPClient.__new__(AsyncMPClient)
    client.client_index = 3
    client.utility_results = {}
    client._ensure_output_queue_task = MagicMock()

    async def fake_send_input(request_type, request, engine=None):
        assert request_type == EngineCoreRequestType.ADD
        client.utility_results[request.admission_call_id].set_result(True)

    client._send_input = fake_send_input

    request = _make_engine_core_request()
    admitted = await client.add_request_with_admission_async(request)

    assert admitted is True
    assert request.client_index == 3
    assert request.admission_call_id == 0
    client._ensure_output_queue_task.assert_called_once()


@pytest.mark.asyncio
async def test_async_llm_add_request_returns_false_when_rejected():
    llm = _make_async_llm()
    request = _make_engine_core_request()
    llm.processor.process_inputs.return_value = ("prompt", request)
    llm.engine_core.add_request_with_admission_async.return_value = {
        "admitted": False,
        "rejection_reason": "CMP",
    }

    admitted, stream, rejection_reason = await llm.add_request(
        prompt="prompt",
        request_id=request.request_id,
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert admitted is False
    assert stream is None
    assert rejection_reason == "CMP"
    llm._run_output_handler.assert_called_once()
    llm.output_processor.add_request.assert_called_once()
    llm.engine_core.add_request_with_admission_async.assert_awaited_once_with(
        request)


@pytest.mark.asyncio
async def test_async_llm_add_request_returns_stream_when_admitted():
    llm = _make_async_llm()
    request = _make_engine_core_request()
    llm.processor.process_inputs.return_value = ("prompt", request)
    llm.engine_core.add_request_with_admission_async.return_value = {
        "admitted": True,
    }

    captured_queue = None

    def capture_queue(*args):
        nonlocal captured_queue
        captured_queue = args[-1]

    llm.output_processor.add_request.side_effect = capture_queue

    admitted, stream, rejection_reason = await llm.add_request(
        prompt="prompt",
        request_id=request.request_id,
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert admitted is True
    assert stream is not None
    assert rejection_reason is None
    assert captured_queue is not None

    captured_queue.put(_make_request_output(request.request_id))
    outputs = [out async for out in stream]

    assert [out.request_id for out in outputs] == [request.request_id]
    assert outputs[0].finished is True


@pytest.mark.asyncio
async def test_async_mp_client_queues_engine_state_only_outputs():

    class FakeOutputSocket:

        def __init__(self):
            self.calls = 0
            self.blocker = asyncio.Future()

        async def recv_multipart(self, copy=False):
            self.calls += 1
            if self.calls == 1:
                return [b"frame"]
            return await self.blocker

    client = AsyncMPClient.__new__(AsyncMPClient)
    client.utility_results = {}
    client.outputs_queue = asyncio.Queue()
    client.decoder = SimpleNamespace(decode=MagicMock(
        return_value=EngineCoreOutputs(
            engine_state_snapshot={
                "exec_plan": None,
                "load_stats": {
                    "num_free_blocks": 3,
                    "n_waitings": 1,
                    "n_running": 2,
                },
            })))
    client.resources = SimpleNamespace(
        output_queue_task=None,
        output_socket=FakeOutputSocket(),
        validate_alive=MagicMock(),
        engine_dead=False,
    )

    AsyncMPClient._ensure_output_queue_task(client)

    outputs = await asyncio.wait_for(client.outputs_queue.get(), timeout=1)

    assert outputs.engine_state_snapshot == {
        "exec_plan": None,
        "load_stats": {
            "num_free_blocks": 3,
            "n_waitings": 1,
            "n_running": 2,
        },
    }

    client.resources.output_queue_task.cancel()
    await client.resources.output_queue_task


@pytest.mark.asyncio
async def test_async_llm_output_handler_publishes_engine_state_snapshot():
    llm = _make_output_handler_llm()
    outputs = EngineCoreOutputs(
        engine_state_snapshot={
            "exec_plan": {
                "req_plans": {
                    "req-0": [[1, 0]],
                },
                "batch_times": [1.5],
                "num_free_blocks": 7,
            },
            "load_stats": {
                "num_free_blocks": 6,
                "n_waitings": 2,
                "n_running": 1,
            },
        })
    llm.engine_core.get_output_async.side_effect = [
        outputs,
        RuntimeError("stop"),
    ]

    publish_remote = MagicMock()
    bus = SimpleNamespace(publish=SimpleNamespace(remote=publish_remote))
    AsyncLLM.set_engine_state_publisher(llm, bus, 4)

    AsyncLLM._run_output_handler(llm)
    await llm.output_handler

    publish_remote.assert_called_once()
    args = publish_remote.call_args.args
    kwargs = publish_remote.call_args.kwargs

    assert args[0] == 4
    assert args[1] == outputs.timestamp
    assert isinstance(args[2], ExecPlan)
    assert args[2].req_plans["req-0"] == [(1, 0)]
    assert args[2].batch_times == [1.5]
    assert args[2].num_free_blocks == 7
    assert kwargs["load_stats"] == {
        "num_free_blocks": 6,
        "effective_num_free_blocks": 0,
        "n_waitings": 2,
        "n_running": 1,
        "n_regular_waitings": 0,
        "n_regular_running": 0,
        "n_best_effort_waitings": 0,
        "n_best_effort_running": 0,
    }
    llm.output_processor.process_outputs.assert_called_once_with(
        [], outputs.timestamp, None)
    llm.engine_core.abort_requests_async.assert_awaited_once_with([])
    llm.output_processor.propagate_error.assert_called_once()


@pytest.mark.asyncio
async def test_generate_uses_private_output_collector_path():
    llm = _make_async_llm()
    queue = MagicMock()
    queue.get_nowait.side_effect = [
        _make_request_output(),
    ]
    llm._add_request_output_collector = AsyncMock(return_value=queue)

    outputs = [out async for out in llm.generate(
        prompt="prompt",
        sampling_params=SamplingParams(max_tokens=1),
        request_id="req-0",
    )]

    assert [out.request_id for out in outputs] == ["req-0"]
    llm._add_request_output_collector.assert_awaited_once()

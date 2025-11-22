# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union
from dataclasses import dataclass
import math
import asyncio
import json
import bisect

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorOutput
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

import SLOsServe_C
from dataclasses import dataclass, field



logger = init_logger(__name__)
        
# class PerfModel:
#     # _HW_PARAMS = { # model, para_config, hardware -> [k1, k2, b] 
#     #     ('Qwen/Qwen2.5-7B-Instruct', '1-1', 'a100'): [4.1e-5, 0, 1.3e-2],
#     #     ('TestModel', '1-1', 'a100'): [1,0,4],
#     #     ('facebook/opt-125m', '1-1', 'a100'): [4.1e-5, 0, 1.3e-2],
#     #     ('google/gemma-3-27b-it', '2-1', 'a100'): [4.1e-5, 0, 1.3e-2],
#     #     ('google/gemma-3-7b-it', '1-1', 'a100'): [4.1e-5, 0, 1.3e-2],
#     #     ('openai/gpt-oss-20b', '2-1', 'a100'): [4.1e-5, 0, 1.3e-2],
#     # }
    
#     def __init__(self, hardware_params: list[float]):
#         self.hardware_params = hardware_params
#         assert len(hardware_params) % 5 == 0 and len(hardware_params) > 0
    
#     def estimate_batch_time(self, num_tokens: list[tuple[int, int]]) -> float:
#         num_reqs = len(num_tokens)
#         num_tot_tokens = sum([x[1] for x in num_tokens], start = 0)
#         num_past_tokens = sum([x[0] for x in num_tokens], start = 0)
#         num_decode_steps = 1
#         times = []
#         for i in range(0, len(self.hardware_params), 5):
#             k1, k2, k3, k4, b = self.hardware_params[i:i+5]
#             time = k1 * num_tot_tokens + k2 * num_reqs + k3 * num_past_tokens + k4 * num_decode_steps + b
#             times.append(time)
#         return max(times)
    
#     def estimate_prefill_time(self, num_tokens: int) -> float:
#         return self.estimate_batch_time([(0, num_tokens)])
    
#     @staticmethod
#     def get_perf_model(
#         vllm_config: VllmConfig,
#     ) -> 'PerfModel':
#         model_name = vllm_config.model_config.model
#         para_config = f'{vllm_config.parallel_config.tensor_parallel_size}-{vllm_config.parallel_config.pipeline_parallel_size}'
#         hardware = 'a100'
#         from motivation.common import get_hardware_params
#         hw_params = get_hardware_params(model_name)
#         assert hw_params is not None
#         # if (model_name, para_config, hardware) not in PerfModel._HW_PARAMS:
#             # logger.warning(f'No hardware parameters found for {model_name}, {para_config}, {hardware}')                
#         return PerfModel(hw_params)

class Timer:
    def __init__(self):
        self.tot = 0
        self.times = defaultdict(float)
        self._time = time.time()
        self.s = 'START'
    
    def stop(self, name: str):
        delta= time.time() - self._time
        self.tot += delta
        self.times[f'{self.s}->{name}'] += delta
        self.s = name
        self._time = time.time()

    def get(self):
        times = list(self.times.items())
        times.sort(key = lambda x: x[1], reverse=True)
        return times

class DefaultTimer:
    async def process(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        pass 
    
    def current_time(self) -> float:
        return time.time()

class LogicalTimer:
    def __init__(self, perf_model: 'PerfModel'):
        self.perf_model = perf_model
        self.logical_time = 0
    
    async def process(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        batch = []
        for req_data in scheduler_output.scheduled_new_reqs:
            if req_data.req_id not in scheduler_output.num_scheduled_tokens:
                continue
            batch.append((req_data.num_computed_tokens, 
                          scheduler_output.num_scheduled_tokens[req_data.req_id]))
        for req_id, num_computed_tokens in zip(
            scheduler_output.scheduled_cached_reqs.req_ids,
            scheduler_output.scheduled_cached_reqs.num_computed_tokens
        ):
            if req_id not in scheduler_output.num_scheduled_tokens:
                continue
            batch.append((num_computed_tokens, scheduler_output.num_scheduled_tokens[req_id]))
        batch_time = self.perf_model.get_batch_time(batch)
        self.logical_time += batch_time
        return self.logical_time

    def current_time(self) -> float:
        return self.logical_time
    
    def set_time(self, time: float):
        self.logical_time = time

class EnumlateTimer:
    def __init__(self, perf_model: 'PerfModel'):
        self.perf_model = perf_model
    
    async def process(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        batch = []
        for req_data in scheduler_output.scheduled_new_reqs:
            if req_data.req_id not in scheduler_output.num_scheduled_tokens:
                continue
            batch.append((req_data.num_computed_tokens, 
                          scheduler_output.num_scheduled_tokens[req_data.req_id]))
        for req_id, num_computed_tokens in zip(
            scheduler_output.scheduled_cached_reqs.req_ids,
            scheduler_output.scheduled_cached_reqs.num_computed_tokens
        ):
            if req_id not in scheduler_output.num_scheduled_tokens:
                continue
            batch.append((num_computed_tokens, scheduler_output.num_scheduled_tokens[req_id]))
        batch_time = self.perf_model.get_batch_time(batch)
        await asyncio.sleep(batch_time)
        return self.current_time()

    def current_time(self) -> float:
        return time.time()
    
    def set_time(self, time: float):
        raise NotImplementedError

class SchedulerAdmCtrl(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        logger.info('SchedulerAdmCtrl::init')      
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.mm_registry = mm_registry
        self.include_finished_set = include_finished_set
        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.scheduler_config.is_mock_connector:
            self.connector = MockConnector()
        elif self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        # self.waiting = create_request_queue(self.policy)
        self.waiting_kv_xfer: list[Request] = []
        self.waiting_attainable: list[Request] = []
        self.waiting_unattainable: list[Request] = []
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        from motivation.common import PerfModel
        self.perf_model = PerfModel.get_perf_model(self.scheduler_config.model_name, self.scheduler_config.length_pattern)
        self.slosserve_scheduler = SLOsServe_C.AdmCtrlScheduler(self.scheduler_config.scheduling_policy, False)
        self.perf_model.hardware_params[4] += self.scheduler_config.scheduling_overhead
        self.slosserve_scheduler.set_ar_planner(
            tpots = [self.get_tpot_slo()],
            hardware_params = self.perf_model.hardware_params,
            fixed_bs = False 
        )
        logger.info(f'fetching perf model for model_name: {self.scheduler_config.model_name} and length_pattern: {self.scheduler_config.length_pattern}')
        logger.info(f'updating slosserve scheduler with TPOT: {self.get_tpot_slo()} and hardware_params: {self.perf_model.hardware_params}')
        if self.scheduler_config.scheduling_policy in ["dp", "edf"]:
            self.stateless_schedule_fn = self._schedule_stateless_slosserve
        elif 'vllm' in self.scheduler_config.scheduling_policy:
            self.stateless_schedule_fn = self._schedule_stateless_vllm
        else:
            raise ValueError(f"Unknown admission policy: {self.scheduler_config.scheduling_policy}")
        
        if self.vllm_config.is_simulation == 'logical':
            self.timer = LogicalTimer(self.perf_model)
        elif self.vllm_config.is_simulation == 'emulate':
            self.timer = EnumlateTimer(self.perf_model)
        else:
            self.timer = DefaultTimer()
        logger.info(f"Timer: {self.timer}")
        
        self.rejected_reqs: list[Request] = []
        self._req_cached_tokens: dict[str, int] = {}
        self._profile_events: list[dict] = []
        print('kv cache manager initialized', self.kv_cache_manager)
        self._load_statistics: list = []
        self._timer = Timer()
    
    def get_load_statistics(self, t: float = 5) -> list[dict[str, Any]]:
        earliest_time = time.time() - t
        idx = len(self._load_statistics) - 1
        while idx >= 0 and self._load_statistics[idx]['timestamp'] >= earliest_time:
            idx -= 1
        self._load_statistics = self._load_statistics[idx+1:]
        return self._load_statistics

    def reset(self, profile_events: dict | None = None):
        logger.info('SchedulerAdmCtrl::reset')
        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self._profile_events = profile_events
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if self.include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        
        if self.scheduler_config.is_mock_connector:
            if self.connector is not None:
                self.connector.reset()
            else:
                self.connector = MockConnector()

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        # self.connector = None
        # if self.vllm_config.kv_transfer_config is not None:
        #     assert len(self.kv_cache_config.kv_cache_groups) == 1, (
        #         "Multiple KV cache groups are not currently supported "
        #         "with KV connectors")
        #     self.connector = KVConnectorFactory.create_connector(
        #         config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        # self.kv_event_publisher = EventPublisherFactory.create(
        #     self.kv_events_config,
        #     self.parallel_config.data_parallel_rank,
        # )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        # self.waiting = create_request_queue(self.policy)
        self.waiting_kv_xfer: list[Request] = []
        self.waiting_attainable: list[Request] = []
        self.waiting_unattainable: list[Request] = []
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=self.vllm_config.model_config,
            scheduler_config=self.vllm_config.scheduler_config,
            mm_registry=self.mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = self.vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=self.kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        from motivation.common import PerfModel
        self.perf_model = PerfModel.get_perf_model(self.scheduler_config.model_name, self.scheduler_config.length_pattern)
        self.perf_model.hardware_params[4] += self.scheduler_config.scheduling_overhead
        self.slosserve_scheduler = SLOsServe_C.AdmCtrlScheduler(self.scheduler_config.scheduling_policy, False)
        self.slosserve_scheduler.set_ar_planner(
            tpots = [self.get_tpot_slo()],
            hardware_params = self.perf_model.hardware_params,
            fixed_bs = False 
        )
        logger.info(f'fetching perf model for model_name: {self.scheduler_config.model_name} and length_pattern: {self.scheduler_config.length_pattern}')
        logger.info(f'updating slosserve scheduler with TPOT: {self.get_tpot_slo()} and hardware_params: {self.perf_model.hardware_params}')
        if self.scheduler_config.scheduling_policy in ["dp", "edf"]:
            self.stateless_schedule_fn = self._schedule_stateless_slosserve
        elif 'vllm' in self.scheduler_config.scheduling_policy:
            self.stateless_schedule_fn = self._schedule_stateless_vllm
        else:
            raise ValueError(f"Unknown admission policy: {self.scheduler_config.scheduling_policy}")
        
        if self.vllm_config.is_simulation == 'logical':
            self.timer = LogicalTimer(self.perf_model)
        elif self.vllm_config.is_simulation == 'emulate':
            self.timer = EnumlateTimer(self.perf_model)
        else:
            self.timer = DefaultTimer()
        logger.info(f"Timer: {self.timer}")
        
        self.rejected_reqs: list[Request] = []
        self._req_cached_tokens: dict[str, int] = {}
        logger.info(f'SchedulerAdmCtrl::reset {self.scheduler_config}, max_num_batched_tokens: {self.max_num_scheduled_tokens}')
    
    def get_tpot_slo(self):
        return self.scheduler_config.slo_tpot

    def current_time(self) -> float:
        return self.timer.current_time()

    def _is_attainable(self, request: Request) -> bool:
        '''
        This function decides whether a request's SLO is attainable. 
        The decision is a necessary but not a sufficient condition for the request to be rejected.
        '''
        remain_prefill_tokens = max(request.num_prompt_tokens - request.num_computed_tokens, 0)
        if remain_prefill_tokens == 0: return True
        # estimated_prefill_time = self.perf_model.get_batch_time([(request.num_computed_tokens, remain_prefill_tokens)])
        
        return self.timer.current_time() < request.prefill_ddl

    def _schedule_stateless_vllm(self) -> tuple[tuple[list[Request], list[Request], list[Request], list[Request]], dict[str, int]]:
        '''
        waiting_attainable
        running
        waiting_unattainable
        
        the schedule needs to decide for each request in waiting_attainable, whether promote them to waiting_unattainable or running or stay, 
        for every request in running, it decides whether to stay or preempt.
        for every request in waiting_unattainable, it decides whether add them to best_effort.
        
        output
        change the status of the requests 
        num_scheduled_tokens:
        
        '''
        preempted_reqs: list[Request] = []
        admitted_reqs: list[Request] = []
        rejected_reqs: list[Request] = []
        resumed_reqs: list[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        
        # For logging.
        # scheduled_timestamp = time.monotonic()
        # First, schedule the RUNNING requests.
        
        req_index = 0
        token_budget = self.max_num_scheduled_tokens
        if 'sarathi+' in self.scheduler_config.scheduling_policy and \
            all(req.num_prompt_tokens > req.num_computed_tokens for req in self.running):
                token_budget = 8192
        
        num_running_reqs = len(self.running)
        num_free_blocks = self.kv_cache_manager.get_num_free_blocks()
        
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            num_blocks_to_allocate = self.kv_cache_manager.get_num_slots_to_allocate(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens)
            
            while True:
                if num_blocks_to_allocate > num_free_blocks:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        # Priority policy
                        preempted_req = max(
                            self.running,
                            key=lambda r: ((not r in preempted_reqs) and (not r == request) and not (r.request_id in num_scheduled_tokens),
                                           r.priority, - r.arrival_time),
                        )
                    else:
                        # FCFS policy
                        preempted_req = max(
                            self.running, 
                            key = lambda r: ((not r in preempted_reqs) and (not r == request) and not (r.request_id in num_scheduled_tokens), - r.arrival_time),
                        )
                        
                    if preempted_req == request or preempted_req in preempted_reqs or preempted_req.request_id in num_scheduled_tokens:
                        # No more request to preempt.
                        can_schedule = False
                        break
                    
                    preempted_reqs.append(preempted_req)
                    num_running_reqs -= 1
                    num_free_blocks += self.kv_cache_manager.get_num_freed_blocks_after_free(preempted_req)
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break

            # Schedule the request.
            token_budget -= num_new_tokens
            num_free_blocks -= num_blocks_to_allocate
            num_scheduled_tokens[request.request_id] = num_new_tokens
            req_index += 1

            # Speculative decode related.
            # if request.spec_token_ids:
            #     num_scheduled_spec_tokens = (num_new_tokens +
            #                                  request.num_computed_tokens -
            #                                  request.num_tokens)
            #     if num_scheduled_spec_tokens > 0:
            #         # Trim spec_token_ids list to num_scheduled_spec_tokens.
            #         del request.spec_token_ids[num_scheduled_spec_tokens:]
            #         scheduled_spec_decode_tokens[request.request_id] = (
            #             request.spec_token_ids)
        
        if self.scheduler_config.queue_length_threshold is not None:
            rejected_reqs.extend(self.waiting_attainable[self.scheduler_config.queue_length_threshold:])
            self.waiting_attainable = self.waiting_attainable[:self.scheduler_config.queue_length_threshold]
            
        if 'edf' in self.scheduler_config.scheduling_policy:
            self.waiting_attainable = sorted(self.waiting_attainable,\
                key= lambda r: r.sampling_params.extra_args['slo_ttft'] + r.arrival_time)
        
        if 'sarathi' in self.scheduler_config.scheduling_policy:
            # prioritize decode requests
            self.waiting_attainable = sorted(self.waiting_attainable,\
                key= lambda r: r.num_prompt_tokens - r.num_computed_tokens)
        else: 
            # prioritize prefill requests
            self.waiting_attainable = sorted(self.waiting_attainable,\
                key= lambda r: - r.num_prompt_tokens + r.num_computed_tokens)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            req_index = 0
            while token_budget > 0 and req_index < len(self.waiting_attainable) + len(self.waiting_unattainable):
                if num_running_reqs == self.max_num_running_reqs:
                    break
                is_waiting_attainable = req_index < len(self.waiting_attainable)
                if is_waiting_attainable:
                    request = self.waiting_attainable[req_index]
                    if self.scheduler_config.allow_rejection and (not self._is_attainable(request)):
                        rejected_reqs.append(request)
                        req_index += 1
                        continue
                else:
                    request = self.waiting_unattainable[req_index - len(self.waiting_attainable)]

                # Check that adding the request still respects the max_loras
                # constraint.
                # if (self.lora_config and request.lora_request and
                #     (len(scheduled_loras) == self.lora_config.max_loras and
                #      request.lora_request.lora_int_id not in scheduled_loras)):
                #     # Scheduling would exceed max_loras, skip.
                #     self.waiting.pop_request()
                #     skipped_waiting_requests.prepend_request(request)
                #     continue

                # num_external_computed_tokens = 0
                # load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0: # this can be new request or resumed request
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)
                    # print('req', request.request_id, 'cache hit', num_new_local_computed_tokens)
                    # Get externally-cached tokens if using a KVConnector.
                    # if self.connector is not None:
                    #     num_external_computed_tokens, load_kv_async = (
                    #         self.connector.get_num_new_matched_tokens(
                    #             request, num_new_local_computed_tokens))

                    # Total computed tokens (local + external).
                    num_computed_tokens = num_new_local_computed_tokens
                    # num_computed_tokens = (num_new_local_computed_tokens +
                    #                        num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    # print('abnormal request', request.request_id, 
                    #       'num_computed_tokens', request.num_computed_tokens,
                    #       'num_tokens', request.num_tokens,
                    #       'status', request.status,
                    #       'is_waiting_attainable', is_waiting_attainable)
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens
                # print('req', request.request_id, 'num_tokens', request.num_tokens, 'num_computed_tokens', num_computed_tokens)
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (0 < self.scheduler_config.long_prefill_token_threshold
                        < num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)

                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                if not self.scheduler_config.chunked_prefill_enabled and \
                    num_new_tokens > token_budget:
                    # self.waiting.pop_request()
                    # skipped_waiting_requests.prepend_request(request)
                    req_index += 1
                    continue

                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                num_blocks_to_allocate = self.kv_cache_manager.get_num_slots_to_allocate(
                    request,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                )

                if num_blocks_to_allocate > num_free_blocks:
                    # The request cannot be scheduled.
                    break
                
                if is_waiting_attainable:
                    admitted_reqs.append(request)
                else:
                    resumed_reqs.append(request)

                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                num_free_blocks -= num_blocks_to_allocate
                num_running_reqs += 1
                req_index += 1

                if num_new_tokens > 1 and self.scheduler_config.scheduling_policy in ('vllm', 'vllm+', 'vllm-edf'):
                    break

        return (preempted_reqs, # Running to waiting_unattainable
                admitted_reqs, # Waiting_unattainable to running
                rejected_reqs, # waiting_attainable to unattainable
                resumed_reqs # waiting_unattainable to running
        ), num_scheduled_tokens
    
    def _schedule_stateless_slosserve(self, waiting_reqs: list[Request] | None = None) -> tuple[tuple[list[Request], list[Request], list[Request], list[Request]], dict[str, int]]:
        self._timer.stop('slosserve_start')
        is_batch_sch = False
        if waiting_reqs is None:
            is_batch_sch = True
            waiting_reqs = self.waiting_attainable
        
        if not len(waiting_reqs) and all(req.num_prompt_tokens <= req.num_computed_tokens for req in self.running):
            # if len(self.running): 
            #     logger.info(f'scheduling {len(self.running)} requests')
            bs = self.perf_model.get_bs(
                t = self.scheduler_config.slo_tpot,
                num_reqs = len(self.running),
                num_past_tokens = sum([req.num_computed_tokens for req in self.running]),
                num_decode_steps = 1
            )
            self._timer.stop('shortcut')
            self._load_statistics.append({
                'type': 'slosserve',
                'timestamp': time.time(),
                'future_batches': [{
                    'n_tokens': len(self.running),
                    'prefill_bs': bs - len(self.running),
                    'estimated_time': self.scheduler_config.slo_tpot,
                    'next': 0
                }]
            })
                        
            return ([], [], [], []), {req.request_id: 1 for req in self.running if req not in waiting_reqs}
        
        promax_reqs = []
        num_free_blocks = self.kv_cache_manager.get_num_free_blocks()
        # mem_per_request = self.max_model_len // self.kv_cache_manager.block_size
        
        for req in waiting_reqs:
            if req.num_computed_tokens == 0:
                new_computed_blocks, num_new_local_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(req)
                input_length = req.num_tokens - num_new_local_computed_tokens
            else:
                input_length = req.num_tokens - req.num_computed_tokens
                
            # over provision the memory 
            mem = math.ceil((req.num_prompt_tokens + self.scheduler_config.max_decoding_length) / self.block_size)
            prefill_mem = math.ceil((req.num_prompt_tokens) / self.block_size)
            # Prefer a client-specified prefill deadline if provided via extra_args.
            profit = req.sampling_params.extra_args.get('profit', 1)
            promax_reqs.append(SLOsServe_C.Request(
                id = req.request_id,
                is_new_req = True, 
                ddl = req.prefill_ddl,
                input_length = input_length,
                n_computed_tokens = req.num_computed_tokens,
                profit = profit,
                mem = mem,
                tpot_idx = 0,
                prefill_mem = prefill_mem,
                prefill_device_id = 0, 
                decode_device_id = 0,
                prefill_only = req.max_tokens == 1
            ))
            
            if is_batch_sch and self.scheduler_config.record_events:
                self._profile_events.append({
                    'event_type': 'req_state',
                    'timestamp': time.time(),
                    'request_id': req.request_id,
                    'state': 'waiting_attainable',
                    'num_prompt_tokens': req.num_prompt_tokens,
                    'num_computed_tokens': req.num_computed_tokens,
                    'num_output_tokens': req.num_output_tokens,
                    'ddl': req.prefill_ddl,
                })
        self._timer.stop('prepare_waitings')

        for req in self.running:
            mem = math.ceil((req.num_prompt_tokens + req.max_tokens) / self.block_size)
            prefill_mem = math.ceil((req.num_prompt_tokens) / self.block_size)
            profit = req.sampling_params.extra_args.get('profit', 1)
            ddl = req.prefill_ddl + max(req.num_output_tokens - self.scheduler_config.slosserve_token_headroom, 0) * self.get_tpot_slo()
            ddl = max(ddl, self.scheduled_timestamp)
            promax_reqs.append(SLOsServe_C.Request(
                id = req.request_id,
                is_new_req = False,
                ddl = ddl,
                input_length = max(req.num_prompt_tokens \
                    - req.num_computed_tokens, 0),
                profit = profit,
                n_computed_tokens = req.num_computed_tokens,
                mem = mem,
                tpot_idx = 0,
                prefill_mem = prefill_mem,
                prefill_device_id = 0, 
                decode_device_id = 0,
                prefill_only = req.max_tokens == 1
            ))
            if is_batch_sch and self.scheduler_config.record_events:
                self._profile_events.append({
                    'event_type': 'req_state',
                    'timestamp': time.time(),
                    'request_id': req.request_id,
                    'state': 'running',
                    'num_prompt_tokens': req.num_prompt_tokens,
                    'num_computed_tokens': req.num_computed_tokens,
                    'num_output_tokens': req.num_output_tokens,
                    'ddl': ddl,
                })
        
        self._timer.stop('prepare_runnings')
        for req in promax_reqs:
            req.ddl -= self.scheduled_timestamp
        
        start_time = time.time()
        is_feasible, accpeted_ids, batch_schedules = self.slosserve_scheduler.schedule(
            promax_reqs, num_free_blocks, 0.0, False
        )
        overhead = time.time() - start_time
        
        self._timer.stop('backend_schedule')
        
        
        if not hasattr(self, 'i'):
            self.i = 0
        else:
            self.i += 1
        
        preempted_reqs: list[Request] = []
        admitted_reqs: list[Request] = []
        rejected_reqs: list[Request] = []
        resumed_reqs: list[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        
        for request in waiting_reqs:
            if request.request_id in accpeted_ids:
                admitted_reqs.append(request)
            elif self.scheduler_config.admission_mode == 'instant' or not self._is_attainable(request):
                rejected_reqs.append(request)
                
        for request in self.running:
            if request.request_id not in accpeted_ids:
                preempted_reqs.append(request)
        
        if not len(batch_schedules):
            self._timer.stop('no_batch_schedules')
            return (preempted_reqs, admitted_reqs, rejected_reqs, resumed_reqs), num_scheduled_tokens
        
        if is_batch_sch and self.scheduler_config.record_events:
            self._profile_events.append({
                'event_type': 'schedule_problem', 
                'timestamp': time.time(),
                'batch_id': getattr(self, 'batch_id', -1),
                'reqs': [{
                        'id': req.id,
                        'is_new_req': req.is_new_req,
                        'ddl': req.ddl,
                        'input_length': req.input_length,
                        'profit': req.profit,
                        'mem': req.mem,
                        'n_computed_tokens': req.n_computed_tokens,
                        'tpot_idx': req.tpot_idx,
                        'prefill_mem': req.prefill_mem,
                        'prefill_device_id': req.prefill_device_id,
                        'decode_device_id': req.decode_device_id,
                        'prefill_only': req.prefill_only,
                    } for req in promax_reqs],
                'num_free_blocks': num_free_blocks,
                'is_feasible': is_feasible,
                'estimated_time': batch_schedules[0].estimated_time,
                'accepted_ids': accpeted_ids,
                'batch_schedule': [{'id': batch.id, 'n': batch.n} for batch in batch_schedules[0].req_batches],
                'overhead': overhead
            })

        self._timer.stop('schedule_problem')
        for req_batch in batch_schedules[0].req_batches:
            if req_batch.n > 0:
                num_scheduled_tokens[req_batch.id] = req_batch.n

        future_batches = []
        for batch in batch_schedules[:10]:
            n_tokens = sum([req_batch.n for req_batch in batch.req_batches])
            future_batches.append({
                'n_tokens': n_tokens,
                'prefill_bs': batch.prefill_bs,
                'estimated_time': batch.estimated_time,
                'next': batch.next
            })


        self._load_statistics.append({
            'timestamp': time.time(),
            'type': 'slosserve',
            'future_batches': future_batches            
        })
        
        self._timer.stop(f'prepare_future_batches {len(batch_schedules)}')
        
        return (preempted_reqs, admitted_reqs, rejected_reqs, resumed_reqs), num_scheduled_tokens

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.
        
        # logger.info('SchedulerAdmCtrl::schedule')

        # scheduled_new_reqs: list[Request] = []
        # scheduled_resumed_reqs: list[Request] = []
        # scheduled_running_reqs: list[Request] = []
        # preempted_reqs: list[Request] = []
        
        self._timer = Timer()
        
        self._load_statistics.append({
            'type': 'pool',
            'timestamp': time.time(),
            'waiting_size': len(self.waiting_attainable),
            'running_size': len(self.running)
        })

        
        # num_scheduled_tokens: dict[str, int] = {}
        # token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        # encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        
        # Check if the scheduling constraints are satisfied.
        # total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        # assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        # assert token_budget >= 0
        # assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        # assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
        #         len(scheduled_running_reqs) <= len(self.running))
        
        # for every request in waiting_kv_xfer, we check if it is finished_recving
        requests_kv_xfer_ready = [request for request in self.waiting_kv_xfer if self._update_waiting_for_remote_kv(request)]
        for request in requests_kv_xfer_ready:
            self._profile_events.append({
                "event_type": "kv_xfer_ready",
                "request_id": request.request_id,
                "timestamp": time.time(),
            })
            self.waiting_kv_xfer.remove(request)
            if self.policy == SchedulingPolicy.FCFS:
                self.waiting_attainable.append(request)
            else:
                assert self.policy == SchedulingPolicy.PRIORITY
                idx = 0
                while idx < len(self.waiting_attainable) and self.waiting_attainable[idx].priority < request.priority:
                    idx += 1
                self.waiting_attainable.insert(idx, request)
            request.status = RequestStatus.WAITING
        
        self._timer.stop('kv_xfer')
            
        # print('---------------------BEFORE-----------------------------')
        # for queue_name, queue in [('Q[waiting_attainable]', self.waiting_attainable), ('Q[waiting_unattainable]', self.waiting_unattainable), ('Q[running]', self.running)]:
        #     print(queue_name, [req.request_id for req in queue])
        
            
        self.scheduled_timestamp = self.timer.current_time()
        (preempted_reqs, admitted_reqs, rejected_reqs, resumed_reqs), num_scheduled_tokens = self.stateless_schedule_fn()
        scheduled_running_reqs = list([self.requests[x] for x in num_scheduled_tokens.keys() \
            if self.requests[x].status == RequestStatus.RUNNING and self.requests[x].scheduled])
       
        self._timer.stop('stateless_schedule_fn')
       
        # for name, reqs in [('admitted', admitted_reqs), ('resumed', resumed_reqs), ('preempted', preempted_reqs), ('rejected', rejected_reqs)]:
        #     print(name, [req.request_id for req in reqs])
        
        args = []
        
        if self.scheduler_config.allow_rejection:
            # if len(rejected_reqs) > 0 or len(preempted_reqs) > 0:
            #     print(f'reject {len(rejected_reqs)} requests, preempt {len(preempted_reqs)} requests')
            self.reject_requests(
                [req.request_id for req in rejected_reqs] + [req.request_id for req in preempted_reqs],
            )
            
        else:
            args.extend([
                (preempted_reqs, self.running, self.waiting_unattainable, RequestStatus.PREEMPTED, EngineCoreEventType.PREEMPTED, 'preempted'),
                (rejected_reqs, self.waiting_attainable, self.waiting_unattainable, RequestStatus.PREEMPTED, EngineCoreEventType.REJECTED, 'rejected')
            ])
        
        # the order is important here, since finish_requests may change self.running / self.waiting_attainable to new instances
        args.extend([
            (admitted_reqs, self.waiting_attainable, self.running, RequestStatus.RUNNING, EngineCoreEventType.SCHEDULED, 'admitted'),
            (resumed_reqs, self.waiting_unattainable, self.running, RequestStatus.RUNNING, EngineCoreEventType.RESUMED, 'resumed')
        ])
            
        # Stateful update
        for reqs, from_pool, to_pool, status, event_type, event_type_str in args:
            # print(event_type_str, reqs)
            for req in reqs:
                assert req in from_pool, f"{event_type_str} req {req.request_id} not in from_pool {from_pool}"
                from_pool.remove(req)
                assert req not in from_pool, f"{event_type_str} req {req.request_id} still in from_pool {from_pool}"
                assert req not in to_pool, f"{event_type_str} req {req.request_id} already in to_pool {to_pool}"
                to_pool.append(req)
                req.status = status
                if self.log_stats:
                    req.record_event(event_type, self.scheduled_timestamp)
        # for queue_name, queue in [('Q[waiting_attainable]', self.waiting_attainable), ('Q[waiting_unattainable]', self.waiting_unattainable), ('Q[running]', self.running)]:
        #     print(queue_name, [req.request_id for req in queue])
        # print('---------------------AFTER-----------------------------')
       # for preempted requests, we need to free the blockss
        for req in preempted_reqs:
            self.kv_cache_manager.free(req)
            req.num_computed_tokens = 0
        
        self._timer.stop('preempted_reqs')
        
            
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        
        new_requests: list[Request] = []
        # for request entering running, we need to allocate the blocks
        failed_requests: list[Request] = []
        for request_id, num_new_tokens in num_scheduled_tokens.items():
            request = self.requests[request_id]
            if request in admitted_reqs or request in resumed_reqs:
                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = \
                                self.kv_cache_manager.get_computed_blocks(
                                    request)
                    self._timer.stop('get_computed_blocks')
                    # print('req', request.request_id, 'num_new_tokens', num_new_tokens, 
                    #       'num_new_local_computed_tokens', num_new_local_computed_tokens)
                    max_num_new_tokens = request.num_tokens - num_new_local_computed_tokens
                    # assert max_num_new_tokens <= num_new_tokens
                    if max_num_new_tokens < num_new_tokens:
                        num_new_tokens = max_num_new_tokens
                        num_scheduled_tokens[request_id] = max_num_new_tokens
                    
                    # logger.info(
                    #     f"allocate_slots: engine_id={self.scheduler_config.engine_id}, "
                    #     f"request_id={request.request_id}, "
                    #     f"num_new_tokens={num_new_tokens}, "
                    #     f"num_new_local_computed_tokens={num_new_local_computed_tokens}, "
                    #     f"new_computed_blocks={new_computed_blocks}, "
                    #     f"request.num_computed_tokens={request.num_computed_tokens}"
                    # )
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        new_computed_blocks,
                        num_lookahead_tokens=0,
                        delay_cache_blocks=False,
                    )
                    self._timer.stop('allocate_slots_1')
                    request.num_computed_tokens = num_new_local_computed_tokens
                    
                    if request.num_cached_tokens < 0:
                        request.num_cached_tokens = num_new_local_computed_tokens
                        self._req_cached_tokens[request.request_id] = num_new_local_computed_tokens
                        
                # KV Xfer request after async KV recvs are completed
                else:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens
                    )
                    self._timer.stop('allocate_slots_2')
            else:
                if num_new_tokens == 0:
                    new_blocks = self.kv_cache_manager.create_empty_block_list()
                else:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens
                    )
                    self._timer.stop('allocate_slots_3')
            # assert new_blocks is not None
            if new_blocks is None:
                logger.error(f"Request {request.request_id} cannot be scheduled")
                failed_requests.append(request)
                continue
            
            req_to_new_blocks[request.request_id] = new_blocks
            if not request.scheduled:
                new_requests.append(request)
            request.scheduled = True 
            
        self._timer.stop(f'allocate_memory_{len(num_scheduled_tokens)}')
        
        for request in failed_requests:
            num_scheduled_tokens.pop(request.request_id)
            if request in scheduled_running_reqs:
                scheduled_running_reqs.remove(request)
        
        self._timer.stop('failed_requests')
        
        
        scheduled_resumed_reqs = list(resumed_reqs)
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        
        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        self._timer.stop('get_common_prefix_blocks')

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in new_requests
        ]
        
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(self.running,
                                     scheduled_spec_decode_tokens))
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        self._timer.stop('get_scheduler_output')

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)
            

        self._update_after_schedule(scheduler_output)
        self._timer.stop('end')
        
        if self._timer.tot > 0.1:
            logger.warning(f'LONG SCHEDULING: {self._timer.get()}, time: {self._timer.tot}, # waiting: {len(self.waiting_attainable)}, # running: {len(self.running)}')
        return scheduler_output


    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[Optional[tuple[list[int], ...]]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to its index in the batch.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[req.request_id] = i

        if not structured_output_request_ids:
            bitmask = None
        else:
            bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
        return structured_output_request_ids, bitmask

    def get_rejected_requests(self) -> list[Request]:
        rejected_reqs = self.rejected_reqs
        if self.finished_req_ids_dict is not None:
            for req in rejected_reqs:
                assert req.client_index in self.finished_req_ids_dict
                assert req.request_id in self.finished_req_ids_dict[req.client_index]
                self.finished_req_ids_dict[req.client_index].remove(req.request_id)
        self.rejected_reqs = []
        return rejected_reqs

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
        elapsed_time: float = 0.0,
    ) -> dict[int, EngineCoreOutputs]:

        for stat in self._load_statistics[::-1]:
            if stat['type'] == 'slosserve':
                stat.update({'elapsed': elapsed_time})
        
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_reqs: set[Request] = set()
        # stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            # assert num_tokens_scheduled > 0
            if num_tokens_scheduled == 0: continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                stopped_reqs.add(request)
                # if status_before_stop == RequestStatus.RUNNING:
                #     stopped_running_reqs.add(request)
                # else:
                #     stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    ))

            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_reqs:
            self.waiting_attainable = remove_all(self.waiting_attainable, stopped_reqs)
            self.running = remove_all(self.running, stopped_reqs)
            self.waiting_unattainable = remove_all(self.waiting_unattainable, stopped_reqs)
        # if stopped_preempted_reqs:
        #     # This is a rare case and unlikely to impact performance.
        #     self.waiting_unattainable.remove_requests(stopped_preempted_reqs)
        
        for req in self.rejected_reqs:
            outputs[req.client_index].append(
                EngineCoreOutput(
                    request_id=req.request_id,
                    new_token_ids=[],
                    finish_reason=req.get_finished_reason(),
                )
            )
        self.rejected_reqs.clear()
        
        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output or self.scheduler_config.is_mock_connector:
            if self.scheduler_config.is_mock_connector:
                model_runner_output.kv_connector_output = KVConnectorOutput()
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
                draft_token_ids.req_ids,
                draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if not spec_token_ids:
                # NOTE(woosuk): request.spec_token_ids should be updated.
                request.spec_token_ids.clear()
            elif self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids)
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting_attainable) + len(self.waiting_unattainable)

    def add_request(self, request: Request) -> bool:
        # logger.info(
        #     f"Adding request: engine_id={self.scheduler_config.engine_id}, "
        #     f"request_id={request.request_id}, max_tokens={request.max_tokens}, "
        #     f"prompt_tokens={request.num_prompt_tokens}, sampling_params={request.sampling_params}"
        # )
        
        if request.request_id in self.requests:
            # logger.info(f"Received request {request.request_id} that is already in the scheduler, finishing the request sending on local server")
            # logger.info(f"New: {request}")
            old_request = self.requests[request.request_id]
            # logger.info(f"Old: {old_request}")
            # logger.info(f"Block Ids: {self.kv_cache_manager.get_block_ids(request.request_id)}")
            old_request.sampling_params = request.sampling_params
            old_request.priority = request.priority
            old_request.arrival_time = request.arrival_time
            old_request.status = RequestStatus.WAITING
            old_request.stop_reason = None
            for k in ['do_remote_prefill', 'do_remote_decode']:
                if request.kv_transfer_params and k in request.kv_transfer_params:
                    old_request.kv_transfer_params[k] = False
            old_request.max_tokens = request.max_tokens
            old_request.scheduled = False
            request = old_request
        
        load_kv_async = False
        if self.connector is not None:
            new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)
            num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))
            # logger.info(f"Request {request.request_id} has {num_external_computed_tokens} new external computed tokens, load_kv_async: {load_kv_async}")
            if load_kv_async: # we only allocate for requests that are waiting for remote kv
                new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_external_computed_tokens,
                        num_new_local_computed_tokens,
                        new_computed_blocks,
                        num_lookahead_tokens=0,
                        delay_cache_blocks=True,
                    )
                if new_blocks is None:
                    # we cannot schedule this request
                    self.finish_requests(request.request_id, RequestStatus.FINISHED_REJECTED)
                    self._profile_events.append({
                        "event_type": "finish",
                        "request_id": request.request_id,
                        "timestamp": time.time(),
                        "finish_reason": "rejected-oom",
                    })
                    return False
                
                self.connector.update_state_after_alloc(
                    request,
                    new_blocks,
                    num_external_computed_tokens,
                )
        
        if self.vllm_config.is_simulation == 'emulate' and request.num_computed_tokens > 0:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                request.num_computed_tokens,
            )
            if new_blocks is None:
                return False

        if load_kv_async:
            self.waiting_kv_xfer.append(request)
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        else:
            if self.scheduler_config.scheduling_policy in ['dp', 'edf']:
                self.scheduled_timestamp = time.time()
                (preempted_reqs, admitted_reqs, rejected_reqs, resumed_reqs), num_scheduled_tokens = self._schedule_stateless_slosserve(waiting_reqs = [request])
                if not len(admitted_reqs): 
                    return False
            else:
                if self.scheduler_config.queue_length_threshold is not None:
                    if len(self.waiting_attainable) + len(self.running) >= self.scheduler_config.queue_length_threshold:
                        return False

            if self.policy == SchedulingPolicy.FCFS:
                self.waiting_attainable.append(request)
            else:
                assert self.policy == SchedulingPolicy.PRIORITY
                idx = 0
                while idx < len(self.waiting_attainable) and self.waiting_attainable[idx].priority < request.priority:
                    idx += 1
                self.waiting_attainable.insert(idx, request)
            request.status = RequestStatus.WAITING
        
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)
        return True

    def reject_requests(
        self, 
        request_ids: Union[str, Iterable[str]],
    ) -> None:
        """Handles the reject signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            elif request.status == RequestStatus.WAITING:
                waiting_requests_to_remove.append(request)
            else:
                raise ValueError(f"Invalid request status: {request.status}")

        # print('#waiting_requests_to_remove', len(waiting_requests_to_remove))
        # print('#running_requests_to_remove', len(running_requests_to_remove))

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting_attainable = remove_all(self.waiting_attainable, waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = RequestStatus.FINISHED_REJECTED
            self.rejected_reqs.append(request)
            self._free_request(request)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        waiting_kv_xfer_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            elif request.status == RequestStatus.PREEMPTED:
                if request in self.running:
                    running_requests_to_remove.add(request)
                else:
                    waiting_requests_to_remove.append(request)
            elif request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                waiting_kv_xfer_requests_to_remove.append(request)
            elif request.status == RequestStatus.WAITING:
                waiting_requests_to_remove.append(request)
            else:
                raise ValueError(f"Invalid request status: {request.status}")

        # print('#waiting_requests_to_remove', len(waiting_requests_to_remove))
        # print('#running_requests_to_remove', len(running_requests_to_remove))

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting_attainable = remove_all(self.waiting_attainable, waiting_requests_to_remove)
        if waiting_kv_xfer_requests_to_remove:
            self.waiting_kv_xfer = remove_all(self.waiting_kv_xfer, waiting_kv_xfer_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting_attainable) + len(self.running) + len(self.waiting_unattainable) + len(self.waiting_kv_xfer)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting_attainable) + len(self.waiting_unattainable),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted
                                   for req in self.running),
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None
        results = self.kv_cache_manager.get_block_ids(request.request_id)
        try:
            (block_ids, ) = results
        except Exception as e:
            logger.error(f"Error getting block ids for request {request.request_id}: {e}, {results}")
            return False, None
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less then one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            scheduler the request during the next step.
        """
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in (kv_connector_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (kv_connector_output.finished_sending or ()):
            if not req_id in self.requests:
                continue
            logger.debug("Finished sending KV transfer for request %s, request status: %s", req_id, self.requests[req_id].status)
            request = self.requests[req_id]
            if request.is_finished():
                self._free_blocks(request)

from abc import abstractmethod


class MockConnectorMetadata(KVConnectorMetadata):
    pass

class MockConnector:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.pending_requests = []
    
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the 
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.
        """
        if request.kv_transfer_params is not None and request.kv_transfer_params.get('do_remote_prefill'):
            if len(request.prompt_token_ids) - num_computed_tokens <= 0:
                return 0, False
            arrival_time = request.kv_transfer_params.get('arrival_time', time.time())
            # Insert (arrival_time, request) into self.pending_requests to keep the list sorted by arrival_time.
            
            arrival_times = [x[0] for x in self.pending_requests]
            insert_idx = bisect.bisect_right(arrival_times, arrival_time)
            self.pending_requests.insert(insert_idx, (arrival_time, request))
            return len(request.prompt_token_ids) - num_computed_tokens, True
        return 0, False
    
    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        pass

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> MockConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        assert connector_output is not None
        
        current_time = time.time()
        idx = 0
        while idx < len(self.pending_requests):
            arrival_time, request = self.pending_requests[idx]
            if current_time < arrival_time:
                break
            if connector_output.finished_recving is None:
                connector_output.finished_recving = set()
            connector_output.finished_recving.add(request.request_id)
            idx += 1
        self.pending_requests = self.pending_requests[idx:]

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        if request.kv_transfer_params is not None and request.kv_transfer_params.get('do_remote_decode'):
            return False, {
                'do_remote_decode': False,
                'do_remote_prefill': True,
                'dispatch_time': time.time(),
                'num_blocks': len(block_ids),
                'num_tokens': request.num_prompt_tokens,
            }
        return False, None
        
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Worker Manager for parallel kernel verification and refinement."""

import tempfile
import shutil
import multiprocessing as mp
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from contextlib import contextmanager


class WorkerManager:
    """Manages multiple verification workers for parallel kernel testing."""

    def __init__(
        self,
        num_workers: int = 4,
        max_rounds: int = 10,
        history_size: int = 8,
        log_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-5",
        high_reasoning_effort: bool = True,
    ):
        """
        Initialize the worker manager.

        Args:
            num_workers: Number of parallel workers
            max_rounds: Maximum rounds of refinement per worker
            history_size: Number of recent rounds to keep in history
            log_dir: Directory for logging (creates temp dir if None)
            openai_api_key: OpenAI API key for LLM refinement
            openai_model: OpenAI model name
            high_reasoning_effort: Whether to use high reasoning effort for OpenAI models
        """
        self.num_workers = num_workers
        self.max_rounds = max_rounds
        self.history_size = history_size
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort

        # Setup logging
        if log_dir is None:
            self.log_dir = Path(tempfile.mkdtemp(prefix="triton_kernel_logs_"))
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Create workers subdirectory
        self.workers_dir = self.log_dir / "workers"
        self.workers_dir.mkdir(exist_ok=True, parents=True)

        # Setup multiprocessing
        self.success_event = mp.Event()  # Shared event to signal success
        self.result_queue = mp.Queue()  # Queue for collecting results
        self.workers: List[mp.Process] = []

        # Setup logger
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = (
            self.log_dir / f"manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def temp_workdirs(self) -> List[Path]:
        """Create temporary working directories for workers."""
        workdirs = []
        try:
            for i in range(self.num_workers):
                workdir = Path(tempfile.mkdtemp(prefix=f"worker_{i}_"))
                workdirs.append(workdir)
                self.logger.info(f"Created workdir for worker {i}: {workdir}")
            yield workdirs
        finally:
            # Cleanup
            for workdir in workdirs:
                if workdir.exists():
                    shutil.rmtree(workdir)
                    self.logger.info(f"Cleaned up workdir: {workdir}")

    def run_verification(
        self,
        kernel_seeds: List[str],
        test_code: str,
        problem_description: str,
        session_log_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run parallel verification on multiple kernel seeds.

        Args:
            kernel_seeds: List of initial kernel implementations
            test_code: Test code to verify kernel correctness
            problem_description: Description of the problem
            session_log_dir: Optional session directory for worker logs

        Returns:
            Dictionary with successful kernel and metadata, or None
        """
        self.logger.info(f"Starting verification with {len(kernel_seeds)} seeds")
        # Reset cross-worker success signal for a fresh run
        try:
            self.success_event.clear()
        except Exception:
            pass
        # Reset workers list to avoid holding stale processes between runs
        self.workers = []

        # Determine where to put worker logs
        if session_log_dir:
            workers_parent_dir = Path(session_log_dir) / "workers"
            workers_parent_dir.mkdir(exist_ok=True)
        else:
            workers_parent_dir = self.workers_dir

        with self.temp_workdirs() as workdirs:
            # Start workers
            for i, (kernel, workdir) in enumerate(zip(kernel_seeds, workdirs)):
                worker_log_dir = workers_parent_dir / f"worker_{i}"
                worker_log_dir.mkdir(exist_ok=True)

                args = (
                    i,
                    kernel,
                    test_code,
                    problem_description,
                    workdir,
                    worker_log_dir,
                    self.max_rounds,
                    self.history_size,
                    self.success_event,
                    self.result_queue,
                    self.openai_api_key,
                    self.openai_model,
                    self.high_reasoning_effort,
                )

                process = mp.Process(target=worker_process, args=args)
                process.start()
                self.workers.append(process)
                self.logger.info(f"Started worker {i}")

            # Wait for any worker to succeed or all to finish
            successful_result = None

            while any(w.is_alive() for w in self.workers):
                try:
                    # Check for results with timeout
                    result = self.result_queue.get(timeout=1.0)
                    if result["success"]:
                        successful_result = result
                        self.logger.info(f"Worker {result['worker_id']} succeeded!")
                        # Signal all workers to stop
                        self.success_event.set()
                        break
                except queue.Empty:
                    continue

            # Wait for all workers to finish
            for worker in self.workers:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    self.logger.warning(f"Terminating worker {worker.pid}")
                    worker.terminate()

            # Collect any remaining results
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    if result["success"] and successful_result is None:
                        successful_result = result
                except queue.Empty:
                    break

            return successful_result

    def cleanup(self):
        """Clean up resources."""
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
        self.workers.clear()


def worker_process(
    worker_id: int,
    kernel_code: str,
    test_code: str,
    problem_description: str,
    workdir: Path,
    log_dir: Path,
    max_rounds: int,
    history_size: int,
    success_event: mp.Event,
    result_queue: mp.Queue,
    openai_api_key: Optional[str],
    openai_model: str,
    high_reasoning_effort: bool,
):
    """
    Worker process for kernel verification and refinement.

    This is run in a separate process.
    """
    # Import here to avoid issues with multiprocessing
    from .worker import VerificationWorker

    worker = VerificationWorker(
        worker_id=worker_id,
        workdir=workdir,
        log_dir=log_dir,
        max_rounds=max_rounds,
        history_size=history_size,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        high_reasoning_effort=high_reasoning_effort,
    )

    result = worker.run(
        kernel_code=kernel_code,
        test_code=test_code,
        problem_description=problem_description,
        success_event=success_event,
    )

    result_queue.put(result)

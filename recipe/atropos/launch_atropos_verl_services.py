#!/usr/bin/env python3
"""
Launch script to synchronize Atropos and VeRL services for integrated training.

This script handles:
1. Starting Atropos environment servers
2. Launching VeRL inference engines
3. Coordinating service discovery
4. Managing weight synchronization
5. Graceful shutdown

Usage:
    python launch_atropos_verl_services.py --config config/gsm8k_grpo_example.yaml
"""

import argparse
import asyncio
import logging
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict

import requests
import yaml

logger = logging.getLogger(__name__)


class AtroposVeRLLauncher:
    """Manages the lifecycle of Atropos and VeRL services for integrated training."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.processes: Dict[str, subprocess.Popen] = {}
        self.service_urls: Dict[str, str] = {}
        self.shutdown_event = asyncio.Event()

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Set defaults
        config.setdefault("atropos", {})
        config["atropos"].setdefault("api_url", "http://localhost:8000")
        config["atropos"].setdefault("environments", ["gsm8k"])

        config.setdefault("verl", {})
        config["verl"].setdefault("inference_engines", ["vllm"])
        config["verl"].setdefault("num_gpus", 8)

        return config

    async def start_atropos_server(self) -> bool:
        """Start the Atropos API server."""
        logger.info("Starting Atropos API server...")

        # Check if already running
        if self._check_service_health(self.config["atropos"]["api_url"]):
            logger.info("Atropos server already running")
            return True

        # Find atropos directory
        atropos_dir = Path.cwd().parent / "atropos"
        if not atropos_dir.exists():
            logger.error(f"Atropos directory not found at {atropos_dir}")
            return False

        # Start API server
        cmd = [sys.executable, "-m", "atroposlib.api", "--port", "8000", "--host", "0.0.0.0"]

        process = subprocess.Popen(cmd, cwd=str(atropos_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes["atropos_api"] = process

        # Wait for startup
        for _ in range(30):
            if self._check_service_health(self.config["atropos"]["api_url"]):
                logger.info("✓ Atropos API server started successfully")
                return True
            await asyncio.sleep(1)

        logger.error("Failed to start Atropos API server")
        return False

    async def start_atropos_environments(self) -> bool:
        """Start configured Atropos environments."""
        atropos_dir = Path.cwd().parent / "atropos"
        environments = self.config["atropos"]["environments"]

        for env_name in environments:
            logger.info(f"Starting {env_name} environment...")

            # Map environment names to scripts
            env_scripts = {"gsm8k": "environments/gsm8k_server.py", "humaneval": "environments/humaneval_server.py", "tool_calling": "environments/tool_calling_server.py"}

            if env_name not in env_scripts:
                logger.warning(f"Unknown environment: {env_name}")
                continue

            script_path = atropos_dir / env_scripts[env_name]
            if not script_path.exists():
                logger.warning(f"Environment script not found: {script_path}")
                continue

            # Start environment
            cmd = [sys.executable, str(script_path), "serve", "--slurm", "false"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes[f"env_{env_name}"] = process

            # Brief wait for initialization
            await asyncio.sleep(2)

        # Verify environments are registered
        try:
            response = requests.get(f"{self.config['atropos']['api_url']}/status")
            if response.status_code == 200:
                status = response.json()
                logger.info(f"✓ Registered environments: {status.get('environments', [])}")
                return True
        except:
            pass

        return False

    async def start_verl_inference_engines(self) -> bool:
        """Start VeRL inference engines (vLLM/SGLang)."""
        logger.info("Starting VeRL inference engines...")

        engines = self.config["verl"]["inference_engines"]
        model_path = self.config.get("actor_rollout_ref", {}).get("model", {}).get("path", "")

        for engine in engines:
            if engine == "vllm":
                await self._start_vllm_server(model_path)
            elif engine == "sglang":
                await self._start_sglang_server(model_path)
            else:
                logger.warning(f"Unknown inference engine: {engine}")

        return len(self.service_urls) > 0

    async def _start_vllm_server(self, model_path: str) -> bool:
        """Start vLLM inference server."""
        port = 8001

        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path, "--port", str(port), "--gpu-memory-utilization", "0.8", "--max-model-len", "2048"]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes["vllm"] = process

        # Wait for startup
        service_url = f"http://localhost:{port}"
        for _ in range(60):
            if self._check_service_health(f"{service_url}/health"):
                self.service_urls["vllm"] = service_url
                logger.info(f"✓ vLLM server started at {service_url}")
                return True
            await asyncio.sleep(1)

        return False

    async def _start_sglang_server(self, model_path: str) -> bool:
        """Start SGLang inference server."""
        # Similar implementation for SGLang
        pass

    def _check_service_health(self, url: str) -> bool:
        """Check if a service is healthy."""
        try:
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except:
            return False

    async def register_inference_endpoints(self) -> bool:
        """Register VeRL inference endpoints with Atropos."""
        logger.info("Registering inference endpoints with Atropos...")

        endpoints = list(self.service_urls.values())
        if not endpoints:
            logger.error("No inference endpoints to register")
            return False

        # Register with Atropos
        try:
            response = requests.post(f"{self.config['atropos']['api_url']}/register_inference_endpoints", json={"endpoints": endpoints})
            if response.status_code == 200:
                logger.info(f"✓ Registered {len(endpoints)} inference endpoints")
                return True
        except Exception as e:
            logger.error(f"Failed to register endpoints: {e}")

        return False

    async def run(self) -> None:
        """Main orchestration loop."""
        logger.info("=" * 60)
        logger.info("Atropos-VeRL Service Launcher")
        logger.info("=" * 60)

        # Start services in order
        if not await self.start_atropos_server():
            logger.error("Failed to start Atropos server")
            return

        if not await self.start_atropos_environments():
            logger.error("Failed to start environments")
            return

        if not await self.start_verl_inference_engines():
            logger.error("Failed to start inference engines")
            return

        if not await self.register_inference_endpoints():
            logger.error("Failed to register endpoints")
            return

        # Log service status
        logger.info("\n" + "=" * 60)
        logger.info("All services started successfully!")
        logger.info("=" * 60)
        logger.info("Services:")
        logger.info(f"  Atropos API: {self.config['atropos']['api_url']}")
        for name, url in self.service_urls.items():
            logger.info(f"  {name}: {url}")
        logger.info("\nReady for training. Press Ctrl+C to shutdown.")
        logger.info("=" * 60 + "\n")

        # Wait for shutdown signal
        await self.shutdown_event.wait()

    async def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        logger.info("\nShutting down services...")

        # Terminate processes in reverse order
        for name, process in reversed(list(self.processes.items())):
            logger.info(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        logger.info("All services stopped.")
        self.shutdown_event.set()

    def handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        asyncio.create_task(self.shutdown())


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch Atropos-VeRL services")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create and run launcher
    launcher = AtroposVeRLLauncher(args.config)

    # Setup signal handlers
    signal.signal(signal.SIGINT, launcher.handle_signal)
    signal.signal(signal.SIGTERM, launcher.handle_signal)

    try:
        await launcher.run()
    except Exception as e:
        logger.error(f"Launcher error: {e}")
        await launcher.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

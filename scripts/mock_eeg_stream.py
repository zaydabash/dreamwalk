#!/usr/bin/env python3
"""
Mock EEG Stream Script

Generates realistic synthetic EEG data and streams it to the DreamWalk system.
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import redis.asyncio as redis
import json
import numpy as np
import structlog
from services.signal_processor.streamers.mock_streamer import MockStreamer
from services.signal_processor.models.signal_models import SignalData, EEGConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger(__name__)


class MockEEGStreamer:
    """Mock EEG data streamer that connects to DreamWalk services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_id = config.get("session_id", "mock_demo_session")
        self.signal_type = config.get("signal_type", "mock")
        self.update_rate = config.get("update_rate", 10)
        
        # Service URLs
        self.realtime_server_url = config.get("realtime_server_url", "http://localhost:8003")
        self.signal_processor_url = config.get("signal_processor_url", "http://localhost:8001")
        
        # HTTP client
        self.http_client = None
        
        # Redis client
        self.redis_client = None
        
        # Mock streamer
        self.mock_streamer = None
        
        # State
        self.is_streaming = False
        self.stream_task = None
    
    async def initialize(self):
        """Initialize connections and services"""
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Initialize Redis client
            self.redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize mock streamer
            eeg_config = EEGConfig(
                sampling_rate=self.config.get("sampling_rate", 250),
                channels=self.config.get("channels", [
                    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
                ])
            )
            
            mock_config = {
                "sampling_rate": eeg_config.sampling_rate,
                "channels": eeg_config.channels,
                "window_size": self.config.get("window_size", 4.0),
                "update_rate": self.update_rate
            }
            
            self.mock_streamer = MockStreamer(mock_config)
            
            logger.info("Mock EEG streamer initialized", session_id=self.session_id)
            
        except Exception as e:
            logger.error("Failed to initialize mock EEG streamer", error=str(e))
            raise
    
    async def start_streaming(self):
        """Start the mock EEG streaming process"""
        try:
            # Start session with real-time server
            await self._start_session()
            
            # Start mock data generation
            self.is_streaming = True
            self.stream_task = asyncio.create_task(self._stream_loop())
            
            logger.info("Mock EEG streaming started", session_id=self.session_id)
            
        except Exception as e:
            logger.error("Failed to start streaming", error=str(e))
            raise
    
    async def stop_streaming(self):
        """Stop the mock EEG streaming process"""
        try:
            self.is_streaming = False
            
            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass
            
            # Stop session
            await self._stop_session()
            
            logger.info("Mock EEG streaming stopped", session_id=self.session_id)
            
        except Exception as e:
            logger.error("Failed to stop streaming", error=str(e))
    
    async def _start_session(self):
        """Start a new session with the real-time server"""
        try:
            session_request = {
                "session_id": self.session_id,
                "signal_type": self.signal_type,
                "config": {
                    "sampling_rate": self.config.get("sampling_rate", 250),
                    "channels": self.config.get("channels", [
                        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
                    ]),
                    "window_size": self.config.get("window_size", 4.0),
                    "update_rate": self.update_rate
                }
            }
            
            async with self.http_client.post(
                f"{self.realtime_server_url}/sessions/start",
                json=session_request
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Session started", result=result)
                else:
                    raise Exception(f"Failed to start session: {response.text}")
                    
        except Exception as e:
            logger.error("Failed to start session", error=str(e))
            raise
    
    async def _stop_session(self):
        """Stop the session"""
        try:
            async with self.http_client.post(
                f"{self.realtime_server_url}/sessions/stop/{self.session_id}"
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    logger.info("Session stopped", result=result)
                else:
                    logger.warning("Failed to stop session properly", status_code=response.status_code)
                    
        except Exception as e:
            logger.error("Failed to stop session", error=str(e))
    
    async def _stream_loop(self):
        """Main streaming loop"""
        try:
            while self.is_streaming:
                # Generate mock EEG data
                signal_data = await self.mock_streamer._generate_signal_window()
                
                # Send to signal processor
                await self._send_signal_data(signal_data)
                
                # Store in Redis for other services
                await self._store_signal_data(signal_data)
                
                # Wait for next update
                await asyncio.sleep(1.0 / self.update_rate)
                
        except asyncio.CancelledError:
            logger.info("Stream loop cancelled")
        except Exception as e:
            logger.error("Stream loop error", error=str(e))
    
    async def _send_signal_data(self, signal_data: SignalData):
        """Send signal data to the signal processor"""
        try:
            # Convert signal data to JSON-serializable format
            signal_dict = {
                "data": signal_data.data.tolist(),
                "sampling_rate": signal_data.sampling_rate,
                "channels": signal_data.channels,
                "timestamp": signal_data.timestamp.isoformat(),
                "metadata": signal_data.metadata
            }
            
            # Send to signal processor
            async with self.http_client.post(
                f"{self.signal_processor_url}/process/batch",
                params={"signal_type": "eeg"},
                json=signal_dict
            ) as response:
                if response.status_code != 200:
                    logger.warning("Signal processing failed", status_code=response.status_code)
                    
        except Exception as e:
            logger.error("Failed to send signal data", error=str(e))
    
    async def _store_signal_data(self, signal_data: SignalData):
        """Store signal data in Redis for other services"""
        try:
            # Create features dictionary (simplified)
            features = {
                "delta_power": np.random.normal(0.5, 0.1, len(signal_data.channels)).tolist(),
                "theta_power": np.random.normal(0.8, 0.2, len(signal_data.channels)).tolist(),
                "alpha_power": np.random.normal(1.0, 0.3, len(signal_data.channels)).tolist(),
                "beta_power": np.random.normal(0.7, 0.2, len(signal_data.channels)).tolist(),
                "gamma_power": np.random.normal(0.3, 0.1, len(signal_data.channels)).tolist(),
                "hjorth_activity": np.random.normal(1.0, 0.1, len(signal_data.channels)).tolist(),
                "hjorth_mobility": np.random.normal(0.5, 0.1, len(signal_data.channels)).tolist(),
                "hjorth_complexity": np.random.normal(0.3, 0.05, len(signal_data.channels)).tolist(),
                "frontal_asymmetry": np.random.normal(0.0, 0.2),
                "parietal_asymmetry": np.random.normal(0.0, 0.15),
                "artifact_ratio": np.random.uniform(0.0, 0.1),
                "eye_blink_count": np.random.randint(0, 3),
                "mean_amplitude": np.random.normal(0.0, 0.1, len(signal_data.channels)).tolist(),
                "std_amplitude": np.random.normal(1.0, 0.2, len(signal_data.channels)).tolist(),
                "skewness": np.random.normal(0.0, 0.5, len(signal_data.channels)).tolist(),
                "kurtosis": np.random.normal(3.0, 1.0, len(signal_data.channels)).tolist()
            }
            
            # Store in Redis
            await self.redis_client.setex(
                f"features:{self.session_id}:latest",
                60,  # 1 minute TTL
                json.dumps(features)
            )
            
        except Exception as e:
            logger.error("Failed to store signal data", error=str(e))
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Mock EEG streamer cleaned up")
            
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Mock EEG Stream for DreamWalk")
    parser.add_argument("--session-id", default="mock_demo_session", help="Session ID")
    parser.add_argument("--sampling-rate", type=int, default=250, help="EEG sampling rate")
    parser.add_argument("--channels", nargs="+", default=[
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"
    ], help="EEG channels")
    parser.add_argument("--window-size", type=float, default=4.0, help="Analysis window size")
    parser.add_argument("--update-rate", type=float, default=10.0, help="Update rate in Hz")
    parser.add_argument("--duration", type=int, default=0, help="Stream duration in seconds (0 for infinite)")
    parser.add_argument("--realtime-server-url", default="http://localhost:8003", help="Real-time server URL")
    parser.add_argument("--signal-processor-url", default="http://localhost:8001", help="Signal processor URL")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "session_id": args.session_id,
        "signal_type": "mock",
        "sampling_rate": args.sampling_rate,
        "channels": args.channels,
        "window_size": args.window_size,
        "update_rate": args.update_rate,
        "duration": args.duration,
        "realtime_server_url": args.realtime_server_url,
        "signal_processor_url": args.signal_processor_url
    }
    
    # Create and initialize streamer
    streamer = MockEEGStreamer(config)
    
    try:
        await streamer.initialize()
        await streamer.start_streaming()
        
        # Run for specified duration or until interrupted
        if args.duration > 0:
            logger.info(f"Streaming for {args.duration} seconds...")
            await asyncio.sleep(args.duration)
            await streamer.stop_streaming()
        else:
            logger.info("Streaming indefinitely. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                await streamer.stop_streaming()
        
    except Exception as e:
        logger.error("Mock EEG streaming failed", error=str(e))
    finally:
        await streamer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

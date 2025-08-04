# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor


# Global service tracker
class ServiceTracker:
    def __init__(self, servers):
        self.servers = servers
        self.lock = threading.Lock()  # Thread safety for metrics updates
        # Initialize metrics for each server
        self.metrics = {
            server: {
                "in_flight": 0,  # Current number of processing requests
                "response_times": deque(maxlen=100),  # Circular buffer of recent response times
                "error_count": 0,  # Consecutive error counter
                "active": True,  # Service availability status
                "complete_requests": 0,  # Total completed requests
            }
            for server in servers
        }

    def update_metrics(self, server, response_time, success):
        """Update server metrics after request completion"""
        with self.lock:
            metrics = self.metrics[server]
            if success:
                metrics["response_times"].append(response_time)
                metrics["error_count"] = 0  # Reset error counter
            else:
                metrics["error_count"] += 1
                # Mark as inactive after 5 consecutive errors
                if metrics["error_count"] > 5:
                    metrics["active"] = False

    def start_request(self, server):
        """Increment in-flight counter before sending request"""
        with self.lock:
            self.metrics[server]["in_flight"] += 1

    def complete_request(self, server):
        """Decrement in-flight counter after request finishes"""
        with self.lock:
            self.metrics[server]["in_flight"] -= 1
            self.metrics[server]["complete_requests"] += 1

    def print_current_state(self):
        def print_metrics(metrics, width=130):
            """Prints the current service metrics in a formatted table"""
            # Print table header

            print("\n" + "=" * width)
            print(
                f"{'Service':<50} | {'Active':<6} | {'In Flight':<10} | "
                f"{'Errors':<6} | {'Avg Resp':<8} | {'completed':<10} | "
                f"{'Samples':<7} | Recent Response Times"
            )
            print("-" * width)

            for server, data in metrics.items():
                # Calculate average response time if available
                avg_response = "N/A"
                sample_count = len(data["response_times"])
                if sample_count > 0:
                    avg_ms = sum(data["response_times"]) / sample_count * 1000
                    avg_response = f"{avg_ms:.2f}ms"

                # Prepare recent response times (last 5 samples)
                recent = list(data["response_times"])[-5:]
                recent_times = ", ".join([f"{t * 1000:.1f}ms" for t in recent]) if recent else "None"

                # Print service row
                print(
                    f"{server:<50} | "
                    f"{'✓' if data['active'] else '✗':<6} | "
                    f"{data['in_flight']:<10} | "
                    f"{data['error_count']:<6} | "
                    f"{avg_response:<8} | "
                    f"{data['complete_requests']:<10} | "
                    f"{sample_count:<7} | "
                    f"{recent_times}"
                )
            print("=" * width + "\n")

        with self.lock:
            print_metrics(self.metrics)

    def get_best_server(self):
        """Select optimal server using weighted scoring algorithm"""
        with self.lock:
            # Filter out inactive servers
            candidates = [s for s in self.servers if self.metrics[s]["active"]]

            # Return none if none available
            if not candidates:
                return None  # No servers configured

            # Calculate weighted score for each candidate
            scores = {}
            for server in candidates:
                metrics = self.metrics[server]
                # Use default response time if no history available
                avg_time = 0.5  # Default 500ms response time
                if metrics["response_times"]:
                    avg_time = (
                        sum(metrics["response_times"]) / len(metrics["response_times"]) * 0.15
                    )  # Weight response time by 0.15
                # Load factor based on current in-flight requests
                load_factor = metrics["in_flight"]
                # Composite score (lower is better)
                scores[server] = avg_time + load_factor
            # Return server with minimum score
            return min(scores, key=scores.get)


def request_loop():
    """Continuous request generator"""
    request_id = 1
    while True:
        # Use thread pool for concurrency control
        with ThreadPoolExecutor(max_workers=500) as executor:
            # Send batch of requests with random intervals
            for _ in range(20):
                executor.submit(send_request, request_id)
                request_id += 1
                # Random delay between requests (0-100ms)
                time.sleep(random.uniform(0, 0.1))


def send_request(request_id, tracker):
    """Send request to backend service with load balancing"""
    # Select optimal server using scoring algorithm
    server = tracker.get_best_server()
    if not server:
        print(f"Request {request_id} failed: No available servers")
        return

    # Update in-flight counter
    tracker.start_request(server)
    start_time = time.time()
    success = False
    try:
        # Send actual request to backend
        time.sleep(2 * random.uniform(0.1, 0.5))  # Simulate network delay
        success = True
        print(f"Request {request_id} to {server} successful")
    except Exception as e:
        print(f"Request {request_id} to {server} failed: {str(e)}")
    finally:
        # Always update metrics regardless of success/failure
        end_time = time.time()
        response_time = end_time - start_time
        tracker.update_metrics(server, response_time, success)
        tracker.complete_request(server)
        tracker.print_current_state()

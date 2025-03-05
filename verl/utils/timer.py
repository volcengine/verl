import time
import sys

class TimeoutChecker:
    def __init__(self, initial_interval_hours=4, backoff_minutes=15):
        super().__init__()
        self.last_save_time = initial_interval_hours * 3600 - backoff_minutes * 60
        self.start_time = time.time()
        self.last_saved = False

    def check_save(self):
        # Flush
        sys.stdout.flush()
        sys.stderr.flush()

        # Already saved after timeout
        if self.last_saved:
            return False

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= self.last_save_time:
            self.last_saved = True
            return True

        return False

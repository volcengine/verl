import time
import sys

def convert_to_seconds(time_string):
    # Split the string into components
    days, hours, minutes, seconds = map(int, time_string.split(':'))

    # Calculate total seconds
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds

    return total_seconds


class TimeoutChecker:
    def __init__(self, timeout: str = '00:03:45:00'):
        super().__init__()
        self.last_save_time = convert_to_seconds(timeout)
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

import time
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    def __init__(self, name="Timer", log=True):
        """
        Initialize the timer.

        :param name: A name for the timer (useful for logging).
        :param log: If True, prints the elapsed time automatically.
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None  # Reset end_time in case of reuse

    def stop(self):
        """Stop the timer and return the elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer was not started. Use .start() before .stop()")
        
        self.end_time = time.time()
        elapsed = self.elapsed_time()
        
        if self.log:
            print(f"[{self.name}] Elapsed time: {elapsed:.4f} seconds")
        
        return elapsed

    def elapsed_time(self):
        """Return the elapsed time without stopping the timer."""
        if self.start_time is None:
            return 0
        return (self.end_time or time.time()) - self.start_time

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timer when entering a `with` block."""
        self.start()
        return self

    def __exit__(self, *args):
        """Stop timer when exiting a `with` block."""
        self.stop()

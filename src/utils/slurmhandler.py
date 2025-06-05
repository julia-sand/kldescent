import signal

class TimeLimitException(Exception):
    pass

def handle_signal(signal_num, frame):
    if signal_num == signal.SIGTERM:
        raise TimeLimitException()

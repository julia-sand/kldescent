import signal



class TimeLimitException(Exception):
    pass

def handle_signal(signal_num, frame):
    """
    Handles kill signal error from job timeouts
    """
    
    if signal_num == signal.SIGTERM:
        raise TimeLimitException()

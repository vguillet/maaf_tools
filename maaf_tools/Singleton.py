
import threading


class Singleton(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SLogger(metaclass=Singleton):
    logger = None

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

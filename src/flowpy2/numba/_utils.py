import logging
import inspect
import sys
from typing import Callable

_log = logging.getLogger(__name__)


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        self.level(sys.stdout)


class redirect_stdout_to_log:
    def __init__(self, logger_func: Callable):
        self._logger_func = logger_func

    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stdout.flush()
        sys.stdout = LoggerWriter(self._logger_func)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.flush()
        sys.stdout = self._old_stdout


def _log_simd(func, simd_key='subp'):
    if _log.level >= logging.INFO:
        count = 0
        file = inspect.getfile(inspect.getmodule(func))
        mod = inspect.getmodulename(file)
        _log.info("%s.%s: Checking simd" % (mod, func.__name__))
        for sig in func.signatures:
            for lo in func.inspect_asm(sig).split('\n'):
                if simd_key in lo:
                    count += 1
                    _log.info(lo)

        if count == 0:
            _log.info("No simd instructions found")


def _log_parallel(func):
    if _log.level >= logging.INFO:
        file = inspect.getfile(inspect.getmodule(func))
        mod = inspect.getmodulename(file)
        _log.info("%s.%s: Checking parallel" % (mod, func.__name__))
        with redirect_stdout_to_log(_log.info):
            func.parallel_diagnostics(level=4)


def log_jit(func, keyword='subp'):
    _log_simd(func, keyword)
    _log_parallel(func)

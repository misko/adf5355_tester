"""Console entry point.

The ``_lgpio`` extension creates a daemon thread at import time that never
retires -- not when a device is closed, not when gpiozero's pin factory is
closed.  The interpreter therefore always begins shutting down with it running,
which caused two distinct problems:

1. CPython aborted with "could not acquire lock for
   <_io.BufferedWriter name='<stderr>'> at interpreter shutdown", so the
   process exited 134 after a command had already succeeded -- 14 of 15 runs.
   Anything checking the exit status saw a failure that never happened.
   Leaving through ``os._exit`` skips finalization entirely.  Nothing here needs
   it: ``ADF5355.close()`` mutes the outputs and releases the SPI and GPIO
   handles, and the kernel reclaims the descriptors.

2. Every run then printed "Exception in thread Thread-1" with a truncated
   traceback.  The thread raises ``lgpio.error('unknown handle')`` on the way
   out, because it touches the chip handle after the process has finished with
   it.  That is not reproducible by closing devices, closing the pin factory, or
   leaving them open -- it belongs to lgpio's own teardown, so only that exact
   error is ignored here.  Anything else a background thread raises is still
   reported, through ``sys.__stderr__`` so it cannot deadlock on the buffered
   stream that finalization is tearing down.
"""
import os
import sys
import threading
import traceback


def _is_lgpio_teardown_noise(args) -> bool:
    """True for the benign lgpio shutdown error, and nothing else."""
    exc_type = args.exc_type
    if exc_type is None or issubclass(exc_type, SystemExit):
        return True
    return (getattr(exc_type, "__module__", "") == "lgpio"
            and exc_type.__name__ == "error"
            and "unknown handle" in str(args.exc_value))


def _thread_excepthook(args) -> None:
    if _is_lgpio_teardown_noise(args):
        return
    try:
        name = args.thread.name if args.thread is not None else "unknown"
        print(f"adf5355: exception in background thread {name}",
              file=sys.__stderr__)
        traceback.print_exception(args.exc_type, args.exc_value,
                                  args.exc_traceback, file=sys.__stderr__)
    except Exception:
        pass


threading.excepthook = _thread_excepthook

from .cli import main  # noqa: E402  -- the hook must be installed first

rc = main()
sys.stdout.flush()
sys.stderr.flush()
os._exit(rc)

"""Console entry point.

Exits via os._exit so the interpreter never finalizes with gpiozero's lgpio
daemon thread still alive.  That race intermittently aborts the process with
"could not acquire lock for <_io.BufferedWriter name='<stderr>'> at interpreter
shutdown", returning 134 from a command that had already succeeded -- which
silently breaks any caller checking the exit status.  Closing the pin factory
first is not sufficient; the thread outlives it.  Nothing here needs interpreter
cleanup: files are closed by ADF5355.close() and the SPI/GPIO handles are
released by the kernel on exit.
"""
import os
import sys

from .cli import main

rc = main()
sys.stdout.flush()
sys.stderr.flush()
os._exit(rc)

"""ADF5355 wideband synthesizer control for Raspberry Pi.

Layered so that everything except :mod:`adf5355.device` is pure computation and
can be tested with no hardware attached:

    registers.py  bit positions, the single source of truth
    plan.py       frequency -> field values (solver + register assembly)
    device.py     spidev + GPIO, write ordering, autocal, lock detect
    cli.py        command line front end
"""
from .registers import MuxOut, OutputPower, RegisterFile, Field, FIELDS
from .plan import SynthConfig, Plan, Solution, plan, Channel

__all__ = [
    "MuxOut", "OutputPower", "RegisterFile", "Field", "FIELDS",
    "SynthConfig", "Plan", "Solution", "plan", "Channel",
]

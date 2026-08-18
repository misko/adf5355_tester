"""Hardware layer: SPI transport, chip enable, write ordering, lock detect.

Wiring assumed (Raspberry Pi 40-pin header):

    GPIO10 / MOSI   pin 19  ->  DAT
    GPIO11 / SCLK   pin 23  ->  CLK
    GPIO8  / CE0    pin 24  ->  LE
    GPIO23          pin 16  ->  CE     (chip enable, driven high to run)
    GPIO19          pin 35  <-  MUXOUT (lock detect / link probe)
    GND             pin 6   ->  GND

The ADF5355 cannot be read back: DAT is an input, there is no MISO, and no
register is readable.  MUXOUT is the only signal the chip can drive back, so
without it the part is entirely open-loop -- writes cannot be confirmed, lock
cannot be detected, and the only way to know the synthesizer did what you asked
is to measure the RF output.  With it, probe() can prove the chip is receiving
writes and wait_for_lock() can confirm the loop actually locked.

CE0 is the right pin for LE by construction: the Pi holds it low for the whole
transfer and releases it high afterwards, so the 32 bits shift in with LE low
and the trailing rising edge latches them.  This requires all four bytes to go
out in a single transfer -- never one byte at a time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .plan import (Channel, Plan, SynthConfig, plan as build_plan,
                   reference_word)
from .registers import FIELDS, N_REGS, MuxOut, to_bytes

DEFAULT_CE_GPIO = 23
DEFAULT_MUXOUT_GPIO = 19     # header pin 35
DEFAULT_SPI_HZ = 1_000_000

_AUTOCAL = FIELDS["autocal"]
_COUNTER_RESET = FIELDS["counter_reset"]
_OUTB_DISABLE = FIELDS["rf_outb_disable"]
_OUTA_ENABLE = FIELDS["rf_out_enable"]


class LockTimeout(RuntimeError):
    """Autocal finished but MUXOUT never reported digital lock."""


@dataclass
class WriteRecord:
    reg: int
    word: int

    def __str__(self) -> str:
        return f"R{self.reg:<2} 0x{self.word:08X}"


class ADF5355:
    """Controller for one ADF5355 on a Pi SPI bus."""

    def __init__(
        self,
        config: SynthConfig,
        bus: int = 0,
        device: int = 0,
        spi_hz: int = DEFAULT_SPI_HZ,
        ce_gpio: int | None = DEFAULT_CE_GPIO,
        muxout_gpio: int | None = DEFAULT_MUXOUT_GPIO,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.bus = bus
        self.device = device
        self.spi_hz = spi_hz
        self.ce_gpio = ce_gpio
        self.muxout_gpio = muxout_gpio
        self.dry_run = dry_run

        self._spi = None
        self._ce = None
        self._muxout = None
        self._synced = False
        self._current: Plan | None = None
        self.trace: list[WriteRecord] = []

    # --- resource management ------------------------------------------------
    def open(self) -> "ADF5355":
        if self.dry_run:
            return self
        import spidev

        self._spi = spidev.SpiDev()
        self._spi.open(self.bus, self.device)
        self._spi.max_speed_hz = self.spi_hz
        self._spi.mode = 0          # CPOL=0, CPHA=0: data latched on rising CLK
        self._spi.lsbfirst = False  # ADF5355 shifts MSB first
        self._spi.bits_per_word = 8

        from gpiozero import DigitalInputDevice, DigitalOutputDevice

        if self.ce_gpio is not None:
            # CE low is chip power-down; come up disabled, enable explicitly.
            self._ce = DigitalOutputDevice(self.ce_gpio, initial_value=False)
        if self.muxout_gpio is not None:
            self._muxout = DigitalInputDevice(self.muxout_gpio)
        return self

    def close(self) -> None:
        try:
            self.mute()
        except Exception:
            pass
        if self._ce is not None:
            self._ce.off()
            self._ce.close()
            self._ce = None
        if self._muxout is not None:
            self._muxout.close()
            self._muxout = None
        if self._spi is not None:
            self._spi.close()
            self._spi = None

    def __enter__(self) -> "ADF5355":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- low level ----------------------------------------------------------
    def _write(self, reg: int, word: int) -> None:
        word = (word | (reg & 0xF)) & 0xFFFFFFFF
        self.trace.append(WriteRecord(reg, word))
        if self.dry_run or self._spi is None:
            return
        # One transfer for all four bytes: CE0/LE must stay low across the
        # whole 32-bit word and rise exactly once, at the end.
        self._spi.xfer2(to_bytes(word))

    def enable_chip(self, settle_s: float = 0.010) -> None:
        """Raise CE and let the regulators settle before any register write."""
        if self._ce is not None:
            self._ce.on()
        time.sleep(settle_s)

    @property
    def chip_enabled(self) -> bool:
        return self._ce is None or bool(self._ce.value)

    # --- programming --------------------------------------------------------
    def program(self, p: Plan) -> Plan:
        """Cold start: write R12..R1 descending, settle, then R0 with autocal.

        Order matters.  R0 carries the autocal trigger and must be last, and
        the VCO band-select ADC needs its clock configured (R10) and more than
        16 ADC cycles of settling before that trigger arrives.
        """
        if not self.chip_enabled:
            self.enable_chip()
        words = p.words
        for reg in range(N_REGS - 1, 0, -1):
            self._write(reg, words[reg])
        time.sleep(p.delay_us / 1e6)
        self._write(0, words[0])
        self._synced = True
        self._current = p
        return p

    def retune(self, p: Plan) -> Plan:
        """Frequency change on an already-initialized part.

        Shorter than a cold start but still ordered: the counter reset brackets
        the divider update, R0 is written once without autocal to load the new
        dividers, then again with autocal to recalibrate the VCO band.
        """
        if not self._synced:
            return self.program(p)
        words = p.words
        self._write(10, words[10])
        self._write(6, words[6])
        self._write(4, words[4] | _COUNTER_RESET.encode(1))
        self._write(2, words[2])
        self._write(1, words[1])
        self._write(0, words[0] & ~_AUTOCAL.mask)
        self._write(4, words[4])
        time.sleep(p.delay_us / 1e6)
        self._write(0, words[0])
        self._current = p
        return p

    def set_frequency(self, freq_hz: int,
                      channel: Channel = Channel.A) -> Plan:
        p = build_plan(self.config, freq_hz, channel)
        return self.retune(p) if self._synced else self.program(p)

    # --- output gating ------------------------------------------------------
    def _rewrite_r6(self, mutate) -> None:
        if self._current is None:
            raise RuntimeError("nothing programmed yet")
        word = mutate(self._current.words[6])
        self._write(6, word)

    def set_output(self, channel: Channel, enabled: bool) -> None:
        """Gate an output without disturbing the loop."""
        if channel is Channel.B:
            # DB10 is active-low: 0 enables RFoutB.
            def mutate(w):
                return (w & ~_OUTB_DISABLE.mask) | _OUTB_DISABLE.encode(int(not enabled))
        else:
            def mutate(w):
                return (w & ~_OUTA_ENABLE.mask) | _OUTA_ENABLE.encode(int(enabled))
        self._rewrite_r6(mutate)

    def mute(self) -> None:
        """Disable both outputs.  Safe to call when nothing is programmed."""
        if self._current is None:
            return
        word = self._current.words[6]
        word = (word & ~_OUTA_ENABLE.mask) | _OUTB_DISABLE.encode(1)
        self._write(6, word)

    # --- talking back -------------------------------------------------------
    def set_muxout(self, mux: MuxOut) -> None:
        """Retarget the MUXOUT pin.  Valid before the PLL is configured."""
        self._write(4, reference_word(self.config, mux))

    def probe(self, settle_s: float = 0.002) -> dict:
        """Confirm the chip is receiving writes, using MUXOUT as a return path.

        The ADF5355 is write-only -- there is no register readback and no MISO.
        MUXOUT is the only thing it can say back, and two of its settings are
        static levels.  Commanding it high and then low and watching the GPIO
        follow exercises the whole chain: CE, CLK, DAT, LE, and the chip's own
        register decode.  A pin stuck at one level means the writes are not
        landing; a floating pin usually means CE is low or MUXOUT is unwired.
        """
        if self._muxout is None:
            raise RuntimeError(
                "cannot probe: MUXOUT is not wired. This is the only way the "
                "ADF5355 can signal back -- connect the board's MUX pin to a "
                "free GPIO and pass muxout_gpio")
        if not self.chip_enabled:
            self.enable_chip()

        readings = {}
        for name, mux in (("high", MuxOut.DVDD), ("low", MuxOut.GND)):
            self.set_muxout(mux)
            time.sleep(settle_s)
            readings[name] = self.locked  # raw pin state

        # Leave MUXOUT where the configuration wants it.
        self.set_muxout(self.config.muxout)
        time.sleep(settle_s)

        if self.dry_run:
            readings = {"high": True, "low": False}
        readings["ok"] = bool(readings["high"]) and not bool(readings["low"])
        return readings

    # --- lock detect --------------------------------------------------------
    @property
    def locked(self) -> bool | None:
        """MUXOUT state, or None if MUXOUT is not wired/configured."""
        if self._muxout is None:
            return None
        return bool(self._muxout.value)

    def settle(self, seconds: float = 0.010) -> None:
        """Blind wait, for when MUXOUT is not wired and lock cannot be read."""
        time.sleep(seconds)

    @property
    def can_detect_lock(self) -> bool:
        return self._muxout is not None

    def wait_for_lock(self, timeout_s: float = 0.5,
                      poll_s: float = 0.001) -> float:
        """Block until digital lock detect asserts.  Returns seconds waited.

        Raises LockTimeout on failure -- a silent no-lock is the failure mode
        that wastes the most bench time, so it is made loud here.
        """
        if self._muxout is None:
            raise RuntimeError(
                "cannot detect lock: MUXOUT is not wired. Connect the board's "
                "MUX pin to a free GPIO and pass muxout_gpio, or call settle()"
            )
        deadline = time.monotonic() + timeout_s
        start = time.monotonic()
        while time.monotonic() < deadline:
            if self._muxout.value:
                return time.monotonic() - start
            time.sleep(poll_s)
        raise LockTimeout(
            f"no digital lock within {timeout_s:.3f} s on GPIO{self.muxout_gpio}"
            + ("" if self._current is None else
               f" at {float(self._current.solution.achieved_hz)/1e9:.6f} GHz")
        )

    def dump_trace(self) -> str:
        return "\n".join(str(r) for r in self.trace)

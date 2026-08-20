/* Turn the ADF5355 outputs off, and say only what that guarantees.
 *
 * The part's SPI is write-only -- MUXOUT is the one signal it can drive back,
 * and MUXOUT reports lock state, not output state. So this tool deliberately
 * does NOT claim to verify the output is off: after a power-down MUXOUT going
 * low is indistinguishable from MUXOUT saying nothing, and a check that cannot
 * fail is worse than no check. Confirming silence needs a receiver, which this
 * tool does not assume exists.
 *
 * What it does guarantee is that the two register writes were accepted by the
 * SPI layer, and it reports the exact words written so the caller can audit
 * them. Register state is what makes the off durable: driving CE low would
 * power the part down harder, but a GPIO line request is released when the
 * process exits, so CE would float back and the off would not outlive the
 * tool. The registers survive.
 *
 * Order matters. R6 disables both outputs first, so the RF stops while the
 * synthesiser is still running normally. R4 then asserts power_down. Doing it
 * the other way round powers down the loop with the output stage still keyed.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

/* Emitted by the register model in adf5355/registers.py:
 *   R6 = both outputs disabled (rf_out_enable 0, rf_outb_disable 1)
 *   R4 = as planned, with power_down asserted
 * Overridable so a caller with a different configuration can pass its own. */
#define DEFAULT_R6_OFF 0x35016406u
#define DEFAULT_R4_OFF 0x300149C4u

static int spi_fd = -1;
static uint32_t spi_hz = 20000000u;

static int wr(uint32_t word)
{
    uint8_t tx[4] = { (uint8_t)(word >> 24), (uint8_t)(word >> 16),
                      (uint8_t)(word >> 8),  (uint8_t)word };
    struct spi_ioc_transfer t;
    memset(&t, 0, sizeof t);
    t.tx_buf = (unsigned long)tx;
    t.len = 4;
    t.speed_hz = spi_hz;
    t.bits_per_word = 8;
    /* All four bytes in one transfer: CE0 idles high, falls for the transfer
     * and rises at the end, and that rising edge is what latches the word. */
    return ioctl(spi_fd, SPI_IOC_MESSAGE(1), &t) < 0 ? -1 : 0;
}

static void usage(const char *me)
{
    fprintf(stderr,
        "usage: %s [/dev/spidevX.Y] [spi_hz] [r6_off] [r4_off]\n"
        "  defaults: /dev/spidev0.0 %u 0x%08X 0x%08X\n",
        me, spi_hz, DEFAULT_R6_OFF, DEFAULT_R4_OFF);
}

int main(int argc, char **argv)
{
    const char *dev = (argc > 1) ? argv[1] : "/dev/spidev0.0";
    if (argc > 2) spi_hz = (uint32_t)strtoul(argv[2], NULL, 0);
    uint32_t r6 = (argc > 3) ? (uint32_t)strtoul(argv[3], NULL, 0) : DEFAULT_R6_OFF;
    uint32_t r4 = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 0) : DEFAULT_R4_OFF;
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        usage(argv[0]);
        return 2;
    }

    if ((r6 & 0xFu) != 6u || (r4 & 0xFu) != 4u) {
        fprintf(stderr, "refusing: r6 must end in 0x6 and r4 in 0x4 "
                        "(the low nibble is the register address)\n");
        return 2;
    }

    spi_fd = open(dev, O_RDWR);
    if (spi_fd < 0) { perror(dev); return 1; }
    uint8_t mode = SPI_MODE_0, bits = 8;
    if (ioctl(spi_fd, SPI_IOC_WR_MODE, &mode) < 0 ||
        ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_hz) < 0) {
        perror("spi configure");
        close(spi_fd);
        return 1;
    }

    if (wr(r6) < 0) { perror("write R6"); close(spi_fd); return 1; }
    if (wr(r4) < 0) { perror("write R4"); close(spi_fd); return 1; }
    close(spi_fd);

    printf("outputs disabled via %s at %u Hz\n", dev, spi_hz);
    printf("  R6 0x%08X  both outputs disabled\n", r6);
    printf("  R4 0x%08X  power_down asserted\n", r4);
    printf("both writes accepted. This confirms the words reached the SPI\n"
           "controller -- the part cannot be read back, so it is not a\n"
           "measurement that the output is silent. Confirm with a receiver\n"
           "if silence matters.\n");
    return 0;
}

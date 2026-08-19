/* Duration-coded ladder transmitter for the ADF5355.
 *
 * Each frequency is held for its OWN dwell, so a point is identified by how
 * long it lasts rather than by where it sits in a shared schedule. With four
 * points at 100/50/25/12.5 ms the durations are 2:1 apart, which a receiver
 * can separate without knowing the seed, the epoch, or the point order --
 * none of the alignment machinery the fixed-dwell decoder needs.
 *
 * Like hop_tx it consumes a plan emitted by Python (tools/emit_ladder_plan.py)
 * rather than reimplementing the divider maths, holds only one period in
 * memory, and loops until signalled.
 *
 * Plan file (little endian):
 *   magic "AD57" u32 | points u32 | r6_on u32 | r6_off u32
 *   delay_us u32 | autocal_every u32 | boot 13 x u32
 *   points x { R1 u32, R2 u32, R0 u32, R0cal u32, dwell_ns u64 }
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

static long spin_ns = 30000L;
static int spi_fd;
static uint32_t spi_speed = 20000000u;
static volatile sig_atomic_t stop_now = 0;
static void on_signal(int s){ (void)s; stop_now = 1; }

static void wr(uint32_t w)
{
    uint8_t tx[4] = { (uint8_t)(w>>24), (uint8_t)(w>>16),
                      (uint8_t)(w>>8),  (uint8_t)w };
    struct spi_ioc_transfer t;
    memset(&t, 0, sizeof t);
    t.tx_buf = (unsigned long)tx;
    t.len = 4;
    t.speed_hz = spi_speed;
    t.bits_per_word = 8;
    ioctl(spi_fd, SPI_IOC_MESSAGE(1), &t);       /* CE0 rising edge latches */
}

static int64_t ns_of(const struct timespec *t)
{ return (int64_t)t->tv_sec*1000000000LL + t->tv_nsec; }

static void wait_until(int64_t deadline)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t left = deadline - ns_of(&now);
    if (left > spin_ns){
        struct timespec ts = { .tv_sec = (deadline - spin_ns)/1000000000LL,
                               .tv_nsec = (deadline - spin_ns)%1000000000LL };
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
    }
    do { clock_gettime(CLOCK_MONOTONIC, &now); } while (ns_of(&now) < deadline);
}

int main(int argc, char **argv)
{
    if (argc < 3){
        fprintf(stderr, "usage: %s PLAN.bin SPI_HZ [/dev/spidevX.Y] [spin_us]\n", argv[0]);
        return 2;
    }
    const char *dev = (argc>3)? argv[3] : "/dev/spidev0.0";
    if (argc>4) spin_ns = atol(argv[4]) * 1000L;
    spi_speed = (uint32_t)strtoul(argv[2], NULL, 10);

    FILE *f = fopen(argv[1], "rb");
    if (!f){ perror("plan"); return 1; }
    uint32_t magic, points, r6_on, r6_off, delay_us, autocal_every, boot[13];
    if (fread(&magic,4,1,f)!=1 || magic != 0x37354441u){   /* "AD57" LE */
        fprintf(stderr, "bad plan magic (need AD57)\n"); return 1; }
    fread(&points,4,1,f); fread(&r6_on,4,1,f); fread(&r6_off,4,1,f);
    fread(&delay_us,4,1,f); fread(&autocal_every,4,1,f);
    fread(boot, 4, 13, f);
    if (points == 0 || points > 4096){ fprintf(stderr,"bad point count\n"); return 1; }
    uint32_t *w = malloc(points*4*sizeof(uint32_t));
    uint64_t *dw = malloc(points*sizeof(uint64_t));
    for (uint32_t i=0;i<points;i++){
        fread(&w[i*4], 4, 4, f);
        fread(&dw[i], 8, 1, f);
    }
    fclose(f);

    spi_fd = open(dev, O_RDWR);
    if (spi_fd < 0){ perror("spidev"); return 1; }
    uint8_t mode = SPI_MODE_0, bits = 8;
    ioctl(spi_fd, SPI_IOC_WR_MODE, &mode);
    ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
    ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed);

    struct sched_param sp; sp.sched_priority = 80;
    int rt = (sched_setscheduler(0, SCHED_FIFO, &sp) == 0);
    int locked = (mlockall(MCL_CURRENT|MCL_FUTURE) == 0);
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
    for (uint32_t i=0;i<points*4;i++) (void)w[i];

    /* Cold start so the VCO band is calibrated for this span. */
    for (int r = 12; r >= 1; r--) wr(boot[r]);
    usleep(delay_us);
    wr(boot[0]);
    usleep(20000);

    wr(w[0]); wr(w[1]); wr(w[autocal_every?3:2]);
    if (autocal_every) usleep(delay_us);
    wr(r6_on);

    struct timespec st; clock_gettime(CLOCK_MONOTONIC, &st);
    int64_t deadline = ns_of(&st);
    uint64_t played = 0;
    while (!stop_now){
        uint32_t p = (uint32_t)(played % points);
        deadline += (int64_t)dw[p];
        wait_until(deadline);
        uint32_t n = (uint32_t)((played+1) % points);
        const uint32_t *v = &w[n*4];
        wr(v[0]); wr(v[1]); wr(v[autocal_every?3:2]);
        if (autocal_every) usleep(delay_us);
        played++;
    }
    wr(r6_off);
    struct timespec en; clock_gettime(CLOCK_MONOTONIC, &en);
    fprintf(stderr, "{\"impl\":\"c-ladder\",\"points\":%u,\"held\":%llu,"
            "\"elapsed_s\":%.3f,\"rt\":%d,\"mlock\":%d,\"autocal\":%u}\n",
            points, (unsigned long long)played,
            (ns_of(&en)-ns_of(&st))/1e9, rt, locked, autocal_every);
    close(spi_fd);
    return 0;
}

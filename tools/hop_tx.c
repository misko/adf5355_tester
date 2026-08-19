/* C transmitter for the seeded hop schedule.
 *
 * Deliberately does NOT reimplement the schedule or the divider maths. It
 * consumes a plan emitted by adf5355 (tools/emit_hop_plan.py), so the seeded
 * protocol has exactly one implementation and the two transmitters cannot
 * drift apart. What C provides here is determinism, not cleverness:
 *
 *   - SCHED_FIFO so the loop is not preempted by ordinary work
 *   - mlockall so no page fault lands mid-dwell
 *   - clock_nanosleep(TIMER_ABSTIME) against a fixed origin, so error cannot
 *     accumulate, with a short spin to cover the timer's own granularity
 *   - no allocator and no collector in the timed path at all
 *
 * Plan file v4 (little endian):
 *   magic "AD56" u32 | points u32 | period_len u32 | total_hops u64
 *   dwell_ns u64 | r6_on u32 | r6_off u32 | delay_us u32 | autocal_every u32
 *   boot 13 x u32
 *   points x 4 u32       (R1, R2, R0-no-autocal, R0-with-autocal per point)
 *   period_len x u16     (point index per hop, ONE period)
 *
 * The plan carries one period, not the whole run: the schedule repeats, so
 * the sequence is indexed modulo period_len and memory is set by the period
 * rather than by how long the transmitter runs. total_hops == 0 means loop
 * until signalled. Lateness is accumulated into a fixed histogram for the
 * same reason -- a per-hop array is unbounded in a run meant to continue
 * indefinitely, which is exactly what sank the Python transmitter.
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
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <signal.h>
#include <linux/spi/spidev.h>

/* Lateness histogram: 1 us buckets. Fixed 256 KiB, independent of run length. */
#define HB 65536
static uint32_t hist[HB];
static uint64_t hist_n, hist_over;
static int64_t  hist_max = 0;

static void hist_add(int64_t late_ns)
{
    if (late_ns > hist_max) hist_max = late_ns;
    int64_t us = late_ns < 0 ? 0 : late_ns / 1000;
    if (us >= HB) hist_over++; else hist[us]++;
    hist_n++;
}

/* Smallest bucket whose cumulative count reaches the requested quantile. */
static double hist_q(double q)
{
    uint64_t want = (uint64_t)(q * (double)hist_n), acc = 0;
    for (uint32_t i = 0; i < HB; i++){
        acc += hist[i];
        if (acc >= want) return (double)i;
    }
    return (double)HB;
}

static volatile sig_atomic_t stop_now = 0;
static void on_signal(int sig){ (void)sig; stop_now = 1; }

/* Spin margin. Longer spin costs CPU but removes the wake-up from the path.
 * Overridable so the trade can be measured rather than guessed. */
static long spin_ns = 30000L;
static int  cpu_pin = -1;
static int  latency_fd = -1;

static int spi_fd;
static uint32_t spi_speed;

static inline void put32(uint8_t *b, uint32_t v){
    b[0]=v>>24; b[1]=v>>16; b[2]=v>>8; b[3]=v;
}
static inline void wr(uint32_t word){
    uint8_t buf[4]; put32(buf, word);
    struct spi_ioc_transfer tr;
    memset(&tr, 0, sizeof tr);
    tr.tx_buf=(unsigned long)buf; tr.len=4;
    tr.speed_hz=spi_speed; tr.bits_per_word=8;
    ioctl(spi_fd, SPI_IOC_MESSAGE(1), &tr);
}
static inline int64_t ns_of(const struct timespec *t){
    return (int64_t)t->tv_sec*1000000000LL + t->tv_nsec;
}
static inline void ts_of(int64_t ns, struct timespec *t){
    t->tv_sec = ns/1000000000LL; t->tv_nsec = ns%1000000000LL;
}
/* Hold the CPU out of deep idle. A core that has dropped into a low-power
 * state takes tens of microseconds to wake, and that shows up directly as
 * jitter on a sleeping timed loop -- which is why p99 tracked the spin margin
 * at low hop rates and vanished at 3200 hops/s, where the core never idles. */
static void pm_qos_hold(void){
    int32_t target = 0;
    latency_fd = open("/dev/cpu_dma_latency", O_RDWR);
    if (latency_fd >= 0)
        if (write(latency_fd, &target, sizeof target) != sizeof target){
            close(latency_fd); latency_fd = -1;
        }
}

static void wait_until(int64_t deadline_ns){
    struct timespec now, target;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int64_t remain = deadline_ns - ns_of(&now);
    if (remain > spin_ns){
        ts_of(deadline_ns - spin_ns, &target);
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, NULL) == EINTR) {}
    }
    do { clock_gettime(CLOCK_MONOTONIC, &now); } while (ns_of(&now) < deadline_ns);
}

int main(int argc, char **argv){
    if (argc < 3){
        fprintf(stderr, "usage: %s PLAN.bin SPI_HZ [/dev/spidevX.Y] [spin_us] [cpu]\n",
                argv[0]);
        return 2;
    }
    const char *dev = (argc>3)? argv[3] : "/dev/spidev0.0";
    if (argc>4) spin_ns = atol(argv[4]) * 1000L;
    if (argc>5) cpu_pin = atoi(argv[5]);
    spi_speed = (uint32_t)strtoul(argv[2], NULL, 10);

    FILE *f = fopen(argv[1], "rb");
    if (!f){ perror("plan"); return 1; }
    uint32_t magic, points, period_len, r6_on, r6_off;
    uint64_t total_hops, dwell_ns;
    if (fread(&magic,4,1,f)!=1 || magic != 0x36354441u){ /* "AD56" LE */
        fprintf(stderr, "bad plan magic (need v4; re-run emit_hop_plan.py)\n");
        return 1; }
    fread(&points,4,1,f); fread(&period_len,4,1,f);
    fread(&total_hops,8,1,f); fread(&dwell_ns,8,1,f);
    fread(&r6_on,4,1,f);  fread(&r6_off,4,1,f);
    if (period_len == 0){ fprintf(stderr, "empty period\n"); return 1; }
    uint32_t delay_us = 0, autocal_every = 0, boot[13];
    fread(&delay_us,4,1,f);
    fread(&autocal_every,4,1,f);
    fread(boot, sizeof(uint32_t), 13, f);
    uint32_t *tbl = malloc(points*4*sizeof(uint32_t));
    uint16_t *seq = malloc(period_len*sizeof(uint16_t));
    fread(tbl, sizeof(uint32_t), points*4, f);
    fread(seq, sizeof(uint16_t), period_len, f);
    fclose(f);

    spi_fd = open(dev, O_RDWR);
    if (spi_fd < 0){ perror("spidev"); return 1; }
    uint8_t mode = SPI_MODE_0, bits = 8;
    ioctl(spi_fd, SPI_IOC_WR_MODE, &mode);
    ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
    ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &spi_speed);

    int rt = 0;
    struct sched_param sp; sp.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) rt = 1;
    int locked = (mlockall(MCL_CURRENT|MCL_FUTURE) == 0);
    pm_qos_hold();
    int pinned = 0;
    if (cpu_pin >= 0){
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu_pin, &set);
        pinned = (sched_setaffinity(0, sizeof set, &set) == 0);
    }
    /* Touch the whole working set so no page fault lands mid-dwell. */
    for (uint32_t i=0;i<points*4;i++) (void)tbl[i];
    for (uint32_t i=0;i<period_len;i++) (void)seq[i];
    memset(hist, 0, sizeof hist);            /* fault it in before timing */

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    /* Cold start: R12 down to R1, wait out the VCO band-select ADC, then R0
     * with autocal. Without this the part keeps whatever band the previous
     * command calibrated and simply never locks at these frequencies. */
    for (int r = 12; r >= 1; r--) wr(boot[r]);
    usleep(delay_us);
    wr(boot[0]);
    usleep(20000);                       /* let the loop settle before keying */

    /* first point, then key the output */
    wr(tbl[seq[0]*4+0]); wr(tbl[seq[0]*4+1]);
    wr(tbl[seq[0]*4 + (autocal_every?3:2)]);
    if (autocal_every) usleep(delay_us);
    wr(r6_on);

    struct timespec start; clock_gettime(CLOCK_MONOTONIC, &start);
    int64_t t0 = ns_of(&start);
    uint64_t played = 0;
    for (uint64_t i = 0; total_hops == 0 || i < total_hops; i++){
        if (stop_now) break;
        int64_t deadline = t0 + (int64_t)(i+1) * (int64_t)dwell_ns;
        wait_until(deadline);
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        hist_add(ns_of(&now) - deadline);
        played++;
        if (total_hops == 0 || i+1 < total_hops){
            const uint32_t *w = &tbl[seq[(i+1) % period_len]*4];
            wr(w[0]); wr(w[1]); wr(w[autocal_every ? 3 : 2]);
            if (autocal_every) usleep(delay_us);
        }
    }
    wr(r6_off);
    struct timespec done; clock_gettime(CLOCK_MONOTONIC, &done);
    double elapsed = (ns_of(&done) - t0)/1e9;

    uint64_t half_us = dwell_ns/2000; uint64_t over = hist_over;
    for (uint64_t i = half_us; i < HB; i++) over += hist[i];
    fprintf(stderr,
      "{\"impl\":\"c\",\"hops\":%llu,\"period\":%u,\"dwell_us\":%.3f,"
      "\"elapsed_s\":%.6f,\"sched_s\":%.6f,\"median_us\":%.3f,"
      "\"p99_us\":%.3f,\"max_us\":%.3f,\"late\":%llu,\"rt\":%d,"
      "\"mlock\":%d,\"spi_hz\":%u,\"spin_us\":%ld,\"cpu\":%d,"
      "\"pinned\":%d,\"pmqos\":%d,\"autocal\":%u}\n",
      (unsigned long long)played, period_len, dwell_ns/1000.0, elapsed,
      (double)played*dwell_ns/1e9, hist_q(0.5), hist_q(0.99),
      hist_max/1000.0, (unsigned long long)over, rt, locked, spi_speed,
      spin_ns/1000L, cpu_pin, pinned, latency_fd >= 0, autocal_every);
    close(spi_fd);
    if (latency_fd >= 0) close(latency_fd);   /* releases the PM QoS hold */
    return 0;
}

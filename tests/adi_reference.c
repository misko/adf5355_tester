/* Differential-test oracle: the arithmetic of the Analog Devices no-OS ADF5355
 * driver (drivers/frequency/adf5355/adf5355.c), transcribed without changes so
 * that the Python implementation can be compared against it directly.
 *
 * Deliberately NOT corrected -- this is the reference, warts included.  The
 * Python side documents where and why it diverges.
 *
 *   build: cc -O2 -o adi_reference adi_reference.c
 *   run:   ./adi_reference <ref_hz> <freq_hz> <chan 0|1> <cp_ua> <doubler> <div2>
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define MODULUS1              16777216ULL
#define MAX_MODULUS2          16384
#define MAX_FREQ_PFD          75000000UL
#define MIN_INT_PRESCALER_89  75
#define MIN_VCO_FREQ          3400000000ULL

#define REG5_DEFAULT   0x00800025
#define REG6_DEFAULT   0x14000006
#define REG10_DEFAULT  0x00C0000A

static uint32_t div_round_up(uint64_t n, uint64_t d) { return (uint32_t)((n + d - 1) / d); }
static uint32_t div_round_closest(uint64_t n, uint64_t d) { return (uint32_t)((n + d / 2) / d); }
static uint32_t clampu(uint32_t v, uint32_t lo, uint32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }
static uint64_t gcd64(uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; }

static void pll_fract_n_compute(uint64_t vco, uint64_t pfd, uint32_t *integer,
                                uint32_t *fract1, uint32_t *fract2, uint32_t *mod2)
{
    uint64_t tmp = vco % pfd;
    uint64_t quot = vco / pfd;

    tmp = tmp * MODULUS1;
    *fract2 = (uint32_t)(tmp % pfd);
    tmp = tmp / pfd;

    *integer = (uint32_t)quot;
    *fract1  = (uint32_t)tmp;
    *mod2    = (uint32_t)pfd;

    while (*mod2 > MAX_MODULUS2) { *mod2 >>= 1; *fract2 >>= 1; }

    uint32_t g = (uint32_t)gcd64(*fract2, *mod2);
    if (g) { *mod2 /= g; *fract2 /= g; }
}

int main(int argc, char **argv)
{
    if (argc != 7) { fprintf(stderr, "usage: %s ref_hz freq_hz chan cp_ua doubler div2\n", argv[0]); return 2; }

    uint64_t clkin   = strtoull(argv[1], NULL, 10);
    uint64_t freq    = strtoull(argv[2], NULL, 10);
    int      chan    = atoi(argv[3]);
    uint32_t cp_ua   = (uint32_t)strtoul(argv[4], NULL, 10);
    int      doubler = atoi(argv[5]);
    int      div2    = atoi(argv[6]);

    /* --- adf5355_setup: maximize the PFD frequency --- */
    uint32_t ref_div_factor = 0, fpfd;
    do {
        ref_div_factor++;
        fpfd = (uint32_t)((clkin * (doubler ? 2 : 1)) / (ref_div_factor * (div2 ? 2 : 1)));
    } while (fpfd > MAX_FREQ_PFD);

    uint32_t cp_code = clampu(div_round_closest(cp_ua - 315, 315), 0, 15);

    /* --- timeouts --- */
    uint32_t tmo = clampu(div_round_up(fpfd, 20000U * 30U), 1, 1023);
    uint32_t synth_tmo = div_round_up((uint64_t)fpfd * 2U, 100000ULL * tmo);
    uint32_t alc_tmo   = div_round_up((uint64_t)fpfd * 5U, 100000ULL * tmo);
    uint32_t band_div  = div_round_up(fpfd, 2400000U);

    /* --- VCO band-select ADC clock --- */
    uint32_t adc_div = clampu(div_round_up(fpfd / 100000U - 2, 4), 1, 255);
    uint32_t delay_us = div_round_up(16000000UL, fpfd / (4 * adc_div + 2));

    /* --- adf5355_set_freq --- */
    uint32_t rf_div_sel = 0;
    if (chan == 0) {
        while (freq < MIN_VCO_FREQ) { freq <<= 1; rf_div_sel++; }
    } else {
        freq >>= 1;                      /* RFoutB is 2 x VCO */
    }

    uint32_t integer, fract1, fract2, mod2;
    pll_fract_n_compute(freq, fpfd, &integer, &fract1, &fract2, &mod2);

    int prescaler = (integer >= MIN_INT_PRESCALER_89);
    int neg_bleed = 1;
    if (fpfd > 100000000UL || (fract1 == 0 && fract2 == 0)) neg_bleed = 0;

    uint32_t cp_bleed = clampu(div_round_up(400U * cp_ua, (uint64_t)integer * 375U), 1, 255);

    /* --- register words (the computed ones; R3/R5/R7/R8/R11/R12 are constant) --- */
    int outa_en = (chan == 0), outb_en = (chan == 1), outa_pwr = 3;
    uint32_t r0 = ((integer & 0xFFFF) << 4) | ((uint32_t)prescaler << 20) | (1u << 21);
    uint32_t r1 = (fract1 & 0xFFFFFF) << 4;
    uint32_t r2 = ((mod2 & 0x3FFF) << 4) | ((fract2 & 0x3FFF) << 18);
    uint32_t r4 = (1u << 7) | (1u << 8) | ((cp_code & 0xF) << 10) | (1u << 14)
                | ((ref_div_factor & 0x3FF) << 15) | ((uint32_t)(div2 ? 1 : 0) << 25)
                | ((uint32_t)(doubler ? 1 : 0) << 26) | (6u << 27);
    uint32_t r6 = ((outa_pwr & 3u) << 4) | ((uint32_t)outa_en << 6)
                | ((uint32_t)(!outb_en) << 10) | (1u << 11)
                | ((cp_bleed & 0xFF) << 13) | ((rf_div_sel & 7u) << 21)
                | (1u << 24) | ((uint32_t)neg_bleed << 29) | REG6_DEFAULT;
    uint32_t r9 = ((synth_tmo & 0x1F) << 4) | ((alc_tmo & 0x1F) << 9)
                | ((tmo & 0x3FF) << 14) | ((band_div & 0xFF) << 24);
    uint32_t r10 = (1u << 4) | (1u << 5) | ((adc_div & 0xFF) << 6) | REG10_DEFAULT;

    printf("fpfd=%u\n", fpfd);
    printf("r_counter=%u\n", ref_div_factor);
    printf("integer=%u\n", integer);
    printf("frac1=%u\n", fract1);
    printf("frac2=%u\n", fract2);
    printf("mod2=%u\n", mod2);
    printf("rf_div_sel=%u\n", rf_div_sel);
    printf("prescaler=%d\n", prescaler);
    printf("cp_bleed=%u\n", cp_bleed);
    printf("neg_bleed=%d\n", neg_bleed);
    printf("adc_div=%u\n", adc_div);
    printf("delay_us=%u\n", delay_us);
    printf("r0=%u\nr1=%u\nr2=%u\nr4=%u\nr6=%u\nr9=%u\nr10=%u\n",
           r0, r1 | 1u, r2 | 2u, r4 | 4u, r6, r9 | 9u, r10);
    return 0;
}

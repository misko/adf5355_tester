/* Red/green tests for the on-air frequency layer.
 *
 * The suite is split into two kinds of test on purpose:
 *
 *   SELF-CONSISTENT  encode then decode and require agreement. These are
 *                    cheap and catch typos, but they CANNOT catch a scaling
 *                    error applied equally in both directions -- the very
 *                    fault that shipped here. They are shown failing to
 *                    catch it, deliberately.
 *
 *   ANCHORED         tie the arithmetic to something outside the module: the
 *                    VCO's physical band, a hand-computed datasheet vector,
 *                    the definition of the doubler. These are the ones that
 *                    catch it.
 *
 * Run with --broken to enable the fault and watch which tests notice.
 */
#include "adf_freq.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int passed, failed;
static const char *kind_now = "";

static void ok(int cond, const char *name, const char *detail)
{
    if (cond) { passed++; printf("  \033[32mPASS\033[0m %-8s %-34s %s\n", kind_now, name, detail); }
    else      { failed++; printf("  \033[31mFAIL\033[0m %-8s %-34s %s\n", kind_now, name, detail); }
}

#define FPFD 125000000.0

/* ---- self-consistent ---------------------------------------------------- */

static void t_round_trip_b(void)
{
    const double f[] = {6.9e9, 8.0e9, 10.0e9, 11.3e9, 13.5e9};
    double worst = 0; int errs = 0;
    for (unsigned i = 0; i < sizeof f/sizeof f[0]; i++) {
        adf_tune_t t;
        if (adf_encode(FPFD, ADF_OUT_B, f[i], &t) != ADF_OK) { errs++; continue; }
        double d = fabs(adf_decode(FPFD, &t) - f[i]);
        if (d > worst) worst = d;
    }
    char m[96]; snprintf(m, sizeof m, "worst %.3f Hz over 5 points, %d encode errors", worst, errs);
    ok(errs == 0 && worst < 1.0, "round_trip_b", m);
}

static void t_step_linearity(void)
{
    double prev = 0, worst = 0;
    for (int i = 0; i < 10; i++) {
        adf_tune_t t;
        double want = 11.3e9 + i * 100e3;
        if (adf_encode(FPFD, ADF_OUT_B, want, &t) != ADF_OK) { worst = 1e9; break; }
        double got = adf_decode(FPFD, &t);
        if (i) { double step = got - prev; if (fabs(step - 100e3) > worst) worst = fabs(step - 100e3); }
        prev = got;
    }
    char m[96]; snprintf(m, sizeof m, "worst step error %.3f Hz vs 100 kHz nominal", worst);
    ok(worst < 1.0, "step_linearity_b", m);
}

static void t_monotonic(void)
{
    double prev = -1; int bad = 0;
    for (int i = 0; i < 200; i++) {
        adf_tune_t t; double want = 7.0e9 + i * 25e6;
        if (adf_encode(FPFD, ADF_OUT_B, want, &t) != ADF_OK) continue;
        double got = adf_decode(FPFD, &t);
        if (got < prev) bad++;
        prev = got;
    }
    char m[64]; snprintf(m, sizeof m, "%d inversions over 200 points", bad);
    ok(bad == 0, "monotonic_b", m);
}

/* ---- anchored to physical truth ----------------------------------------- */

static void t_doubler_definition(void)
{
    double v = 5.65e9, got = adf_output_from_vco(ADF_OUT_B, 0, v);
    char m[80]; snprintf(m, sizeof m, "RFoutB(%.2f GHz VCO) = %.3f GHz, want 11.300", v/1e9, got/1e9);
    ok(fabs(got - 2.0*v) < 1e-6, "doubler_is_2x_vco", m);
}

static void t_vco_in_band(void)
{
    int out_of_band = 0; double worst = 0;
    for (int i = 0; i < 60; i++) {
        adf_tune_t t; double want = 6.9e9 + i * 110e6;
        if (want > ADF_OUTB_MAX_HZ) break;
        if (adf_encode(FPFD, ADF_OUT_B, want, &t) != ADF_OK) continue;
        double n = t.integer + (t.frac1 + (double)t.frac2/t.mod2)/ADF_MOD1;
        double vco = n * FPFD;
        if (vco < ADF_VCO_MIN_HZ || vco > ADF_VCO_MAX_HZ) { out_of_band++; if (vco > worst) worst = vco; }
    }
    char m[96]; snprintf(m, sizeof m, "%d of 60 encode a VCO outside 3.4-6.8 GHz (worst %.2f GHz)", out_of_band, worst/1e9);
    ok(out_of_band == 0, "vco_within_band", m);
}

static void t_known_answer(void)
{
    /* Hand-computed: RFoutB 11.3 GHz, fPFD 125 MHz -> VCO 5.65 GHz,
     * N = 45.2, so INT = 45 and FRAC1 = 0.2 * 2^24 = 3355443. */
    adf_tune_t t;
    adf_err_t e = adf_encode(FPFD, ADF_OUT_B, 11.3e9, &t);
    char m[110];
    snprintf(m, sizeof m, "INT=%u FRAC1=%u (want INT=45 FRAC1=3355443) [%s]",
             t.integer, t.frac1, adf_strerror(e));
    ok(e == ADF_OK && t.integer == 45 && t.frac1 == 3355443, "known_answer_11G3", m);
}

static void t_channel_parity(void)
{
    /* RFoutB at 2f and RFoutA at f with no division are the same VCO. */
    adf_tune_t b, a;
    adf_err_t eb = adf_encode(FPFD, ADF_OUT_B, 11.3e9, &b);
    adf_err_t ea = adf_encode(FPFD, ADF_OUT_A, 5.65e9, &a);
    int same = (eb == ADF_OK && ea == ADF_OK &&
                b.integer == a.integer && b.frac1 == a.frac1);
    char m[110];
    snprintf(m, sizeof m, "B@11.3G INT=%u F1=%u vs A@5.65G INT=%u F1=%u",
             b.integer, b.frac1, a.integer, a.frac1);
    ok(same, "channel_parity", m);
}

static void t_rejects_out_of_band(void)
{
    adf_tune_t t;
    int lo = adf_encode(FPFD, ADF_OUT_B, 6.0e9, &t) != ADF_OK;
    int hi = adf_encode(FPFD, ADF_OUT_B, 14.0e9, &t) != ADF_OK;
    char m[80]; snprintf(m, sizeof m, "6.0 GHz rejected=%d, 14.0 GHz rejected=%d", lo, hi);
    ok(lo && hi, "rejects_out_of_band", m);
}

static void t_resolution(void)
{
    double lsb = adf_resolution_hz(FPFD, ADF_OUT_B, 0);
    adf_tune_t t; adf_encode(FPFD, ADF_OUT_B, 11.3e9 + 3.7, &t);
    double err = fabs(adf_decode(FPFD, &t) - (11.3e9 + 3.7));
    char m[96]; snprintf(m, sizeof m, "LSB %.3f Hz, achieved within %.3f Hz", lsb, err);
    ok(err <= lsb + 1e-6, "resolution_within_lsb", m);
}

int main(int argc, char **argv)
{
    int broken = (argc > 1 && strcmp(argv[1], "--broken") == 0);
    adf_emulate_missing_doubler = broken;
    printf("\n  ADF5355 on-air frequency layer -- %s\n\n",
           broken ? "\033[31mFAULT INJECTED (output stage skipped for RFoutB)\033[0m"
                  : "\033[32mcorrect implementation\033[0m");

    kind_now = "[self]";
    t_round_trip_b(); t_step_linearity(); t_monotonic();
    printf("\n");
    kind_now = "[anchor]";
    t_doubler_definition(); t_vco_in_band(); t_known_answer();
    t_channel_parity(); t_rejects_out_of_band(); t_resolution();

    printf("\n  %d passed, %d failed\n\n", passed, failed);
    return failed ? 1 : 0;
}

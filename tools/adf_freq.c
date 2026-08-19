#include "adf_freq.h"
#include <math.h>

int adf_emulate_missing_doubler = 0;

double adf_vco_from_output(adf_out_t out, uint8_t rf_div_sel, double out_hz)
{
    if (out == ADF_OUT_B)
        return adf_emulate_missing_doubler ? out_hz : out_hz / 2.0;
    return out_hz * (double)(1u << rf_div_sel);
}

double adf_output_from_vco(adf_out_t out, uint8_t rf_div_sel, double vco_hz)
{
    if (out == ADF_OUT_B)
        return adf_emulate_missing_doubler ? vco_hz : vco_hz * 2.0;
    return vco_hz / (double)(1u << rf_div_sel);
}

double adf_resolution_hz(double fpfd_hz, adf_out_t out, uint8_t rf_div_sel)
{
    double vco_step = fpfd_hz / ADF_MOD1;
    return adf_output_from_vco(out, rf_div_sel, vco_step)
         - adf_output_from_vco(out, rf_div_sel, 0.0);
}

adf_err_t adf_encode(double fpfd_hz, adf_out_t out, double want_hz,
                     adf_tune_t *t)
{
    if (!t || fpfd_hz <= 0.0) return ADF_ERR_ARG;

    uint8_t div = 0;
    if (out == ADF_OUT_B) {
        if (want_hz < ADF_OUTB_MIN_HZ || want_hz > ADF_OUTB_MAX_HZ)
            return ADF_ERR_RANGE;
    } else {
        if (want_hz < ADF_OUTA_MIN_HZ || want_hz > ADF_OUTA_MAX_HZ)
            return ADF_ERR_RANGE;
        /* smallest divider that lifts the VCO into its band */
        while (div <= 6 && want_hz * (double)(1u << div) < ADF_VCO_MIN_HZ)
            div++;
        if (div > 6) return ADF_ERR_VCO;
    }

    double vco = adf_vco_from_output(out, div, want_hz);
    if (!adf_emulate_missing_doubler &&
        (vco < ADF_VCO_MIN_HZ || vco > ADF_VCO_MAX_HZ))
        return ADF_ERR_VCO;

    double n = vco / fpfd_hz;
    uint32_t integer = (uint32_t)floor(n);
    if (integer < 23 || integer > 32767) return ADF_ERR_INT;

    double frac = (n - (double)integer) * ADF_MOD1;
    uint32_t frac1 = (uint32_t)floor(frac);
    double rem = frac - (double)frac1;

    /* MOD2 fixed at its maximum so the residue lands as finely as the part
     * allows; FRAC2 is the residue measured in those steps. */
    uint32_t mod2 = 16383;
    uint32_t frac2 = (uint32_t)llround(rem * (double)mod2);
    if (frac2 >= mod2) { frac2 = 0; frac1 += 1; }
    if (frac1 >= (uint32_t)ADF_MOD1) { frac1 = 0; integer += 1; }

    t->integer = integer; t->frac1 = frac1; t->frac2 = frac2;
    t->mod2 = mod2; t->rf_div_sel = div; t->out = out;
    return ADF_OK;
}

double adf_decode(double fpfd_hz, const adf_tune_t *t)
{
    if (!t || t->mod2 == 0) return 0.0;
    double n = (double)t->integer
             + ((double)t->frac1 + (double)t->frac2 / (double)t->mod2)
               / ADF_MOD1;
    return adf_output_from_vco(t->out, t->rf_div_sel, n * fpfd_hz);
}

const char *adf_strerror(adf_err_t e)
{
    switch (e) {
    case ADF_OK:        return "ok";
    case ADF_ERR_RANGE: return "requested output outside the part's range";
    case ADF_ERR_VCO:   return "VCO outside 3.4-6.8 GHz";
    case ADF_ERR_INT:   return "INT outside the fractional-N range";
    default:            return "bad argument";
    }
}

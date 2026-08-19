/* On-air frequency abstraction for the ADF5355.
 *
 * Callers state the frequency they want to appear at the connector. Exactly
 * one pair of functions -- vco_from_output / output_from_vco -- knows that
 * RFoutB is the frequency doubler and RFoutA is the divider chain. Everything
 * above works in real frequencies and nothing above needs to know.
 *
 * That pair is deliberately an inverse pair rather than a single conversion,
 * because it makes the whole layer testable with no hardware: encode a
 * frequency to registers, decode those registers back, and the two must agree.
 * A scaling mistake in the output stage cannot survive that round trip, which
 * is precisely the class of fault that put a 2x error on every RFoutB comb in
 * this project and was only caught on-air, weeks late.
 */
#ifndef ADF_FREQ_H
#define ADF_FREQ_H

#include <stdint.h>

typedef enum { ADF_OUT_A = 0, ADF_OUT_B = 1 } adf_out_t;

typedef struct {
    uint32_t  integer;      /* INT   */
    uint32_t  frac1;        /* FRAC1, scaled by 2^24 */
    uint32_t  frac2;        /* FRAC2 */
    uint32_t  mod2;         /* MOD2  */
    uint8_t   rf_div_sel;   /* RFoutA divider exponent; 0 for RFoutB */
    adf_out_t out;
} adf_tune_t;

typedef enum {
    ADF_OK = 0,
    ADF_ERR_RANGE,          /* requested output outside the part's reach */
    ADF_ERR_VCO,            /* no divider puts the VCO in its band       */
    ADF_ERR_INT,            /* INT outside the legal fractional-N range  */
    ADF_ERR_ARG,
} adf_err_t;

#define ADF_VCO_MIN_HZ   3400000000.0
#define ADF_VCO_MAX_HZ   6800000000.0
#define ADF_OUTB_MIN_HZ  (ADF_VCO_MIN_HZ * 2.0)
#define ADF_OUTB_MAX_HZ  (ADF_VCO_MAX_HZ * 2.0)
#define ADF_OUTA_MIN_HZ  (ADF_VCO_MIN_HZ / 64.0)
#define ADF_OUTA_MAX_HZ  ADF_VCO_MAX_HZ
#define ADF_MOD1         16777216.0     /* 2^24 */

/* The output stage, in one place. */
double adf_vco_from_output(adf_out_t out, uint8_t rf_div_sel, double out_hz);
double adf_output_from_vco(adf_out_t out, uint8_t rf_div_sel, double vco_hz);

/* want_hz is the frequency at the connector, not the VCO. */
adf_err_t adf_encode(double fpfd_hz, adf_out_t out, double want_hz,
                     adf_tune_t *t);

/* What will actually appear at the connector for these registers. */
double adf_decode(double fpfd_hz, const adf_tune_t *t);

/* Smallest output step at this setting -- one FRAC1 LSB through the stage. */
double adf_resolution_hz(double fpfd_hz, adf_out_t out, uint8_t rf_div_sel);

const char *adf_strerror(adf_err_t e);

/* Test seam: when set, the output stage is skipped for RFoutB, reproducing
 * the fault this layer exists to prevent. Never set in production code. */
extern int adf_emulate_missing_doubler;

#endif

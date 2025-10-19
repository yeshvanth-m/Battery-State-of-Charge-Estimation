#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "test_data.h"
#include "soc_fcn_weights.h"
#include "scaler_with_rows.h"
#include <inttypes.h>

#define NUM_INPUTS          5
#define NEURONS_LAYER_1     128
#define NEURONS_LAYER_2     64
#define NUM_OUTPUTS         1
#define MAX_NEURONS        10

typedef struct 
{
    uint32_t num_layers;
    float weights_layer_1[NEURONS_LAYER_1][NUM_INPUTS];
    float weights_layer_2[NEURONS_LAYER_2][NEURONS_LAYER_1];
    float weights_output[NUM_OUTPUTS][NEURONS_LAYER_2];
    float biases_layer_1[NEURONS_LAYER_1];
    float biases_layer_2[NEURONS_LAYER_2];
    float biases_output[NUM_OUTPUTS];
} FeedForwardNN;

FILE *fpwb;

/*
FeedForwardNN FCN = 
{
    .num_layers = 3,
    .weights_layer_1 = 
    {
        {-0.790833873752347, -0.293611662252038,  0.839281232229617, -0.847619804394231,   0.135862174733978},
        {0.40825113221851,    0.39705806376579,  -0.290357720187592,  0.18472923024971,    0.185955912916414},
        {0.671931533819604,  -0.391433402191457, -0.157767635104551, -0.0176598182920163, -0.210830553464044},
        {0.59321694613389,   -0.23969260483893,   0.17657495083671,   0.855148447093332,  -0.510257912300629},
        {0.314671165004648,  -0.162040389075539,  0.280583557825099, -0.407333783382596,   0.742788581740702},
        {0.398124746090035,   0.101446110695795, -0.921896975281057, -0.693203115921896,   0.571582333566267},
        {0.332482480273737,  -0.218877077142823, -1.02550482305152,   0.272644748084382,  -0.32564495452459},
        {-0.107128966909749, -0.286177853382451, -0.317794167820117, -0.00981799067106903, 0.142394838578472},
        {-0.208556881008102, -0.273976857863,    -0.255712715566176, -0.0501788414506202,  0.739297835892161},
        {-0.146289184211499,  0.0850778228180522,-0.815533759052087, -0.192547703664811,   0.718696367043104}
    },
    .biases_layer_1 = 
    {
        0.0309259452878632, -0.61039586384407, 0.366851958205962, -0.103112241142919, 0.198603331272367,
        0.0759108569330868, 0.539282159520218,-0.0265894001948924,-0.152150965836352,-0.13633677940557
    },
    .weights_layer_2 = 
    {
        {-1.0096241260219,0.384839880673814,0.467489258494656,0.399665788674379,-0.39986765373004,-0.242523062637488,0.170338236703085,-0.449073187338085,-0.110985596008292,-0.327832640183828},
        {-0.469344741398983,0.554041599804513,0.841198746349013,-0.316541357867751,0.234096829052763,-0.328537705584302,0.0613938167478977,0.512860441961598,-0.059814192142893,0.728672050652683},
        {-0.33366701426426,-0.376882788510052,0.311414821534382,-0.278544013114602,-0.263311150637896,0.158776985157421,-0.410385588957597,0.180507082241326,-0.529198405916037,0.398178938922273},
        {0.0446342653285069,-0.0302322057616488,-0.470136641203741,-0.407445898023458,0.0332251547667959,-0.00937424348337765,-0.128118817997811,-0.423731673920195,-0.289804555819791,0.0239534032731946},
        {0.253857351746338,-0.00308646332152365,-0.149916365830585,-0.161873950257283,0.19603428978973,-0.370238984233905,-0.828160246161552,-0.457056736574776,0.00272092166756452,0.000353041428104929},
        {-0.260857746111831,-0.131945890540378,-0.161023618106293,-0.207100215739002,-0.474371648552192,0.180145499155662,0.426320585849852,-0.508445308077668,-0.0778113293142873,0.196189801567282},
        {-0.161293032289329,-0.150229798505492,0.0770366797942338,-0.766050729710707,-0.671044988982937,0.459437787314645,0.620901087069555,0.0789235457702105,-0.360142473399782,0.109853893526042},
        {0.258737680137938,0.367739684643942,0.543511872213477,-0.421064102551739,-0.252528921228268,0.382260412024738,0.255376748039719,0.372032819395783,0.480566680225967,0.141453987316601},
        {-0.375267643896386,-0.435796432567931,-0.749239928331824,-0.555311904446642,0.26691235156032,0.603001476055316,-0.138194574982889,0.293817713197324,0.0510996450821657,-0.179351061325503},
        {0.119897219519705,-0.54880824537398,-0.0942368609059711,0.21852322760021,0.186733605929516,-0.271414688201577,0.155425079612475,0.547298977624833,0.593868207145585,0.313177357842453}
    },
    .biases_layer_2 = 
    {
        0.36217399315038, -0.269483735764346, 0, -0.108147271375755, 0.0191867085858886, 
        -0.105379385604212, 0.5247585209603, -0.251412013633538, 0.704773205004789, -0.110139501520608
    },
    .weights_output = 
    {
        0.789633340219721,-0.793390371333755,0.537834389979798,-0.331225134865427,0.381909043420518,
        -0.525491098040812,0.650008320905375,-0.298958799039973,-0.576567653790747,0.352706015208224
    },
    .biases_output = {0.516276464685792}
};
 */
// helper relu for float
static inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

// helper relu for fixed Q (Q1.3.27)
static inline int32_t relu_q(int32_t x) { return x > 0 ? x : 0; }

// convert float -> signed Q with frac_bits fractional bits, saturate to int32
static inline int32_t float_to_q(float v, int frac_bits)
{
    // multiplication in 64-bit to avoid overflow for large shift amounts
    int64_t scale = (int64_t)1 << frac_bits;
    int64_t tmp = (int64_t) llroundf(v * (double)scale);

    if (tmp > INT32_MAX) return INT32_MAX;
    if (tmp < INT32_MIN) return INT32_MIN;
    return (int32_t)tmp;
}

// convert signed Q -> float
static inline float q_to_float(int32_t qv, int frac_bits)
{
    return (float)qv / (float)((int64_t)1 << frac_bits);
}

float min_input = __FLT_MAX__, max_input = __FLT_MIN__, min_weight = __FLT_MAX__, max_weight = __FLT_MIN__, 
min_bias = __FLT_MAX__, max_bias = __FLT_MIN__;

int32_t max_acc = INT32_MIN;
int32_t min_acc = INT32_MAX;

/*
 Q formats used:
 - input:  Q1.2.13  -> frac_bits = 13
 - weight: Q1.1.14  -> frac_bits = 14
 - acc:    Q1.3.27  -> frac_bits = 27
 - bias:   Q1.3.27  -> frac_bits = 27
*/

// Example: external functions/data expected by your code (placeholders)
// extern float FCN_L1_W[][NUM_INPUTS];
// extern float FCN_L1_b[];
// etc.

void fcn_layer(const float *input,
               int16_t *input_q,
               const float *weights,   // row-major: [num_neurons][input_size]
               const float *biases,
               uint32_t num_neurons,
               uint32_t input_size,
               float *output,
               int32_t *output_i,
               int apply_relu)
{
    // Perform floating point matrix-vector multiplication and add biases
    for (uint32_t i = 0; i < num_neurons; i++) 
    {
        const float *wrow = &weights[i * input_size];   // <-- correct stride
        float acc = biases[i];
        for (uint32_t j = 0; j < input_size; j++) 
        {
            acc += input[j] * wrow[j];
        }
        output[i] = apply_relu ? relu_f(acc) : acc;
        //printf(" Neuron %2d: Weighted sum: %f, Output: %f\n", i, acc, output[i]);
    }

    // Perform fixed point matrix-vector multiplication and add biases
    for (uint32_t i = 0; i < num_neurons; i++) 
    {
        const float *wrow = &weights[i * input_size];   // <-- correct stride
        
        int32_t acc;

        if (biases[i] > max_bias) max_bias = biases[i];
        if (biases[i] < min_bias) min_bias = biases[i];
        /*
        if (biases[i] < 0)
        {
            acc = (1 << 32) | (int32_t)((-1 * biases[i]) * (1 << 14)); // Q2.14
        }
        else
        {
            acc = (int32_t)(biases[i] * (1 << 14)); // Q2.14
        } */

        for (uint32_t j = 0; j < input_size; j++) 
        {
            if (input[j] > max_input) max_input = input[j];
            if (input[j] < min_input) min_input = input[j];
            if (weights[j] > max_weight) max_weight = weights[j];
            if (weights[j] < min_weight) min_weight = weights[j];

        }
        //printf(" Neuron %2d: Weighted sum: %f, Output: %f\n", i, acc, output[i]);
    }

    // Perform fixed point conversion matrix-vector multiplication and add biases 
    // Q map:
    // input  -> Q1.2.13  (frac_bits_in  = 13)
    // weight -> Q1.1.14  (frac_bits_wt  = 14)
    // product (input*weight) -> Q1.3.27 (13+14 = 27 fractional bits)
    // accumulator & bias -> Q1.3.27 (frac_bits_acc = 27)

    const int frac_bits_in  = 13;
    const int frac_bits_wt  = 14;
    const int frac_bits_acc = 27; // product fractional bits

    // Option A: convert whole input vector to fixed once (recommended)
    // allocate temporary buffer for inputs in fixed Q1.2.13
    // NOTE: if input_size is large, consider static/stack allocation limits.

    // Now process each neuron
    for (uint32_t i = 0; i < num_neurons; i++) 
    {
        const float *wrow = &weights[i * input_size];   // <-- correct stride

        // convert bias to Q1.3.27
        int32_t bias_q = float_to_q(biases[i], frac_bits_acc);

        // initialize accumulator in 64-bit using bias (Q1.3.27)
        int32_t acc = bias_q;

        // For each input: multiply input_q (Q1.2.13) * weight_q (Q1.1.14)
        // product has frac_bits = 13 + 14 = 27 -> already aligned with acc Q1.3.27
        for (uint32_t j = 0; j < input_size; j++) 
        {
            int16_t w_q = (int16_t)float_to_q(wrow[j], frac_bits_wt);       // Q1.1.14
            // multiply in 64-bit to avoid overflow: (int64_t) * (int64_t)
            int32_t prod = (int32_t)(input_q[j] * w_q);    // result is Q1.3.27
            acc += prod;
        }
        
        if (acc > max_acc) max_acc = (int32_t)acc;
        if (acc < min_acc) min_acc = (int32_t)acc;
        // saturate acc into int32 (the accumulator format is Q1.3.27 stored in int32)
        output_i[i] = (int32_t)acc;

        // apply ReLU in fixed domain if requested
        if (apply_relu) {
            output_i[i] = relu_q(output_i[i]);
        }
    }
}

void export_weights_biases()
{
 
    // Perform fixed point conversion matrix-vector multiplication and add biases 
    // Q map:
    // input  -> Q1.2.13  (frac_bits_in  = 13)
    // weight -> Q1.1.14  (frac_bits_wt  = 14)
    // product (input*weight) -> Q1.3.27 (13+14 = 27 fractional bits)
    // accumulator & bias -> Q1.3.27 (frac_bits_acc = 27)

    const int frac_bits_in  = 13;
    const int frac_bits_wt  = 14;
    const int frac_bits_acc = 27; // product fractional bits

    // Option A: convert whole input vector to fixed once (recommended)
    // allocate temporary buffer for inputs in fixed Q1.2.13
    // NOTE: if input_size is large, consider static/stack allocation limits.
    for (uint32_t i = 0; i < FCN_L1_OUT; i++) 
    {
        for (uint32_t j = 0; j < FCN_L1_IN; j++) 
        {
            int32_t w_q = float_to_q(FCN_L1_W[i][j], frac_bits_wt);       // Q1.1.14
            fprintf(fpwb, "%x\n", w_q);
        }
    }

    for (uint32_t i = 0; i < FCN_L1_OUT; i++) 
    {
        int32_t b_q = float_to_q(FCN_L1_b[i], frac_bits_acc);
        fprintf(fpwb, "%x\n", b_q);
    }

    for (uint32_t i = 0; i < FCN_L2_OUT; i++) 
    {
        for (uint32_t j = 0; j < FCN_L2_IN; j++) 
        {
            int32_t w_q = float_to_q(FCN_L2_W[i][j], frac_bits_wt);       // Q1.1.14
            fprintf(fpwb, "%x\n", w_q);
        }
    }

    for (uint32_t i = 0; i < FCN_L2_OUT; i++) 
    {
        int32_t b_q = float_to_q(FCN_L2_b[i], frac_bits_acc);
        fprintf(fpwb, "%x\n", b_q);
    }

    for (uint32_t i = 0; i < 1; i++) 
    {
        for (uint32_t j = 0; j < FCN_L2_OUT; j++) 
        {
            int32_t w_q = float_to_q(FCN_OUT_W[i][j], frac_bits_wt);       // Q1.1.14
            fprintf(fpwb, "%x\n", w_q);
        }
    }
    for (uint32_t i = 0; i < 1; i++) 
    {
        int32_t b_q = float_to_q(FCN_OUT_b[i], frac_bits_acc);
        fprintf(fpwb, "%x\n", b_q);
    }
}

// Helper: convert float -> signed integer with saturation using frac_bits
static inline int64_t float_to_q_sat64(float v, int frac_bits, int out_bits)
{
    int64_t scale = (int64_t)1 << frac_bits;
    int64_t tmp = llroundf((double)v * (double)scale);
    int64_t minv = - (1LL << (out_bits - 1));
    int64_t maxv =   (1LL << (out_bits - 1)) - 1;
    if (tmp < minv) tmp = minv;
    if (tmp > maxv) tmp = maxv;
    return tmp;
}

// Helper: produce two's complement hex string for given signed integer and bitwidth
// Writes into buf (must be large enough). Returns pointer to buf.
static inline char *to_twos_hex(char *buf, int64_t value, int bits)
{
    uint64_t u;
    if (value < 0) {
        // two's complement
        u = ((uint64_t)1 << bits) + (uint64_t)value;
    } else {
        u = (uint64_t)value;
    }
    // hex digits required
    int hexlen = (bits + 3) / 4;
    // format as hex with leading zeros, e.g. 16'h00ab
    // We'll return just the hex digits (without 0x) because we'll produce "16'hhhhh"
    // Write into buf
    // snprintf width: hexlen characters
    char fmt[16];
    snprintf(fmt, sizeof(fmt), "%%0%dx", hexlen);
    snprintf(buf, hexlen + 1 + 2, fmt, (unsigned int)u);
    return buf;
}

// The function that writes the verilog file using arrays linked into your program.
// It expects these symbols to exist (they come from soc_fcn_weights.h):
//   float FCN_L1_W[NEURONS_LAYER_1][NUM_INPUTS];
//   float FCN_L1_b[NEURONS_LAYER_1];
//   float FCN_L2_W[NEURONS_LAYER_2][NEURONS_LAYER_1];
//   float FCN_L2_b[NEURONS_LAYER_2];
//   float FCN_OUT_W[NUM_OUTPUTS][NEURONS_LAYER_2];
//   float FCN_OUT_b[NUM_OUTPUTS];
//
// If your symbols are named differently, adjust the names below.
void emit_verilog_mem_from_c(const char *out_path)
{
    FILE *vf = fopen(out_path, "w");
    if (!vf) {
        perror(out_path);
        return;
    }

    // Fixed-point config (match code comments in this file)
    const int wt_bits = 16;  // weights stored as signed 16-bit (Q1.1.14)
    const int wt_frac = 14;
    const int b_bits  = 32;  // biases stored as signed 32-bit (Q1.3.27)
    const int b_frac  = 27;

    // Write header & module
    fprintf(vf, "// Auto-generated by FCN.c at runtime\n");
    fprintf(vf, "// Module containing ROM initializations (weights and biases)\n\n");

    fprintf(vf,
        "module fcn_weights_mem_from_c (\n"
        "    // read-only memories (flattened row-major)\n"
        "    output logic [%d:0] dummy // placeholder to avoid empty port list\n"
        ");\n\n", 0);

    // L1 weights (NEURONS_LAYER_1 x NUM_INPUTS)
    fprintf(vf, "// L1 weights: %d x %d -> %d-bit fixed (Q1.1.14)\n",
            NEURONS_LAYER_1, NUM_INPUTS, wt_bits);
    fprintf(vf, "localparam L1_W_DEPTH = %d;\n", NEURONS_LAYER_1 * NUM_INPUTS);
    fprintf(vf, "reg signed [%d:0] mem_l1_w [0:L1_W_DEPTH-1];\n\n", wt_bits-1);

    // L1 biases
    fprintf(vf, "// L1 biases: %d -> %d-bit fixed (Q1.3.27)\n", NEURONS_LAYER_1, b_bits);
    fprintf(vf, "reg signed [%d:0] mem_l1_b [0:%d-1];\n\n", b_bits-1, NEURONS_LAYER_1);

    // L2 weights
    fprintf(vf, "// L2 weights: %d x %d -> %d-bit fixed (Q1.1.14)\n",
            NEURONS_LAYER_2, NEURONS_LAYER_1, wt_bits);
    fprintf(vf, "localparam L2_W_DEPTH = %d;\n", NEURONS_LAYER_2 * NEURONS_LAYER_1);
    fprintf(vf, "reg signed [%d:0] mem_l2_w [0:L2_W_DEPTH-1];\n\n", wt_bits-1);

    // L2 biases
    fprintf(vf, "// L2 biases: %d -> %d-bit fixed (Q1.3.27)\n", NEURONS_LAYER_2, b_bits);
    fprintf(vf, "reg signed [%d:0] mem_l2_b [0:%d-1];\n\n", b_bits-1, NEURONS_LAYER_2);

    // OUT weights
    fprintf(vf, "// OUT weights: %d x %d -> %d-bit fixed (Q1.1.14)\n",
            NUM_OUTPUTS, NEURONS_LAYER_2, wt_bits);
    fprintf(vf, "localparam OUT_W_DEPTH = %d;\n", NUM_OUTPUTS * NEURONS_LAYER_2);
    fprintf(vf, "reg signed [%d:0] mem_out_w [0:OUT_W_DEPTH-1];\n\n", wt_bits-1);

    // OUT biases
    fprintf(vf, "// OUT biases: %d -> %d-bit fixed (Q1.3.27)\n", NUM_OUTPUTS, b_bits);
    fprintf(vf, "reg signed [%d:0] mem_out_b [0:%d-1];\n\n", b_bits-1, NUM_OUTPUTS);

    // Begin initial block with inline hex values
    fprintf(vf, "initial begin\n");

    // Helper buffers for hex strings
    char hexbuf[40];

    // --- emit L1 weights (flatten row-major: neuron * NUM_INPUTS + input) ---
    for (int n = 0; n < NEURONS_LAYER_1; ++n) {
        for (int j = 0; j < NUM_INPUTS; ++j) {
            float val = FCN_L1_W[n][j];
            int64_t q = float_to_q_sat64(val, wt_frac, wt_bits);
            to_twos_hex(hexbuf, q, wt_bits);
            fprintf(vf, "    mem_l1_w[%d] = %d'h%s;\n", n * NUM_INPUTS + j, wt_bits, hexbuf);
        }
    }
    fprintf(vf, "\n");

    // --- emit L1 biases ---
    for (int n = 0; n < NEURONS_LAYER_1; ++n) {
        float val = FCN_L1_b[n];
        int64_t q = float_to_q_sat64(val, b_frac, b_bits);
        to_twos_hex(hexbuf, q, b_bits);
        fprintf(vf, "    mem_l1_b[%d] = %d'h%s;\n", n, b_bits, hexbuf);
    }
    fprintf(vf, "\n");

    // --- emit L2 weights ---
    for (int n = 0; n < NEURONS_LAYER_2; ++n) {
        for (int j = 0; j < NEURONS_LAYER_1; ++j) {
            float val = FCN_L2_W[n][j];
            int64_t q = float_to_q_sat64(val, wt_frac, wt_bits);
            to_twos_hex(hexbuf, q, wt_bits);
            fprintf(vf, "    mem_l2_w[%d] = %d'h%s;\n", n * NEURONS_LAYER_1 + j, wt_bits, hexbuf);
        }
    }
    fprintf(vf, "\n");

    // --- emit L2 biases ---
    for (int n = 0; n < NEURONS_LAYER_2; ++n) {
        float val = FCN_L2_b[n];
        int64_t q = float_to_q_sat64(val, b_frac, b_bits);
        to_twos_hex(hexbuf, q, b_bits);
        fprintf(vf, "    mem_l2_b[%d] = %d'h%s;\n", n, b_bits, hexbuf);
    }
    fprintf(vf, "\n");

    // --- emit OUT weights ---
    for (int n = 0; n < NUM_OUTPUTS; ++n) {
        for (int j = 0; j < NEURONS_LAYER_2; ++j) {
            float val = FCN_OUT_W[n][j];
            int64_t q = float_to_q_sat64(val, wt_frac, wt_bits);
            to_twos_hex(hexbuf, q, wt_bits);
            fprintf(vf, "    mem_out_w[%d] = %d'h%s;\n", n * NEURONS_LAYER_2 + j, wt_bits, hexbuf);
        }
    }
    fprintf(vf, "\n");

    // --- emit OUT biases ---
    for (int n = 0; n < NUM_OUTPUTS; ++n) {
        float val = FCN_OUT_b[n];
        int64_t q = float_to_q_sat64(val, b_frac, b_bits);
        to_twos_hex(hexbuf, q, b_bits);
        fprintf(vf, "    mem_out_b[%d] = %d'h%s;\n", n, b_bits, hexbuf);
    }

    fprintf(vf, "end\n\nendmodule\n");
    fclose(vf);
    printf("Wrote Verilog ROM file: %s\n", out_path);
}

/* -------------------------
   mem emitter for $readmemh
   Paste into FCN.c and call:
     emit_mem_files_and_wrapper("fcn");
   This writes:
     fcn_l1_w.mem, fcn_l1_b.mem,
     fcn_l2_w.mem, fcn_l2_b.mem,
     fcn_out_w.mem, fcn_out_b.mem,
     fcn_weights_mem.v
   ------------------------- */

/* --- CONFIG: fixed-point formats --- */
#define WT_BITS  16   /* weights: signed 16-bit */
#define WT_FRAC  14   /* Q1.1.14 */
#define B_BITS   32   /* biases: signed 32-bit */
#define B_FRAC   27   /* Q1.3.27 */

/* NOTE:
   This code expects these arrays (or equivalent) to be available in your C file:
     float FCN_L1_W[NEURONS_LAYER_1][NUM_INPUTS];
     float FCN_L1_b[NEURONS_LAYER_1];
     float FCN_L2_W[NEURONS_LAYER_2][NEURONS_LAYER_1];
     float FCN_L2_b[NEURONS_LAYER_2];
     float FCN_OUT_W[NUM_OUTPUTS][NEURONS_LAYER_2];
     float FCN_OUT_b[NUM_OUTPUTS];
   and these macros:
     NUM_INPUTS, NEURONS_LAYER_1, NEURONS_LAYER_2, NUM_OUTPUTS
   Rename references below if your symbols are different.
*/

/* helper: convert float -> signed integer with saturation */


/* helper: write unsigned hex line padded to hexlen chars (lowercase) */
static void fprintf_hex_line(FILE *f, uint64_t u, int bits) {
    int hexlen = (bits + 3) / 4;
    /* width with leading zeros, lowercase hex */
    fprintf(f, "%0*llx\n", hexlen, (unsigned long long)u);
}

/* convert signed two's complement integer to unsigned representation for hex */
static uint64_t to_unsigned_twos(int64_t val, int bits) {
    if (val < 0) {
        uint64_t u = ((uint64_t)1 << bits) + (uint64_t)val;
        return u;
    } else {
        return (uint64_t)val;
    }
}

/* write a single mem file given float array pointer and dimensions */
static int write_mem_flat_from_2d(const char *fname, const float *arr, size_t rows, size_t cols, int bits, int frac) {
    FILE *f = fopen(fname, "w");
    if (!f) {
        perror(fname);
        return -1;
    }
    size_t cnt = 0;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float v = arr[r * cols + c];
            int64_t q = float_to_q_sat64(v, frac, bits);
            uint64_t u = to_unsigned_twos(q, bits);
            fprintf_hex_line(f, u, bits);
            ++cnt;
        }
    }
    fclose(f);
    /* optional: print count */
    /* printf("Wrote %s (%zu entries)\n", fname, cnt); */
    return 0;
}

/* write a single mem file from 1d float array */
static int write_mem_flat_from_1d(const char *fname, const float *arr, size_t len, int bits, int frac) {
    FILE *f = fopen(fname, "w");
    if (!f) {
        perror(fname);
        return -1;
    }
    for (size_t i = 0; i < len; ++i) {
        float v = arr[i];
        int64_t q = float_to_q_sat64(v, frac, bits);
        uint64_t u = to_unsigned_twos(q, bits);
        fprintf_hex_line(f, u, bits);
    }
    fclose(f);
    return 0;
}

/* Main emitter:
   prefix -> string placed in filenames, e.g. "fcn" gives fcn_l1_w.mem and fcn_weights_mem.v
*/
void emit_mem_files_and_wrapper(const char *prefix) {
    char fname[256];
    /* 1) L1 weights: NEURONS_LAYER_1 x NUM_INPUTS */
    snprintf(fname, sizeof(fname), "%s_l1_w.mem", prefix);
    /* FCN_L1_W is expected as 2D array: [NEURONS_LAYER_1][NUM_INPUTS]
       We'll treat it as flattened row-major pointer.
       Cast required because arr may be declared as float[][]; take address of first element.
    */
    if (write_mem_flat_from_2d(fname, &FCN_L1_W[0][0], (size_t)NEURONS_LAYER_1, (size_t)NUM_INPUTS, WT_BITS, WT_FRAC) != 0) {
        fprintf(stderr, "Failed to write %s\n", fname);
        return;
    }

    /* 2) L1 biases */
    snprintf(fname, sizeof(fname), "%s_l1_b.mem", prefix);
    if (write_mem_flat_from_1d(fname, &FCN_L1_b[0], (size_t)NEURONS_LAYER_1, B_BITS, B_FRAC) != 0) {
        fprintf(stderr, "Failed to write %s\n", fname);
        return;
    }

    /* 3) L2 weights: NEURONS_LAYER_2 x NEURONS_LAYER_1 */
    snprintf(fname, sizeof(fname), "%s_l2_w.mem", prefix);
    if (write_mem_flat_from_2d(fname, &FCN_L2_W[0][0], (size_t)NEURONS_LAYER_2, (size_t)NEURONS_LAYER_1, WT_BITS, WT_FRAC) != 0) {
        fprintf(stderr, "Failed to write %s\n", fname);
        return;
    }

    /* 4) L2 biases */
    snprintf(fname, sizeof(fname), "%s_l2_b.mem", prefix);
    if (write_mem_flat_from_1d(fname, &FCN_L2_b[0], (size_t)NEURONS_LAYER_2, B_BITS, B_FRAC) != 0) {
        fprintf(stderr, "Failed to write %s\n", fname);
        return;
    }

    /* 5) OUT weights: NUM_OUTPUTS x NEURONS_LAYER_2 */
    snprintf(fname, sizeof(fname), "%s_out_w.mem", prefix);
    if (write_mem_flat_from_2d(fname, &FCN_OUT_W[0][0], (size_t)NUM_OUTPUTS, (size_t)NEURONS_LAYER_2, WT_BITS, WT_FRAC) != 0) {
        fprintf(stderr, "Failed to write %s\n", fname);
        return;
    }

    /* 6) OUT biases */
    snprintf(fname, sizeof(fname), "%s_out_b.mem", prefix);
    if (write_mem_flat_from_1d(fname, &FCN_OUT_b[0], (size_t)NUM_OUTPUTS, B_BITS, B_FRAC) != 0) {
        fprintf(stderr, "Failed to write %s\n", fname);
        return;
    }

    /* 7) Emit small Verilog wrapper using $readmemh */
    snprintf(fname, sizeof(fname), "%s_weights_mem.v", prefix);
    FILE *vf = fopen(fname, "w");
    if (!vf) {
        perror(fname);
        return;
    }

    /* Prepare sizes as numbers for Verilog */
    int L1_W_DEPTH = NEURONS_LAYER_1 * NUM_INPUTS;
    int L2_W_DEPTH = NEURONS_LAYER_2 * NEURONS_LAYER_1;
    int OUT_W_DEPTH = NUM_OUTPUTS * NEURONS_LAYER_2;

    fprintf(vf, "// Auto-generated wrapper to load mem files using $readmemh\n");
    fprintf(vf, "// Generated by emit_mem_files_and_wrapper in FCN.c\n\n");

    fprintf(vf, "`timescale 1ns/1ps\n");
    fprintf(vf, "module %s_weights_mem #(\n", prefix);
    fprintf(vf, "    parameter NUM_INPUTS = %d,\n", NUM_INPUTS);
    fprintf(vf, "    parameter NEURONS_L1 = %d,\n", NEURONS_LAYER_1);
    fprintf(vf, "    parameter NEURONS_L2 = %d,\n", NEURONS_LAYER_2);
    fprintf(vf, "    parameter NUM_OUTPUTS = %d,\n", NUM_OUTPUTS);
    fprintf(vf, "    parameter WEIGHT_WIDTH = %d,\n", WT_BITS);
    fprintf(vf, "    parameter BIAS_WIDTH = %d\n", B_BITS);
    fprintf(vf, ") (\n");
    fprintf(vf, "    input  wire [$clog2(NEURONS_L1*NUM_INPUTS)-1:0] addr_l1_w,\n");
    fprintf(vf, "    output wire signed [WEIGHT_WIDTH-1:0] dout_l1_w,\n\n");

    fprintf(vf, "    input  wire [$clog2(NEURONS_L1)-1:0] addr_l1_b,\n");
    fprintf(vf, "    output wire signed [BIAS_WIDTH-1:0] dout_l1_b,\n\n");

    fprintf(vf, "    input  wire [$clog2(NEURONS_L2*NEURONS_L1)-1:0] addr_l2_w,\n");
    fprintf(vf, "    output wire signed [WEIGHT_WIDTH-1:0] dout_l2_w,\n\n");

    fprintf(vf, "    input  wire [$clog2(NEURONS_L2)-1:0] addr_l2_b,\n");
    fprintf(vf, "    output wire signed [BIAS_WIDTH-1:0] dout_l2_b,\n\n");

    fprintf(vf, "    input  wire [$clog2(NUM_OUTPUTS*NEURONS_L2)-1:0] addr_out_w,\n");
    fprintf(vf, "    output wire signed [WEIGHT_WIDTH-1:0] dout_out_w,\n\n");

    fprintf(vf, "    input  wire [$clog2(NUM_OUTPUTS)-1:0] addr_out_b,\n");
    fprintf(vf, "    output wire signed [BIAS_WIDTH-1:0] dout_out_b\n");
    fprintf(vf, ");\n\n");

    /* declare memories */
    fprintf(vf, "    // Depths\n");
    fprintf(vf, "    localparam L1_W_DEPTH = %d;\n", L1_W_DEPTH);
    fprintf(vf, "    localparam L2_W_DEPTH = %d;\n", L2_W_DEPTH);
    fprintf(vf, "    localparam OUT_W_DEPTH = %d;\n\n", OUT_W_DEPTH);

    fprintf(vf, "    // memories\n");
    fprintf(vf, "    reg signed [WEIGHT_WIDTH-1:0] mem_l1_w [0:L1_W_DEPTH-1];\n");
    fprintf(vf, "    reg signed [BIAS_WIDTH-1:0]   mem_l1_b [0:NEURONS_L1-1];\n\n");

    fprintf(vf, "    reg signed [WEIGHT_WIDTH-1:0] mem_l2_w [0:L2_W_DEPTH-1];\n");
    fprintf(vf, "    reg signed [BIAS_WIDTH-1:0]   mem_l2_b [0:NEURONS_L2-1];\n\n");

    fprintf(vf, "    reg signed [WEIGHT_WIDTH-1:0] mem_out_w [0:OUT_W_DEPTH-1];\n");
    fprintf(vf, "    reg signed [BIAS_WIDTH-1:0]   mem_out_b [0:NUM_OUTPUTS-1];\n\n");

    fprintf(vf, "    initial begin\n");
    fprintf(vf, "        // load weights/biases from mem files (hex, two's complement)\n");
    fprintf(vf, "        $readmemh(\"%s_l1_w.mem\", mem_l1_w);\n", prefix);
    fprintf(vf, "        $readmemh(\"%s_l1_b.mem\", mem_l1_b);\n", prefix);
    fprintf(vf, "        $readmemh(\"%s_l2_w.mem\", mem_l2_w);\n", prefix);
    fprintf(vf, "        $readmemh(\"%s_l2_b.mem\", mem_l2_b);\n", prefix);
    fprintf(vf, "        $readmemh(\"%s_out_w.mem\", mem_out_w);\n", prefix);
    fprintf(vf, "        $readmemh(\"%s_out_b.mem\", mem_out_b);\n", prefix);
    fprintf(vf, "    end\n\n");

    /* combinational read outputs */
    fprintf(vf, "    assign dout_l1_w  = mem_l1_w[addr_l1_w];\n");
    fprintf(vf, "    assign dout_l1_b  = mem_l1_b[addr_l1_b];\n\n");

    fprintf(vf, "    assign dout_l2_w  = mem_l2_w[addr_l2_w];\n");
    fprintf(vf, "    assign dout_l2_b  = mem_l2_b[addr_l2_b];\n\n");

    fprintf(vf, "    assign dout_out_w = mem_out_w[addr_out_w];\n");
    fprintf(vf, "    assign dout_out_b = mem_out_b[addr_out_b];\n\n");

    fprintf(vf, "endmodule\n");
    fclose(vf);

    printf("Wrote mem files and Verilog wrapper with prefix '%s_'\n", prefix);
    printf(" - %s_l1_w.mem  (%d entries)\n", prefix, L1_W_DEPTH);
    printf(" - %s_l1_b.mem  (%d entries)\n", prefix, NEURONS_LAYER_1);
    printf(" - %s_l2_w.mem  (%d entries)\n", prefix, L2_W_DEPTH);
    printf(" - %s_l2_b.mem  (%d entries)\n", prefix, NEURONS_LAYER_2);
    printf(" - %s_out_w.mem (%d entries)\n", prefix, OUT_W_DEPTH);
    printf(" - %s_out_b.mem (%d entries)\n", prefix, NUM_OUTPUTS);
    printf(" - %s_weights_mem.v\n", prefix);
}

/* -------------------------
   Usage:
   1) Insert this code into FCN.c
   2) At the end of main(), after arrays are available, call:
        emit_mem_files_and_wrapper(\"fcn\");
   3) Rebuild and run your program:
        gcc -std=c11 -O2 -o fcn_run FCN.c -lm
        ./fcn_run
   4) You will get mem files and fcn_weights_mem.v in the working dir.
   ------------------------- */

/*
void CnnForwardPass (const float *input, const FeedForwardNN *nn, float *output_scalar)
{
    float layer_1[NEURONS_LAYER_1];
    float layer_2[NEURONS_LAYER_2];

    fcn_layer(input,  &nn->weights_layer_1[0][0], nn->biases_layer_1,
              NEURONS_LAYER_1, NUM_INPUTS, layer_1, 1);

    fcn_layer(layer_1, &nn->weights_layer_2[0][0], nn->biases_layer_2,
              NEURONS_LAYER_2, NEURONS_LAYER_1, layer_2, 1);

    float out1[NUM_OUTPUTS];
    fcn_layer(layer_2, &nn->weights_output[0][0], nn->biases_output,
              NUM_OUTPUTS, NEURONS_LAYER_2, out1, 0);   // <-- linear
    *output_scalar = out1[0];
} */
/* Convert a single value */
/* Convert single Q1.4.27 (int32_t) -> Q1.2.13 (int16_t)
   Round-to-nearest (ties away from zero), saturate.
*/
static inline int16_t q14_27_to_q2_13_safe(int32_t src_q14_27)
{
    const int SHIFT = 27 - 13;               // 14
    const int64_t ROUND = 1LL << (SHIFT - 1); // 1 << 13

    /* widen to avoid overflow & to preserve sign when shifting */
    int64_t tmp = (int64_t) src_q14_27;

    /* sign-aware rounding: add for positive, subtract for negative */
    if (tmp >= 0) tmp += ROUND;
    else          tmp -= ROUND;

    /* arithmetic right shift on signed 64-bit (preserves sign) */
    tmp = tmp >> SHIFT;

    /* saturate to signed 16-bit range (Q1.2.13 fits in int16_t) */
    const int64_t TGT_MIN = (int64_t)INT16_MIN; /* -32768 */
    const int64_t TGT_MAX = (int64_t)INT16_MAX; /* +32767 */

    if (tmp < TGT_MIN) tmp = TGT_MIN;
    if (tmp > TGT_MAX) tmp = TGT_MAX;

    return (int16_t) tmp;
}

void DnnForwardPass (const float *input, float *output_scalar, int32_t *output)
{
    float layer_1[NEURONS_LAYER_1];
    float layer_2[NEURONS_LAYER_2];

    int32_t layer_1_i[NEURONS_LAYER_1];
    int32_t layer_2_i[NEURONS_LAYER_2];
    
    int16_t layer_1_o[NEURONS_LAYER_1];
    int16_t layer_2_o[NEURONS_LAYER_2];

    const int frac_bits_in  = 13;
    const int frac_bits_wt  = 14;

    int16_t input_q[NUM_INPUTS];
    for (uint32_t j = 0; j < NUM_INPUTS; ++j) {
        input_q[j] = (int16_t)float_to_q(input[j], frac_bits_in);
    }

    fcn_layer(input, input_q, &FCN_L1_W[0][0], FCN_L1_b,
              NEURONS_LAYER_1, NUM_INPUTS, layer_1, layer_1_i, /*ReLU*/1);
    
    for (uint32_t j = 0; j < NEURONS_LAYER_1; ++j) {
        layer_1_o[j] = q14_27_to_q2_13_safe(layer_1_i[j]); // adjust Q format for next layer input
    }

    fcn_layer(layer_1, layer_1_o, &FCN_L2_W[0][0], FCN_L2_b,
              NEURONS_LAYER_2, NEURONS_LAYER_1, layer_2, layer_2_i,/*ReLU*/1);

    for (uint32_t j = 0; j < NEURONS_LAYER_2; ++j) {
        layer_2_o[j] = q14_27_to_q2_13_safe(layer_2_i[j]); // adjust Q format for next layer input
    }

    float out1[NUM_OUTPUTS];
    int32_t out1_i[NUM_OUTPUTS];

    fcn_layer(layer_2, layer_2_o, &FCN_OUT_W[0][0], FCN_OUT_b,
              NUM_OUTPUTS, NEURONS_LAYER_2, out1, out1_i, /*ReLU*/0);   // <-- linear
    
    *output_scalar = out1[0];
    *output = out1_i[0];
}

// ----------------------
// Call emitter at the end of main()
// ----------------------
// In main(), after creating predictions.csv and before exiting, add:
//
//    emit_verilog_mem_from_c(\"fcn_weights_mem.v\");
//
// This will create 'fcn_weights_mem.v' in the working directory.
//
// Example: modify the bottom of main() to call it before return 0.
// (If you want to always emit only for debug builds, guard with a compile-time define.)
//

// Example usage
int main() 
{
    float output;
    int32_t output_i;
    float row[FEATURE_COUNT];
    float input[NUM_INPUTS];
    FILE *fp = fopen("predictions.csv", "w");
    if (!fp) { perror("predictions.csv"); return 1; }

    fpwb = fopen("fixed_point_weights_biases.csv", "w");
    if (!fpwb) { perror("fixed_point_weights_biases.csv"); return 1; }

    float acc_error = 0.0f;
    float error = 0.0f;

    for (uint32_t i = 0; i < ROW_COUNT; i++) 
    {
        //uint32_t i = 0;
        output = 0.0;
        output_i = 0;
        //memcpy(&input, &test_inputs[i], sizeof(test_inputs[i]));
        memcpy(&row, &NORMALIZED_DATA[i], FEATURE_COUNT * sizeof(float));
        //printf("Input: Voltage: %f, Current: %f, Temp: %f, V_Avg: %f, I_Avg: %f\n", 
        //       input[0], input[1], input[2], input[3], input[4]);
        //printf("Input: Voltage: %f, Current: %f, Temp: %f, Capacity: %f, Cum Capacity: %f\n", 
        //       row[2], row[3], row[4], row[5], row[7]);
        for (int j = 0; j < NUM_INPUTS - 1; j++) {
            input[j] = row[j + 2];   // take features starting from index 2
        }
        input[4] = row[7]; // Cum Capacity
        //CnnForwardPass(input, &FCN, &output);
        DnnForwardPass(input, &output, &output_i);
        //printf("Output: %f\n", output);
        //printf("Output (fixed Q1.3.27): %d -> %f\n", output_i, q_to_float(output_i, 27));
        error = fabsf(output - q_to_float(output_i, 27));
        acc_error += error;
        // write one CSV row
        fprintf(fp, "%.6f, %.6f, %.6f\n", output, q_to_float(output_i, 27), error);
    }

    printf("Average absolute error: %.6f\n", acc_error / ROW_COUNT);

    export_weights_biases();
    emit_verilog_mem_from_c("fcn_weights_mem.v");
    emit_mem_files_and_wrapper("fcn");

    fclose(fp);
    printf("Max input: %f, Min input: %f\n", max_input, min_input);
    printf("Max weight: %f, Min weight: %f\n", max_weight, min_weight);
    printf("Max bias: %f, Min bias: %f\n", max_bias, min_bias);
    printf("Max acc: %.6f, Min acc: %.6f\n", q_to_float(max_acc, 27), q_to_float(min_acc, 27));
    
    fclose(fpwb);
    return 0;
}

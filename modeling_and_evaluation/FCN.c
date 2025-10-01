#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "test_data.h"
#include "soc_fcn_weights.h"
#include "scaler_with_rows.h"

#define NUM_INPUTS          5
#define FEATURE_SIZE        10
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
static inline float relu(float x) { return x > 0.f ? x : 0.f; }

void fcn_layer(const float *input,
               const float *weights,   // row-major: [num_neurons][input_size]
               const float *biases,
               uint32_t num_neurons,
               uint32_t input_size,
               float *output,
               int apply_relu)
{
    for (uint32_t i = 0; i < num_neurons; i++) 
    {
        const float *wrow = &weights[i * input_size];   // <-- correct stride
        float acc = biases[i];
        for (uint32_t j = 0; j < input_size; j++) 
        {
            acc += input[j] * wrow[j];
        }
        output[i] = apply_relu ? relu(acc) : acc;
        //printf(" Neuron %2d: Weighted sum: %f, Output: %f\n", i, acc, output[i]);
    }
}

void CnnForwardPass (const float *input, const FeedForwardNN *nn, float *output_scalar)
{
    float layer_1[NEURONS_LAYER_1];
    float layer_2[NEURONS_LAYER_2];

    fcn_layer(input,  &nn->weights_layer_1[0][0], nn->biases_layer_1,
              NEURONS_LAYER_1, NUM_INPUTS, layer_1, /*ReLU*/1);

    fcn_layer(layer_1, &nn->weights_layer_2[0][0], nn->biases_layer_2,
              NEURONS_LAYER_2, NEURONS_LAYER_1, layer_2, /*ReLU*/1);

    float out1[NUM_OUTPUTS];
    fcn_layer(layer_2, &nn->weights_output[0][0], nn->biases_output,
              NUM_OUTPUTS, NEURONS_LAYER_2, out1, /*ReLU*/0);   // <-- linear
    *output_scalar = out1[0];
}

void DnnForwardPass (const float *input, float *output_scalar)
{
    float layer_1[NEURONS_LAYER_1];
    float layer_2[NEURONS_LAYER_2];

    fcn_layer(input,  &FCN_L1_W[0][0], FCN_L1_b,
              NEURONS_LAYER_1, NUM_INPUTS, layer_1, /*ReLU*/1);

    fcn_layer(layer_1, &FCN_L2_W[0][0], FCN_L2_b,
              NEURONS_LAYER_2, NEURONS_LAYER_1, layer_2, /*ReLU*/1);

    float out1[NUM_OUTPUTS];
    fcn_layer(layer_2, &FCN_OUT_W[0][0], FCN_OUT_b,
              NUM_OUTPUTS, NEURONS_LAYER_2, out1, /*ReLU*/0);   // <-- linear
    *output_scalar = out1[0];
}

// Example usage
int main() 
{
    float output;
    float row[FEATURE_SIZE];
    float input[NUM_INPUTS];
    FILE *fp = fopen("predictions.csv", "w");
    if (!fp) { perror("predictions.csv"); return 1; }

    for (uint32_t i = 0; i < ROW_COUNT; i++) 
    {
        //uint32_t i = 0;
        output = 0.0;
        //memcpy(&input, &test_inputs[i], sizeof(test_inputs[i]));
        memcpy(&row, &NORMALIZED_DATA[i], FEATURE_SIZE * sizeof(float));
        //printf("Input: Voltage: %f, Current: %f, Temp: %f, V_Avg: %f, I_Avg: %f\n", 
        //       input[0], input[1], input[2], input[3], input[4]);
        //printf("Input: Voltage: %f, Current: %f, Temp: %f, Capacity: %f, Cum Capacity: %f\n", 
        //       row[2], row[3], row[4], row[5], row[7]);
        for (int j = 0; j < NUM_INPUTS - 1; j++) {
            input[j] = row[j + 2];   // take features starting from index 2
        }
        input[4] = row[7]; // Cum Capacity
        //CnnForwardPass(input, &FCN, &output);
        DnnForwardPass(input, &output);
        printf("Output: %f\n", output);
        // write one CSV row
        fprintf(fp, "%.6f\n", output);
    }
    return 0;
}

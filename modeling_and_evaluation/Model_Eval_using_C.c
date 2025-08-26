#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 10000
#define NUM_FEATURES 5

// Dummy function to represent model inference
float model_inference(float features[NUM_FEATURES]) {
    // Replace with actual model code or call to a C deep learning library
    // For demonstration, just sum the features
    float result = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        result += features[i];
        printf("Feature %d: %f\n", i, features[i]);
    }
    return result;
}

int main() 
{
    FILE *fp = fopen("../dataset/LG_HG2_processed/25degC/549_HPPC_processed.csv", "r");
    if (!fp) {
        printf("Failed to open file.\n");
        return 1;
    }

    char line[1024];
    int row = 0;
    float features[MAX_ROWS][NUM_FEATURES];

    // Skip header
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp) && row < MAX_ROWS) 
    {
        // Columns: Timestamp,Time [min],Time [s],Voltage [V],Current [A],Temperature [degC],Capacity [Ah],Time_diff,Cumulative_Capacity_Ah,SOC [-],Rounded_Time
        char timestamp[32];
        float time_min, time_s, voltage, current, temperature, capacity_ah, time_diff, cumulative_capacity_ah, soc, rounded_time;
        sscanf(line, "%31[^,],%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
        timestamp, &time_min, &time_s, &voltage, &current, &temperature, &capacity_ah, &time_diff, &cumulative_capacity_ah, &soc, &rounded_time);

        // Calculate power
        float power = voltage * current;

        // Assign features
        features[row][0] = voltage;
        features[row][1] = current;
        features[row][2] = temperature;
        features[row][3] = power;
        features[row][4] = cumulative_capacity_ah;

        row++;
    }

    fclose(fp);

    // Feed each row to the model
    for (int i = 0; i < row; i++) {
        float soc = model_inference(features[i]);
        //printf("Row %d: Predicted SOC = %f\n", i, soc);
    }

    return 0;
}
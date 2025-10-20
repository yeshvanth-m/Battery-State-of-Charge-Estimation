`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 20.10.2025 09:32:01
// Design Name: 
// Module Name: mac
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module mac
#(
    parameter NUM_INPUTS = 5,
    parameter NEURONS_L1 = 128,
    parameter NEURONS_L2 = 64,
    parameter NUM_OUTPUTS = 1,
    parameter WEIGHT_WIDTH = 16,
    parameter BIAS_WIDTH = 32,
    parameter BATCH_SIZE = 8
)
(
    input logic clk,
    input logic rst,
    input logic start,
    output logic done
);

    // Depths
    localparam L1_W_DEPTH = 640;
    localparam L2_W_DEPTH = 8192;
    localparam OUT_W_DEPTH = 64;
    localparam WEIGHT_DEPTH = (640 + 8192 + 64) / 8;
    localparam BIAS_DEPTH = (NEURONS_L1 + NEURONS_L2 + 8) / 8;
    
    // memories
    logic signed [WEIGHT_WIDTH-1:0] mem_w [0:WEIGHT_DEPTH-1][0:BATCH_SIZE-1];
    logic signed [BIAS_WIDTH-1:0] mem_b [0:BIAS_DEPTH-1][0:BATCH_SIZE-1];
    logic signed [BIAS_WIDTH-1:0] acc [0:BATCH_SIZE-1]; // 8 accumalators 

    initial begin
        // load weights/biases from mem files (hex, two's complement)
        $readmemh("weights.mem", mem_w);
        $readmemh("biases.mem", mem_b);
    end
    
    // -----------------------------------------
    // FSM definition
    // -----------------------------------------
    typedef enum logic [2:0] {
        S_IDLE          = 3'd0,
        S_INIT_BIAS_L1  = 3'd1,
        S_COMPUTE_L1    = 3'd2,
        S_INIT_BIAS_L2  = 3'd3,
        S_COMPUTE_L2    = 3'd4,
        S_OUTPUT_LAYER  = 3'd5
    } state_t;
    
    state_t state;
    var logic [$clog2(BIAS_DEPTH)-1:0]   bias_addr;
    var logic [$clog2(WEIGHT_DEPTH)-1:0] weight_addr;
    
    always @(posedge clk or posedge rst)
    begin
        if (rst)
        begin
            state <= S_IDLE;
        end
        else
        begin
            case (state)
                S_IDLE:
                begin
                    if (start)
                    begin 
                        state <= S_INIT_BIAS_L1;
                        bias_addr <= 0;
                    end
                end
                S_INIT_BIAS_L1:
                begin
                    //for (logic [2:0] i = 0; i < 8; i++) 
                    //begin
                        
                    //end
                    //state <= S_COMPUTE_L1;
                end
                S_COMPUTE_L1:
                begin
                
                end
            endcase
        end
    end

endmodule

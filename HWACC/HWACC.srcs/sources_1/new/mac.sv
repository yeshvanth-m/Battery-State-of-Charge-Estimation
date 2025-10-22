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
    parameter IN_WIDTH   = 16,
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
    localparam L1_W_DEPTH = 640 / 8;
    localparam L2_W_DEPTH = 8192 / 8;
    localparam OUT_W_DEPTH = 64 / 8;
    localparam WEIGHT_DEPTH = (640 + 8192 + 64) / 8;
    localparam BIAS_DEPTH = (NEURONS_L1 + NEURONS_L2 + 8) / 8;
    
    // memories
    logic signed [0:BATCH_SIZE-1][WEIGHT_WIDTH-1:0] mem_w [0:WEIGHT_DEPTH-1];
    logic signed [0:BATCH_SIZE-1][BIAS_WIDTH-1:0]   mem_b [0:BIAS_DEPTH-1];
    logic signed [BIAS_WIDTH-1:0]   acc_ff   [0:BATCH_SIZE-1]; // 8 accumalators
    logic signed [BIAS_WIDTH-1:0]   acc_comb [0:BATCH_SIZE-1]; // 8 accumalators
    logic signed [IN_WIDTH-1:0]     mac_op_1 [0:BATCH_SIZE-1]; // 8 accumalators
    logic signed [WEIGHT_WIDTH-1:0] mac_op_2 [0:BATCH_SIZE-1]; // 8 accumalators
    logic signed [IN_WIDTH-1:0]     out_buf_l1  [0:NEURONS_L1-1];
    logic signed [IN_WIDTH-1:0]     out_buf_l2  [0:NEURONS_L2-1];

    logic signed [0:NUM_INPUTS-1][15:0] in;
    initial begin
        // load weights/biases from mem files (hex, two's complement)
        $readmemh("weights.mem", mem_w);
        $readmemh("biases.mem", mem_b);
        
        in[0] = 16'h2d62;
        in[1] = 16'hffcb;
        in[2] = 16'hc8fc; 
        in[3] = 16'h2f2c; 
        in[4] = 16'h2f32;
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
    var logic [$clog2(NEURONS_L1):0] in_addr;
    var logic [$clog2(NEURONS_L1)-1:0] out_buf_addr;
    
    // 8 MACs, combinatorialy defined
    always_comb
    begin
        foreach (acc_comb[i])
        begin
            acc_comb[i] = acc_ff[i] + (mac_op_1[i] * mac_op_2[i]);
        end
    end
    
    always_ff @(posedge clk or posedge rst)
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
                        state       <= S_INIT_BIAS_L1;
                        bias_addr   <= 0;
                        weight_addr <= 0;
                        in_addr     <= 0;
                        out_buf_addr <= 0;
                    end
                end
                S_INIT_BIAS_L1:
                begin                 
                    state <= S_COMPUTE_L1;
                    foreach (acc_ff[i])
                    begin 
                        mac_op_1[i] <= in[in_addr];
                        mac_op_2[i] <= mem_w[weight_addr][i];
                        acc_ff[i]   <= mem_b[bias_addr][i];
                    end
                    in_addr++;
                    weight_addr++;
                    bias_addr++;
                end
                S_COMPUTE_L1:
                begin
                    if (in_addr < NUM_INPUTS)
                    begin
                        foreach (mac_op_1[i])
                        begin
                            mac_op_1[i] <= in[in_addr];
                            mac_op_2[i] <= mem_w[weight_addr][i];
                            acc_ff[i]   <= acc_comb[i];
                        end
                        in_addr++;
                        weight_addr++;
                    end
                    else
                    begin
                        in_addr <= 0;
                        if (weight_addr < L1_W_DEPTH) 
                            state <= S_INIT_BIAS_L1;
                        else 
                            state <= S_INIT_BIAS_L2;
                        foreach (acc_comb[i])
                        begin
                            out_buf_l1[out_buf_addr + i] <= (acc_comb[i] > 0) ? ((acc_comb[i] + (1 << 13)) >> 14) : 0;
                        end
                        out_buf_addr <= out_buf_addr + 8;
                    end 
                end
                S_INIT_BIAS_L2:
                begin
                    state <= S_COMPUTE_L2;
                    foreach (acc_ff[i])
                    begin 
                        mac_op_1[i] <= out_buf_l1[in_addr];
                        mac_op_2[i] <= mem_w[weight_addr][i];
                        acc_ff[i]   <= mem_b[bias_addr][i];
                    end
                    in_addr++;
                    weight_addr++;
                    bias_addr++;
                end
                S_COMPUTE_L2:
                begin
                    if (in_addr < NEURONS_L1)
                    begin
                        foreach (mac_op_1[i])
                        begin
                            mac_op_1[i] <= out_buf_l1[in_addr];
                            mac_op_2[i] <= mem_w[weight_addr][i];
                            acc_ff[i]   <= acc_comb[i];
                        end
                        in_addr++;
                        weight_addr++;
                    end
                    else
                    begin
                        in_addr <= 0;
                        if (weight_addr < L2_W_DEPTH) 
                            state <= S_INIT_BIAS_L2;
                        else 
                            state <= S_OUTPUT_LAYER;
                        foreach (acc_comb[i])
                        begin
                            out_buf_l2[out_buf_addr + i] <= (acc_comb[i] > 0) ? ((acc_comb[i] + (1 << 13)) >> 14) : 0;
                            //acc_ff[i] <= 0;
                        end
                        out_buf_addr <= out_buf_addr + 8;
                    end 
                end
                S_OUTPUT_LAYER:
                begin
                    if (in_addr < NEURONS_L2)
                    begin
                        foreach (mac_op_1[i])
                        begin
                            mac_op_1[i] <= out_buf_l2[in_addr + i];
                            mac_op_2[i] <= mem_w[weight_addr][i];
                            acc_ff[i] <= in_addr > 0 ? acc_comb[i] : 0;
                        end
                        in_addr <= in_addr + 8;
                        weight_addr++;
                        out_buf_addr <= 1;
                    end
                    else
                    begin
                        if (out_buf_addr < BATCH_SIZE)
                        begin
                            if (weight_addr == 1112)
                            begin
                                foreach (acc_ff[i])
                                    acc_ff[i] <= acc_comb[i];
                                    weight_addr <= 0;
                            end
                            else
                            begin
                                acc_ff[0] <= acc_ff[0] + acc_ff[out_buf_addr];
                                out_buf_addr++;
                            end
                        end
                        else
                        begin
                            acc_ff[0] <= acc_ff[0] + mem_b[bias_addr][0];
                            state <= S_IDLE;
                            done <= 1;
                        end
                    end
                end
            endcase
        end
    end

endmodule

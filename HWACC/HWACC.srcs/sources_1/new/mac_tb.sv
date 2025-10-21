`timescale 1ns / 1ps

module mac_tb;

    // Clock and reset
    reg clk = 0;
    reg rst = 0;
    reg start = 0;
    wire done;

    // Instantiate the DUT
    mac dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .done(done)
    );

    // Generate clock
    always #5 clk = ~clk;   // 100 MHz clock
    
    real weight_value, bias_value;
    
      // convert signed 16-bit fixed (two's complement) to real given frac bits
    function real fixed16_to_real(input logic [15:0] bits, input int frac);
      begin
        fixed16_to_real = $signed(bits) / (2.0 ** frac);
      end
    endfunction
  
    // convert signed 32-bit fixed to real
    function real fixed32_to_real(input logic [31:0] bits, input int frac);
      begin
        fixed32_to_real = $signed(bits) / (2.0 ** frac);
      end
    endfunction

    initial begin
        $display("----- MAC Testbench Start -----");

        // Reset pulse
        rst = 1;
        #20;
        rst = 0;

        // Start signal
        #10;
        start = 1;
        #10;
        start = 0;

        // Wait a little for initialization
        #50;

        // Display a few memory contents
        $display("\n=== Weight Memory Preview ===");
        for (int j = 0; j < 1112; j++) begin
            for (int i = 0; i < 8; i++) begin
                weight_value = fixed16_to_real(dut.mem_w[j][i], 14);
                $display("mem_w[%0d] = %f", i, weight_value);
            end
        end

        $display("\n=== Bias Memory Preview ===");
        for (int i = 0; i < 4; i++) begin
            $display("mem_b[%0d] = %h", i, dut.mem_b[i][0]);
        end

        $display("\n----- MAC Testbench End -----");
        
        #100;
        $finish;
    end

endmodule

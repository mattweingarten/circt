// Top-level module
module top_module (
    input logic clk,
    input logic rst,
    input logic [3:0] in_data,
    output logic [3:0] out_data
);
    
    logic [3:0] interconnect1, interconnect2;
    logic [7:0] interconnect3, interconnect4;
    logic valid, ready;
    
    module1 u1 (
        .clk(clk),
        .rst(rst),
        .data_in(in_data),
        .data_out(interconnect1)
    );
    
    module2 u2 (
        .clk(clk),
        .rst(rst),
        .data_in(interconnect1),
        .data_out(interconnect2)
    );
    
    module3 u3 (
        .clk(clk),
        .reset(rst),
        .valid(valid),
        .in_data({4'b0, interconnect2}),
        .out_data(interconnect3)
    );
    
    module4 u4 (
        .clk(clk),
        .reset(rst),
        .valid(valid),
        .ready(ready),
        .in_data(interconnect3),
        .out_data(interconnect4),
        .done()
    );
    
    assign out_data = interconnect4[3:0];
endmodule


// Module 1
module module1 (
    input logic clk,
    input logic rst,
    input logic [3:0] data_in,
    output logic [3:0] data_out
);
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            data_out <= 4'b0;
        else
            data_out <= (data_in + 1) ^ 4'b0011;
    end
endmodule

// Module 2
module module2 (
    input logic clk,
    input logic rst,
    input logic [3:0] data_in,
    output logic [3:0] data_out
);
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            data_out <= 4'b0;
        else
            data_out <= (data_in << 1) | 4'b0001;
    end
endmodule


module module3 (
    input logic clk,
    input logic reset,
    input logic valid,
    input logic [7:0] in_data,
    output logic [7:0] out_data
);
    always_ff @(posedge clk or posedge reset) begin
        if (reset)
            out_data <= 8'b0;
        else if (valid)  // Only update when valid is high
            out_data <= in_data;
    end
endmodule

module module4 (
    input logic clk,
    input logic reset,
    input logic valid,
    input logic ready,
    input logic [7:0] in_data,
    output logic [7:0] out_data,
    output logic done
);
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            out_data <= 8'b0;
            done <= 1'b0;
        end else if (valid && ready) begin  // Process data only when valid and ready are high
            out_data <= in_data;
            done <= 1'b1;
        end else begin
            done <= 1'b0;
        end
    end
endmodule


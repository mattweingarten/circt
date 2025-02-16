// Top-level module
module top_module (
    input logic clk,
    input logic rst,
    input logic [3:0] in_data,
    output logic [3:0] out_data
);
    
    logic [3:0] interconnect1, interconnect2;
    
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
        .rst(rst),
        .data_in(interconnect2),
        .data_out(out_data)
    );
    
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

// Module 3
module module3 (
    input logic clk,
    input logic rst,
    input logic [3:0] data_in,
    output logic [3:0] data_out
);
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            data_out <= 4'b0;
        else
            data_out <= (data_in ^ 4'b1010) + 4'b0011;
    end
endmodule

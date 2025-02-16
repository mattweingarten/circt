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

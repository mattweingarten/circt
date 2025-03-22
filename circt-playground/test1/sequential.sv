module simple_sequential (
    input logic in_data,
    output logic out_data
);

    logic intermediate_value;

    always_comb begin
        intermediate_value = in_data + 1;
        out_data = intermediate_value; 
    end

endmodule

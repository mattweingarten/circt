module {
  firrtl.circuit "TopModule" attributes {rawAnnotations = [{class = "firrtl.transforms.DedupGroupAnnotation", group = "Module1", target = "~TopModule|Module1"}, {class = "firrtl.transforms.DedupGroupAnnotation", group = "Module2", target = "~TopModule|Module2"}, {class = "firrtl.transforms.DedupGroupAnnotation", group = "Module3", target = "~TopModule|Module3"}, {class = "firrtl.transforms.DedupGroupAnnotation", group = "TopModule", target = "~TopModule|TopModule"}]} {
    firrtl.module private @Module1(in %clock: !firrtl.clock, in %reset: !firrtl.reset, out %io: !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>) {
      %0 = firrtl.subfield %io[data_out] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %1 = firrtl.subfield %io[data_in] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %2 = firrtl.subfield %io[rst] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %c0_ui4 = firrtl.constant 0 : !firrtl.const.uint<4>
      %reg = firrtl.regreset interesting_name %clock, %reset, %c0_ui4 : !firrtl.clock, !firrtl.reset, !firrtl.const.uint<4>, !firrtl.uint<4>
      %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
      %c1_ui1 = firrtl.constant 1 : !firrtl.const.uint<1>
      %c3_ui4 = firrtl.constant 3 : !firrtl.const.uint<4>
      firrtl.when %2 : !firrtl.uint<1> {
        %3 = firrtl.pad %c0_ui1, 4 : (!firrtl.const.uint<1>) -> !firrtl.const.uint<4>
        %4 = firrtl.constCast %3 : (!firrtl.const.uint<4>) -> !firrtl.uint<4>
        firrtl.matchingconnect %reg, %4 : !firrtl.uint<4>
      } else {
        %3 = firrtl.add %1, %c1_ui1 : (!firrtl.uint<4>, !firrtl.const.uint<1>) -> !firrtl.uint<5>
        %_reg_T = firrtl.node interesting_name %3 : !firrtl.uint<5>
        %4 = firrtl.tail %_reg_T, 1 : (!firrtl.uint<5>) -> !firrtl.uint<4>
        %_reg_T_1 = firrtl.node interesting_name %4 : !firrtl.uint<4>
        %5 = firrtl.xor %_reg_T_1, %c3_ui4 : (!firrtl.uint<4>, !firrtl.const.uint<4>) -> !firrtl.uint<4>
        %_reg_T_2 = firrtl.node interesting_name %5 : !firrtl.uint<4>
        firrtl.matchingconnect %reg, %_reg_T_2 : !firrtl.uint<4>
      }
      firrtl.matchingconnect %0, %reg : !firrtl.uint<4>
    }
    firrtl.module private @Module2(in %clock: !firrtl.clock, in %reset: !firrtl.reset, out %io: !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>) {
      %0 = firrtl.subfield %io[data_out] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %1 = firrtl.subfield %io[data_in] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %2 = firrtl.subfield %io[rst] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %c0_ui4 = firrtl.constant 0 : !firrtl.const.uint<4>
      %reg = firrtl.regreset interesting_name %clock, %reset, %c0_ui4 : !firrtl.clock, !firrtl.reset, !firrtl.const.uint<4>, !firrtl.uint<4>
      %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
      %c1_ui4 = firrtl.constant 1 : !firrtl.const.uint<4>
      firrtl.when %2 : !firrtl.uint<1> {
        %3 = firrtl.pad %c0_ui1, 4 : (!firrtl.const.uint<1>) -> !firrtl.const.uint<4>
        %4 = firrtl.constCast %3 : (!firrtl.const.uint<4>) -> !firrtl.uint<4>
        firrtl.matchingconnect %reg, %4 : !firrtl.uint<4>
      } else {
        %3 = firrtl.shl %1, 1 : (!firrtl.uint<4>) -> !firrtl.uint<5>
        %_reg_T = firrtl.node interesting_name %3 : !firrtl.uint<5>
        %4 = firrtl.or %_reg_T, %c1_ui4 : (!firrtl.uint<5>, !firrtl.const.uint<4>) -> !firrtl.uint<5>
        %_reg_T_1 = firrtl.node interesting_name %4 : !firrtl.uint<5>
        %5 = firrtl.tail %_reg_T_1, 1 : (!firrtl.uint<5>) -> !firrtl.uint<4>
        firrtl.matchingconnect %reg, %5 : !firrtl.uint<4>
      }
      firrtl.matchingconnect %0, %reg : !firrtl.uint<4>
    }
    firrtl.module private @Module3(in %clock: !firrtl.clock, in %reset: !firrtl.reset, out %io: !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>) {
      %0 = firrtl.subfield %io[data_out] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %1 = firrtl.subfield %io[data_in] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %2 = firrtl.subfield %io[rst] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %c0_ui4 = firrtl.constant 0 : !firrtl.const.uint<4>
      %reg = firrtl.regreset interesting_name %clock, %reset, %c0_ui4 : !firrtl.clock, !firrtl.reset, !firrtl.const.uint<4>, !firrtl.uint<4>
      %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
      %c10_ui4 = firrtl.constant 10 : !firrtl.const.uint<4>
      %c3_ui4 = firrtl.constant 3 : !firrtl.const.uint<4>
      firrtl.when %2 : !firrtl.uint<1> {
        %3 = firrtl.pad %c0_ui1, 4 : (!firrtl.const.uint<1>) -> !firrtl.const.uint<4>
        %4 = firrtl.constCast %3 : (!firrtl.const.uint<4>) -> !firrtl.uint<4>
        firrtl.matchingconnect %reg, %4 : !firrtl.uint<4>
      } else {
        %3 = firrtl.xor %1, %c10_ui4 : (!firrtl.uint<4>, !firrtl.const.uint<4>) -> !firrtl.uint<4>
        %_reg_T = firrtl.node interesting_name %3 : !firrtl.uint<4>
        %4 = firrtl.add %_reg_T, %c3_ui4 : (!firrtl.uint<4>, !firrtl.const.uint<4>) -> !firrtl.uint<5>
        %_reg_T_1 = firrtl.node interesting_name %4 : !firrtl.uint<5>
        %5 = firrtl.tail %_reg_T_1, 1 : (!firrtl.uint<5>) -> !firrtl.uint<4>
        %_reg_T_2 = firrtl.node interesting_name %5 : !firrtl.uint<4>
        firrtl.matchingconnect %reg, %_reg_T_2 : !firrtl.uint<4>
      }
      firrtl.matchingconnect %0, %reg : !firrtl.uint<4>
    }
    firrtl.module @TopModule(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io: !firrtl.bundle<clk flip: clock, rst flip: uint<1>, in_data flip: uint<4>, out_data: uint<4>>) {
      %0 = firrtl.subfield %io[out_data] : !firrtl.bundle<clk flip: clock, rst flip: uint<1>, in_data flip: uint<4>, out_data: uint<4>>
      %1 = firrtl.subfield %io[clk] : !firrtl.bundle<clk flip: clock, rst flip: uint<1>, in_data flip: uint<4>, out_data: uint<4>>
      %2 = firrtl.subfield %io[rst] : !firrtl.bundle<clk flip: clock, rst flip: uint<1>, in_data flip: uint<4>, out_data: uint<4>>
      %3 = firrtl.subfield %io[in_data] : !firrtl.bundle<clk flip: clock, rst flip: uint<1>, in_data flip: uint<4>, out_data: uint<4>>
      %u1_clock, %u1_reset, %u1_io = firrtl.instance u1 interesting_name @Module1(in clock: !firrtl.clock, in reset: !firrtl.reset, out io: !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>)
      %4 = firrtl.subfield %u1_io[data_out] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %5 = firrtl.subfield %u1_io[clk] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %6 = firrtl.subfield %u1_io[rst] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %7 = firrtl.subfield %u1_io[data_in] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      firrtl.matchingconnect %u1_clock, %clock : !firrtl.clock
      %8 = firrtl.resetCast %reset : (!firrtl.uint<1>) -> !firrtl.reset
      firrtl.matchingconnect %u1_reset, %8 : !firrtl.reset
      %u2_clock, %u2_reset, %u2_io = firrtl.instance u2 interesting_name @Module2(in clock: !firrtl.clock, in reset: !firrtl.reset, out io: !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>)
      %9 = firrtl.subfield %u2_io[data_out] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %10 = firrtl.subfield %u2_io[clk] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %11 = firrtl.subfield %u2_io[rst] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %12 = firrtl.subfield %u2_io[data_in] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      firrtl.matchingconnect %u2_clock, %clock : !firrtl.clock
      %13 = firrtl.resetCast %reset : (!firrtl.uint<1>) -> !firrtl.reset
      firrtl.matchingconnect %u2_reset, %13 : !firrtl.reset
      %u3_clock, %u3_reset, %u3_io = firrtl.instance u3 interesting_name @Module3(in clock: !firrtl.clock, in reset: !firrtl.reset, out io: !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>)
      %14 = firrtl.subfield %u3_io[data_out] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %15 = firrtl.subfield %u3_io[clk] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %16 = firrtl.subfield %u3_io[rst] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      %17 = firrtl.subfield %u3_io[data_in] : !firrtl.bundle<data_in flip: uint<4>, data_out: uint<4>, rst flip: uint<1>, clk flip: clock>
      firrtl.matchingconnect %u3_clock, %clock : !firrtl.clock
      %18 = firrtl.resetCast %reset : (!firrtl.uint<1>) -> !firrtl.reset
      firrtl.matchingconnect %u3_reset, %18 : !firrtl.reset
      firrtl.matchingconnect %7, %3 : !firrtl.uint<4>
      firrtl.matchingconnect %6, %2 : !firrtl.uint<1>
      firrtl.matchingconnect %5, %1 : !firrtl.clock
      firrtl.matchingconnect %12, %4 : !firrtl.uint<4>
      firrtl.matchingconnect %11, %2 : !firrtl.uint<1>
      firrtl.matchingconnect %10, %1 : !firrtl.clock
      firrtl.matchingconnect %17, %9 : !firrtl.uint<4>
      firrtl.matchingconnect %16, %2 : !firrtl.uint<1>
      firrtl.matchingconnect %15, %1 : !firrtl.clock
      firrtl.matchingconnect %0, %14 : !firrtl.uint<4>
    }
  }
}

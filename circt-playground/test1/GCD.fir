module {
  firrtl.circuit "GCD" attributes {rawAnnotations = [{class = "firrtl.transforms.DedupGroupAnnotation", group = "GCD", target = "~GCD|GCD"}]} {
    firrtl.module @GCD(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io: !firrtl.bundle<value1 flip: uint<16>, value2 flip: uint<16>, loadingValues flip: uint<1>, outputGCD: uint<16>, outputValid: uint<1>>) {
      %0 = firrtl.subfield %io[outputValid] : !firrtl.bundle<value1 flip: uint<16>, value2 flip: uint<16>, loadingValues flip: uint<1>, outputGCD: uint<16>, outputValid: uint<1>>
      %1 = firrtl.subfield %io[outputGCD] : !firrtl.bundle<value1 flip: uint<16>, value2 flip: uint<16>, loadingValues flip: uint<1>, outputGCD: uint<16>, outputValid: uint<1>>
      %2 = firrtl.subfield %io[value2] : !firrtl.bundle<value1 flip: uint<16>, value2 flip: uint<16>, loadingValues flip: uint<1>, outputGCD: uint<16>, outputValid: uint<1>>
      %3 = firrtl.subfield %io[value1] : !firrtl.bundle<value1 flip: uint<16>, value2 flip: uint<16>, loadingValues flip: uint<1>, outputGCD: uint<16>, outputValid: uint<1>>
      %4 = firrtl.subfield %io[loadingValues] : !firrtl.bundle<value1 flip: uint<16>, value2 flip: uint<16>, loadingValues flip: uint<1>, outputGCD: uint<16>, outputValid: uint<1>>
      %x = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint
      %y = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint
      %5 = firrtl.gt %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
      %_T = firrtl.node interesting_name %5 : !firrtl.uint<1>
      firrtl.when %_T : !firrtl.uint<1> {
        %7 = firrtl.sub %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
        %_x_T = firrtl.node interesting_name %7 : !firrtl.uint
        %8 = firrtl.tail %_x_T, 1 : (!firrtl.uint) -> !firrtl.uint
        %_x_T_1 = firrtl.node interesting_name %8 : !firrtl.uint
        firrtl.connect %x, %_x_T_1 : !firrtl.uint
      } else {
        %7 = firrtl.sub %y, %x : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
        %_y_T = firrtl.node interesting_name %7 : !firrtl.uint
        %8 = firrtl.tail %_y_T, 1 : (!firrtl.uint) -> !firrtl.uint
        %_y_T_1 = firrtl.node interesting_name %8 : !firrtl.uint
        firrtl.connect %y, %_y_T_1 : !firrtl.uint
      }
      firrtl.when %4 : !firrtl.uint<1> {
        firrtl.connect %x, %3 : !firrtl.uint, !firrtl.uint<16>
        firrtl.connect %y, %2 : !firrtl.uint, !firrtl.uint<16>
      }
      firrtl.connect %1, %x : !firrtl.uint<16>, !firrtl.uint
      %c0_ui1 = firrtl.constant 0 : !firrtl.const.uint<1>
      %6 = firrtl.eq %y, %c0_ui1 : (!firrtl.uint, !firrtl.const.uint<1>) -> !firrtl.uint<1>
      %_io_outputValid_T = firrtl.node interesting_name %6 : !firrtl.uint<1>
      firrtl.matchingconnect %0, %_io_outputValid_T : !firrtl.uint<1>
    }
  }
}

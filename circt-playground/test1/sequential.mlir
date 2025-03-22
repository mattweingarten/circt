module {
  hw.module @simple_sequential(in %in_data : i1, out out_data : i1) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i31 = hw.constant 0 : i31
    %c1_i32 = hw.constant 1 : i32
    %false = hw.constant false
    %in_data_0 = llhd.sig name "in_data" %false : i1
    %out_data = llhd.sig %false : i1
    %intermediate_value = llhd.sig %false : i1
    %1 = llhd.prb %in_data_0 : !hw.inout<i1>
    %2 = llhd.prb %intermediate_value : !hw.inout<i1>
    %3 = llhd.prb %out_data : !hw.inout<i1>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      %5 = llhd.prb %in_data_0 : !hw.inout<i1>
      %6 = comb.concat %c0_i31, %5 : i31, i1
      %7 = comb.add %6, %c1_i32 : i32
      %8 = comb.extract %7 from 0 : (i32) -> i1
      llhd.drv %intermediate_value, %8 after %0 : !hw.inout<i1>
      %9 = llhd.prb %intermediate_value : !hw.inout<i1>
      llhd.drv %out_data, %9 after %0 : !hw.inout<i1>
      llhd.wait (%1, %2, %3 : i1, i1, i1), ^bb1
    }
    llhd.drv %in_data_0, %in_data after %0 : !hw.inout<i1>
    %4 = llhd.prb %out_data : !hw.inout<i1>
    hw.output %4 : i1
  }
}


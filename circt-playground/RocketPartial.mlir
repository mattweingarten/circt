module {
  firrtl.circuit "IntXbar_i1_o1" {
    firrtl.module public @IntXbar_i1_o1(out %auto: !firrtl.bundle<anon_in flip: vector<uint<1>, 2>, anon_out: vector<uint<1>, 2>>) {
      %0 = firrtl.subfield %auto[anon_in] : !firrtl.bundle<anon_in flip: vector<uint<1>, 2>, anon_out: vector<uint<1>, 2>>
    }
  }
}
//===- EmitAutoCounterAnnotations.cpp
//--------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Support/InstanceGraphInterface.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "perf-emit-autocounter-annotations"

namespace circt {
namespace perf {
#define GEN_PASS_DEF_EMITAUTOCOUNTERANNOTATIONS
#include "circt/Dialect/Perf/PerfPasses.h.inc"
} // namespace perf
} // namespace circt

using namespace circt;
using namespace perf;

namespace {
struct EmitAutoCounterAnnotationsPass
    : public circt::perf::impl::EmitAutoCounterAnnotationsBase<
          EmitAutoCounterAnnotationsPass> {
  using EmitAutoCounterAnnotationsBase::outputFilename;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::perf::PerfDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();
    registry.insert<circt::seq::SeqDialect>();
  }
private:
  void runOnOperation() override;
};


static void startJSON(llvm::raw_ostream &os) { os << "[\n"; }

static void endJSON(llvm::raw_ostream &os) { os << "]\n"; }

struct AutoCounterAnnotation {
  std::string target;
  std::string circuit;
  std::string module;
  std::string clock;
  std::string reset;
  std::string label;
  std::string description;
  bool coverGenerated = false;

  static constexpr const char *AnnotationClass =
      "midas.targetutils.AutoCounterFirrtlAnnotation";

  AutoCounterAnnotation(std::string target, std::string circuit,
                        std::string module, std::string clock,
                        std::string reset, std::string label,
                        std::string description = "",
                        bool coverGenerated = false)
      : target(target), circuit(circuit), module(module), clock(clock),
        reset(reset), label(label), description(description),
        coverGenerated(coverGenerated) {}
  virtual ~AutoCounterAnnotation() = default;

  virtual llvm::StringRef getOpTypeClass() const = 0;

  std::string getModulePref() const { return module + ">"; }

  void printAsJSON(llvm::raw_ostream &os, int annoCount) const {
    if (annoCount > 0)
      os << ",\n";
    os << "{\n";
    os << "  \"class\":\"" << AnnotationClass << "\",\n";
    os << "  \"target\":\"~" << circuit << "|" << getModulePref() << target
       << "\",\n";
    os << "  \"clock\":\"~" << circuit << "|" << getModulePref() << clock
       << "\",\n";
    os << "  \"reset\":\"~" << circuit << "|" << getModulePref() << reset
       << "\",\n";
    os << "  \"label\":\"" << label << "\",\n";
    os << "  \"description\":\""
       << label << "<--(circt autogen)" 
       << (!description.empty() ? " " : "")
       <<  description << "\",\n";
    os << "  \"opType\":{\n";
    os << "    \"class\":\"" << getOpTypeClass() << "\"\n";
    os << "  },\n";
    os << "  \"coverGenerated\":" << (coverGenerated ? "true" : "false")
       << "\n";
    os << "}\n";
  }
};

struct AccumulateCounterAnnotation : public AutoCounterAnnotation {

  AccumulateCounterAnnotation(std::string target, std::string circuit,
                              std::string module, std::string clock,
                              std::string reset, std::string label,
                              std::string description = "",
                              bool coverGenerated = false)
      : AutoCounterAnnotation(target, circuit, module, clock, reset, label,
                              description, coverGenerated) {}

  llvm::StringRef getOpTypeClass() const override {
    return "midas.targetutils.PerfCounterOps$Accumulate$";
  }
  /*  */
  static llvm::Expected<AccumulateCounterAnnotation>
  createAccumuluateCounterAnnotation(perf::PerfCounterOp op) {

    ::circt::firrtl::CircuitOp circuitOp =
        op->getParentOfType<circt::firrtl::CircuitOp>();

    std::string circuitName;
    if (!circuitOp) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[AUTOCOUNTER] PerfCounterOp has no parent firrtl.circuit OP for "
             "emitting AutoCounter annotations. Setting default to FireSim.\n");
      circuitName = "FireSim"; // Default Name
    } else {
      circuitName = circuitOp.getName().str();
    }

    // Get generic Module name.
    ::circt::igraph::ModuleOpInterface parentModule =
        op->getParentOfType<::circt::igraph::ModuleOpInterface>();

    LLVM_DEBUG(llvm::dbgs()
               << "[AUTOCOUNTER] Creating "
                  "AccumulateCounterAnnotation for PerfCounterOp: \n"
               << op << "\n");

    if (!parentModule) {
      return llvm::make_error<llvm::StringError>(
          "PerfCounterOp has no parent module", llvm::errc::invalid_argument);
    }

    std::string parentName = parentModule.getModuleName().str();

    std::string clkName = perf::getSSAName(op.getClk(), parentModule);

    std::string resetName =
        op.getReset() ? perf::getSSAName(op.getReset(), parentModule) : "reset";

    std::string label = op.getName() ? op.getName()->str() : "<unkown>";

    AccumulateCounterAnnotation annotation(
        label,       /*target signal, for now asssume same as label*/
        circuitName, /*circuit*/
        parentName,  /*module*/
        clkName,     // clock
        resetName,   // TODO hardcode reset for now
        label        /*label*/
    );
    return annotation;
  }
};

struct IdentityCounterAnnotation : public AutoCounterAnnotation {
  IdentityCounterAnnotation(std::string target, std::string circuit,
                            std::string module, std::string clock,
                            std::string reset, std::string label,
                            std::string description = "",
                            bool coverGenerated = false)
      : AutoCounterAnnotation(target, circuit, module, clock, reset, label,
                              description, coverGenerated) {}

  llvm::StringRef getOpTypeClass() const override {
    return "midas.targetutils.PerfCounterOps$Identity$";
  }
};

} // namespace

void EmitAutoCounterAnnotationsPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "[AUTOCOUNTER] Running EmitAutoCounterAnnotationsPass\n");

  if (this->outputFilename.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[AUTOCOUNTER] No output filename specified.\n");
    signalPassFailure();
    return;
  }
  std::error_code ec;
  llvm::raw_fd_ostream os(outputFilename, ec);
  if (ec) {
    LLVM_DEBUG(llvm::dbgs()
               << "[AUTOCOUNTER] Error opening file: " << ec.message() << "\n");
    signalPassFailure();
    return;
  }

  startJSON(os);

  int annoCount = 0;
  getOperation().walk([&](perf::PerfCounterOp op) {
    LLVM_DEBUG(llvm::dbgs()
               << "[AUTOCOUNTER] Emitting annotation for PerfCounterOp: "
               << op.getName() << "\n");
    auto annoOrErr =
        AccumulateCounterAnnotation::createAccumuluateCounterAnnotation(op);
    if (!annoOrErr) {
      llvm::errs() << "[AUTOCOUNTER] Error creating annotation: "
                   << llvm::toString(annoOrErr.takeError()) << "\n";
      return;
    }
    AccumulateCounterAnnotation anno = *annoOrErr;
    anno.printAsJSON(os, annoCount);
    annoCount++;
    op.erase();
  });

  endJSON(os);

  return;
}

std::unique_ptr<Pass>
circt::perf::createEmitAutoCounterAnnotationsPass(StringRef outputFilename) {
  auto pass = std::make_unique<EmitAutoCounterAnnotationsPass>();

  if (!outputFilename.empty())
    pass->outputFilename = outputFilename.str();
  return pass;
}

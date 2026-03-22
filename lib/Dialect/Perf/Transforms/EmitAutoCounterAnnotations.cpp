//===- EmitAutoCounterAnnotations.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"
#include "circt/Support/InstanceGraphInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
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

static std::string sanitizeFilename(llvm::StringRef s) {
  std::string out = s.str();
  for (char &c : out) {
    if (!(llvm::isAlnum(c) || c == '_' || c == '-' || c == '.'))
      c = '_';
  }
  return out;
}

static llvm::json::Value attrToJson(mlir::Attribute attr);

static llvm::json::Object dictAttrToJsonObject(mlir::DictionaryAttr dict) {
  llvm::json::Object obj;
  for (auto named : dict) {
    obj[named.getName().str()] = attrToJson(named.getValue());
  }
  return obj;
}

static llvm::json::Array arrayAttrToJsonArray(mlir::ArrayAttr arr) {
  llvm::json::Array out;
  for (auto elt : arr)
    out.push_back(attrToJson(elt));
  return out;
}

static llvm::json::Value attrToJson(mlir::Attribute attr) {
  if (!attr)
    return nullptr;

  if (auto s = llvm::dyn_cast<mlir::StringAttr>(attr))
    return s.getValue().str();

  if (auto b = llvm::dyn_cast<mlir::BoolAttr>(attr))
    return b.getValue();

  if (auto i = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return static_cast<int64_t>(i.getInt());

  if (auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr))
    return dictAttrToJsonObject(dict);

  if (auto arr = llvm::dyn_cast<mlir::ArrayAttr>(attr))
    return arrayAttrToJsonArray(arr);

  llvm::SmallString<64> buf;
  llvm::raw_svector_ostream os(buf);
  attr.print(os);
  return os.str().str();
}
static mlir::LogicalResult writeAnnotationsToFileForModule(
    circt::firrtl::FModuleOp module, llvm::StringRef outputDir,
    llvm::ArrayRef<mlir::DictionaryAttr> annotations) {
  if (outputDir.empty())
    return mlir::success();

  if (std::error_code ec = llvm::sys::fs::create_directories(outputDir)) {
    module.emitError() << "failed to create output directory '" << outputDir
                       << "': " << ec.message();
    return mlir::failure();
  }

  std::string moduleName = sanitizeFilename(module.getModuleName());
  llvm::SmallString<256> path(outputDir);
  llvm::sys::path::append(path, moduleName + ".autocounter.json");

  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    module.emitError() << "failed to open annotation output file '" << path
                       << "': " << ec.message();
    return mlir::failure();
  }

  llvm::json::Array arr;
  for (auto anno : annotations)
    arr.push_back(dictAttrToJsonObject(anno));

  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(arr))) << "\n";
  return mlir::success();
}

struct AutoCounterAnnotationHelper {
  static constexpr llvm::StringLiteral annotationClass =
      "midas.targetutils.AutoCounterFirrtlAnnotation";
  static constexpr llvm::StringLiteral accumulateOpTypeClass =
      "midas.targetutils.PerfCounterOps$Accumulate$";

  static mlir::DictionaryAttr
  build(mlir::MLIRContext *ctx, llvm::StringRef clock, llvm::StringRef reset,
        llvm::StringRef label, llvm::StringRef description = "",
        bool coverGenerated = false) {
    mlir::Builder b(ctx);

    auto opType = mlir::DictionaryAttr::get(
        ctx, {b.getNamedAttr("class", b.getStringAttr(accumulateOpTypeClass))});

    std::string desc = (label + "<--(circt autogen)").str();
    if (!description.empty()) {
      desc += " ";
      desc += description.str();
    }

    return mlir::DictionaryAttr::get(
        ctx,
        {
            b.getNamedAttr("class", b.getStringAttr(annotationClass)),
            b.getNamedAttr("clock", b.getStringAttr(clock)),
            b.getNamedAttr("reset", b.getStringAttr(reset)),
            b.getNamedAttr("label", b.getStringAttr(label)),
            b.getNamedAttr("description", b.getStringAttr(desc)),
            b.getNamedAttr("opType", opType),
            b.getNamedAttr("coverGenerated", b.getBoolAttr(coverGenerated)),
        });
  }

  static llvm::Expected<mlir::DictionaryAttr> buildFor(perf::PerfCounterOp op) {
    auto parentModule =
        op->getParentOfType<::circt::igraph::ModuleOpInterface>();

    if (!parentModule) {
      op.emitError("PerfCounterOp has no parent module");
      return llvm::make_error<llvm::StringError>(
          "PerfCounterOp has no parent module",
          std::make_error_code(std::errc::invalid_argument));
    }

    std::string clkName = perf::getSSAName(op.getClk(), parentModule);
    std::string resetName =
        op.getReset() ? perf::getSSAName(op.getReset(), parentModule) : "reset";
    std::string label = op.getName() ? op.getName()->str() : "<unknown>";

    return build(op->getContext(), clkName, resetName, label);
  }
};

static void addAnnotationToOp(mlir::Operation *targetOp,
                              mlir::DictionaryAttr anno) {
  firrtl::AnnotationSet annos(targetOp);
  annos.addAnnotations(mlir::ArrayAttr::get(targetOp->getContext(), {anno}));
  annos.applyToOperation(targetOp);
}
static mlir::LogicalResult addAnnotationToBlockArg(mlir::BlockArgument arg,
                                                   mlir::DictionaryAttr anno) {
  auto *block = arg.getOwner();
  if (!block)
    return mlir::failure();

  auto module =
      llvm::dyn_cast_or_null<circt::firrtl::FModuleOp>(block->getParentOp());
  if (!module)
    return mlir::failure();

  unsigned portIdx = arg.getArgNumber();
  auto *ctx = module->getContext();

  llvm::SmallVector<mlir::Attribute> allPortAnnos;
  if (auto existing = module.getPortAnnotationsAttr())
    allPortAnnos.assign(existing.begin(), existing.end());
  else
    allPortAnnos.assign(module.getNumPorts(), mlir::ArrayAttr::get(ctx, {}));

  if (portIdx >= allPortAnnos.size())
    return mlir::failure();

  llvm::SmallVector<mlir::Attribute> thisPortAnnos;
  if (auto arr = llvm::dyn_cast<mlir::ArrayAttr>(allPortAnnos[portIdx]))
    thisPortAnnos.assign(arr.begin(), arr.end());

  thisPortAnnos.push_back(anno);
  allPortAnnos[portIdx] = mlir::ArrayAttr::get(ctx, thisPortAnnos);

  module->setAttr("portAnnotations", mlir::ArrayAttr::get(ctx, allPortAnnos));
  return mlir::success();
}

static mlir::LogicalResult
annotateInputSource(circt::perf::PerfCounterOp perfOp,
                    mlir::DictionaryAttr anno) {
  if (perfOp->getNumOperands() == 0) {
    perfOp.emitError("PerfCounterOp has no operands");
    return mlir::failure();
  }

  mlir::Value input = perfOp->getOperand(0);

  if (mlir::Operation *defOp = input.getDefiningOp()) {
    addAnnotationToOp(defOp, anno);
    return mlir::success();
  }

  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(input)) {
    if (mlir::failed(addAnnotationToBlockArg(blockArg, anno))) {
      perfOp.emitError("failed to attach annotation to input block argument");
      return mlir::failure();
    }
    return mlir::success();
  }

  perfOp.emitError("unsupported PerfCounter input kind for annotation target");
  return mlir::failure();
}
struct EmitAutoCounterAnnotationsPass
    : public circt::perf::impl::EmitAutoCounterAnnotationsBase<
          EmitAutoCounterAnnotationsPass> {
  using Base = circt::perf::impl::EmitAutoCounterAnnotationsBase<
      EmitAutoCounterAnnotationsPass>;
  using Base::Base;
  using Base::outputDir;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::perf::PerfDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();
    registry.insert<circt::seq::SeqDialect>();
  }

  void runOnOperation() override;
};

void EmitAutoCounterAnnotationsPass::runOnOperation() {
  auto module = getOperation();

  bool anyFailure = false;
  llvm::SmallVector<mlir::Operation *> opsToErase;
  llvm::SmallVector<mlir::DictionaryAttr> emittedAnnotations;

  module.walk([&](perf::PerfCounterOp op) {
    auto annoOrErr = AutoCounterAnnotationHelper::buildFor(op);
    if (!annoOrErr) {
      op.emitError() << "failed to create AutoCounter annotation: "
                     << llvm::toString(annoOrErr.takeError());
      anyFailure = true;
      return;
    }

    mlir::DictionaryAttr anno = *annoOrErr;

    if (mlir::failed(annotateInputSource(op, anno))) {
      anyFailure = true;
      return;
    }

    emittedAnnotations.push_back(anno);
    opsToErase.push_back(op);
  });

  if (anyFailure) {
    signalPassFailure();
    return;
  }

  if (!emittedAnnotations.empty()) {
    if (mlir::failed(writeAnnotationsToFileForModule(module, outputDir,
                                                     emittedAnnotations))) {
      signalPassFailure();
      return;
    }
  }

  for (auto *op : opsToErase)
    op->erase();
}
} // namespace

std::unique_ptr<mlir::Pass>
circt::perf::createEmitAutoCounterAnnotationsPass(llvm::StringRef outputDir) {
  auto pass = std::make_unique<EmitAutoCounterAnnotationsPass>();

  if (!outputDir.empty())
    pass->outputDir = outputDir.str();
  return pass;
}
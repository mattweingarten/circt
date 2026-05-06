//===- EmitFireSimAnnotations.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Support/InstanceGraphInterface.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <system_error>

#define DEBUG_TYPE "perf-emit-firesim-annotations"

namespace circt {
namespace perf {
#define GEN_PASS_DEF_EMITFIRESIMANNOTATIONS
#include "circt/Dialect/Perf/PerfPasses.h.inc"
} // namespace perf
} // namespace circt

using namespace circt;
using namespace perf;

namespace {

static std::string sanitizeFilename(llvm::StringRef s) {
  std::string out = s.str();
  for (char &c : out) {
    if (!(llvm::isAlnum(static_cast<unsigned char>(c)) || c == '_' ||
          c == '-' || c == '.'))
      c = '_';
  }
  return out;
}

static llvm::json::Value attrToJson(mlir::Attribute attr);

static llvm::json::Object dictAttrToJsonObject(mlir::DictionaryAttr dict) {
  llvm::json::Object obj;
  for (auto named : dict)
    obj[named.getName().str()] = attrToJson(named.getValue());
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

static mlir::FailureOr<llvm::SmallVector<circt::firrtl::InstanceOp>>
getInstancesFromHierPathContext(mlir::Operation *anchorOp,
                                mlir::FlatSymbolRefAttr context) {
  llvm::SmallVector<circt::firrtl::InstanceOp> instances;

  if (!context)
    return instances;

  mlir::Operation *target =
      mlir::SymbolTable::lookupNearestSymbolFrom(anchorOp, context);
  if (!target)
    return mlir::failure();

  auto hierPath = llvm::dyn_cast<circt::hw::HierPathOp>(target);
  if (!hierPath)
    return mlir::failure();

  auto circuit = anchorOp->getParentOfType<circt::firrtl::CircuitOp>();
  if (!circuit)
    return mlir::failure();

  mlir::SymbolTable symbolTable(circuit);

  for (auto attr : hierPath.getNamepath()) {
    auto innerRef = llvm::dyn_cast<circt::hw::InnerRefAttr>(attr);
    if (!innerRef)
      return mlir::failure();

    auto moduleRef = innerRef.getModuleRef();
    auto innerName = innerRef.getName();

    auto parentModule =
        symbolTable.lookup<circt::firrtl::FModuleOp>(moduleRef.getValue());
    if (!parentModule)
      return mlir::failure();

    circt::firrtl::InstanceOp matchedInst;

    parentModule.walk([&](circt::firrtl::InstanceOp inst) {
      if (matchedInst)
        return;

      auto innerSym = inst.getInnerSymAttr();
      if (!innerSym)
        return;

      if (innerSym.getSymName() == innerName)
        matchedInst = inst;
    });

    if (!matchedInst)
      return mlir::failure();

    instances.push_back(matchedInst);
  }

  return instances;
}

static std::optional<circt::firrtl::AnnoPathValue>
getAnnoPathValueForValue(mlir::Operation *anchorOp, mlir::Value value,
                         mlir::FlatSymbolRefAttr context) {
  if (!value)
    return std::nullopt;

  circt::firrtl::AnnoTarget ref;
  circt::firrtl::FModuleOp localModule;

  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    localModule = llvm::dyn_cast<circt::firrtl::FModuleOp>(
        blockArg.getOwner()->getParentOp());
    if (!localModule)
      return std::nullopt;

    unsigned portIdx = blockArg.getArgNumber();
    if (portIdx >= localModule.getNumPorts())
      return std::nullopt;

    ref = circt::firrtl::PortAnnoTarget(localModule, portIdx);
  } else if (auto *defOp = value.getDefiningOp()) {
    localModule = defOp->getParentOfType<circt::firrtl::FModuleOp>();
    if (!localModule)
      return std::nullopt;

    if (!llvm::isa<circt::firrtl::FNamableOp>(defOp))
      return std::nullopt;

    ref = circt::firrtl::OpAnnoTarget(defOp);
  } else {
    return std::nullopt;
  }

  llvm::SmallVector<circt::firrtl::InstanceOp> instances;
  if (context) {
    auto instancesOrFailure =
        getInstancesFromHierPathContext(anchorOp, context);
    if (mlir::failed(instancesOrFailure))
      return std::nullopt;

    instances = *instancesOrFailure;
  }

  return circt::firrtl::AnnoPathValue(instances, ref, /*fieldIdx=*/0);
}

static std::optional<std::string>
annoPathValueToFIRRTLTarget(const circt::firrtl::AnnoPathValue &pathValue) {
  circt::firrtl::FModuleOp targetModule;
  std::string refName;

  if (auto opRef = llvm::dyn_cast<circt::firrtl::OpAnnoTarget>(pathValue.ref)) {
    mlir::Operation *op = opRef.getOp();

    targetModule = op->getParentOfType<circt::firrtl::FModuleOp>();
    if (!targetModule)
      return std::nullopt;

    auto namable = llvm::dyn_cast<circt::firrtl::FNamableOp>(op);
    if (!namable)
      return std::nullopt;

    refName = namable.getName().str();
  } else if (auto portRef =
                 llvm::dyn_cast<circt::firrtl::PortAnnoTarget>(pathValue.ref)) {
    auto moduleLike = portRef.getModule();

    targetModule =
        llvm::dyn_cast<circt::firrtl::FModuleOp>(moduleLike.getOperation());
    if (!targetModule)
      return std::nullopt;

    unsigned portIdx = portRef.getPortNo();

    if (portIdx >= targetModule.getNumPorts())
      return std::nullopt;

    refName = targetModule.getPortName(portIdx).str();
  } else {
    return std::nullopt;
  }

  auto circuit = targetModule->getParentOfType<circt::firrtl::CircuitOp>();
  if (!circuit)
    return std::nullopt;

  std::string target = "~";
  target += circuit.getName().str();
  target += "|";

  if (pathValue.instances.empty()) {
    target += targetModule.getModuleName().str();
  } else {
    auto rootModule = pathValue.instances.front()
                          ->getParentOfType<circt::firrtl::FModuleOp>();
    if (!rootModule)
      return std::nullopt;

    target += rootModule.getModuleName().str();

    for (auto inst : pathValue.instances) {
      target += "/";
      target += inst.getName().str();
      target += ":";
      target += inst.getModuleName().str();
    }
  }

  target += ">";
  target += refName;

  return target;
}

struct PerfOpInfo {
  mlir::Operation *op = nullptr;
  mlir::Value cond;
  mlir::Value clk;
  mlir::Value reset;
  mlir::FlatSymbolRefAttr context;
  std::string label;
  std::string description;
  bool isTrace = false;
};

static PerfOpInfo getPerfOpInfo(perf::PerfCounterOp op) {
  return PerfOpInfo{
      /*op=*/op.getOperation(),
      /*cond=*/op.getCond(),
      /*clk=*/op.getClk(),
      /*reset=*/op.getReset(),
      /*context=*/op->getAttrOfType<mlir::FlatSymbolRefAttr>("context"),
      /*label=*/op.getName().value_or("<unknown>").str(),
      /*description=*/op.getDesc().value_or("").str(),
      /*isTrace=*/false,
  };
}

static PerfOpInfo getPerfOpInfo(perf::PerfTraceOp op) {
  return PerfOpInfo{
      /*op=*/op.getOperation(),
      /*cond=*/op.getCond(),
      /*clk=*/op.getClk(),
      /*reset=*/op.getReset(),
      /*context=*/op->getAttrOfType<mlir::FlatSymbolRefAttr>("context"),
      /*label=*/op.getName().value_or("<unknown>").str(),
      /*description=*/op.getDesc().value_or("").str(),
      /*isTrace=*/true,
  };
}

struct FireSimAnnotationHelper {
  static constexpr llvm::StringLiteral autoCounterAnnoClass =
      "midas.targetutils.AutoCounterFirrtlAnnotation";
  static constexpr llvm::StringLiteral counterOpTypeClass =
      "midas.targetutils.PerfCounterOps$Accumulate$";
  static constexpr llvm::StringLiteral traceDoctorAnnoClass =
      "midas.targetutils.TraceDoctorFirrtlAnnotation";

  static std::string buildDescription(llvm::StringRef description) {
    return description.str();
  }

  static mlir::DictionaryAttr
  buildAutoCounterAttr(mlir::MLIRContext *ctx, llvm::StringRef clock,
                       llvm::StringRef reset, llvm::StringRef label,
                       llvm::StringRef description = "",
                       bool coverGenerated = false) {
    mlir::Builder b(ctx);

    auto opType = mlir::DictionaryAttr::get(
        ctx, {b.getNamedAttr("class", b.getStringAttr(counterOpTypeClass))});

    return mlir::DictionaryAttr::get(
        ctx,
        {
            b.getNamedAttr("class", b.getStringAttr(autoCounterAnnoClass)),
            b.getNamedAttr("clock", b.getStringAttr(clock)),
            b.getNamedAttr("reset", b.getStringAttr(reset)),
            b.getNamedAttr("label", b.getStringAttr(label)),
            b.getNamedAttr("description",
                           b.getStringAttr(buildDescription(description))),
            b.getNamedAttr("opType", opType),
            b.getNamedAttr("coverGenerated", b.getBoolAttr(coverGenerated)),
        });
  }

  static mlir::DictionaryAttr
  buildTraceDoctorAttr(mlir::MLIRContext *ctx, llvm::StringRef clock,
                       llvm::StringRef reset, llvm::StringRef label,
                       llvm::StringRef description = "",
                       bool coverGenerated = false) {
    mlir::Builder b(ctx);

    return mlir::DictionaryAttr::get(
        ctx,
        {
            b.getNamedAttr("class", b.getStringAttr(traceDoctorAnnoClass)),
            b.getNamedAttr("clock", b.getStringAttr(clock)),
            b.getNamedAttr("reset", b.getStringAttr(reset)),
            b.getNamedAttr("label", b.getStringAttr(label)),
            b.getNamedAttr("description",
                           b.getStringAttr(buildDescription(description))),
            b.getNamedAttr("coverGenerated", b.getBoolAttr(coverGenerated)),
        });
  }

  static llvm::Expected<mlir::DictionaryAttr>
  buildForPerfOp(const PerfOpInfo &info) {
    auto clkPath = getAnnoPathValueForValue(info.op, info.clk, info.context);
    if (!clkPath) {
      return llvm::make_error<llvm::StringError>(
          "failed to compute AnnoPathValue for PerfOp clock",
          std::make_error_code(std::errc::invalid_argument));
    }

    auto clkTarget = annoPathValueToFIRRTLTarget(*clkPath);
    if (!clkTarget) {
      return llvm::make_error<llvm::StringError>(
          "failed to stringify FIRRTL target for PerfOp clock",
          std::make_error_code(std::errc::invalid_argument));
    }

    std::string resetTarget;
    if (info.reset) {
      auto rstPath =
          getAnnoPathValueForValue(info.op, info.reset, info.context);
      if (!rstPath) {
        return llvm::make_error<llvm::StringError>(
            "failed to compute AnnoPathValue for PerfOp reset",
            std::make_error_code(std::errc::invalid_argument));
      }

      auto rstTarget = annoPathValueToFIRRTLTarget(*rstPath);
      if (!rstTarget) {
        return llvm::make_error<llvm::StringError>(
            "failed to stringify FIRRTL target for PerfOp reset",
            std::make_error_code(std::errc::invalid_argument));
      }

      resetTarget = *rstTarget;
    } else {
      resetTarget = "~<unknown>|<unknown>>reset";
    }

    if (info.isTrace)
      return buildTraceDoctorAttr(info.op->getContext(), *clkTarget,
                                  resetTarget, info.label, info.description);

    return buildAutoCounterAttr(info.op->getContext(), *clkTarget, resetTarget,
                                info.label, info.description);
  }

  static llvm::Expected<mlir::DictionaryAttr> buildFor(perf::PerfCounterOp op) {
    return buildForPerfOp(getPerfOpInfo(op));
  }

  static llvm::Expected<mlir::DictionaryAttr> buildFor(perf::PerfTraceOp op) {
    return buildForPerfOp(getPerfOpInfo(op));
  }
};

struct DontTouchAnnotationHelper {
  static constexpr llvm::StringLiteral annotationClass =
      "firrtl.transforms.DontTouchAnnotation";

  static mlir::DictionaryAttr build(mlir::MLIRContext *ctx) {
    mlir::Builder b(ctx);
    return mlir::DictionaryAttr::get(
        ctx, {b.getNamedAttr("class", b.getStringAttr(annotationClass))});
  }
};

static void
addAnnotationsToOp(mlir::Operation *targetOp,
                   llvm::ArrayRef<mlir::DictionaryAttr> annosToAdd) {
  firrtl::AnnotationSet annos(targetOp);

  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(annosToAdd.size());
  for (auto anno : annosToAdd)
    attrs.push_back(anno);

  annos.addAnnotations(mlir::ArrayAttr::get(targetOp->getContext(), attrs));
  annos.applyToOperation(targetOp);
}

static mlir::LogicalResult
addAnnotationsToBlockArg(mlir::BlockArgument arg,
                         llvm::ArrayRef<mlir::DictionaryAttr> annosToAdd) {
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

  for (auto anno : annosToAdd)
    thisPortAnnos.push_back(anno);

  allPortAnnos[portIdx] = mlir::ArrayAttr::get(ctx, thisPortAnnos);

  module->setAttr("portAnnotations", mlir::ArrayAttr::get(ctx, allPortAnnos));
  return mlir::success();
}

static mlir::LogicalResult
annotateInputSource(mlir::Operation *perfOp, mlir::Value input,
                    llvm::ArrayRef<mlir::DictionaryAttr> annos) {
  if (!input) {
    perfOp->emitError("Perf op has no input operand");
    return mlir::failure();
  }

  if (mlir::Operation *defOp = input.getDefiningOp()) {
    addAnnotationsToOp(defOp, annos);
    return mlir::success();
  }

  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(input)) {
    if (mlir::failed(addAnnotationsToBlockArg(blockArg, annos))) {
      perfOp->emitError("failed to attach annotations to input block argument");
      return mlir::failure();
    }
    return mlir::success();
  }

  perfOp->emitError("unsupported Perf input kind for annotation target");
  return mlir::failure();
}

template <typename PerfOpTy>
static mlir::LogicalResult
processPerfOp(PerfOpTy op, llvm::SmallVectorImpl<mlir::Operation *> &opsToErase,
              llvm::function_ref<void()> signalFailure) {
  PerfOpInfo info = getPerfOpInfo(op);

  auto fireSimAnnoOrErr = FireSimAnnotationHelper::buildFor(op);
  if (!fireSimAnnoOrErr) {
    llvm::errs() << "EmitFireSimAnnotations: failed to create "
                 << (info.isTrace ? "TraceDoctor" : "AutoCounter")
                 << " annotation: "
                 << llvm::toString(fireSimAnnoOrErr.takeError()) << "\n";
    llvm::errs() << "Offending PerfOp:\n";
    op.print(llvm::errs());
    llvm::errs() << "\n";

    opsToErase.push_back(op.getOperation());
    signalFailure();
    return mlir::failure();
  }

  mlir::DictionaryAttr fireSimAnno = *fireSimAnnoOrErr;
  mlir::DictionaryAttr dontTouchAnno =
      DontTouchAnnotationHelper::build(op->getContext());

  llvm::SmallVector<mlir::DictionaryAttr, 2> annos = {fireSimAnno,
                                                      dontTouchAnno};

  if (mlir::failed(annotateInputSource(op.getOperation(), info.cond, annos))) {
    llvm::errs()
        << "EmitFireSimAnnotations: failed to attach annotations to source\n";
    llvm::errs() << "Offending PerfOp:\n";
    op.print(llvm::errs());
    llvm::errs() << "\n";

    opsToErase.push_back(op.getOperation());
    return mlir::failure();
  }

  opsToErase.push_back(op.getOperation());
  return mlir::success();
}

struct EmitFireSimAnnotationsPass
    : public circt::perf::impl::EmitFireSimAnnotationsBase<
          EmitFireSimAnnotationsPass> {
  using Base =
      circt::perf::impl::EmitFireSimAnnotationsBase<EmitFireSimAnnotationsPass>;
  using Base::Base;
  using Base::outputDir;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::perf::PerfDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();
    registry.insert<circt::seq::SeqDialect>();
  }

  void runOnOperation() override;
};

void EmitFireSimAnnotationsPass::runOnOperation() {
  auto module = getOperation();

  llvm::SmallVector<mlir::Operation *> opsToErase;

  auto signalFailure = [&]() { signalPassFailure(); };

  module.walk([&](perf::PerfCounterOp op) {
    (void)processPerfOp(op, opsToErase, signalFailure);
  });

  module.walk([&](perf::PerfTraceOp op) {
    (void)processPerfOp(op, opsToErase, signalFailure);
  });

  for (mlir::Operation *op : opsToErase)
    op->erase();
}

} // namespace

std::unique_ptr<mlir::Pass>
circt::perf::createEmitFireSimAnnotationsPass(llvm::StringRef outputDir) {
  auto pass = std::make_unique<EmitFireSimAnnotationsPass>();
  if (!outputDir.empty())
    pass->outputDir = outputDir.str();
  return pass;
}
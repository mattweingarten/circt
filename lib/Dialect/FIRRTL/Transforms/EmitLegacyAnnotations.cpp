//===- EmitLegacyAnnotations.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace circt;
using namespace circt::firrtl;

namespace {

static llvm::json::Value attrToJson(mlir::Attribute attr) {
  if (!attr)
    return nullptr;

  return llvm::TypeSwitch<mlir::Attribute, llvm::json::Value>(attr)
      .Case<mlir::StringAttr>(
          [](auto a) { return llvm::json::Value(a.getValue().str()); })
      .Case<mlir::BoolAttr>(
          [](auto a) { return llvm::json::Value(a.getValue()); })
      .Case<mlir::IntegerAttr>(
          [](auto a) { return llvm::json::Value(a.getInt()); })
      .Case<mlir::FloatAttr>(
          [](auto a) { return llvm::json::Value(a.getValueAsDouble()); })
      .Case<mlir::ArrayAttr>([](auto a) {
        llvm::json::Array arr;
        for (auto elem : a)
          arr.push_back(attrToJson(elem));
        return llvm::json::Value(std::move(arr));
      })
      .Case<mlir::DictionaryAttr>([](auto a) {
        llvm::json::Object obj;
        for (auto named : a)
          obj[named.getName().str()] = attrToJson(named.getValue());
        return llvm::json::Value(std::move(obj));
      })
      .Default([](mlir::Attribute a) {
        std::string s;
        llvm::raw_string_ostream os(s);
        a.print(os);
        os.flush();
        return llvm::json::Value(s);
      });
}

static FModuleLike getEnclosingModule(mlir::Operation *op) {
  while (op) {
    if (auto mod = mlir::dyn_cast<FModuleLike>(op))
      return mod;
    op = op->getParentOp();
  }
  return {};
}


// Very hacky, but for some reason these have an outdated ModuleName-style target that doesn't match the new convention? What is a good fix for this?
/// Annotation classes that should keep the old ModuleName-style textual target,
/// e.g. "Circuit.Module".
static const llvm::StringSet<> oldModuleNameStyleAnnoClasses = {
    "firrtl.transforms.BlackBoxInlineAnno",
    "freechips.rocketchip.util.AddressMapAnnotation",
    "freechips.rocketchip.util.ParamsAnnotation",
    "freechips.rocketchip.util.RegFieldDescMappingAnnotation",
    "freechips.rocketchip.util.RetimeModuleAnnotation",
    "freechips.rocketchip.util.SRAMAnnotation",
    "chisel3.experimental.EnumAnnotations$EnumVecAnnotation",
    "firrtl.passes.InlineAnnotation",
};

static bool usesOldModuleNameStyle(mlir::DictionaryAttr anno) {
  auto cls = anno.getAs<mlir::StringAttr>("class");
  if (!cls)
    return false;
  return oldModuleNameStyleAnnoClasses.contains(cls.getValue());
}

static std::string getOldModuleNameTarget(CircuitOp circuit,
                                          FModuleLike module) {
  return circuit.getName().str() + "." + module.getModuleName().str();
}

static std::string getLegacyModuleTarget(CircuitOp circuit, FModuleLike module) {
  return "~" + circuit.getName().str() + "|" + module.getModuleName().str();
}

static std::string getAnnotationTarget(CircuitOp circuit, mlir::Operation *op,
                                       mlir::DictionaryAttr anno) {
  auto circuitName = circuit.getName().str();

  if (op == circuit.getOperation())
    return "~" + circuitName;

  if (auto module = mlir::dyn_cast<FModuleLike>(op)) {
    if (usesOldModuleNameStyle(anno))
      return getOldModuleNameTarget(circuit, module);
    return getLegacyModuleTarget(circuit, module);
  }

  if (auto inst = mlir::dyn_cast<InstanceOp>(op)) {
    auto parentModule = getEnclosingModule(op->getParentOp());
    if (!parentModule)
      return "~" + circuitName;

    return "~" + circuitName + "|" + parentModule.getModuleName().str() + "/" +
           inst.getName().str() + ":" + inst.getModuleName().str();
  }

  auto parentModule = getEnclosingModule(op);
  if (!parentModule)
    return "~" + circuitName;

  if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name"))
    return "~" + circuitName + "|" + parentModule.getModuleName().str() + ">" +
           nameAttr.getValue().str();

  return "~" + circuitName + "|" + parentModule.getModuleName().str();
}

static llvm::json::Object convertAnnotation(CircuitOp circuit,
                                            mlir::Operation *op,
                                            mlir::DictionaryAttr anno) {
  llvm::json::Object obj;
  for (auto named : anno) {
    if (named.getName() == "target")
      continue;
    obj[named.getName().str()] = attrToJson(named.getValue());
  }
  obj["target"] = getAnnotationTarget(circuit, op, anno);
  return obj;
}

static std::optional<std::string>
getTargetSuffixFromFieldID(FIRRTLBaseType type, uint64_t fieldID) {
  std::string suffix;

  while (fieldID != 0) {
    if (auto bundle = type_dyn_cast<BundleType>(type)) {
      auto index = bundle.getIndexForFieldID(fieldID);
      fieldID -= bundle.getFieldID(index);
      suffix += ".";
      suffix += bundle.getElement(index).name.getValue().str();
      type = bundle.getElementType(index);
      continue;
    }

    if (auto vector = type_dyn_cast<FVectorType>(type)) {
      auto index = vector.getIndexForFieldID(fieldID);
      fieldID -= vector.getFieldID(index);
      suffix += "[";
      suffix += std::to_string(index);
      suffix += "]";
      type = vector.getElementType();
      continue;
    }

    return std::nullopt;
  }

  return suffix;
}

static std::optional<std::string>
getPortTarget(CircuitOp circuit, FModuleLike module, llvm::StringRef portName,
              FIRRTLBaseType portType, mlir::DictionaryAttr anno) {
  std::string target = "~" + circuit.getName().str() + "|" +
                       module.getModuleName().str() + ">" + portName.str();

  auto fieldID = anno.getAs<IntegerAttr>("circt.fieldID");
  if (!fieldID)
    return target;

  uint64_t id = fieldID.getInt();
  if (id == 0)
    return target;

  auto suffix = getTargetSuffixFromFieldID(portType, id);
  if (!suffix)
    return std::nullopt;

  target += *suffix;
  return target;
}

struct EmitLegacyAnnotationsPass
    : public EmitLegacyAnnotationsBase<EmitLegacyAnnotationsPass> {
  using EmitLegacyAnnotationsBase::EmitLegacyAnnotationsBase;
  using EmitLegacyAnnotationsBase::outputFilename;
  void runOnOperation() override;
};

} // namespace

void EmitLegacyAnnotationsPass::runOnOperation() {
  auto circuit = getOperation();

  if (outputFilename.empty()) {
    circuit.emitError("missing output filename; use "
                      "--firrtl-emit-legacy-annotations=file=<path>");
    signalPassFailure();
    return;
  }

  llvm::json::Array annotations;

  circuit.walk([&](mlir::Operation *op) {
    auto annos = op->getAttrOfType<mlir::ArrayAttr>("annotations");
    if (!annos)
      return;

    for (auto attr : annos) {
      auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
      if (!dict)
        continue;
      annotations.push_back(convertAnnotation(circuit, op, dict));
    }
  });

  circuit.walk([&](FModuleLike module) {
    unsigned numPorts = module.getNumPorts();
    for (unsigned i = 0; i < numPorts; ++i) {
      auto annos = module.getAnnotationsForPort(i);
      if (annos.empty())
        continue;

      auto portNameAttr = module.getPortNameAttr(i);
      if (!portNameAttr)
        continue;

      auto portName = portNameAttr.getValue();
      auto portType = mlir::dyn_cast<FIRRTLBaseType>(module.getPortType(i));
      if (!portType)
        continue;

      for (auto attr : annos) {
        auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
        if (!dict)
          continue;

        auto target = getPortTarget(circuit, module, portName, portType, dict);
        if (!target) {
          module->emitWarning()
              << "skipping port annotation on '" << portName
              << "' in module '" << module.getModuleName()
              << "' because circt.fieldID could not be decoded";
          continue;
        }

        llvm::json::Object obj;
        for (auto named : dict) {
          if (named.getName() == "target" || named.getName() == "circt.fieldID")
            continue;
          obj[named.getName().str()] = attrToJson(named.getValue());
        }
        obj["target"] = *target;
        annotations.push_back(std::move(obj));
      }
    }
  });

  std::error_code ec;
  llvm::raw_fd_ostream os(outputFilename, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    circuit.emitError() << "failed to open output file '" << outputFilename
                        << "': " << ec.message();
    signalPassFailure();
    return;
  }

  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(annotations)))
     << "\n";
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createEmitLegacyAnnotationsPass(llvm::StringRef outputFilename) {
  auto pass = std::make_unique<EmitLegacyAnnotationsPass>();
  if (!outputFilename.empty())
    pass->outputFilename = outputFilename.str();
  return pass;
}
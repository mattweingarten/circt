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

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
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
  return cls && oldModuleNameStyleAnnoClasses.contains(cls.getValue());
}

static std::string getCircuitTarget(CircuitOp circuit) {
  return "~" + circuit.getName().str();
}

static std::string getOldModuleNameTarget(CircuitOp circuit,
                                          FModuleLike module) {
  return circuit.getName().str() + "." + module.getModuleName().str();
}

static std::string getLegacyModuleTarget(CircuitOp circuit,
                                         FModuleLike module) {
  return getCircuitTarget(circuit) + "|" + module.getModuleName().str();
}

static std::string getInstanceTarget(CircuitOp circuit, InstanceOp inst) {
  auto parentModule = getEnclosingModule(inst->getParentOp());
  if (!parentModule)
    return getCircuitTarget(circuit);

  return getCircuitTarget(circuit) + "|" + parentModule.getModuleName().str() +
         "/" + inst.getName().str() + ":" + inst.getModuleName().str();
}

static std::string getLocalTarget(CircuitOp circuit, mlir::Operation *op,
                                  mlir::DictionaryAttr anno) {
  if (op == circuit.getOperation())
    return getCircuitTarget(circuit);

  if (auto module = mlir::dyn_cast<FModuleLike>(op)) {
    if (usesOldModuleNameStyle(anno))
      return getOldModuleNameTarget(circuit, module);
    return getLegacyModuleTarget(circuit, module);
  }

  if (auto inst = mlir::dyn_cast<InstanceOp>(op))
    return getInstanceTarget(circuit, inst);

  auto parentModule = getEnclosingModule(op);
  if (!parentModule)
    return getCircuitTarget(circuit);

  auto target = getLegacyModuleTarget(circuit, parentModule);
  if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name"))
    target += ">" + nameAttr.getValue().str();

  return target;
}

static hw::HierPathOp getNonLocalAnnotationHierPath(CircuitOp circuit,
                                                    mlir::DictionaryAttr anno) {
  auto nonLocal = anno.getAs<mlir::FlatSymbolRefAttr>("circt.nonlocal");
  if (!nonLocal)
    return {};

  auto res = mlir::SymbolTable::lookupNearestSymbolFrom<hw::HierPathOp>(
      circuit.getOperation(), nonLocal);
  if (!res)
    return {};

  return res;
}

static FModuleLike lookupModule(CircuitOp circuit, mlir::StringAttr moduleName) {
  FModuleLike result;
  circuit.walk([&](FModuleLike module) {
    if (result)
      return;
    if (module.getModuleNameAttr() == moduleName)
      result = module;
  });
  return result;
}

static std::optional<std::string>
getInnerRefTargetName(CircuitOp circuit, hw::InnerRefAttr ref) {
  auto module = lookupModule(circuit, ref.getModule());
  if (!module)
    return std::nullopt;

  mlir::Operation *innerOp = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (innerOp)
      return;

    auto innerSym = op->getAttrOfType<hw::InnerSymAttr>("inner_sym");
    if (!innerSym)
      return;

    if (innerSym.getSymName() == ref.getName())
      innerOp = op;
  });

  if (!innerOp)
    return ref.getName().getValue().str();

  if (auto inst = mlir::dyn_cast<InstanceOp>(innerOp))
    return inst.getName().str();

  if (auto nameAttr = innerOp->getAttrOfType<mlir::StringAttr>("name"))
    return nameAttr.getValue().str();

  return ref.getName().getValue().str();
}

static std::optional<std::string>
getNonLocalAnnotationTarget(CircuitOp circuit, mlir::DictionaryAttr anno) {
  auto hierPath = getNonLocalAnnotationHierPath(circuit, anno);
  if (!hierPath)
    return std::nullopt;

  auto namepath = hierPath.getNamepath();
  if (namepath.size() == 0)
    return std::nullopt;

  auto firstRef = mlir::dyn_cast<hw::InnerRefAttr>(namepath[0]);
  if (!firstRef)
    return std::nullopt;

  std::string target = getCircuitTarget(circuit);
  target += "|";
  target += firstRef.getModule().getValue().str();

  for (unsigned i = 0, e = namepath.size(); i < e; ++i) {
    auto ref = mlir::dyn_cast<hw::InnerRefAttr>(namepath[i]);
    if (!ref) {
      if (mlir::dyn_cast<mlir::FlatSymbolRefAttr>(namepath[i]))
        return target;

      return std::nullopt;
    }

    auto refName = getInnerRefTargetName(circuit, ref);
    if (!refName)
      refName = ref.getName().getValue().str();

    if (i + 1 == e) {
      target += ">";
      target += *refName;
      break;
    }

    auto nextRef = mlir::dyn_cast<hw::InnerRefAttr>(namepath[i + 1]);
    if (!nextRef) {
      if (auto moduleRef =
              mlir::dyn_cast<mlir::FlatSymbolRefAttr>(namepath[i + 1])) {
        target += "/";
        target += *refName;
        target += ":";
        target += moduleRef.getValue().str();
        continue;
      }

      return std::nullopt;
    }

    target += "/";
    target += *refName;
    target += ":";
    target += nextRef.getModule().getValue().str();
  }

  return target;
}

static std::string getAnnotationTarget(CircuitOp circuit, mlir::Operation *op,
                                       mlir::DictionaryAttr anno) {
  if (auto nonLocalTarget = getNonLocalAnnotationTarget(circuit, anno))
    return *nonLocalTarget;

  return getLocalTarget(circuit, op, anno);
}

static bool shouldDropAnnotationField(llvm::StringRef name) {
  return name == "target" || name == "circt.nonlocal";
}

static llvm::json::Object
convertAnnotationWithTarget(mlir::DictionaryAttr anno, llvm::StringRef target,
                            bool dropFieldID = false) {
  llvm::json::Object obj;

  for (auto named : anno) {
    auto name = named.getName().strref();
    if (shouldDropAnnotationField(name))
      continue;
    if (dropFieldID && name == "circt.fieldID")
      continue;

    obj[name.str()] = attrToJson(named.getValue());
  }

  obj["target"] = target.str();
  return obj;
}

static llvm::json::Object convertAnnotation(CircuitOp circuit,
                                            mlir::Operation *op,
                                            mlir::DictionaryAttr anno) {
  return convertAnnotationWithTarget(anno,
                                     getAnnotationTarget(circuit, op, anno));
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

static std::optional<std::string> getFieldIDSuffix(FIRRTLBaseType type,
                                                   mlir::DictionaryAttr anno) {
  auto fieldID = anno.getAs<mlir::IntegerAttr>("circt.fieldID");
  if (!fieldID)
    return std::string();

  uint64_t id = fieldID.getInt();
  if (id == 0)
    return std::string();

  return getTargetSuffixFromFieldID(type, id);
}

static std::optional<std::string> getPortBaseTarget(CircuitOp circuit,
                                                    FModuleLike module,
                                                    llvm::StringRef portName,
                                                    mlir::DictionaryAttr anno) {
  if (auto nonLocalTarget = getNonLocalAnnotationTarget(circuit, anno))
    return *nonLocalTarget;

  return getLegacyModuleTarget(circuit, module) + ">" + portName.str();
}

static std::optional<std::string>
getPortTarget(CircuitOp circuit, FModuleLike module, llvm::StringRef portName,
              FIRRTLBaseType portType, mlir::DictionaryAttr anno) {
  auto target = getPortBaseTarget(circuit, module, portName, anno);
  if (!target)
    return std::nullopt;

  auto suffix = getFieldIDSuffix(portType, anno);
  if (!suffix)
    return std::nullopt;

  *target += *suffix;
  return target;
}

static std::optional<llvm::json::Object>
convertPortAnnotation(CircuitOp circuit, FModuleLike module,
                      llvm::StringRef portName, FIRRTLBaseType portType,
                      mlir::DictionaryAttr anno) {
  auto target = getPortTarget(circuit, module, portName, portType, anno);
  if (!target)
    return std::nullopt;

  return convertAnnotationWithTarget(anno, *target, true);
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

      auto portType = mlir::dyn_cast<FIRRTLBaseType>(module.getPortType(i));
      if (!portType)
        continue;

      auto portName = portNameAttr.getValue();

      for (auto attr : annos) {
        auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
        if (!dict)
          continue;

        auto obj =
            convertPortAnnotation(circuit, module, portName, portType, dict);
        if (!obj) {
          module->emitWarning()
              << "skipping port annotation on '" << portName << "' in module '"
              << module.getModuleName()
              << "' because circt.fieldID could not be decoded";
          continue;
        }

        annotations.push_back(std::move(*obj));
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
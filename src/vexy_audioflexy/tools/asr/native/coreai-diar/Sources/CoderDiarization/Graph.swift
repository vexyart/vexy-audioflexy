// this_file: src/vexy_audioflexy/tools/asr/native/coreai-diar/Sources/CoderDiarization/Graph.swift
// CoreAI array access follows N3DGraph.swift, BSD-3-Clause Daisuke Majima 2026.
import CoreAI
import Foundation

struct Graph {
    let function: InferenceFunction
    let inputs: [(String, NDArrayDescriptor)]
    let output: String
    let outputShape: [Int]

    init(url: URL, preference: String) async throws {
        let options = SpecializationOptions(preferredComputeUnitKind: preference == "ane" ? .neuralEngine : .gpu)
        let model = try await AIModel(contentsOf: url, options: options)
        guard let descriptor = model.functionDescriptor(for: "main"),
              let function = try model.loadFunction(named: "main"), descriptor.outputNames.count == 1 else {
            throw HostError.contract("Expected one main function output")
        }
        self.function = function
        var inputs: [(String, NDArrayDescriptor)] = []
        for name in descriptor.inputNames {
            guard case .ndArray(let array) = descriptor.inputDescriptor(of: name) else {
                throw HostError.contract("Expected tensor input \(name)")
            }
            inputs.append((name, array))
        }
        self.inputs = inputs.sorted { $0.1.shape.reduce(1, *) > $1.1.shape.reduce(1, *) }
        output = descriptor.outputNames[0]
        guard case .ndArray(let array) = descriptor.outputDescriptor(of: output) else {
            throw HostError.contract("Expected tensor output")
        }
        outputShape = array.shape
    }

    func run(_ values: [[Float]]) async throws -> [Float] {
        guard values.count == inputs.count else { throw HostError.contract("Wrong input count") }
        var arrays: [String: NDArray] = [:]
        for ((name, descriptor), data) in zip(inputs, values) {
            guard data.count == descriptor.shape.reduce(1, *), data.allSatisfy({ $0.isFinite }) else {
                throw HostError.contract("Invalid input \(name)")
            }
            var array = NDArray(descriptor: descriptor)
            switch descriptor.scalarType {
            case .float32:
                var view = array.mutableView(as: Float.self)
                view.copyElements(fromContentsOf: data)
            case .float16:
                var view = array.mutableView(as: Float16.self)
                view.copyElements(fromContentsOf: data.map { Float16($0) })
            default: throw HostError.contract("Unsupported input scalar type")
            }
            arrays[name] = array
        }
        var outputs = try await function.run(inputs: arrays)
        guard let array = outputs.remove(output)?.ndArray, array.shape == outputShape else {
            throw HostError.contract("Invalid output shape")
        }
        let result: [Float]
        switch array.scalarType {
        case .float32:
            result = array.view(as: Float.self).withUnsafePointer { ptr, _, _ in Self.gather(ptr, array.shape, array.strides) { $0 } }
        case .float16:
            result = array.view(as: Float16.self).withUnsafePointer { ptr, _, _ in Self.gather(ptr, array.shape, array.strides) { Float($0) } }
        default: throw HostError.contract("Unsupported output scalar type")
        }
        guard result.allSatisfy({ $0.isFinite }) else { throw HostError.contract("Nonfinite graph output") }
        return result
    }

    static func gather<T>(_ pointer: UnsafePointer<T>, _ shape: [Int], _ strides: [Int], _ convert: (T) -> Float) -> [Float] {
        var dense = [Int](repeating: 1, count: shape.count)
        for dimension in stride(from: shape.count - 2, through: 0, by: -1) { dense[dimension] = dense[dimension + 1] * shape[dimension + 1] }
        return (0..<shape.reduce(1, *)).map { flat in
            var remaining = flat, offset = 0
            for dimension in 0..<shape.count {
                let index = remaining / dense[dimension]
                remaining -= index * dense[dimension]
                offset += index * (strides.isEmpty ? dense[dimension] : strides[dimension])
            }
            return convert(pointer[offset])
        }
    }
}

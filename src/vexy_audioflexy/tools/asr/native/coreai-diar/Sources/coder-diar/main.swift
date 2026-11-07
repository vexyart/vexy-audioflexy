// this_file: src/vexy_audioflexy/tools/asr/native/coreai-diar/Sources/coder-diar/main.swift
import CoderDiarization
import Foundation
import N3DGateSupport

do {
    let args = CommandLine.arguments
    guard args.count == 5 || args.count == 6 else {
        throw HostError.contract("Usage: coder-diar ASSETS WAV JSON PROBS [gpu|ane]")
    }
    let samples = try loadWav16kMono(URL(fileURLWithPath: args[2]))
    let model = try await Diarizer(directory: URL(fileURLWithPath: args[1]), preference: args.count == 6 ? args[5] : "gpu")
    let (receipt, probabilities) = try await model.process(samples: samples)
    try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: args[3]))
    try writeF32(probabilities, to: URL(fileURLWithPath: args[4]))
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}

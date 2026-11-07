// swift-tools-version: 6.0
// this_file: src/vexy_audioflexy/tools/asr/native/coreai-diar/Package.swift
import PackageDescription
import Foundation

guard let zooRoot = ProcessInfo.processInfo.environment["COREAI_ZOO_ROOT"], !zooRoot.isEmpty else {
    fatalError("Set COREAI_ZOO_ROOT to the pinned coreai-model-zoo checkout")
}

let package = Package(
    name: "CoderDiarization",
    platforms: [.macOS("27.0")],
    products: [.executable(name: "coder-diar", targets: ["coder-diar"])],
    dependencies: [.package(path: zooRoot + "/conversion/nemotron3_diar/swift")],
    targets: [
        .target(name: "CoderDiarization", dependencies: [.product(name: "NemotronDiarizer", package: "swift")],
                linkerSettings: [.linkedFramework("CoreAI")]),
        .executableTarget(name: "coder-diar", dependencies: ["CoderDiarization",
            .product(name: "N3DGateSupport", package: "swift")]),
        .testTarget(name: "CoderDiarizationTests", dependencies: ["CoderDiarization"], resources: [.copy("Fixtures")])
    ]
)

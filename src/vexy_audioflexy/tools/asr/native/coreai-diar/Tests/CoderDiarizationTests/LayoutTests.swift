// this_file: src/vexy_audioflexy/tools/asr/native/coreai-diar/Tests/CoderDiarizationTests/LayoutTests.swift
import XCTest
@testable import CoderDiarization

final class LayoutTests: XCTestCase {
    func testChannelFirstPaddingAndInverse() throws {
        let rows: [Float] = [1, 2, 3, 4]
        let packed = try Layout.channelsFirst(rows, channels: 2, capacity: 3)
        XCTAssertEqual(packed, [1, 3, 0, 2, 4, 0])
        XCTAssertEqual(try Layout.timeMajor(packed, channels: 2, subframes: 1, capacity: 3, frames: 2), rows)
    }
    func testEncoderSubframesBecomeChronological() throws {
        let output = try Layout.timeMajor([0, 1, 2, 3, 10, 11, 12, 13], channels: 2, subframes: 2, capacity: 2, frames: 2)
        XCTAssertEqual(output, [0, 10, 2, 12, 1, 11, 3, 13])
    }
    func testMalformedAndOversizedRowsAreRejected() {
        XCTAssertThrowsError(try Layout.channelsFirst([1, 2, 3], channels: 2, capacity: 4))
        XCTAssertThrowsError(try Layout.channelsFirst([1, 2], channels: 1, capacity: 1))
        XCTAssertThrowsError(try Layout.timeMajor([1], channels: 1, subframes: 1, capacity: 1, frames: 2))
    }
    func testStridedTensorReadUsesActualStrides() {
        let physical: [Float] = [1, 2, 99, 3, 4, 99]
        let values = physical.withUnsafeBufferPointer { Graph.gather($0.baseAddress!, [2, 2], [3, 1]) { $0 } }
        XCTAssertEqual(values, [1, 2, 3, 4])
    }
    func testMetadataRejectsCachePolicyMismatch() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "fast32", withExtension: "json", subdirectory: "Fixtures"))
        XCTAssertNoThrow(try Metadata.read(url))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        object["prediction_score_threshold"] = 0.3
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        try JSONSerialization.data(withJSONObject: object).write(to: temporary)
        XCTAssertThrowsError(try Metadata.read(temporary))
    }
}

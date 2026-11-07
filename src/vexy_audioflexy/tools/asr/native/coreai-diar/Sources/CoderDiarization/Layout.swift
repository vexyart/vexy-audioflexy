// this_file: src/vexy_audioflexy/tools/asr/native/coreai-diar/Sources/CoderDiarization/Layout.swift
import Foundation

public enum HostError: Error {
    case contract(String)
}

public enum Layout {
    /// Time-major host rows -> [1, channels, 1, capacity] graph input, zero padded.
    public static func channelsFirst(_ rows: [Float], channels: Int, capacity: Int) throws -> [Float] {
        guard channels > 0, capacity > 0, rows.count % channels == 0, rows.count / channels <= capacity else {
            throw HostError.contract("Invalid channels-first geometry")
        }
        var result = [Float](repeating: 0, count: channels * capacity)
        for t in 0..<(rows.count / channels) {
            for c in 0..<channels { result[c * capacity + t] = rows[t * channels + c] }
        }
        return result
    }

    /// [1, channels, subframes, capacity] graph output -> chronological host rows.
    public static func timeMajor(_ values: [Float], channels: Int, subframes: Int,
                                 capacity: Int, frames: Int) throws -> [Float] {
        guard channels > 0, subframes > 0, capacity > 0, frames > 0, frames <= capacity,
              values.count == channels * subframes * capacity else {
            throw HostError.contract("Invalid time-major geometry")
        }
        var result = [Float](repeating: 0, count: frames * subframes * channels)
        for t in 0..<frames {
            for k in 0..<subframes {
                for c in 0..<channels { result[(t * subframes + k) * channels + c] = values[(c * subframes + k) * capacity + t] }
            }
        }
        return result
    }
}

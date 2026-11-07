// this_file: src/vexy_audioflexy/tools/asr/native/coreai-diar/Sources/CoderDiarization/Diarizer.swift
import Foundation
import NemotronDiarizer

public struct Metadata: Decodable {
    let format: String
    let chunk_frames: Int
    let right_context_frames: Int
    let speaker_cache_frames: Int
    let fifo_frames: Int
    let cache_update_frames: Int
    let encoder_capacity: Int
    let embedding_capacity: Int
    let hidden_size: Int
    let speakers: Int
    let subsampling_factor: Int
    let sample_rate: Int
    let hop_samples: Int
    let prediction_score_threshold: Double
    let latest_frames_score_boost: Double
    let strong_boost_rate: Double
    let weak_boost_rate: Double
    let min_positive_scores_rate: Double
    let silence_frames_per_speaker: Int

    public static func read(_ url: URL) throws -> Metadata {
        let value = try JSONDecoder().decode(Metadata.self, from: Data(contentsOf: url))
        guard value.format == "nemotron3-diarization-coreai-v1", value.hidden_size == 512, value.speakers == 8,
              value.subsampling_factor == 8, value.sample_rate == 16000, value.hop_samples == 160,
              value.speaker_cache_frames == 264, value.prediction_score_threshold == 0.25,
              value.latest_frames_score_boost == 0.05, value.strong_boost_rate == 0.75,
              value.weak_boost_rate == 1.5, value.min_positive_scores_rate == 0.5,
              value.silence_frames_per_speaker == 1, value.chunk_frames > 0, value.right_context_frames >= 0,
              value.fifo_frames > 0, value.cache_update_frames > 0,
              value.embedding_capacity == value.chunk_frames + value.right_context_frames,
              value.encoder_capacity >= value.speaker_cache_frames + value.fifo_frames + value.embedding_capacity else {
            throw HostError.contract("Metadata does not match reused Nemotron primitives")
        }
        return value
    }
}

public struct Diarizer {
    let metadata: Metadata
    let embedding: Graph
    let encoder: Graph
    let mel: N3DMel
    let silence: [Float]
    public let loadSeconds: Double

    public init(directory: URL, preference: String = "gpu") async throws {
        guard ["gpu", "ane"].contains(preference) else { throw HostError.contract("Expected gpu or ane") }
        let start = ContinuousClock.now
        metadata = try Metadata.read(directory.appendingPathComponent("metadata.json"))
        embedding = try await Graph(url: directory.appendingPathComponent("embedding.aimodel"), preference: preference)
        encoder = try await Graph(url: directory.appendingPathComponent("encoder.aimodel"), preference: preference)
        guard embedding.inputs.map({ $0.1.shape }) == [[1, 1024, 1, metadata.embedding_capacity]],
              embedding.outputShape == [1, 512, 1, metadata.embedding_capacity],
              encoder.inputs.map({ $0.1.shape }) == [[1, 512, 1, metadata.encoder_capacity], [1, 1, 1, metadata.encoder_capacity]],
              encoder.outputShape == [1, 8, 8, metadata.encoder_capacity] else {
            throw HostError.contract("Published graph I/O does not match metadata")
        }
        let filters = try N3DAssets.readF32LE(directory.appendingPathComponent("mel-filter.f32"), count: 128 * 257)
        let window = try N3DAssets.readF32LE(directory.appendingPathComponent("window.f32"), count: 400)
        silence = try N3DAssets.readF32LE(directory.appendingPathComponent("silence.f32"), count: 512)
        guard (filters + window + silence).allSatisfy({ $0.isFinite }) else { throw HostError.contract("Nonfinite frontend weights") }
        mel = N3DMel(melFilters: filters, hannWindow: window)
        loadSeconds = Self.elapsed(start)
    }

    public func process(samples: [Float]) async throws -> ([String: Any], [Float]) {
        guard samples.count >= 160, samples.allSatisfy({ $0.isFinite }) else { throw HostError.contract("Require finite 16k PCM") }
        let start = ContinuousClock.now
        let (features, frames) = samples.withUnsafeBufferPointer { mel.logMel($0, center: true) }
        let rowCount = (frames + 7) / 8
        let stacked = features + [Float](repeating: 0, count: rowCount * 1024 - features.count)
        var cache = N3DSpeakerCache(fifoLength: metadata.fifo_frames, updatePeriod: metadata.cache_update_frames)
        var logits: [Float] = [], traces: [[String: Int]] = []
        for first in stride(from: 0, to: rowCount, by: metadata.chunk_frames) {
            let committed = min(metadata.chunk_frames, rowCount - first)
            let lookahead = min(metadata.right_context_frames, rowCount - first - committed)
            let count = committed + lookahead
            let input = try Layout.channelsFirst(Array(stacked[(first * 1024)..<((first + count) * 1024)]), channels: 1024, capacity: metadata.embedding_capacity)
            let projected = try await embedding.run([input])
            let chunk = try Layout.timeMajor(projected, channels: 512, subframes: 1, capacity: metadata.embedding_capacity, frames: count)
            let prefix = cache.cacheFrames + cache.fifoFrames
            let rows = cache.rows + chunk
            let length = rows.count / 512
            let hidden = try Layout.channelsFirst(rows, channels: 512, capacity: metadata.encoder_capacity)
            let mask: [Float] = (0..<metadata.encoder_capacity).map { $0 < length ? 1 : 0 }
            let prediction = try await encoder.run([hidden, mask])
            let chronological = try Layout.timeMajor(prediction, channels: 8, subframes: 8, capacity: metadata.encoder_capacity, frames: length)
            cache.update(rows: rows, logits: chronological, frames: length, silence: silence, chunkFrames: committed)
            let valid = min(committed * 8, frames - first * 8)
            logits.append(contentsOf: chronological[(prefix * 64)..<(prefix * 64 + valid * 8)])
            traces.append(["first_row": first, "committed": committed, "lookahead": lookahead,
                           "cache_frames": cache.cacheFrames, "fifo_frames": cache.fifoFrames])
        }
        guard logits.count == frames * 8 else { throw HostError.contract("Final flush lost frames") }
        let probabilities = logits.map { N3DSpeakerCache.sigmoid($0) }
        let turns: [[String: Any]] = N3DDiarizer.segments(from: probabilities).map {
            ["start": $0.start, "end": $0.end, "speaker": $0.speaker]
        }
        return (["segments": turns, "frames": frames, "wall_s": Self.elapsed(start),
                 "load_s": loadSeconds, "trace": traces, "compressions": cache.compressions], probabilities)
    }

    static func elapsed(_ start: ContinuousClock.Instant) -> Double {
        let duration = ContinuousClock.now - start
        return Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }
}

package com.iykyk.collage.pipeline

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** One appearance, flattened for offline threshold sweeps. */
@Serializable
data class TrackletRecord(
    val index: Int,
    val startMs: Long,
    val endMs: Long,
    val visibleFrames: Int,
    val bestScore: Float,
    val bestTimestampMs: Long,
    val embedding: List<Float>,
)

/**
 * Everything the tuning harness needs to replay clustering offline: the appearances, their
 * embeddings, and the cannot-link pairs the video geometry implies.
 *
 * Re-running the device pipeline per threshold experiment costs minutes; replaying this
 * costs milliseconds.
 */
@Serializable
data class AnalysisDump(
    val sourceName: String,
    val durationMs: Long,
    val tracklets: List<TrackletRecord>,
    /** Index pairs that must not merge, as two-element lists. */
    val cannotLink: List<List<Int>>,
) {
    fun toJson(): String = JSON.encodeToString(serializer(), this)

    companion object {
        private val JSON = Json { prettyPrint = false; ignoreUnknownKeys = true }

        fun fromJson(text: String): AnalysisDump = JSON.decodeFromString(serializer(), text)
    }
}

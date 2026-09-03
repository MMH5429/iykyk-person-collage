package com.iykyk.collage.video

import com.iykyk.collage.core.frame.Nv21Buffer
import kotlinx.coroutines.flow.Flow

/**
 * One sampled frame ready for analysis.
 *
 * [nv21] is in *raw* orientation; [rotationDegrees] is what must be applied to make it
 * upright. ML Kit is given both, so it reports boxes in upright coordinates.
 */
data class AnalysisFrame(
    val index: Int,
    val timestampMs: Long,
    val nv21: Nv21Buffer,
    val rotationDegrees: Int,
)

/**
 * Supplies sampled frames of a video in presentation order.
 *
 * Implementations must be cold: collecting [frames] starts a fresh decode, and cancelling
 * the collector releases the codec.
 */
interface FrameSource {
    /** Total clip duration, used to drive determinate progress. */
    val durationMs: Long

    fun frames(): Flow<AnalysisFrame>
}

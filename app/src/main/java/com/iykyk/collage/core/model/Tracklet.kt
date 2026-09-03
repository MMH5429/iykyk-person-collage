package com.iykyk.collage.core.model

import com.iykyk.collage.core.PipelineConfig

/** A detection with its quality assessment attached. */
data class ScoredObservation(
    val observation: FaceObservation,
    val signals: QualitySignals,
    val clearlyVisible: Boolean,
    val score: Float,
)

/**
 * One continuous run of a single face across consecutive sampled frames.
 *
 * A tracklet that passes [isAppearance] is exactly one "appearance" as the assignment
 * defines it: a continuous segment during which the person is clearly visible.
 */
data class Tracklet(
    val id: Int,
    val observations: List<ScoredObservation>,
) {
    private val visible: List<ScoredObservation> = observations.filter { it.clearlyVisible }

    /** Bounds of the *clearly visible* span — the appearance proper, not the blurry approach. */
    val visibleStartMs: Long get() = visible.firstOrNull()?.observation?.timestampMs ?: -1L
    val visibleEndMs: Long get() = visible.lastOrNull()?.observation?.timestampMs ?: -1L
    val visibleFrameCount: Int get() = visible.size
    val visibleDurationMs: Long
        get() = if (visible.isEmpty()) 0L else visibleEndMs - visibleStartMs

    /** The best candidate shot in this tracklet, ranked by weighted quality. */
    val best: ScoredObservation? get() = visible.maxByOrNull { it.score }

    fun isAppearance(config: PipelineConfig): Boolean =
        visibleFrameCount >= config.minVisibleFrames &&
            visibleDurationMs >= config.minVisibleDurationMs

    /**
     * True when both tracklets are clearly visible at the same moment.
     *
     * Two people on screen together cannot be the same person, so this drives the
     * cannot-link constraint in clustering.
     */
    fun overlapsInTime(other: Tracklet): Boolean {
        if (visibleFrameCount == 0 || other.visibleFrameCount == 0) return false
        return visibleStartMs <= other.visibleEndMs && other.visibleStartMs <= visibleEndMs
    }
}

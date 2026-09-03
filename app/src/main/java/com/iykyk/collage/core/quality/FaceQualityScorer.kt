package com.iykyk.collage.core.quality

import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.QualitySignals
import kotlin.math.abs
import kotlin.math.min

/** Weighted sum used to rank candidate shots for a person. */
fun QualitySignals.weightedScore(c: PipelineConfig): Float =
    c.wFrontality * frontality +
        c.wSharpness * sharpness +
        c.wEyesOpen * eyesOpen +
        c.wFaceSize * faceSize +
        c.wCompleteness * completeness +
        c.wExpression * expression

/**
 * Turns a raw detection into six normalised quality signals, and decides whether the face
 * is "clearly visible" in the sense the assignment defines an appearance.
 *
 * Pure: no Android types, so the whole gating policy is unit-testable and tunable offline.
 */
class FaceQualityScorer(private val config: PipelineConfig) {

    fun score(obs: FaceObservation): QualitySignals = QualitySignals(
        sharpness = (obs.rawSharpness / config.sharpnessSaturation).coerceIn(0f, 1f),
        frontality = frontality(obs),
        eyesOpen = eyesOpen(obs),
        expression = obs.smileProb ?: NEUTRAL,
        faceSize = faceSize(obs),
        completeness = completeness(obs),
    )

    /**
     * A face counts as clearly visible when it is sharp, reasonably front-facing, big
     * enough to identify, and not cut off by the frame edge.
     *
     * Eyes-open and expression are deliberately excluded: a person mid-blink is still
     * present. Those signals only influence which shot is chosen to represent them.
     */
    fun isClearlyVisible(s: QualitySignals): Boolean =
        s.sharpness >= config.minSharpness &&
            s.frontality >= config.minFrontality &&
            s.faceSize >= config.minFaceSize &&
            s.completeness >= config.minCompleteness

    private fun frontality(obs: FaceObservation): Float {
        val yaw = (abs(obs.headEulerY) / config.maxYawDeg).coerceIn(0f, 1f)
        val pitch = (abs(obs.headEulerX) / config.maxPitchDeg).coerceIn(0f, 1f)
        val roll = (abs(obs.headEulerZ) / config.maxRollDeg).coerceIn(0f, 1f)
        // Yaw dominates: turning away hides identity far faster than nodding or tilting.
        return (1f - (0.60f * yaw + 0.25f * pitch + 0.15f * roll)).coerceIn(0f, 1f)
    }

    private fun eyesOpen(obs: FaceObservation): Float {
        val l = obs.leftEyeOpenProb
        val r = obs.rightEyeOpenProb
        if (l == null && r == null) return NEUTRAL
        return min(l ?: NEUTRAL, r ?: NEUTRAL)
    }

    private fun faceSize(obs: FaceObservation): Float {
        val frameArea = obs.frameWidth.toFloat() * obs.frameHeight.toFloat()
        if (frameArea <= 0f) return 0f
        return ((obs.box.area / frameArea) / config.faceAreaSaturation).coerceIn(0f, 1f)
    }

    /**
     * How much of the face box lies inside the frame, with an extra penalty for sitting
     * right on the edge — a face touching the border is usually mid-exit and partly cut.
     */
    private fun completeness(obs: FaceObservation): Float {
        val frame = BoxF(0f, 0f, obs.frameWidth.toFloat(), obs.frameHeight.toFloat())
        val visible = obs.box.intersect(frame)?.area ?: return 0f
        if (obs.box.area <= 0f) return 0f
        val inside = (visible / obs.box.area).coerceIn(0f, 1f)

        val margin = min(obs.frameWidth, obs.frameHeight) * config.edgeMarginFraction
        val touchesEdge = obs.box.left <= margin ||
            obs.box.top <= margin ||
            obs.box.right >= obs.frameWidth - margin ||
            obs.box.bottom >= obs.frameHeight - margin
        return if (touchesEdge) inside * EDGE_PENALTY else inside
    }

    private companion object {
        const val NEUTRAL = 0.5f
        const val EDGE_PENALTY = 0.6f
    }
}

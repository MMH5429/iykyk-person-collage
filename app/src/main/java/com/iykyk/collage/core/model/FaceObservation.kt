package com.iykyk.collage.core.model

/**
 * One face detected in one sampled frame, in *upright* analysis-frame coordinates.
 *
 * "Upright" means the video's rotation metadata has already been applied, so [box] and
 * the landmarks are in the same space a viewer sees.
 */
data class FaceObservation(
    val frameIndex: Int,
    val timestampMs: Long,
    /** ML Kit tracking id when available; used only as an association hint. */
    val trackingId: Int?,
    val box: BoxF,
    val leftEye: PointF2?,
    val rightEye: PointF2?,
    /** Head pose in degrees: X = pitch (nod), Y = yaw (turn), Z = roll (tilt). */
    val headEulerX: Float,
    val headEulerY: Float,
    val headEulerZ: Float,
    val leftEyeOpenProb: Float?,
    val rightEyeOpenProb: Float?,
    val smileProb: Float?,
    /** Raw variance-of-Laplacian over the face region; normalised by the scorer. */
    val rawSharpness: Float,
    val frameWidth: Int,
    val frameHeight: Int,
)

/** Six independent quality signals, each normalised to [0, 1]. */
data class QualitySignals(
    val sharpness: Float,
    val frontality: Float,
    val eyesOpen: Float,
    val expression: Float,
    val faceSize: Float,
    val completeness: Float,
)

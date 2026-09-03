package com.iykyk.collage.core

/**
 * Every tuned constant in the pipeline, in one place.
 *
 * Defaults here are starting points. The offline sweep in `ThresholdSweepTest` calibrates
 * them against Sample 1's published ground truth (5 people x 4 appearances) and the
 * winning values are written back here and documented in the README.
 */
data class PipelineConfig(
    // --- Sampling -------------------------------------------------------
    /** Frames analysed per second of video. */
    val sampleFps: Int = 8,
    /** Long edge of the analysis frame in pixels; the decoder subsamples to roughly this. */
    val analysisLongEdge: Int = 540,

    // --- Quality normalisation ------------------------------------------
    /** Raw Laplacian variance that maps to a sharpness score of 1.0. */
    val sharpnessSaturation: Float = 350f,
    val maxYawDeg: Float = 60f,
    val maxPitchDeg: Float = 45f,
    val maxRollDeg: Float = 45f,
    /** Face-box area fraction of the frame that maps to a faceSize score of 1.0. */
    val faceAreaSaturation: Float = 0.06f,
    /** Distance from the frame edge, as a fraction of the short edge, below which a face reads as clipped. */
    val edgeMarginFraction: Float = 0.01f,

    // --- "Clearly visible" gates ----------------------------------------
    val minSharpness: Float = 0.20f,
    val minFrontality: Float = 0.35f,
    val minFaceSize: Float = 0.10f,
    val minCompleteness: Float = 0.85f,

    // --- Representative-shot weights (must sum to 1) --------------------
    val wFrontality: Float = 0.25f,
    val wSharpness: Float = 0.25f,
    val wEyesOpen: Float = 0.20f,
    val wFaceSize: Float = 0.10f,
    val wCompleteness: Float = 0.10f,
    val wExpression: Float = 0.10f,

    // --- Tracklet association -------------------------------------------
    /** How long a track may go unseen before it is closed. */
    val gapToleranceMs: Long = 400,
    /** Minimum IoU for a detection to join an open track. */
    val minAssociationIou: Float = 0.20f,
    /** Bonus added to the association score when ML Kit tracking ids agree. */
    val trackingIdBonus: Float = 0.15f,
    /** A tracklet must hold at least this many clearly-visible frames to be an appearance. */
    val minVisibleFrames: Int = 2,
    /** ...and span at least this long. */
    val minVisibleDurationMs: Long = 200,

    // --- Embedding -------------------------------------------------------
    /** Best-quality crops averaged into one embedding per tracklet. */
    val cropsPerTracklet: Int = 5,

    // --- Clustering ------------------------------------------------------
    /** Cosine-distance cut for agglomerative clustering. THE similarity threshold. */
    val clusterThreshold: Float = 0.55f,

    // --- Representative crop ---------------------------------------------
    /** Crop side as a multiple of the face box; deliberately generous, never tight. */
    val representativeCropFactor: Float = 2.2f,
) {
    companion object {
        val Default = PipelineConfig()
    }
}

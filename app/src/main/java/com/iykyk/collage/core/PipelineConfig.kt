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
    /**
     * Raw Laplacian variance that maps to a sharpness score of 1.0.
     *
     * Calibrated from the sample clips: over ~450 real detections the raw variance runs
     * median ~25 and p95 ~180. Large faces on a subsampled frame are mostly smooth skin,
     * so the absolute values are far lower than a tight crop would give.
     */
    val sharpnessSaturation: Float = 150f,
    val maxYawDeg: Float = 60f,
    val maxPitchDeg: Float = 45f,
    val maxRollDeg: Float = 45f,
    /** Face-box area fraction of the frame that maps to a faceSize score of 1.0. */
    val faceAreaSaturation: Float = 0.06f,
    // --- "Clearly visible" gates ----------------------------------------
    /** Raw variance of ~9. Whip-pan blur sits near 1; ordinary frames sit well above. */
    val minSharpness: Float = 0.06f,
    val minFrontality: Float = 0.35f,
    val minFaceSize: Float = 0.10f,
    /** Allows a quarter of the face box off-frame; the weighted score still prefers whole faces. */
    val minCompleteness: Float = 0.75f,

    // --- Representative-shot weights (must sum to 1) --------------------
    val wFrontality: Float = 0.25f,
    val wSharpness: Float = 0.25f,
    val wEyesOpen: Float = 0.20f,
    val wFaceSize: Float = 0.10f,
    val wCompleteness: Float = 0.10f,
    val wExpression: Float = 0.10f,

    // --- Tracklet association -------------------------------------------
    /**
     * Luma-histogram distance above which consecutive frames are treated as a hard cut.
     *
     * These clips are edited montages, so cuts - not motion - are what end most
     * appearances. Measured over all three samples the two populations separate cleanly:
     * within-shot frames peak at 0.215 while the smallest real cut is 0.225, so this sits
     * in the gap.
     */
    val sceneCutDistance: Float = 0.22f,

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
    /**
     * Cosine-distance cut for agglomerative clustering. THE similarity threshold.
     *
     * Calibrated against Sample 1's published ground truth of five people, sweeping in
     * 0.01 steps over embeddings dumped from a real device run.
     *
     *   Sample 1: 5 people for tau in [0.45, 0.64)   <- the only published ground truth
     *   Sample 2: 5 people for tau in [0.40, 0.52)
     *   Sample 3: 5 people for tau in [0.47, 0.54)
     *
     * 0.50 sits inside all three plateaus rather than on any edge, and the three clips
     * share a cast, so agreeing on five across all of them is a meaningful cross-check
     * rather than a coincidence.
     */
    val clusterThreshold: Float = 0.50f,

    // --- Representative crop ---------------------------------------------
    /** Crop side as a multiple of the face box; deliberately generous, never tight. */
    val representativeCropFactor: Float = 2.2f,
) {
    companion object {
        val Default = PipelineConfig()
    }
}

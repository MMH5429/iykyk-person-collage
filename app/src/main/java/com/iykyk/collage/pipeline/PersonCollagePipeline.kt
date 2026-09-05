package com.iykyk.collage.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.iykyk.collage.collage.CollageRenderer
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.cluster.ConstrainedClusterer
import com.iykyk.collage.core.frame.histogramDistance
import com.iykyk.collage.core.frame.lumaHistogram
import com.iykyk.collage.core.model.AnalysisResult
import com.iykyk.collage.core.model.Appearance
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.Person
import com.iykyk.collage.core.model.PointF2
import com.iykyk.collage.core.model.Tracklet
import com.iykyk.collage.core.quality.FaceQualityScorer
import com.iykyk.collage.core.track.TrackletBuilder
import com.iykyk.collage.detect.CropExtractor
import com.iykyk.collage.detect.MlKitFaceDetector
import com.iykyk.collage.embed.EyeAligner
import com.iykyk.collage.embed.FaceEmbedder
import com.iykyk.collage.video.AnalysisFrame
import com.iykyk.collage.video.FullResFrameGrabber
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

/**
 * Runs the whole analysis for one video and emits progress.
 *
 * Stage order matters: tracklets must exist before appearances can be embedded - embedding
 * once per appearance rather than once per frame is what keeps this both fast and stable -
 * and embeddings must exist before clustering can group appearances into people.
 *
 * Memory shape matters just as much. Face crops are cut during the decode pass and the
 * frame itself is dropped immediately after. Retaining whole frames instead costs about
 * 780 KB each, which over a 30-second clip is more than 100 MB of NV21 buffers: enough to
 * hold a mid-range device in continuous GC, and enough that dropped frames made the
 * appearance counts vary between runs of the same video. An aligned 160x160 crop is 100 KB,
 * and only clearly-visible faces get one.
 */
class PersonCollagePipeline(
    private val context: Context,
    private val config: PipelineConfig,
) {

    private val scorer = FaceQualityScorer(config)
    private val aligner = EyeAligner()

    /** Populated after every run; the tuning harness reads this via the debug export. */
    @Volatile
    var lastDump: AnalysisDump? = null
        private set

    fun run(uri: Uri, sourceName: String): Flow<PipelineState> = flow {
        emit(PipelineState.Preparing)

        val source = MediaCodecFrameSource(context, uri, config)
        val durationMs = source.durationMs.coerceAtLeast(1L)

        val observationsByFrame = mutableListOf<List<FaceObservation>>()
        val cropsByFrame = mutableListOf<List<Bitmap?>>()
        val sceneCuts = mutableListOf<Boolean>()

        try {
            FaceEmbedder(context).use { embedder ->

                // --- Stage 1: decode, detect, and cut crops while the frame is in hand ----
                val detector = MlKitFaceDetector(config)
                var previousHistogram: IntArray? = null

                try {
                    source.frames().collect { frame ->
                        currentCoroutineContext().ensureActive()

                        val histogram = lumaHistogram(
                            luma = frame.nv21.bytes,
                            width = frame.nv21.width,
                            height = frame.nv21.height,
                            rowStride = frame.nv21.lumaRowStride,
                        )
                        sceneCuts += previousHistogram
                            ?.let { histogramDistance(it, histogram) >= config.sceneCutDistance }
                            ?: false
                        previousHistogram = histogram

                        val observations = detector.detect(frame)
                        observationsByFrame += observations
                        cropsByFrame += observations.map { obs ->
                            // Only faces that can contribute to an appearance are worth keeping.
                            if (!scorer.isClearlyVisible(scorer.score(obs))) null
                            else alignedCrop(frame, obs, observations, embedder.inputSize)
                        }

                        emit(
                            PipelineState.Analysing(
                                (frame.timestampMs.toFloat() / durationMs).coerceIn(0f, 1f)
                            )
                        )
                    }
                } finally {
                    detector.close()
                }
                emit(PipelineState.Analysing(1f))

                // --- Stage 2: appearances --------------------------------------------------
                emit(PipelineState.GroupingPeople)
                val tracklets =
                    TrackletBuilder(config, scorer).build(observationsByFrame, sceneCuts)
                if (tracklets.isEmpty()) {
                    emit(PipelineState.Failed("No clearly visible faces were found in this video."))
                    return@use
                }

                // --- Stage 3: one averaged embedding per appearance ------------------------
                val embeddings = tracklets.map { tracklet ->
                    currentCoroutineContext().ensureActive()
                    embedder.embedAveraged(tracklet.bestCrops(observationsByFrame, cropsByFrame))
                }

                // --- Stage 4: group appearances into people --------------------------------
                val cannotLink = buildCannotLink(tracklets)
                lastDump = buildDump(
                    sourceName, durationMs, observationsByFrame.size,
                    tracklets, embeddings, cannotLink,
                )
                val clusters =
                    ConstrainedClusterer(config.clusterThreshold).cluster(embeddings, cannotLink)

                // --- Stage 5: representative shots -----------------------------------------
                emit(PipelineState.SelectingShots)
                val people = FullResFrameGrabber(context, uri).use { grabber ->
                    clusters
                        .map { members -> members.map { tracklets[it] } }
                        .sortedBy { group -> group.minOf { it.visibleStartMs } }
                        .mapIndexed { index, group ->
                            currentCoroutineContext().ensureActive()
                            Person(
                                id = index,
                                label = "Person ${index + 1}",
                                appearances = group
                                    .map { Appearance(it.visibleStartMs, it.visibleEndMs) }
                                    .sortedBy { it.startMs },
                                shot = representativeShot(
                                    group, grabber, observationsByFrame, cropsByFrame
                                ),
                            )
                        }
                }

                val analysis = AnalysisResult(people, sourceName, durationMs)

                // --- Stage 6: collage --------------------------------------------------------
                emit(PipelineState.BuildingCollage)
                emit(
                    PipelineState.Done(
                        CollageResult(analysis, CollageRenderer().render(analysis))
                    )
                )
            }
        } finally {
            cropsByFrame.forEach { row -> row.forEach { it?.recycle() } }
        }
    }.flowOn(Dispatchers.Default)

    /** Tracklets that are clearly visible at the same moment cannot be the same person. */
    private fun buildCannotLink(tracklets: List<Tracklet>): Set<Pair<Int, Int>> = buildSet {
        for (i in tracklets.indices) {
            for (j in i + 1 until tracklets.size) {
                if (tracklets[i].overlapsInTime(tracklets[j])) add(i to j)
            }
        }
    }

    private fun buildDump(
        sourceName: String,
        durationMs: Long,
        framesAnalysed: Int,
        tracklets: List<Tracklet>,
        embeddings: List<FloatArray>,
        cannotLink: Set<Pair<Int, Int>>,
    ) = AnalysisDump(
        sourceName = sourceName,
        durationMs = durationMs,
        framesAnalysed = framesAnalysed,
        tracklets = tracklets.mapIndexed { i, t ->
            TrackletRecord(
                index = i,
                startMs = t.visibleStartMs,
                endMs = t.visibleEndMs,
                visibleFrames = t.visibleFrameCount,
                bestScore = t.best?.score ?: 0f,
                bestTimestampMs = t.best?.observation?.timestampMs ?: 0L,
                embedding = embeddings[i].toList(),
            )
        },
        cannotLink = cannotLink.map { listOf(it.first, it.second) },
    )

    /**
     * The crop cut during the decode pass for this exact detection.
     *
     * Observations are stored per frame in detection order and those same instances flow
     * into the tracklets, so an identity match within the frame's row finds the right crop.
     */
    private fun cropFor(
        obs: FaceObservation,
        observationsByFrame: List<List<FaceObservation>>,
        cropsByFrame: List<List<Bitmap?>>,
    ): Bitmap? {
        val row = observationsByFrame.getOrNull(obs.frameIndex) ?: return null
        val index = row.indexOfFirst { it === obs }
        if (index < 0) return null
        return cropsByFrame.getOrNull(obs.frameIndex)?.getOrNull(index)
    }

    /** The top-scoring crops of one appearance, already aligned to the model's input. */
    private fun Tracklet.bestCrops(
        observationsByFrame: List<List<FaceObservation>>,
        cropsByFrame: List<List<Bitmap?>>,
    ): List<Bitmap> {
        val crops = observations
            .filter { it.clearlyVisible }
            .sortedByDescending { it.score }
            .take(config.cropsPerTracklet)
            .mapNotNull { cropFor(it.observation, observationsByFrame, cropsByFrame) }

        if (crops.isNotEmpty()) return crops

        // Should not happen for an admitted tracklet, but never hand an empty list on.
        return observations
            .sortedByDescending { it.score }
            .firstNotNullOfOrNull { cropFor(it.observation, observationsByFrame, cropsByFrame) }
            ?.let { listOf(it) }
            .orEmpty()
    }

    private fun alignedCrop(
        frame: AnalysisFrame,
        obs: FaceObservation,
        neighbours: List<FaceObservation>,
        inputSize: Int,
    ): Bitmap? {
        val crop = runCatching { CropExtractor.faceCrop(frame, obs, EMBED_CROP_FACTOR, neighbours) }
            .getOrNull() ?: return null

        // Landmarks are in upright frame coordinates; the aligner needs crop-local ones.
        val side = CropExtractor.croppedSide(obs, EMBED_CROP_FACTOR, neighbours)
        val originX = obs.box.centerX - side / 2f
        val originY = obs.box.centerY - side / 2f
        val scale = crop.width / side

        fun local(p: PointF2?) =
            p?.let { PointF2((it.x - originX) * scale, (it.y - originY) * scale) }

        val aligned = aligner.align(crop, local(obs.leftEye), local(obs.rightEye), inputSize)
        if (aligned !== crop) crop.recycle()
        return aligned
    }

    /**
     * The best shot across all of a person's appearances, taken from a *full-resolution*
     * re-decode and cropped generously - never tight to the face box.
     *
     * Only one seek happens per person, so the cost is trivial. Doing the same for every
     * embedded frame was tried and rejected: the extra full-size bitmaps reintroduced the
     * memory pressure this pipeline is shaped to avoid.
     */
    private fun representativeShot(
        group: List<Tracklet>,
        grabber: FullResFrameGrabber,
        observationsByFrame: List<List<FaceObservation>>,
        cropsByFrame: List<List<Bitmap?>>,
    ): Bitmap {
        val best = group.mapNotNull { it.best }.maxBy { it.score }
        val obs = best.observation
        val neighbours = observationsByFrame.getOrElse(obs.frameIndex) { emptyList() }

        val fullFrame = grabber.grab(obs.timestampMs)
        if (fullFrame != null) {
            val shot = CropExtractor.faceCropFromFullFrame(
                fullFrame, obs, config.representativeCropFactor, neighbours
            )
            fullFrame.recycle()
            if (shot != null) return shot
        }

        // Fall back to the stored analysis crop if the seek failed.
        return cropFor(obs, observationsByFrame, cropsByFrame)
            ?.copy(Bitmap.Config.ARGB_8888, false)
            ?: Bitmap.createBitmap(FALLBACK_TILE, FALLBACK_TILE, Bitmap.Config.ARGB_8888)
    }

    private companion object {
        /** Tighter than the collage crop: the model wants the face to fill its input. */
        const val EMBED_CROP_FACTOR = 1.6f
        const val FALLBACK_TILE = 160
    }
}

package com.iykyk.collage.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.iykyk.collage.collage.CollageRenderer
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.cluster.ConstrainedClusterer
import com.iykyk.collage.core.model.AnalysisResult
import com.iykyk.collage.core.model.Appearance
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.Person
import com.iykyk.collage.core.model.PointF2
import com.iykyk.collage.core.model.ScoredObservation
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
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Runs the whole analysis for one video and emits progress.
 *
 * Stage order matters: tracklets must exist before anything is embedded — embedding once
 * per appearance rather than once per frame is what keeps this both fast and stable — and
 * embeddings must exist before clustering can group appearances into people.
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

        // --- Stage 1: decode and detect ------------------------------------------------
        val detector = MlKitFaceDetector(config)
        val observationsByFrame = mutableListOf<List<FaceObservation>>()
        val frameCache = HashMap<Int, AnalysisFrame>()

        try {
            source.frames().collect { frame ->
                currentCoroutineContext().ensureActive()
                val observations = detector.detect(frame)
                observationsByFrame += observations
                // Frames with no face are never needed again, so they are not retained.
                if (observations.isNotEmpty()) frameCache[frame.index] = frame
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

        // --- Stage 2: appearances ------------------------------------------------------
        emit(PipelineState.GroupingPeople)
        val tracklets = TrackletBuilder(config, scorer).build(observationsByFrame)
        if (tracklets.isEmpty()) {
            emit(PipelineState.Failed("No clearly visible faces were found in this video."))
            return@flow
        }

        // --- Stage 3: one averaged embedding per appearance ----------------------------
        val embeddings = FaceEmbedder(context).use { embedder ->
            tracklets.map { tracklet ->
                currentCoroutineContext().ensureActive()
                val crops = tracklet.bestCrops(frameCache, embedder.inputSize)
                embedder.embedAveraged(crops).also { crops.forEach(Bitmap::recycle) }
            }
        }

        // --- Stage 4: group appearances into people ------------------------------------
        val cannotLink = buildCannotLink(tracklets)
        lastDump = buildDump(sourceName, durationMs, tracklets, embeddings, cannotLink)
        val clusters = ConstrainedClusterer(config.clusterThreshold).cluster(embeddings, cannotLink)

        // --- Stage 5: representative shots ---------------------------------------------
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
                        shot = representativeShot(group, grabber, frameCache),
                    )
                }
        }

        val analysis = AnalysisResult(people, sourceName, durationMs)

        // --- Stage 6: collage ------------------------------------------------------------
        emit(PipelineState.BuildingCollage)
        emit(PipelineState.Done(CollageResult(analysis, CollageRenderer().render(analysis))))
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
        tracklets: List<Tracklet>,
        embeddings: List<FloatArray>,
        cannotLink: Set<Pair<Int, Int>>,
    ) = AnalysisDump(
        sourceName = sourceName,
        durationMs = durationMs,
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

    /** The top-scoring crops of one appearance, aligned to the model's input. */
    private fun Tracklet.bestCrops(
        frameCache: Map<Int, AnalysisFrame>,
        inputSize: Int,
    ): List<Bitmap> {
        val crops = observations
            .filter { it.clearlyVisible }
            .sortedByDescending { it.score }
            .take(config.cropsPerTracklet)
            .mapNotNull { alignedCrop(it, frameCache, inputSize) }

        if (crops.isNotEmpty()) return crops

        // Should not happen for an admitted tracklet, but never hand an empty list on.
        return listOfNotNull(
            observations.maxByOrNull { it.score }?.let { alignedCrop(it, frameCache, inputSize) }
        )
    }

    private fun alignedCrop(
        scored: ScoredObservation,
        frameCache: Map<Int, AnalysisFrame>,
        inputSize: Int,
    ): Bitmap? {
        val frame = frameCache[scored.observation.frameIndex] ?: return null
        val obs = scored.observation
        val crop = runCatching { CropExtractor.faceCrop(frame, obs, EMBED_CROP_FACTOR) }
            .getOrNull() ?: return null

        // Landmarks are in upright frame coordinates; the aligner needs crop-local ones.
        val side = max(obs.box.width, obs.box.height) * EMBED_CROP_FACTOR
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
     * re-decode and cropped generously — never tight to the face box.
     */
    private fun representativeShot(
        group: List<Tracklet>,
        grabber: FullResFrameGrabber,
        frameCache: Map<Int, AnalysisFrame>,
    ): Bitmap {
        val best = group.mapNotNull { it.best }.maxBy { it.score }
        val obs = best.observation

        val fullFrame = grabber.grab(obs.timestampMs)
        if (fullFrame != null) {
            // Analysis coordinates are downsampled; scale up to the full-res frame.
            val scale = fullFrame.width.toFloat() / obs.frameWidth
            val side = max(obs.box.width, obs.box.height) * config.representativeCropFactor * scale
            val cx = obs.box.centerX * scale
            val cy = obs.box.centerY * scale
            val rect = BoxF(cx - side / 2f, cy - side / 2f, cx + side / 2f, cy + side / 2f)
                .clampTo(fullFrame.width, fullFrame.height)

            val w = (rect.right - rect.left).roundToInt()
            val h = (rect.bottom - rect.top).roundToInt()
            if (w > 1 && h > 1) {
                val cropped = Bitmap.createBitmap(
                    fullFrame, rect.left.roundToInt(), rect.top.roundToInt(), w, h
                )
                if (cropped !== fullFrame) fullFrame.recycle()
                return cropped
            }
            fullFrame.recycle()
        }

        // Fall back to the analysis-resolution crop if the seek failed.
        val frame = frameCache.getValue(obs.frameIndex)
        return CropExtractor.faceCrop(frame, obs, config.representativeCropFactor)
    }

    private companion object {
        /** Tighter than the collage crop: the model wants the face to fill its input. */
        const val EMBED_CROP_FACTOR = 1.6f
    }
}

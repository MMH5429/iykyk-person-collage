package com.iykyk.collage.video

import android.content.Context
import android.graphics.ImageFormat
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.frame.PlaneData
import com.iykyk.collage.core.frame.chooseStep
import com.iykyk.collage.core.frame.downsampleToNv21
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.nio.ByteBuffer

/**
 * Decodes a video once, front to back, emitting one downsampled NV21 frame per sampling
 * interval.
 *
 * Sequential decode rather than seeking: `MediaMetadataRetriever.getFrameAtTime` costs
 * 50-150 ms per seek, so 240 samples would take 15-30 s. Walking the stream once costs a
 * few seconds and keeps the progress bar honest.
 *
 * Only frames we actually want are rendered to the [ImageReader]; the rest are dropped by
 * the decoder without ever being converted.
 */
class MediaCodecFrameSource(
    private val context: Context,
    private val uri: Uri,
    private val config: PipelineConfig,
) : FrameSource {

    override val durationMs: Long by lazy {
        val mmr = MediaMetadataRetriever()
        try {
            mmr.setDataSource(context, uri)
            mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        } catch (e: RuntimeException) {
            0L
        } finally {
            mmr.release()
        }
    }

    override fun frames(): Flow<AnalysisFrame> = flow {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var reader: ImageReader? = null

        try {
            extractor.setDataSource(context, uri, null)
            val trackIndex = (0 until extractor.trackCount).firstOrNull { i ->
                extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: error("no video track in $uri")

            extractor.selectTrack(trackIndex)
            val format = extractor.getTrackFormat(trackIndex)
            val rawWidth = format.getInteger(MediaFormat.KEY_WIDTH)
            val rawHeight = format.getInteger(MediaFormat.KEY_HEIGHT)
            val rotation =
                if (format.containsKey(KEY_ROTATION)) format.getInteger(KEY_ROTATION) else 0
            val step = chooseStep(rawWidth, rawHeight, config.analysisLongEdge)
            val sampleIntervalUs = 1_000_000L / config.sampleFps

            reader = ImageReader.newInstance(rawWidth, rawHeight, ImageFormat.YUV_420_888, MAX_IMAGES)
            codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            codec.configure(format, reader.surface, null, 0)
            codec.start()

            val bufferInfo = MediaCodec.BufferInfo()
            var emitted = 0
            var nextSampleUs = 0L
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!inputDone) {
                    val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val input = codec.getInputBuffer(inputIndex)!!
                        val size = extractor.readSampleData(input, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outputIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
                if (outputIndex >= 0) {
                    val presentationUs = bufferInfo.presentationTimeUs
                    val wanted = bufferInfo.size > 0 && presentationUs >= nextSampleUs
                    codec.releaseOutputBuffer(outputIndex, wanted)

                    if (wanted) {
                        nextSampleUs = presentationUs + sampleIntervalUs
                        val nv21 = reader.awaitImage()?.use { image ->
                            downsampleToNv21(
                                y = image.planes[0].toPlaneData(),
                                u = image.planes[1].toPlaneData(),
                                v = image.planes[2].toPlaneData(),
                                srcWidth = image.width,
                                srcHeight = image.height,
                                step = step,
                            )
                        }
                        if (nv21 != null) {
                            emit(
                                AnalysisFrame(
                                    index = emitted++,
                                    timestampMs = presentationUs / 1000,
                                    nv21 = nv21,
                                    rotationDegrees = rotation,
                                )
                            )
                        }
                    }
                }

                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    outputDone = true
                }
            }
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            runCatching { reader?.close() }
            runCatching { extractor.release() }
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Rendering to the reader's surface is asynchronous, so the image is not always ready
     * the instant `releaseOutputBuffer` returns. Retry briefly rather than dropping frames.
     */
    private fun ImageReader.awaitImage(): Image? {
        repeat(ACQUIRE_ATTEMPTS) {
            acquireNextImage()?.let { return it }
            Thread.sleep(ACQUIRE_BACKOFF_MS)
        }
        return null
    }

    /** Copies a plane out of its direct buffer so it survives `Image.close()`. */
    private fun Image.Plane.toPlaneData(): PlaneData {
        val source: ByteBuffer = buffer
        val bytes = ByteArray(source.remaining())
        source.get(bytes)
        return PlaneData(bytes, rowStride, pixelStride)
    }

    private companion object {
        const val TIMEOUT_US = 10_000L
        const val MAX_IMAGES = 3
        const val ACQUIRE_ATTEMPTS = 25
        const val ACQUIRE_BACKOFF_MS = 2L
        const val KEY_ROTATION = "rotation-degrees"
    }
}

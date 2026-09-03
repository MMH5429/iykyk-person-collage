package com.iykyk.collage.embed

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.cluster.cosineDistance
import com.iykyk.collage.detect.CropExtractor
import com.iykyk.collage.detect.MlKitFaceDetector
import com.iykyk.collage.samples.SampleVideos
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class FaceEmbedderTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    @Test
    fun modelReportsTheExpectedShape() {
        FaceEmbedder(context).use { embedder ->
            assertEquals(160, embedder.inputSize)
            assertEquals(512, embedder.dimensions)
        }
    }

    @Test
    fun embeddingsAreUnitLength() {
        FaceEmbedder(context).use { embedder ->
            val bitmap = Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888)
            val v = embedder.embed(bitmap)
            assertEquals(512, v.size)
            assertEquals(1f, v.fold(0f) { a, x -> a + x * x }, 1e-3f)
        }
    }

    /**
     * The real signal test: two crops of the same person, one sampled frame apart, must be
     * much closer than crops of two different people sharing a frame.
     *
     * If this fails, the input normalisation or the alignment is wrong — not the threshold.
     */
    @Test
    fun sameFaceIsCloserThanDifferentFaces() = runBlocking {
        val sample = SampleVideos.ensureExtracted(context).first()
        val frames = MediaCodecFrameSource(context, sample.uri, config).frames().take(200).toList()
        val detector = MlKitFaceDetector(config)

        FaceEmbedder(context).use { embedder ->
            try {
                // A frame containing two distinct faces at once.
                var found: Pair<Int, List<com.iykyk.collage.core.model.FaceObservation>>? = null
                for ((i, f) in frames.withIndex()) {
                    val obs = detector.detect(f)
                    if (obs.size >= 2 && i + 1 < frames.size) {
                        found = i to obs
                        break
                    }
                }
                assertTrue("no frame with two simultaneous faces found", found != null)
                val (index, observations) = found!!
                val frame = frames[index]

                val a1 = embedder.embed(CropExtractor.faceCrop(frame, observations[0], 1.6f))
                val b1 = embedder.embed(CropExtractor.faceCrop(frame, observations[1], 1.6f))

                // The same first face one sampled frame later.
                val nextFrame = frames[index + 1]
                val nextObs = detector.detect(nextFrame)
                    .minByOrNull { abs(it.box.centerX - observations[0].box.centerX) }
                assertTrue("face vanished on the next frame", nextObs != null)
                val a2 = embedder.embed(CropExtractor.faceCrop(nextFrame, nextObs!!, 1.6f))

                val sameDistance = cosineDistance(a1, a2)
                val differentDistance = cosineDistance(a1, b1)
                assertTrue(
                    "same-person distance $sameDistance was not clearly below " +
                        "different-person distance $differentDistance",
                    sameDistance < differentDistance - 0.15f
                )
            } finally {
                detector.close()
            }
        }
    }

    @Test
    fun averagingProducesAUnitVector() {
        FaceEmbedder(context).use { embedder ->
            val bitmaps = List(3) { Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888) }
            val v = embedder.embedAveraged(bitmaps)
            assertEquals(1f, v.fold(0f) { a, x -> a + x * x }, 1e-3f)
        }
    }
}

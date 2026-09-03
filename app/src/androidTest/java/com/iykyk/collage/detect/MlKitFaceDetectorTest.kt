package com.iykyk.collage.detect

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.samples.SampleVideos
import com.iykyk.collage.video.AnalysisFrame
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MlKitFaceDetectorTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    private fun firstFrames(n: Int): List<AnalysisFrame> = runBlocking {
        val sample = SampleVideos.ensureExtracted(context).first()
        MediaCodecFrameSource(context, sample.uri, config).frames().take(n).toList()
    }

    @Test
    fun detectsAtLeastOneFaceSomewhereInTheOpeningSeconds() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            var total = 0
            firstFrames(24).forEach { total += detector.detect(it).size }
            assertTrue("no faces found in the first 3 seconds", total > 0)
        } finally {
            detector.close()
        }
    }

    @Test
    fun observationsAreInUprightCoordinatesAndInsideTheFrame() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            val observations = firstFrames(24).flatMap { detector.detect(it) }
            assertTrue(observations.isNotEmpty())
            observations.forEach { o ->
                assertTrue("frame must be portrait upright", o.frameHeight > o.frameWidth)
                assertTrue(o.box.centerX in 0f..o.frameWidth.toFloat())
                assertTrue(o.box.centerY in 0f..o.frameHeight.toFloat())
            }
        } finally {
            detector.close()
        }
    }

    @Test
    fun sharpnessIsPopulatedAndVaries() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            val values = firstFrames(40).flatMap { detector.detect(it) }.map { it.rawSharpness }
            assertTrue(values.isNotEmpty())
            assertTrue("sharpness never populated", values.any { it > 0f })
            assertTrue(
                "sharpness is constant, region mapping is probably wrong",
                values.distinct().size > 1
            )
        } finally {
            detector.close()
        }
    }

    @Test
    fun cropIsGenerousRatherThanTightToTheBox() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            val frames = firstFrames(24)
            val pair = frames.firstNotNullOfOrNull { f ->
                detector.detect(f).firstOrNull()?.let { f to it }
            }
            assertTrue("no face found to crop", pair != null)
            val (frame, obs) = pair!!

            val crop = CropExtractor.faceCrop(frame, obs, expandFactor = 2.2f)
            assertTrue("crop is empty", crop.width > 0 && crop.height > 0)
            assertTrue(
                "crop ${crop.width}px is not wider than the ${obs.box.width}px face box",
                crop.width > obs.box.width * 1.5f
            )
        } finally {
            detector.close()
        }
    }
}

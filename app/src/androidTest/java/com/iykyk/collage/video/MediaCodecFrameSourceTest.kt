package com.iykyk.collage.video

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.frame.RotationMapper
import com.iykyk.collage.samples.SampleVideos
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MediaCodecFrameSourceTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    private fun sampleSource(): MediaCodecFrameSource {
        val sample = SampleVideos.ensureExtracted(context).first()
        return MediaCodecFrameSource(context, sample.uri, config)
    }

    @Test
    fun reportsDurationOfRoughlyThirtySeconds() {
        val duration = sampleSource().durationMs
        assertTrue("duration was $duration", duration in 25_000..35_000)
    }

    @Test
    fun emitsRoughlyTheConfiguredNumberOfFrames() = runBlocking {
        val frames = sampleSource().frames().toList()
        val expected = 30 * config.sampleFps
        assertTrue(
            "expected about $expected frames, got ${frames.size}",
            frames.size in (expected * 0.7).toInt()..(expected * 1.3).toInt()
        )
    }

    @Test
    fun timestampsIncreaseMonotonically() = runBlocking {
        val stamps = sampleSource().frames().toList().map { it.timestampMs }
        assertEquals(stamps.sorted(), stamps)
        assertEquals(stamps.distinct().size, stamps.size)
    }

    @Test
    fun downsampledFramesRespectTheAnalysisTarget() = runBlocking {
        val frame = sampleSource().frames().take(1).toList().single()
        val longEdge = maxOf(frame.nv21.width, frame.nv21.height)
        assertTrue("long edge was $longEdge", longEdge <= config.analysisLongEdge * 2)
        assertEquals(frame.nv21.width * frame.nv21.height * 3 / 2, frame.nv21.bytes.size)
    }

    @Test
    fun portraitClipReportsAnUprightPortraitFrame() = runBlocking {
        val frame = sampleSource().frames().take(1).toList().single()
        val (w, h) = RotationMapper.uprightSize(
            frame.nv21.width, frame.nv21.height, frame.rotationDegrees
        )
        assertTrue("upright frame $w x $h is not portrait", h > w)
    }

    @Test
    fun cancellingEarlyStopsDecoding() = runBlocking {
        // take(3) cancels the flow; this must return promptly rather than decode the clip.
        val frames = sampleSource().frames().take(3).toList()
        assertEquals(3, frames.size)
    }
}

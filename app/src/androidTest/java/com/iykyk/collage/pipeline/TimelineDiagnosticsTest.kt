package com.iykyk.collage.pipeline

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.frame.histogramDistance
import com.iykyk.collage.core.frame.lumaHistogram
import com.iykyk.collage.core.quality.FaceQualityScorer
import com.iykyk.collage.detect.MlKitFaceDetector
import com.iykyk.collage.samples.SampleVideos
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Per-frame timeline: how many faces were detected, how many cleared the gates, and how
 * big the scene-cut signal was.
 *
 * This distinguishes the two ways an appearance can go missing — never detected, versus
 * detected and then gated away — which need opposite fixes.
 */
@RunWith(AndroidJUnit4::class)
class TimelineDiagnosticsTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    @Test
    fun writeTimeline() = runBlocking {
        val out = StringBuilder()
        val scorer = FaceQualityScorer(config)

        SampleVideos.ensureExtracted(context).forEach { sample ->
            val detector = MlKitFaceDetector(config)
            out.appendLine("=== ${sample.label} ===")
            out.appendLine("idx      ms  det  vis   cut   maxSharp maxComp")
            var previous: IntArray? = null

            try {
                MediaCodecFrameSource(context, sample.uri, config).frames().collect { f ->
                    val h = lumaHistogram(
                        f.nv21.bytes, f.nv21.width, f.nv21.height, f.nv21.lumaRowStride
                    )
                    val cut = previous?.let { histogramDistance(it, h) } ?: 0f
                    previous = h

                    val obs = detector.detect(f)
                    val sig = obs.map { scorer.score(it) }
                    val vis = sig.count { scorer.isClearlyVisible(it) }
                    val maxSharp = sig.maxOfOrNull { it.sharpness } ?: 0f
                    val maxComp = sig.maxOfOrNull { it.completeness } ?: 0f

                    out.appendLine(
                        "%3d %7d %4d %4d  %.3f    %.3f   %.3f".format(
                            f.index, f.timestampMs, obs.size, vis, cut, maxSharp, maxComp
                        )
                    )
                }
            } finally {
                detector.close()
            }
        }

        File(context.getExternalFilesDir(null), "timeline.txt").writeText(out.toString())
    }
}

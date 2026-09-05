package com.iykyk.collage.pipeline

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.quality.FaceQualityScorer
import com.iykyk.collage.detect.MlKitFaceDetector
import com.iykyk.collage.samples.SampleVideos
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Diagnostic, not an assertion: reports why detections are being rejected.
 *
 * When the appearance count is wrong, the cause is upstream of clustering and no threshold
 * can repair it. This prints the per-gate rejection breakdown and the signal distributions
 * needed to choose gate values from data rather than by guessing.
 */
@RunWith(AndroidJUnit4::class)
class GateDiagnosticsTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    @Test
    fun reportGateRejections() = runBlocking {
        val out = StringBuilder()
        val scorer = FaceQualityScorer(config)

        SampleVideos.ensureExtracted(context).forEachIndexed { idx, sample ->
            val detector = MlKitFaceDetector(config)
            val all = mutableListOf<FaceObservation>()
            var frames = 0
            var framesWithFace = 0
            var framesWithTwo = 0

            try {
                MediaCodecFrameSource(context, sample.uri, config).frames().collect { f ->
                    frames++
                    val obs = detector.detect(f)
                    if (obs.isNotEmpty()) framesWithFace++
                    if (obs.size >= 2) framesWithTwo++
                    all += obs
                }
            } finally {
                detector.close()
            }

            out.appendLine("=== ${sample.label} ===")
            out.appendLine("frames=$frames framesWithFace=$framesWithFace framesWith2+=$framesWithTwo detections=${all.size}")

            if (all.isEmpty()) {
                out.appendLine("NO DETECTIONS AT ALL"); return@forEachIndexed
            }

            val sig = all.map { scorer.score(it) }
            fun pct(v: List<Float>, p: Int) = v.sorted()[(v.size - 1) * p / 100]
            fun line(name: String, v: List<Float>) = out.appendLine(
                "  %-12s p05=%.3f p25=%.3f p50=%.3f p75=%.3f p95=%.3f".format(
                    name, pct(v, 5), pct(v, 25), pct(v, 50), pct(v, 75), pct(v, 95)
                )
            )

            line("sharpness", sig.map { it.sharpness })
            line("frontality", sig.map { it.frontality })
            line("faceSize", sig.map { it.faceSize })
            line("completeness", sig.map { it.completeness })
            line("rawSharp", all.map { it.rawSharpness })
            line("boxWidth", all.map { it.box.width })
            out.appendLine("  frame size: ${all[0].frameWidth}x${all[0].frameHeight}")

            val fs = sig.count { it.sharpness < config.minSharpness }
            val ff = sig.count { it.frontality < config.minFrontality }
            val fz = sig.count { it.faceSize < config.minFaceSize }
            val fc = sig.count { it.completeness < config.minCompleteness }
            val visible = sig.count { scorer.isClearlyVisible(it) }
            out.appendLine("  FAIL sharpness=$fs frontality=$ff faceSize=$fz completeness=$fc")
            out.appendLine("  clearlyVisible=$visible / ${all.size}")

            // Which single gate is the binding constraint?
            val onlySharp = sig.count {
                it.sharpness < config.minSharpness && it.frontality >= config.minFrontality &&
                    it.faceSize >= config.minFaceSize && it.completeness >= config.minCompleteness
            }
            val onlyComplete = sig.count {
                it.completeness < config.minCompleteness && it.sharpness >= config.minSharpness &&
                    it.frontality >= config.minFrontality && it.faceSize >= config.minFaceSize
            }
            val onlySize = sig.count {
                it.faceSize < config.minFaceSize && it.sharpness >= config.minSharpness &&
                    it.frontality >= config.minFrontality && it.completeness >= config.minCompleteness
            }
            val onlyFront = sig.count {
                it.frontality < config.minFrontality && it.sharpness >= config.minSharpness &&
                    it.faceSize >= config.minFaceSize && it.completeness >= config.minCompleteness
            }
            out.appendLine("  SOLE blocker: sharp=$onlySharp front=$onlyFront size=$onlySize complete=$onlyComplete")
        }

        val file = File(context.getExternalFilesDir(null), "gate-diagnostics.txt")
        file.writeText(out.toString())
        println(out.toString())
    }
}

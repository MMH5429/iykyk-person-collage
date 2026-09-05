package com.iykyk.collage.pipeline

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.samples.SampleVideos
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Renders the real collage for each sample and writes it out, so the visual result can be
 * reviewed rather than merely asserted about.
 *
 * Also records per-run wall-clock time, which decides whether the 60-second demo can show
 * all three clips processing or only the first.
 */
@RunWith(AndroidJUnit4::class)
class CollageCaptureTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun captureCollagesForAllSamples() = runBlocking {
        val dir = File(context.getExternalFilesDir(null), "collages").apply { mkdirs() }
        val summary = StringBuilder()

        SampleVideos.ensureExtracted(context).forEachIndexed { index, sample ->
            val started = System.currentTimeMillis()
            val states = mutableListOf<PipelineState>()
            PersonCollagePipeline(context, PipelineConfig.Default)
                .run(sample.uri, sample.label)
                .collect { states += it }
            val elapsed = System.currentTimeMillis() - started

            val done = states.filterIsInstance<PipelineState.Done>().singleOrNull()
            assertTrue("${sample.label} did not complete: ${states.last()}", done != null)

            val analysis = done!!.result.analysis
            summary.appendLine(
                "${sample.label}: ${analysis.people.size} people, " +
                    "${analysis.totalAppearances} appearances, ${elapsed / 1000}s"
            )
            analysis.people.forEach {
                summary.appendLine("   ${it.label}: ${it.appearanceCount} appearances")
            }

            File(dir, "sample${index + 1}.png").outputStream().use { out ->
                done.result.collage.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
        }

        File(dir, "summary.txt").writeText(summary.toString())
        println(summary.toString())
    }
}

package com.iykyk.collage.pipeline

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
 * Not a test of behaviour — a data exporter.
 *
 * Running it writes one analysis dump per sample into the app's external files dir, which
 * is then pulled into `src/test/resources/dumps` so `ThresholdSweepTest` can replay
 * clustering offline instead of rebuilding the APK for every threshold experiment.
 */
@RunWith(AndroidJUnit4::class)
class DumpAnalysisTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun writeDumpsForAllSamples() = runBlocking {
        val outDir = File(context.getExternalFilesDir(null), "dumps").apply { mkdirs() }

        SampleVideos.ensureExtracted(context).forEachIndexed { index, sample ->
            val pipeline = PersonCollagePipeline(context, PipelineConfig.Default)
            pipeline.run(sample.uri, sample.label).collect { }

            val dump = pipeline.lastDump
            assertTrue("no dump produced for ${sample.label}", dump != null)
            File(outDir, "sample${index + 1}.json").writeText(dump!!.toJson())
        }
    }
}

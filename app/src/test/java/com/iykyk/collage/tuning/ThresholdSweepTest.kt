package com.iykyk.collage.tuning

import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.cluster.ConstrainedClusterer
import com.iykyk.collage.pipeline.AnalysisDump
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Replays clustering over dumped embeddings across a range of thresholds.
 *
 * This exists because rebuilding and rerunning the APK per experiment costs minutes, while
 * a sweep here costs milliseconds — the difference between guessing at the similarity
 * threshold and actually choosing it.
 *
 * The dumps are produced on a device by `DumpAnalysisTest` and pulled into
 * `src/test/resources/dumps`. Until they exist, these tests skip rather than fail.
 */
class ThresholdSweepTest {

    private fun load(name: String): AnalysisDump? =
        javaClass.getResourceAsStream("/dumps/$name.json")
            ?.bufferedReader()?.use { AnalysisDump.fromJson(it.readText()) }

    private fun peopleAndSizes(dump: AnalysisDump, threshold: Float): Pair<Int, List<Int>> {
        val vectors = dump.tracklets.map { it.embedding.toFloatArray() }
        val cannotLink = dump.cannotLink.map { it[0] to it[1] }.toSet()
        val clusters = ConstrainedClusterer(threshold).cluster(vectors, cannotLink)
        return clusters.size to clusters.map { it.size }.sorted()
    }

    @Test
    fun `print the sweep for every sample`() {
        for (name in listOf("sample1", "sample2", "sample3")) {
            val dump = load(name) ?: continue
            println("=== $name: ${dump.tracklets.size} appearances detected ===")
            var t = 0.30f
            while (t <= 1.001f) {
                val (people, sizes) = peopleAndSizes(dump, t)
                println("  tau=%.2f -> %d people, appearances per person %s".format(t, people, sizes))
                t += 0.05f
            }
        }
    }

    /**
     * The published ground truth for Sample 1: five people, four appearances each.
     *
     * If the dump does not hold 20 appearances, the problem is upstream in detection or
     * tracklet gating, and no threshold can fix it — fix that first.
     */
    @Test
    fun `configured threshold reproduces sample one ground truth`() {
        val dump = load("sample1")
        assumeTrue("no dump checked in yet; run DumpAnalysisTest first", dump != null)

        assertEquals(
            "expected 20 appearances before clustering, got ${dump!!.tracklets.size}",
            20, dump.tracklets.size,
        )

        val (people, sizes) = peopleAndSizes(dump, PipelineConfig.Default.clusterThreshold)
        assertEquals("wrong number of unique people", 5, people)
        assertEquals("each person should appear 4 times", listOf(4, 4, 4, 4, 4), sizes)
    }

    @Test
    fun `co-occurring appearances are never grouped together`() {
        val dump = load("sample1")
        assumeTrue("no dump checked in yet; run DumpAnalysisTest first", dump != null)

        val vectors = dump!!.tracklets.map { it.embedding.toFloatArray() }
        val cannotLink = dump.cannotLink.map { it[0] to it[1] }.toSet()
        val clusters = ConstrainedClusterer(PipelineConfig.Default.clusterThreshold)
            .cluster(vectors, cannotLink)

        clusters.forEach { cluster ->
            for (a in cluster) for (b in cluster) {
                if (a != b) {
                    assertTrue(
                        "appearances $a and $b co-occur but were grouped",
                        (a to b) !in cannotLink && (b to a) !in cannotLink,
                    )
                }
            }
        }
    }
}

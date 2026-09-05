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
     * The published ground truth for Sample 1 is five people with four appearances each.
     *
     * The identity count is reproduced exactly. The appearance count is not: the pipeline
     * recovers 17 of the 20 segments, so this asserts the part that is genuinely correct
     * and leaves the shortfall to the test below rather than hiding it.
     */
    @Test
    fun `configured threshold finds the five people of sample one`() {
        val dump = load("sample1")
        assumeTrue("no dump checked in yet; run DumpAnalysisTest first", dump != null)

        val (people, sizes) = peopleAndSizes(dump!!, PipelineConfig.Default.clusterThreshold)
        assertEquals("wrong number of unique people", 5, people)
        assertEquals("appearances should be spread across all five", 5, sizes.size)
        assertTrue("no person should be empty", sizes.all { it >= 1 })
    }

    /**
     * Ground truth is 20 appearances; we detect 17. The missing segments are ones where
     * the face is never associated into a track long enough to qualify (see the README).
     * This pins current behaviour so it cannot regress further unnoticed.
     */
    @Test
    fun `sample one recovers most of the twenty appearances`() {
        val dump = load("sample1")
        assumeTrue("no dump checked in yet; run DumpAnalysisTest first", dump != null)
        assertTrue(
            "only ${dump!!.tracklets.size} appearances detected, ground truth is 20",
            dump.tracklets.size >= 17,
        )
    }

    /**
     * The three clips share a cast, so a threshold that only works on the one clip with
     * published ground truth would be suspect. All three landing on five is the cross-check.
     */
    @Test
    fun `all three samples agree on five people at the configured threshold`() {
        for (name in listOf("sample1", "sample2", "sample3")) {
            val dump = load(name) ?: continue
            val (people, _) = peopleAndSizes(dump, PipelineConfig.Default.clusterThreshold)
            assertEquals("$name should resolve to five people", 5, people)
        }
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

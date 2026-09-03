package com.iykyk.collage.core.cluster

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class ConstrainedClustererTest {

    /** A unit vector in the plane at [deg] degrees, embedded in a 4-d space. */
    private fun v(deg: Double): FloatArray {
        val r = Math.toRadians(deg)
        return floatArrayOf(cos(r).toFloat(), sin(r).toFloat(), 0f, 0f).l2Normalized()
    }

    private fun clusterOf(result: List<List<Int>>, index: Int): List<Int> =
        result.first { index in it }

    @Test
    fun `cosine distance is zero for identical unit vectors and one for orthogonal`() {
        assertEquals(0f, cosineDistance(v(0.0), v(0.0)), 1e-4f)
        assertEquals(1f, cosineDistance(v(0.0), v(90.0)), 1e-4f)
    }

    @Test
    fun `l2 normalisation gives unit length`() {
        val n = floatArrayOf(3f, 4f, 0f, 0f).l2Normalized()
        assertEquals(1f, n.fold(0f) { acc, x -> acc + x * x }, 1e-5f)
    }

    @Test
    fun `two tight groups become two clusters`() {
        // 0,1 near 0 degrees; 2,3 near 90 degrees.
        val result = ConstrainedClusterer(threshold = 0.2f)
            .cluster(listOf(v(0.0), v(4.0), v(90.0), v(94.0)), emptySet())
        assertEquals(2, result.size)
        assertEquals(listOf(0, 1), clusterOf(result, 0))
        assertEquals(listOf(2, 3), clusterOf(result, 2))
    }

    @Test
    fun `everything merges when the threshold is loose`() {
        val result = ConstrainedClusterer(threshold = 1.9f)
            .cluster(listOf(v(0.0), v(45.0), v(90.0)), emptySet())
        assertEquals(1, result.size)
    }

    @Test
    fun `nothing merges when the threshold is tight`() {
        val result = ConstrainedClusterer(threshold = 0.0001f)
            .cluster(listOf(v(0.0), v(10.0), v(20.0)), emptySet())
        assertEquals(3, result.size)
    }

    @Test
    fun `a cannot link pair is never merged`() {
        // Identical vectors, but they were on screen at the same time.
        val result = ConstrainedClusterer(threshold = 0.5f)
            .cluster(listOf(v(0.0), v(0.0)), setOf(0 to 1))
        assertEquals(2, result.size)
    }

    @Test
    fun `cannot link propagates through merges`() {
        // A, B, C are all identical. A cannot link with C.
        // A+B may merge, but C must then stay out of that cluster.
        val result = ConstrainedClusterer(threshold = 0.5f)
            .cluster(listOf(v(0.0), v(0.0), v(0.0)), setOf(0 to 2))
        assertEquals(2, result.size)
        assertTrue(2 !in clusterOf(result, 0))
    }

    @Test
    fun `two people alternating on screen stay separate`() {
        // The Sample 1 shape: person A and person B each appear twice, sharing the
        // frame once. Their embeddings are close-ish but the overlap forbids merging.
        val vectors = listOf(v(0.0), v(30.0), v(3.0), v(33.0))  // A1, B1, A2, B2
        val cannotLink = setOf(0 to 1, 2 to 3)                  // A1|B1 and A2|B2 co-occur
        val result = ConstrainedClusterer(threshold = 0.15f).cluster(vectors, cannotLink)
        assertEquals(2, result.size)
        assertEquals(listOf(0, 2), clusterOf(result, 0))
        assertEquals(listOf(1, 3), clusterOf(result, 1))
    }

    @Test
    fun `average linkage is used rather than single linkage`() {
        // A chain 0 - 1 - 2 where neighbours are close but the ends are far apart.
        // Single linkage would merge all three; average linkage must not.
        val result = ConstrainedClusterer(threshold = 0.02f)
            .cluster(listOf(v(0.0), v(10.0), v(20.0)), emptySet())
        assertTrue(result.size >= 2)
    }

    @Test
    fun `empty input yields no clusters`() {
        assertEquals(0, ConstrainedClusterer(0.5f).cluster(emptyList(), emptySet()).size)
    }

    @Test
    fun `a single vector yields one cluster`() {
        assertEquals(listOf(listOf(0)), ConstrainedClusterer(0.5f).cluster(listOf(v(0.0)), emptySet()))
    }
}

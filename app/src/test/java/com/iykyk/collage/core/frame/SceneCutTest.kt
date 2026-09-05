package com.iykyk.collage.core.frame

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SceneCutTest {

    private fun frame(width: Int, height: Int, value: (Int, Int) -> Int): ByteArray {
        val out = ByteArray(width * height)
        for (y in 0 until height) for (x in 0 until width) out[y * width + x] = value(x, y).toByte()
        return out
    }

    private fun hist(bytes: ByteArray, w: Int, h: Int) = lumaHistogram(bytes, w, h, w, step = 1)

    @Test
    fun `histogram has the expected bin count and total`() {
        val h = hist(frame(16, 16) { _, _ -> 128 }, 16, 16)
        assertEquals(HISTOGRAM_BINS, h.size)
        assertEquals(256, h.sum())
    }

    @Test
    fun `identical frames have zero distance`() {
        val a = hist(frame(16, 16) { x, y -> (x * 16 + y) % 256 }, 16, 16)
        assertEquals(0f, histogramDistance(a, a), 1e-5f)
    }

    @Test
    fun `a black frame and a white frame are maximally distant`() {
        val black = hist(frame(16, 16) { _, _ -> 0 }, 16, 16)
        val white = hist(frame(16, 16) { _, _ -> 255 }, 16, 16)
        assertEquals(1f, histogramDistance(black, white), 1e-5f)
    }

    @Test
    fun `a small brightness shift over a spread of tones is a small distance`() {
        // A flat frame is a degenerate case  every pixel shares one bin, so nudging it
        // across a bucket boundary looks maximally different. Real frames have a spread,
        // where a small shift moves only the pixels near each boundary.
        val a = hist(frame(64, 64) { x, y -> (x * 4 + y) % 256 }, 64, 64)
        val b = hist(frame(64, 64) { x, y -> (x * 4 + y + 2) % 256 }, 64, 64)
        assertTrue("distance was ${histogramDistance(a, b)}", histogramDistance(a, b) < 0.2f)
    }

    @Test
    fun `a real cut scores far above a gradual pan`() {
        // A pan shifts content but keeps the overall tone; a cut replaces everything.
        val shot = hist(frame(32, 32) { x, _ -> if (x < 16) 40 else 60 }, 32, 32)
        val panned = hist(frame(32, 32) { x, _ -> if (x < 12) 40 else 60 }, 32, 32)
        val cut = hist(frame(32, 32) { _, _ -> 230 }, 32, 32)

        assertTrue(histogramDistance(shot, cut) > histogramDistance(shot, panned) + 0.5f)
    }

    @Test
    fun `subsampling does not change the distribution much`() {
        val bytes = frame(64, 64) { x, y -> (x + y) % 256 }
        val full = lumaHistogram(bytes, 64, 64, 64, step = 1)
        val sub = lumaHistogram(bytes, 64, 64, 64, step = 4)
        assertTrue(histogramDistance(full, sub) < 0.1f)
    }

    @Test
    fun `empty histograms are treated as identical rather than dividing by zero`() {
        assertEquals(0f, histogramDistance(IntArray(HISTOGRAM_BINS), IntArray(HISTOGRAM_BINS)), 1e-6f)
    }
}

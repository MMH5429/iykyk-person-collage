package com.iykyk.collage.core.frame

import org.junit.Assert.assertEquals
import org.junit.Test

class Nv21Test {

    /** A y-plane where every pixel encodes its own x coordinate, for easy assertions. */
    private fun yPlane(w: Int, h: Int, stride: Int = w) = PlaneData(
        buffer = ByteArray(stride * h) { i -> ((i % stride) % 256).toByte() },
        rowStride = stride,
        pixelStride = 1,
    )

    private fun uvPlane(w: Int, h: Int, fill: Int) = PlaneData(
        buffer = ByteArray((w / 2) * (h / 2)) { fill.toByte() },
        rowStride = w / 2,
        pixelStride = 1,
    )

    @Test
    fun `chooseStep picks the smallest step that fits the target`() {
        assertEquals(1, chooseStep(540, 960, targetLongEdge = 960))
        assertEquals(2, chooseStep(1080, 1920, targetLongEdge = 960))
        assertEquals(2, chooseStep(1080, 1920, targetLongEdge = 540))
        assertEquals(4, chooseStep(2160, 3840, targetLongEdge = 960))
        assertEquals(1, chooseStep(320, 240, targetLongEdge = 960))
    }

    @Test
    fun `step of one preserves dimensions and luma`() {
        val out = downsampleToNv21(yPlane(8, 4), uvPlane(8, 4, 128), uvPlane(8, 4, 200), 8, 4, step = 1)
        assertEquals(8, out.width)
        assertEquals(4, out.height)
        assertEquals(8 * 4 + 8 * 4 / 2, out.bytes.size)
        for (x in 0 until 8) assertEquals(x, out.bytes[x].toInt() and 0xFF)
    }

    @Test
    fun `step of two halves the dimensions and samples every other pixel`() {
        val out = downsampleToNv21(yPlane(8, 4), uvPlane(8, 4, 128), uvPlane(8, 4, 200), 8, 4, step = 2)
        assertEquals(4, out.width)
        assertEquals(2, out.height)
        for (i in 0 until 4) assertEquals(i * 2, out.bytes[i].toInt() and 0xFF)
    }

    @Test
    fun `output size matches the nv21 contract`() {
        val out = downsampleToNv21(
            yPlane(1080, 1920), uvPlane(1080, 1920, 128), uvPlane(1080, 1920, 128), 1080, 1920, step = 2
        )
        assertEquals(540, out.width)
        assertEquals(960, out.height)
        assertEquals(540 * 960 * 3 / 2, out.bytes.size)
    }

    @Test
    fun `chroma is interleaved as v then u`() {
        val out = downsampleToNv21(yPlane(4, 4), uvPlane(4, 4, 111), uvPlane(4, 4, 222), 4, 4, step = 1)
        val chromaStart = 4 * 4
        assertEquals(222, out.bytes[chromaStart].toInt() and 0xFF)      // V first
        assertEquals(111, out.bytes[chromaStart + 1].toInt() and 0xFF)  // then U
    }

    @Test
    fun `row stride padding is respected`() {
        // A plane padded to stride 16 for an 8-wide image must still read the right pixels.
        val out = downsampleToNv21(yPlane(8, 4, stride = 16), uvPlane(8, 4, 128), uvPlane(8, 4, 128), 8, 4, step = 1)
        for (x in 0 until 8) assertEquals(x, out.bytes[x].toInt() and 0xFF)
    }
}

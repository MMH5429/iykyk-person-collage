package com.iykyk.collage.core.quality

import com.iykyk.collage.core.model.BoxF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SharpnessTest {

    private val width = 32
    private val height = 32
    private val region = BoxF(0f, 0f, 32f, 32f)

    private fun luma(pixel: (x: Int, y: Int) -> Int): ByteArray {
        val out = ByteArray(width * height)
        for (y in 0 until height) for (x in 0 until width) {
            out[y * width + x] = pixel(x, y).toByte()
        }
        return out
    }

    @Test
    fun `flat image has near zero laplacian variance`() {
        val flat = luma { _, _ -> 128 }
        assertEquals(0f, laplacianVarianceFromLuma(flat, width, region), 1e-3f)
    }

    @Test
    fun `checkerboard has high laplacian variance`() {
        val checker = luma { x, y -> if ((x + y) % 2 == 0) 0 else 255 }
        assertTrue(laplacianVarianceFromLuma(checker, width, region) > 1000f)
    }

    @Test
    fun `blurred edge scores lower than hard edge`() {
        val hard = luma { x, _ -> if (x < 16) 0 else 255 }
        val soft = luma { x, _ -> ((x.toFloat() / width) * 255f).toInt() }
        assertTrue(
            laplacianVarianceFromLuma(soft, width, region) <
                laplacianVarianceFromLuma(hard, width, region)
        )
    }

    @Test
    fun `region restricts the measurement`() {
        // Sharp only on the left half; measuring the right half must be flat.
        val half = luma { x, y -> if (x < 16) (if ((x + y) % 2 == 0) 0 else 255) else 128 }
        val right = BoxF(17f, 1f, 31f, 31f)
        assertEquals(0f, laplacianVarianceFromLuma(half, width, right), 1e-3f)
    }

    @Test
    fun `degenerate region returns zero`() {
        val flat = luma { _, _ -> 128 }
        assertEquals(0f, laplacianVarianceFromLuma(flat, width, BoxF(5f, 5f, 6f, 6f)), 1e-3f)
    }
}

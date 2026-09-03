package com.iykyk.collage.core.quality

import com.iykyk.collage.core.model.BoxF
import kotlin.math.roundToInt

/**
 * Variance of the 4-neighbour Laplacian over [region] of a single-channel luma plane.
 *
 * Low variance means few high-frequency details, i.e. a soft or motion-blurred face.
 * This is what excludes whip-pan passes: a blurred face scores near zero and never
 * counts as "clearly visible".
 *
 * Operates directly on the Y plane of YUV_420_888, so no colour conversion is needed.
 */
fun laplacianVarianceFromLuma(luma: ByteArray, rowStride: Int, region: BoxF): Float {
    // Interior pixels only — the kernel needs one pixel of margin on every side.
    val x0 = region.left.roundToInt().coerceAtLeast(1)
    val y0 = region.top.roundToInt().coerceAtLeast(1)
    val x1 = region.right.roundToInt().coerceAtMost(rowStride - 2)
    val y1 = region.bottom.roundToInt().coerceAtMost(luma.size / rowStride - 2)
    if (x1 - x0 < 2 || y1 - y0 < 2) return 0f

    var sum = 0.0
    var sumSq = 0.0
    var n = 0

    for (y in y0..y1) {
        val row = y * rowStride
        val up = row - rowStride
        val down = row + rowStride
        for (x in x0..x1) {
            val c = luma[row + x].toInt() and 0xFF
            val l = luma[row + x - 1].toInt() and 0xFF
            val r = luma[row + x + 1].toInt() and 0xFF
            val u = luma[up + x].toInt() and 0xFF
            val d = luma[down + x].toInt() and 0xFF
            val lap = (l + r + u + d - 4 * c).toDouble()
            sum += lap
            sumSq += lap * lap
            n++
        }
    }
    if (n == 0) return 0f
    val mean = sum / n
    return ((sumSq / n) - mean * mean).toFloat().coerceAtLeast(0f)
}

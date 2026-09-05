package com.iykyk.collage.core.frame

import kotlin.math.abs

/** Number of luma buckets. Coarse on purpose: this detects cuts, not subtle grading. */
const val HISTOGRAM_BINS = 32

/**
 * Coarse luma histogram of a frame, sampling every [step]-th pixel in each direction.
 *
 * Subsampling keeps this cheap enough to run on every analysed frame; a cut changes the
 * whole picture, so a 1-in-16 sample is more than enough to see it.
 */
fun lumaHistogram(
    luma: ByteArray,
    width: Int,
    height: Int,
    rowStride: Int,
    step: Int = 4,
): IntArray {
    val bins = IntArray(HISTOGRAM_BINS)
    // 8-bit luma into a power-of-two bin count: 32 bins means shifting right by 3.
    val shift = 8 - HISTOGRAM_BINS.countTrailingZeroBits()
    var y = 0
    while (y < height) {
        val row = y * rowStride
        var x = 0
        while (x < width) {
            val v = luma[row + x].toInt() and 0xFF
            bins[v shr shift]++
            x += step
        }
        y += step
    }
    return bins
}

/**
 * Normalised L1 distance between two histograms, in [0, 1].
 *
 * 0 means identical distributions, 1 means no overlap at all.
 */
fun histogramDistance(a: IntArray, b: IntArray): Float {
    require(a.size == b.size) { "histogram sizes differ: ${a.size} vs ${b.size}" }
    val totalA = a.sum()
    val totalB = b.sum()
    if (totalA == 0 || totalB == 0) return 0f

    var diff = 0f
    for (i in a.indices) {
        diff += abs(a[i].toFloat() / totalA - b[i].toFloat() / totalB)
    }
    // L1 between two probability distributions maxes out at 2.
    return (diff / 2f).coerceIn(0f, 1f)
}

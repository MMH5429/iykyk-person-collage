package com.iykyk.collage.core.frame

/** One plane of a YUV_420_888 image, copied out of its (possibly padded) direct buffer. */
class PlaneData(
    val buffer: ByteArray,
    val rowStride: Int,
    val pixelStride: Int,
)

/** A tightly packed NV21 image: full-resolution luma followed by interleaved V,U. */
class Nv21Buffer(val bytes: ByteArray, val width: Int, val height: Int) {
    /** Luma is packed, so the row stride is simply the width. */
    val lumaRowStride: Int get() = width
}

/**
 * Smallest integer subsampling step whose result still covers [targetLongEdge].
 *
 * Integer stepping keeps the inner loop to a pointer add, and never upsamples.
 */
fun chooseStep(srcWidth: Int, srcHeight: Int, targetLongEdge: Int): Int {
    val longEdge = maxOf(srcWidth, srcHeight)
    var step = 1
    while (longEdge / (step * 2) >= targetLongEdge && step < MAX_STEP) step *= 2
    return step
}

/**
 * Nearest-neighbour subsample of a YUV_420_888 image into a packed NV21 buffer.
 *
 * NV21 is what `InputImage.fromByteBuffer` wants, and it keeps the luma plane contiguous
 * so sharpness can be measured on it directly with no colour conversion.
 *
 * Output dimensions are forced even, as the 4:2:0 chroma layout requires.
 */
fun downsampleToNv21(
    y: PlaneData,
    u: PlaneData,
    v: PlaneData,
    srcWidth: Int,
    srcHeight: Int,
    step: Int,
): Nv21Buffer {
    require(step >= 1) { "step must be positive" }
    val outWidth = (srcWidth / step) and 1.inv()
    val outHeight = (srcHeight / step) and 1.inv()
    require(outWidth >= 2 && outHeight >= 2) { "downsampled frame is degenerate" }

    val lumaSize = outWidth * outHeight
    val out = ByteArray(lumaSize * 3 / 2)

    // Luma.
    var o = 0
    for (row in 0 until outHeight) {
        val srcRow = row * step * y.rowStride
        var srcCol = 0
        for (col in 0 until outWidth) {
            out[o++] = y.buffer[srcRow + srcCol]
            srcCol += step * y.pixelStride
        }
    }

    // Chroma: one V,U pair per 2x2 luma block, so the source is subsampled by step too.
    val chromaWidth = outWidth / 2
    val chromaHeight = outHeight / 2
    for (row in 0 until chromaHeight) {
        val uRow = row * step * u.rowStride
        val vRow = row * step * v.rowStride
        for (col in 0 until chromaWidth) {
            out[o++] = v.buffer.getOrElse(vRow + col * step * v.pixelStride) { NEUTRAL_CHROMA }
            out[o++] = u.buffer.getOrElse(uRow + col * step * u.pixelStride) { NEUTRAL_CHROMA }
        }
    }

    return Nv21Buffer(out, outWidth, outHeight)
}

private const val MAX_STEP = 8

/** 128 is the zero point for chroma; used when a plane is shorter than expected. */
private const val NEUTRAL_CHROMA: Byte = -128

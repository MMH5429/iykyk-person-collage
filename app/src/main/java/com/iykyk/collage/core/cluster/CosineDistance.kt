package com.iykyk.collage.core.cluster

import kotlin.math.sqrt

/** Returns a copy scaled to unit length. A zero vector is returned unchanged. */
fun FloatArray.l2Normalized(): FloatArray {
    var sumSq = 0f
    for (x in this) sumSq += x * x
    val norm = sqrt(sumSq)
    if (norm <= 1e-12f) return copyOf()
    return FloatArray(size) { this[it] / norm }
}

/**
 * Cosine distance in [0, 2]. Both vectors are assumed L2-normalised, which is what
 * `FaceEmbedder` guarantees, so this reduces to `1 - dot`.
 */
fun cosineDistance(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "embedding dimensions differ: ${a.size} vs ${b.size}" }
    var dot = 0f
    for (i in a.indices) dot += a[i] * b[i]
    return 1f - dot
}

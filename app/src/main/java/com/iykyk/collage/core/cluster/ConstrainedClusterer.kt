package com.iykyk.collage.core.cluster

/**
 * Average-linkage agglomerative clustering over embeddings, with hard cannot-link
 * constraints.
 *
 * The constraint is what makes co-occurring people resolve correctly: two tracklets that
 * are clearly visible at the same moment are, by definition, different people, so no
 * similarity score is allowed to merge them. Constraints propagate — once a cluster
 * contains a member that cannot link with x, the whole cluster cannot absorb x.
 *
 * Input sizes here are tens of tracklets, so the straightforward O(n^3) loop is far
 * cheaper than the bookkeeping needed to avoid it.
 */
class ConstrainedClusterer(private val threshold: Float) {

    /**
     * @param vectors one L2-normalised embedding per item.
     * @param cannotLink unordered index pairs that must never share a cluster.
     * @return member indices per cluster, each sorted ascending, ordered by first member.
     */
    fun cluster(vectors: List<FloatArray>, cannotLink: Set<Pair<Int, Int>>): List<List<Int>> {
        if (vectors.isEmpty()) return emptyList()

        val forbidden = HashSet<Long>(cannotLink.size * 2)
        for ((a, b) in cannotLink) {
            forbidden += key(a, b)
            forbidden += key(b, a)
        }

        val clusters = vectors.indices.map { mutableListOf(it) }.toMutableList()

        while (clusters.size > 1) {
            var bestI = -1
            var bestJ = -1
            var bestDistance = Float.MAX_VALUE

            for (i in 0 until clusters.size - 1) {
                for (j in i + 1 until clusters.size) {
                    if (isForbidden(clusters[i], clusters[j], forbidden)) continue
                    val d = averageLinkage(clusters[i], clusters[j], vectors)
                    if (d < bestDistance) {
                        bestDistance = d
                        bestI = i
                        bestJ = j
                    }
                }
            }

            if (bestI < 0 || bestDistance > threshold) break

            clusters[bestI].addAll(clusters[bestJ])
            clusters.removeAt(bestJ)
        }

        return clusters
            .map { it.sorted() }
            .sortedBy { it.first() }
    }

    private fun isForbidden(a: List<Int>, b: List<Int>, forbidden: Set<Long>): Boolean {
        for (x in a) for (y in b) if (key(x, y) in forbidden) return true
        return false
    }

    private fun averageLinkage(a: List<Int>, b: List<Int>, vectors: List<FloatArray>): Float {
        var total = 0f
        for (x in a) for (y in b) total += cosineDistance(vectors[x], vectors[y])
        return total / (a.size * b.size)
    }

    private fun key(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
}

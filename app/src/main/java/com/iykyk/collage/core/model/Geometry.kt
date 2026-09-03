package com.iykyk.collage.core.model

import kotlin.math.max
import kotlin.math.min

/** An axis-aligned rectangle in frame coordinates. Pure data — no Android types. */
data class BoxF(val left: Float, val top: Float, val right: Float, val bottom: Float) {

    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val area: Float get() = max(0f, width) * max(0f, height)
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f

    fun intersect(other: BoxF): BoxF? {
        val l = max(left, other.left)
        val t = max(top, other.top)
        val r = min(right, other.right)
        val b = min(bottom, other.bottom)
        return if (l < r && t < b) BoxF(l, t, r, b) else null
    }

    /** Intersection over union. 0 when disjoint, 1 when identical. */
    fun iou(other: BoxF): Float {
        val inter = intersect(other)?.area ?: return 0f
        val union = area + other.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** Scales the box about its own centre. A factor of 2 doubles each side. */
    fun expand(factor: Float): BoxF {
        val halfW = width * factor / 2f
        val halfH = height * factor / 2f
        return BoxF(centerX - halfW, centerY - halfH, centerX + halfW, centerY + halfH)
    }

    fun clampTo(frameWidth: Int, frameHeight: Int): BoxF = BoxF(
        left = left.coerceIn(0f, frameWidth.toFloat()),
        top = top.coerceIn(0f, frameHeight.toFloat()),
        right = right.coerceIn(0f, frameWidth.toFloat()),
        bottom = bottom.coerceIn(0f, frameHeight.toFloat()),
    )
}

data class PointF2(val x: Float, val y: Float)

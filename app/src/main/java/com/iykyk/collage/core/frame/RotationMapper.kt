package com.iykyk.collage.core.frame

import com.iykyk.collage.core.model.BoxF
import kotlin.math.max
import kotlin.math.min

/**
 * Converts between the *raw* pixel space of a decoded frame and the *upright* space a
 * viewer sees after the container's rotation metadata is applied.
 *
 * ML Kit reports face boxes in upright coordinates, but crops have to be taken from the
 * raw buffer, so every crop needs this inverse mapping.
 */
object RotationMapper {

    /** Dimensions of the frame after [rotationDeg] of clockwise rotation is applied. */
    fun uprightSize(rawW: Int, rawH: Int, rotationDeg: Int): Pair<Int, Int> =
        when (normalise(rotationDeg)) {
            90, 270 -> rawH to rawW
            else -> rawW to rawH
        }

    /**
     * Maps a box given in upright coordinates back into raw buffer coordinates.
     *
     * [rawW] and [rawH] are the dimensions of the undecoded frame, before rotation.
     */
    fun uprightToRaw(box: BoxF, rawW: Int, rawH: Int, rotationDeg: Int): BoxF {
        val corners = when (normalise(rotationDeg)) {
            0 -> listOf(box.left to box.top, box.right to box.bottom)

            // Upright (x, y) came from raw (y, rawH - x).
            90 -> listOf(
                box.top to (rawH - box.left),
                box.bottom to (rawH - box.right),
            )

            180 -> listOf(
                (rawW - box.left) to (rawH - box.top),
                (rawW - box.right) to (rawH - box.bottom),
            )

            // Upright (x, y) came from raw (rawW - y, x).
            else -> listOf(
                (rawW - box.top) to box.left,
                (rawW - box.bottom) to box.right,
            )
        }
        val (x0, y0) = corners[0]
        val (x1, y1) = corners[1]
        return BoxF(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    }

    private fun normalise(deg: Int): Int = ((deg % 360) + 360) % 360
}

package com.iykyk.collage.embed

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import com.iykyk.collage.core.model.PointF2

/**
 * Normalises a face crop into the model's square input.
 *
 * Two implementations exist so the behaviour can be compared rather than assumed:
 * alignment usually improves embeddings, but it also amplifies landmark error on small
 * faces, so the plain resize is a genuine alternative and not just a fallback.
 */
interface FaceAligner {
    /**
     * @param crop a generous face crop, already upright.
     * @param leftEye eye landmark in *crop-local* pixel coordinates, or null.
     * @param rightEye as above.
     * @param size the model's square input edge.
     */
    fun align(crop: Bitmap, leftEye: PointF2?, rightEye: PointF2?, size: Int): Bitmap
}

/** Straight resize. Robust, and the fallback whenever eye landmarks are missing. */
class PlainResizeAligner : FaceAligner {
    override fun align(crop: Bitmap, leftEye: PointF2?, rightEye: PointF2?, size: Int): Bitmap =
        Bitmap.createScaledBitmap(crop, size, size, true)
}

/**
 * Similarity transform placing both eyes at fixed canonical positions, which removes
 * in-plane rotation and scale variation before the model sees the face.
 */
class EyeAligner(private val fallback: FaceAligner = PlainResizeAligner()) : FaceAligner {

    override fun align(crop: Bitmap, leftEye: PointF2?, rightEye: PointF2?, size: Int): Bitmap {
        if (leftEye == null || rightEye == null) return fallback.align(crop, null, null, size)

        val source = floatArrayOf(leftEye.x, leftEye.y, rightEye.x, rightEye.y)
        val target = floatArrayOf(
            LEFT_EYE_X * size, EYE_Y * size,
            RIGHT_EYE_X * size, EYE_Y * size,
        )

        val matrix = Matrix()
        // Two point pairs give a similarity transform: rotation, uniform scale, translation.
        if (!matrix.setPolyToPoly(source, 0, target, 0, 2)) {
            return fallback.align(crop, null, null, size)
        }

        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(
            crop, matrix, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        )
        return out
    }

    private companion object {
        // Canonical FaceNet eye placement: eyes on the upper third, symmetric about centre.
        const val LEFT_EYE_X = 0.34f
        const val RIGHT_EYE_X = 0.66f
        const val EYE_Y = 0.38f
    }
}

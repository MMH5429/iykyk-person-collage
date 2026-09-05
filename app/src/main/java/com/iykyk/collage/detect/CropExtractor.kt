package com.iykyk.collage.detect

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import com.iykyk.collage.core.frame.RotationMapper
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.video.AnalysisFrame
import java.io.ByteArrayOutputStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Cuts an upright RGB crop around a detected face.
 *
 * Only the crop region is colour-converted, so this stays cheap even though it runs for
 * every candidate shot. The crop is deliberately generous — the assignment calls out tight
 * bounding-box crops as a source of low-resolution, ugly tiles.
 */
object CropExtractor {

    /**
     * @param neighbours other faces detected in the same frame. The crop is shrunk so it
     *   never reaches another person's face. These clips use side-by-side two-shots, where
     *   a generous square crop otherwise swallows the neighbour — which both looks wrong in
     *   a collage tile and, far worse, feeds two people into one embedding.
     */
    fun faceCrop(
        frame: AnalysisFrame,
        obs: FaceObservation,
        expandFactor: Float,
        neighbours: List<FaceObservation> = emptyList(),
    ): Bitmap {
        // Square, generous, centred on the face, in upright coordinates.
        val side = croppedSide(obs, expandFactor, neighbours)
        val upright = BoxF(
            obs.box.centerX - side / 2f,
            obs.box.centerY - side / 2f,
            obs.box.centerX + side / 2f,
            obs.box.centerY + side / 2f,
        ).clampTo(obs.frameWidth, obs.frameHeight)

        val raw = RotationMapper
            .uprightToRaw(upright, frame.nv21.width, frame.nv21.height, frame.rotationDegrees)
            .clampTo(frame.nv21.width, frame.nv21.height)

        val rect = Rect(
            raw.left.roundToInt(),
            raw.top.roundToInt(),
            raw.right.roundToInt(),
            raw.bottom.roundToInt(),
        )
        require(rect.width() > 1 && rect.height() > 1) { "degenerate crop rect $rect" }

        val yuv = YuvImage(
            frame.nv21.bytes, ImageFormat.NV21, frame.nv21.width, frame.nv21.height, null
        )
        val jpeg = ByteArrayOutputStream()
            .also { yuv.compressToJpeg(rect, JPEG_QUALITY, it) }
            .toByteArray()
        val rawBitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
            ?: error("could not decode the cropped region")

        if (frame.rotationDegrees % 360 == 0) return rawBitmap
        val matrix = Matrix().apply { postRotate(frame.rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(
            rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true
        )
        if (rotated !== rawBitmap) rawBitmap.recycle()
        return rotated
    }

    /**
     * The same generous, neighbour-aware crop, but taken from an already-decoded
     * full-resolution frame instead of the downsampled analysis buffer.
     *
     * Detection runs at ~540 px for speed, which leaves the face crops too soft for the
     * embedding model to separate similar-looking people. Re-cropping the handful of frames
     * that actually get embedded from full resolution costs a few seeks and gives the model
     * real detail to work with.
     *
     * [source] is expected upright and the same aspect as the analysis frame.
     */
    fun faceCropFromFullFrame(
        source: Bitmap,
        obs: FaceObservation,
        expandFactor: Float,
        neighbours: List<FaceObservation> = emptyList(),
    ): Bitmap? {
        val scale = source.width.toFloat() / obs.frameWidth
        if (scale <= 0f) return null

        val side = croppedSide(obs, expandFactor, neighbours) * scale
        val cx = obs.box.centerX * scale
        val cy = obs.box.centerY * scale
        val rect = BoxF(cx - side / 2f, cy - side / 2f, cx + side / 2f, cy + side / 2f)
            .clampTo(source.width, source.height)

        val left = rect.left.roundToInt()
        val top = rect.top.roundToInt()
        val width = (rect.right - rect.left).roundToInt()
        val height = (rect.bottom - rect.top).roundToInt()
        if (width < 2 || height < 2) return null

        return Bitmap.createBitmap(source, left, top, width, height)
    }

    /**
     * The crop side, reduced until it excludes every neighbouring face centre.
     *
     * A square centred on this face contains another face's centre only when both the
     * horizontal and vertical offsets fall inside the half-side, so keeping the half-side
     * below the larger of the two offsets is enough to exclude it. Never shrinks below
     * [MIN_EXPAND] times the face box, since the face itself must always fit.
     */
    fun croppedSide(
        obs: FaceObservation,
        expandFactor: Float,
        neighbours: List<FaceObservation>,
    ): Float {
        val faceSide = max(obs.box.width, obs.box.height)
        var halfSide = faceSide * expandFactor / 2f

        for (other in neighbours) {
            if (other === obs) continue
            val dx = abs(other.box.centerX - obs.box.centerX)
            val dy = abs(other.box.centerY - obs.box.centerY)
            val separation = max(dx, dy)
            if (separation > 0f) halfSide = min(halfSide, separation * NEIGHBOUR_MARGIN)
        }

        return max(halfSide * 2f, faceSide * MIN_EXPAND)
    }

    /** High enough that the embedding is unaffected; the tile itself comes from full res. */
    private const val JPEG_QUALITY = 95

    /** Stop just short of the neighbour rather than exactly at them. */
    private const val NEIGHBOUR_MARGIN = 0.9f

    /** A crop must always contain the whole face, however close the neighbour is. */
    private const val MIN_EXPAND = 1.1f
}

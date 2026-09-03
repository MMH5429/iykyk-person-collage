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
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Cuts an upright RGB crop around a detected face.
 *
 * Only the crop region is colour-converted, so this stays cheap even though it runs for
 * every candidate shot. The crop is deliberately generous — the assignment calls out tight
 * bounding-box crops as a source of low-resolution, ugly tiles.
 */
object CropExtractor {

    fun faceCrop(frame: AnalysisFrame, obs: FaceObservation, expandFactor: Float): Bitmap {
        // Square, generous, centred on the face, in upright coordinates.
        val side = max(obs.box.width, obs.box.height) * expandFactor
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

    /** High enough that the embedding is unaffected; the tile itself comes from full res. */
    private const val JPEG_QUALITY = 95
}

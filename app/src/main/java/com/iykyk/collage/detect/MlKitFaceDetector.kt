package com.iykyk.collage.detect

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.frame.RotationMapper
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.PointF2
import com.iykyk.collage.core.quality.laplacianVarianceFromLuma
import com.iykyk.collage.video.AnalysisFrame
import kotlinx.coroutines.tasks.await
import java.nio.ByteBuffer

/**
 * Wraps ML Kit face detection and turns each detected face into a plain [FaceObservation]
 * in upright coordinates.
 *
 * Detection is stateless and reproducible: see the tracking note on the options below.
 * Tracklets are formed downstream by IoU association, not by ML Kit's tracker.
 */
class MlKitFaceDetector(private val config: PipelineConfig) {

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            // FAST rather than ACCURATE: with tracking off, every frame is a full detection,
            // and ACCURATE cost ~150s per 30s clip on a 2018 device. The subjects here are
            // large, frontal, foreground faces, which FAST handles well.
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(MIN_FACE_SIZE)
            // Tracking is deliberately NOT enabled. It puts ML Kit into stateful stream
            // mode, which is tuned for a live camera and adapts to CPU load by skipping
            // work - so the same video analysed twice produced different detections, and
            // the appearance count swung between 12 and 19 across runs of one clip.
            // Without it each frame is processed independently and the result is
            // reproducible. Nothing is lost: tracklets are built by IoU association, and
            // the tracking id was only ever a tie-breaking bonus.
            .build()
    )

    suspend fun detect(frame: AnalysisFrame): List<FaceObservation> {
        val image = InputImage.fromByteBuffer(
            ByteBuffer.wrap(frame.nv21.bytes),
            frame.nv21.width,
            frame.nv21.height,
            frame.rotationDegrees,
            InputImage.IMAGE_FORMAT_NV21,
        )
        val faces = detector.process(image).await()
        if (faces.isEmpty()) return emptyList()

        val (uprightW, uprightH) = RotationMapper.uprightSize(
            frame.nv21.width, frame.nv21.height, frame.rotationDegrees
        )

        return faces.map { face -> face.toObservation(frame, uprightW, uprightH) }
    }

    private fun Face.toObservation(
        frame: AnalysisFrame,
        uprightW: Int,
        uprightH: Int,
    ): FaceObservation {
        val box = BoxF(
            boundingBox.left.toFloat(),
            boundingBox.top.toFloat(),
            boundingBox.right.toFloat(),
            boundingBox.bottom.toFloat(),
        )

        // Sharpness is measured on the luma plane, which is in raw orientation.
        val rawRegion = RotationMapper
            .uprightToRaw(box, frame.nv21.width, frame.nv21.height, frame.rotationDegrees)
            .clampTo(frame.nv21.width, frame.nv21.height)

        return FaceObservation(
            frameIndex = frame.index,
            timestampMs = frame.timestampMs,
            trackingId = trackingId,
            // Deliberately NOT clamped to the frame: how far the box extends past the edge
            // is exactly what the completeness signal measures. Consumers that need pixel
            // coordinates (crop extraction, sharpness regions) clamp for themselves.
            box = box,
            leftEye = getLandmark(FaceLandmark.LEFT_EYE)?.position?.let { PointF2(it.x, it.y) },
            rightEye = getLandmark(FaceLandmark.RIGHT_EYE)?.position?.let { PointF2(it.x, it.y) },
            headEulerX = headEulerAngleX,
            headEulerY = headEulerAngleY,
            headEulerZ = headEulerAngleZ,
            leftEyeOpenProb = leftEyeOpenProbability,
            rightEyeOpenProb = rightEyeOpenProbability,
            smileProb = smilingProbability,
            rawSharpness = laplacianVarianceFromLuma(
                luma = frame.nv21.bytes,
                rowStride = frame.nv21.lumaRowStride,
                region = rawRegion,
            ),
            frameWidth = uprightW,
            frameHeight = uprightH,
        )
    }

    fun close() = detector.close()

    private companion object {
        /** Faces smaller than this fraction of the frame are background, not subjects. */
        const val MIN_FACE_SIZE = 0.08f
    }
}

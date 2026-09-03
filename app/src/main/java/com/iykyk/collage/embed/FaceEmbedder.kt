package com.iykyk.collage.embed

import android.content.Context
import android.graphics.Bitmap
import com.iykyk.collage.core.cluster.l2Normalized
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.sqrt

/**
 * FaceNet-512 (Inception-ResNet-v1) running on-device through TFLite.
 *
 * Input and output shapes are read from the model at load time rather than hardcoded, so
 * swapping in a different embedding model needs no code change here.
 */
class FaceEmbedder(context: Context, threads: Int = DEFAULT_THREADS) : Closeable {

    private val interpreter: Interpreter = Interpreter(
        loadModel(context),
        Interpreter.Options().apply {
            numThreads = threads
            setUseXNNPACK(true)
        }
    )

    /** Square input edge the model expects, e.g. 160. */
    val inputSize: Int = interpreter.getInputTensor(0).shape()[1]

    /** Embedding dimensionality, e.g. 512. */
    val dimensions: Int = interpreter.getOutputTensor(0).shape()[1]

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(inputSize * inputSize * CHANNELS * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

    private val pixels = IntArray(inputSize * inputSize)

    /** Returns the L2-normalised embedding of a single aligned face. */
    fun embed(face: Bitmap): FloatArray {
        val scaled = if (face.width == inputSize && face.height == inputSize) {
            face
        } else {
            Bitmap.createScaledBitmap(face, inputSize, inputSize, true)
        }

        writeStandardised(scaled)
        if (scaled !== face) scaled.recycle()

        val output = Array(1) { FloatArray(dimensions) }
        interpreter.run(inputBuffer, output)
        return output[0].l2Normalized()
    }

    /**
     * Mean of several embeddings, renormalised.
     *
     * Averaging a tracklet's best crops is what makes identity robust: a single bad frame
     * can no longer invent a person.
     */
    fun embedAveraged(faces: List<Bitmap>): FloatArray {
        require(faces.isNotEmpty()) { "cannot average zero embeddings" }
        val sum = FloatArray(dimensions)
        for (face in faces) {
            val v = embed(face)
            for (i in sum.indices) sum[i] += v[i]
        }
        return sum.l2Normalized()
    }

    /**
     * FaceNet expects per-image standardisation — (pixel - mean) / adjusted stddev — not a
     * [0,1] or [-1,1] rescale. Getting this wrong degrades similarity silently, so it is
     * worth being explicit about.
     */
    private fun writeStandardised(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        var sum = 0.0
        var sumSq = 0.0
        val n = pixels.size * CHANNELS
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            sum += (r + g + b).toDouble()
            sumSq += (r * r + g * g + b * b).toDouble()
        }
        val mean = sum / n
        val variance = (sumSq / n) - mean * mean
        // The 1/sqrt(N) floor mirrors TensorFlow's per_image_standardization.
        val std = max(sqrt(max(variance, 0.0)), 1.0 / sqrt(n.toDouble())).toFloat()
        val meanF = mean.toFloat()

        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.putFloat((((p shr 16) and 0xFF) - meanF) / std)
            inputBuffer.putFloat((((p shr 8) and 0xFF) - meanF) / std)
            inputBuffer.putFloat(((p and 0xFF) - meanF) / std)
        }
        inputBuffer.rewind()
    }

    override fun close() = interpreter.close()

    private companion object {
        const val MODEL_ASSET = "facenet_512.tflite"
        const val CHANNELS = 3
        const val DEFAULT_THREADS = 4

        fun loadModel(context: Context): ByteBuffer =
            context.assets.openFd(MODEL_ASSET).use { fd ->
                fd.createInputStream().use { stream ->
                    stream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength,
                    )
                }
            }
    }
}

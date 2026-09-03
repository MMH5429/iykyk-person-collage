package com.iykyk.collage.samples

import android.content.Context
import android.net.Uri
import androidx.core.net.toUri
import java.io.File

data class SampleVideo(val label: String, val uri: Uri)

/**
 * Copies the bundled sample clips out of assets on first use.
 *
 * These are a convenience for demoing; they are handed to the pipeline as ordinary file
 * URIs and travel the identical code path as a video the user picks. Nothing about them
 * is special-cased downstream.
 */
object SampleVideos {

    private val ASSET_NAMES = listOf(
        "Sample 1" to "samples/sample1.mp4",
        "Sample 2" to "samples/sample2.mp4",
        "Sample 3" to "samples/sample3.mp4",
    )

    fun ensureExtracted(context: Context): List<SampleVideo> {
        val dir = File(context.cacheDir, "samples").apply { mkdirs() }
        return ASSET_NAMES.map { (label, assetPath) ->
            val out = File(dir, assetPath.substringAfterLast('/'))
            if (!out.exists() || out.length() == 0L) {
                context.assets.open(assetPath).use { input ->
                    out.outputStream().use { output -> input.copyTo(output) }
                }
            }
            SampleVideo(label, out.toUri())
        }
    }
}

package com.iykyk.collage.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import java.io.Closeable

/**
 * Pulls single frames at full resolution by seeking.
 *
 * Seeking is slow, which is why analysis uses a sequential decoder instead — but the
 * representative shots are only a handful of frames, so a few seeks are the cheapest way
 * to get full-quality pixels for the collage tiles.
 */
class FullResFrameGrabber(context: Context, uri: Uri) : Closeable {

    private val retriever = MediaMetadataRetriever().apply { setDataSource(context, uri) }

    fun grab(timestampMs: Long): Bitmap? = runCatching {
        retriever.getFrameAtTime(timestampMs * 1_000L, MediaMetadataRetriever.OPTION_CLOSEST)
    }.getOrNull()

    override fun close() = retriever.release()
}

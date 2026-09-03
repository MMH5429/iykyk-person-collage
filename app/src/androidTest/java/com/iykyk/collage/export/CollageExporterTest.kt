package com.iykyk.collage.export

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CollageExporterTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val exporter = CollageExporter(context)

    private fun bitmap() = Bitmap.createBitmap(1080, 1920, Bitmap.Config.ARGB_8888)
        .also { Canvas(it).drawColor(Color.MAGENTA) }

    @Suppress("DEPRECATION")
    private fun Intent.stream(): Uri? = getParcelableExtra(Intent.EXTRA_STREAM)

    @Test
    fun savingToTheGalleryReturnsAReadableUri() = runBlocking {
        val result = exporter.saveToGallery(bitmap(), "test-collage")
        assertTrue("save failed: ${result.exceptionOrNull()}", result.isSuccess)

        val uri = result.getOrThrow()
        context.contentResolver.openInputStream(uri).use { stream ->
            assertNotNull(stream)
            assertTrue("saved image is empty", stream!!.available() > 0)
        }
    }

    @Test
    fun shareIntentIsASendIntentWithAnImageStream() = runBlocking {
        val intent = exporter.shareIntent(bitmap(), "test-collage")
        assertEquals(Intent.ACTION_SEND, intent.action)
        assertEquals("image/jpeg", intent.type)
        assertNotNull(intent.stream())
        assertTrue(
            "read permission not granted to the receiving app",
            intent.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION != 0
        )
    }

    @Test
    fun sharedFileIsReadableThroughTheProvider() = runBlocking {
        val uri = exporter.shareIntent(bitmap(), "test-collage").stream()!!
        assertEquals("content", uri.scheme)
        context.contentResolver.openInputStream(uri).use { assertNotNull(it) }
    }
}

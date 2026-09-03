package com.iykyk.collage.collage

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.iykyk.collage.core.model.AnalysisResult
import com.iykyk.collage.core.model.Appearance
import com.iykyk.collage.core.model.Person
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CollageRendererTest {

    private fun swatch(color: Int) = Bitmap.createBitmap(400, 500, Bitmap.Config.ARGB_8888)
        .also { Canvas(it).drawColor(color) }

    private fun analysis(count: Int) = AnalysisResult(
        people = (0 until count).map { i ->
            Person(
                id = i,
                label = "Person ${i + 1}",
                appearances = List(i + 1) { k -> Appearance(k * 1000L, k * 1000L + 500) },
                shot = swatch(Color.rgb(40 + i * 20, 90, 200 - i * 15)),
            )
        },
        sourceName = "Sample 1",
        durationMs = 30_000,
    )

    @Test
    fun rendersAtStoryResolution() {
        val bmp = CollageRenderer().render(analysis(5))
        assertEquals(1080, bmp.width)
        assertEquals(1920, bmp.height)
    }

    @Test
    fun everyPersonCountFromOneToNineRendersWithoutError() {
        for (n in 1..9) {
            val bmp = CollageRenderer().render(analysis(n))
            assertTrue("count $n produced an empty bitmap", bmp.width > 0)
            bmp.recycle()
        }
    }

    @Test
    fun tilePixelsActuallyAppearInTheOutput() {
        val bmp = CollageRenderer().render(analysis(4))
        val pixels = IntArray(bmp.width * bmp.height)
        bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val distinct = pixels.toHashSet()
        assertTrue(
            "only ${distinct.size} distinct colours; tiles were probably not drawn",
            distinct.size > 50
        )
    }

    @Test
    fun aspectRatioIsInstagramStory() {
        val bmp = CollageRenderer().render(analysis(3))
        assertEquals(9f / 16f, bmp.width.toFloat() / bmp.height, 1e-3f)
    }
}

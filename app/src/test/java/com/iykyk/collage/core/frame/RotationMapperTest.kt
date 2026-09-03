package com.iykyk.collage.core.frame

import com.iykyk.collage.core.model.BoxF
import org.junit.Assert.assertEquals
import org.junit.Test

class RotationMapperTest {

    @Test
    fun `zero and one eighty keep the frame dimensions`() {
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1080, 1920, 0))
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1080, 1920, 180))
    }

    @Test
    fun `ninety and two seventy swap the frame dimensions`() {
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1920, 1080, 90))
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1920, 1080, 270))
    }

    @Test
    fun `zero rotation is the identity`() {
        val box = BoxF(10f, 20f, 30f, 40f)
        assertEquals(box, RotationMapper.uprightToRaw(box, rawW = 100, rawH = 200, rotationDeg = 0))
    }

    @Test
    fun `ninety degrees maps the upright top left to the raw bottom left`() {
        // Raw 200x100 rotated 90 CW gives an upright 100x200 frame.
        // Upright (0,0)-(10,20) comes from raw (0, 90)-(20, 100).
        val mapped = RotationMapper.uprightToRaw(BoxF(0f, 0f, 10f, 20f), rawW = 200, rawH = 100, rotationDeg = 90)
        assertEquals(BoxF(0f, 90f, 20f, 100f), mapped)
    }

    @Test
    fun `one eighty flips both axes`() {
        val mapped = RotationMapper.uprightToRaw(BoxF(10f, 20f, 30f, 40f), rawW = 100, rawH = 200, rotationDeg = 180)
        assertEquals(BoxF(70f, 160f, 90f, 180f), mapped)
    }

    @Test
    fun `two seventy is the inverse of ninety`() {
        val mapped = RotationMapper.uprightToRaw(BoxF(0f, 0f, 10f, 20f), rawW = 200, rawH = 100, rotationDeg = 270)
        assertEquals(BoxF(180f, 0f, 200f, 10f), mapped)
    }

    @Test
    fun `mapping a full frame box covers the whole raw frame`() {
        for (deg in listOf(0, 90, 180, 270)) {
            val (uw, uh) = RotationMapper.uprightSize(200, 100, deg)
            val mapped = RotationMapper.uprightToRaw(
                BoxF(0f, 0f, uw.toFloat(), uh.toFloat()), rawW = 200, rawH = 100, rotationDeg = deg
            )
            assertEquals("rotation $deg", BoxF(0f, 0f, 200f, 100f), mapped)
        }
    }
}

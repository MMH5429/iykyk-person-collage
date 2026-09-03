package com.iykyk.collage.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GeometryTest {

    @Test
    fun `identical boxes have iou of one`() {
        val a = BoxF(0f, 0f, 10f, 10f)
        assertEquals(1f, a.iou(a), 1e-4f)
    }

    @Test
    fun `disjoint boxes have iou of zero`() {
        val a = BoxF(0f, 0f, 10f, 10f)
        val b = BoxF(20f, 20f, 30f, 30f)
        assertEquals(0f, a.iou(b), 1e-4f)
        assertNull(a.intersect(b))
    }

    @Test
    fun `half overlapping boxes have iou of one third`() {
        // Union 150, intersection 50.
        val a = BoxF(0f, 0f, 10f, 10f)
        val b = BoxF(5f, 0f, 15f, 10f)
        assertEquals(1f / 3f, a.iou(b), 1e-4f)
    }

    @Test
    fun `expand grows the box about its centre`() {
        val e = BoxF(10f, 10f, 20f, 20f).expand(2f)
        assertEquals(BoxF(5f, 5f, 25f, 25f), e)
    }

    @Test
    fun `clampTo keeps the box inside the frame`() {
        val c = BoxF(-5f, -5f, 50f, 50f).clampTo(40, 30)
        assertEquals(BoxF(0f, 0f, 40f, 30f), c)
    }
}

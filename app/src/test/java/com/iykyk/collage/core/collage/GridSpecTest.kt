package com.iykyk.collage.core.collage

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GridSpecTest {

    @Test
    fun `one person fills a single cell`() {
        assertEquals(listOf(Cell(0, 0, 1)), GridSpec.forCount(1))
    }

    @Test
    fun `two people stack in one column on a portrait canvas`() {
        val cells = GridSpec.forCount(2)
        assertEquals(2, cells.size)
        assertEquals(listOf(0, 1), cells.map { it.row })
        assertTrue("both should be in column 0", cells.all { it.column == 0 })
    }

    @Test
    fun `four people form a two by two grid`() {
        val cells = GridSpec.forCount(4)
        assertEquals(4, cells.size)
        assertEquals(setOf(0, 1), cells.map { it.row }.toSet())
        assertEquals(setOf(0, 1), cells.map { it.column }.toSet())
    }

    @Test
    fun `five people use two columns with the last row spanning`() {
        val cells = GridSpec.forCount(5)
        assertEquals(5, cells.size)
        assertEquals(3, cells.maxOf { it.row } + 1)
        // The odd one out spans the full width rather than leaving a hole.
        assertEquals(2, cells.last().columnSpan)
    }

    @Test
    fun `every count from one to twelve produces exactly that many cells`() {
        for (n in 1..12) assertEquals("count $n", n, GridSpec.forCount(n).size)
    }

    @Test
    fun `no two cells occupy the same position`() {
        for (n in 1..12) {
            val positions = GridSpec.forCount(n).map { it.row to it.column }
            assertEquals("count $n has overlapping cells", positions.size, positions.distinct().size)
        }
    }

    @Test
    fun `larger groups stay at most three columns wide`() {
        for (n in 1..12) {
            assertTrue("count $n", GridSpec.forCount(n).all { it.column + it.columnSpan <= 3 })
        }
    }

    @Test
    fun `rows and columns agree with the generated cells`() {
        for (n in 1..12) {
            assertEquals("count $n", GridSpec.rowsFor(n), GridSpec.forCount(n).maxOf { it.row } + 1)
        }
    }

    @Test
    fun `zero people yields no cells`() {
        assertEquals(emptyList<Cell>(), GridSpec.forCount(0))
    }
}

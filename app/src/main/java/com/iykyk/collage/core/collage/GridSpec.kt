package com.iykyk.collage.core.collage

/** A tile position in the collage grid. [columnSpan] lets a short final row stay centred. */
data class Cell(val row: Int, val column: Int, val columnSpan: Int)

/**
 * Chooses a grid for a given number of people on a 9:16 canvas.
 *
 * Portrait-biased: at most three columns, so tiles stay tall enough to show a face well.
 * A final row with leftover space widens its tiles rather than leaving an empty hole.
 */
object GridSpec {

    fun forCount(count: Int): List<Cell> {
        if (count <= 0) return emptyList()

        val columns = columnsFor(count)
        val cells = mutableListOf<Cell>()
        var remaining = count
        var row = 0

        while (remaining > 0) {
            val inThisRow = minOf(columns, remaining)
            // Widen tiles so a short row still fills the canvas.
            val span = columns / inThisRow
            for (i in 0 until inThisRow) {
                cells += Cell(row = row, column = i * span, columnSpan = span)
            }
            remaining -= inThisRow
            row++
        }
        return cells
    }

    /** Number of grid columns used for [count] people. */
    fun columnsFor(count: Int): Int = when {
        count <= 2 -> 1
        count <= 6 -> 2
        else -> 3
    }

    /** Number of grid rows used for [count] people. */
    fun rowsFor(count: Int): Int =
        if (count <= 0) 0 else (count + columnsFor(count) - 1) / columnsFor(count)
}

package com.iykyk.collage.pipeline

import org.junit.Assert.assertEquals
import org.junit.Test

class AnalysisDumpTest {

    private val dump = AnalysisDump(
        sourceName = "Sample 1",
        durationMs = 30_000,
        tracklets = listOf(
            TrackletRecord(0, 100, 900, 7, 0.82f, 400, listOf(0.1f, 0.2f, 0.3f)),
            TrackletRecord(1, 200, 800, 5, 0.71f, 500, listOf(0.4f, 0.5f, 0.6f)),
        ),
        cannotLink = listOf(listOf(0, 1)),
    )

    @Test
    fun `json round trips exactly`() {
        assertEquals(dump, AnalysisDump.fromJson(dump.toJson()))
    }

    @Test
    fun `embeddings survive the round trip`() {
        val back = AnalysisDump.fromJson(dump.toJson())
        assertEquals(listOf(0.1f, 0.2f, 0.3f), back.tracklets[0].embedding)
    }

    @Test
    fun `cannot link pairs survive the round trip`() {
        assertEquals(listOf(listOf(0, 1)), AnalysisDump.fromJson(dump.toJson()).cannotLink)
    }
}

package com.iykyk.collage.core.model

import android.graphics.Bitmap

/** One continuous visible segment of a person. */
data class Appearance(val startMs: Long, val endMs: Long)

/**
 * A unique person found in the video, with every segment they appeared in and the single
 * best shot chosen to represent them.
 */
data class Person(
    val id: Int,
    val label: String,
    val appearances: List<Appearance>,
    val shot: Bitmap,
) {
    val appearanceCount: Int get() = appearances.size
}

data class AnalysisResult(
    val people: List<Person>,
    val sourceName: String,
    val durationMs: Long,
) {
    val totalAppearances: Int get() = people.sumOf { it.appearanceCount }
}

package com.iykyk.collage.pipeline

import android.graphics.Bitmap
import com.iykyk.collage.core.model.AnalysisResult

data class CollageResult(val analysis: AnalysisResult, val collage: Bitmap)

/** Progress of one processing run, surfaced straight to the UI. */
sealed interface PipelineState {
    data object Idle : PipelineState
    data object Preparing : PipelineState

    /** [fraction] in 0..1 across the clip's duration. */
    data class Analysing(val fraction: Float) : PipelineState
    data object GroupingPeople : PipelineState
    data object SelectingShots : PipelineState
    data object BuildingCollage : PipelineState

    data class Done(val result: CollageResult) : PipelineState
    data class Failed(val message: String) : PipelineState

    /** Human-readable stage name for the progress UI. */
    val label: String
        get() = when (this) {
            Idle -> ""
            Preparing -> "Preparing video"
            is Analysing -> "Finding faces"
            GroupingPeople -> "Grouping people"
            SelectingShots -> "Choosing best shots"
            BuildingCollage -> "Building collage"
            is Done -> "Done"
            is Failed -> "Failed"
        }
}

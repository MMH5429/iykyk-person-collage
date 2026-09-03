package com.iykyk.collage.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.export.CollageExporter
import com.iykyk.collage.pipeline.CollageResult
import com.iykyk.collage.pipeline.PersonCollagePipeline
import com.iykyk.collage.pipeline.PipelineState
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

/**
 * Owns one processing run at a time.
 *
 * The pipeline is a cold Flow on a background dispatcher, so cancelling [job] genuinely
 * stops decoding rather than leaving a codec running behind a discarded UI. Surviving
 * configuration changes falls out of the ViewModel outliving the activity.
 */
class CollageViewModel(application: Application) : AndroidViewModel(application) {

    private val pipeline = PersonCollagePipeline(application, PipelineConfig.Default)
    private val exporter = CollageExporter(application)

    private val _state = MutableStateFlow<PipelineState>(PipelineState.Idle)
    val state: StateFlow<PipelineState> = _state.asStateFlow()

    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message.asStateFlow()

    private var job: Job? = null

    fun process(uri: Uri, sourceName: String) {
        job?.cancel()
        _message.value = null
        job = viewModelScope.launch {
            pipeline.run(uri, sourceName)
                .catch { cause ->
                    if (cause is CancellationException) throw cause
                    _state.value = PipelineState.Failed(cause.message ?: "Processing failed.")
                }
                .collect { _state.value = it }
        }
    }

    fun cancel() {
        job?.cancel()
        job = null
        _state.value = PipelineState.Idle
    }

    fun reset() {
        cancel()
        _message.value = null
    }

    fun save() {
        val result = currentResult() ?: return
        viewModelScope.launch {
            val outcome = exporter.saveToGallery(result.collage, slug(result))
            _message.value = if (outcome.isSuccess) {
                "Saved to your gallery"
            } else {
                "Could not save: ${outcome.exceptionOrNull()?.message}"
            }
        }
    }

    fun share(onIntent: (Intent) -> Unit) {
        val result = currentResult() ?: return
        viewModelScope.launch {
            val intent = exporter.shareIntent(result.collage, slug(result))
            onIntent(Intent.createChooser(intent, "Share collage"))
        }
    }

    fun consumeMessage() {
        _message.value = null
    }

    private fun currentResult(): CollageResult? = (_state.value as? PipelineState.Done)?.result

    private fun slug(result: CollageResult): String =
        result.analysis.sourceName.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-')
            .ifEmpty { "collage" }
}

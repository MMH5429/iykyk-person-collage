package com.iykyk.collage.ui

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.iykyk.collage.pipeline.PipelineState
import com.iykyk.collage.samples.SampleVideos

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScreen(viewModel: CollageViewModel, onShare: (Intent) -> Unit) {
    val state by viewModel.state.collectAsState()
    val message by viewModel.message.collectAsState()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let { viewModel.process(it, "Selected video") } }

    LaunchedEffect(message) {
        message?.let {
            snackbar.showSnackbar(it)
            viewModel.consumeMessage()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = { TopAppBar(title = { Text("Who's in this video") }) },
    ) { padding ->
        Box(
            Modifier
                .padding(padding)
                .fillMaxSize()
                .padding(20.dp),
            contentAlignment = Alignment.Center,
        ) {
            when (val s = state) {
                is PipelineState.Idle -> PickerContent(
                    onPickVideo = {
                        picker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
                        )
                    },
                    onPickSample = { index ->
                        val sample = SampleVideos.ensureExtracted(context)[index]
                        viewModel.process(sample.uri, sample.label)
                    },
                )

                is PipelineState.Failed -> FailureContent(s.message) { viewModel.reset() }

                is PipelineState.Done -> ResultContent(
                    state = s,
                    onSave = viewModel::save,
                    onShare = { viewModel.share(onShare) },
                    onStartOver = viewModel::reset,
                )

                else -> ProcessingContent(s) { viewModel.cancel() }
            }
        }
    }
}

@Composable
private fun PickerContent(onPickVideo: () -> Unit, onPickSample: (Int) -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            "Pick a video and I'll find everyone in it.",
            style = MaterialTheme.typography.bodyLarge,
        )
        Spacer(Modifier.height(8.dp))
        Button(onClick = onPickVideo, modifier = Modifier.fillMaxWidth()) {
            Text("Choose a video")
        }
        Text("or try a sample", style = MaterialTheme.typography.labelMedium)
        repeat(3) { i ->
            OutlinedButton(onClick = { onPickSample(i) }, modifier = Modifier.fillMaxWidth()) {
                Text("Sample ${i + 1}")
            }
        }
    }
}

@Composable
private fun ProcessingContent(state: PipelineState, onCancel: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(state.label, style = MaterialTheme.typography.titleMedium)

        if (state is PipelineState.Analysing) {
            LinearProgressIndicator(
                progress = { state.fraction },
                modifier = Modifier.fillMaxWidth(),
            )
            Text("${(state.fraction * 100).toInt()}%", style = MaterialTheme.typography.labelLarge)
        } else {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        TextButton(onClick = onCancel) { Text("Cancel") }
    }
}

@Composable
private fun FailureContent(message: String, onStartOver: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(message, style = MaterialTheme.typography.bodyLarge)
        Button(onClick = onStartOver) { Text("Try another video") }
    }
}

@Composable
private fun ResultContent(
    state: PipelineState.Done,
    onSave: () -> Unit,
    onShare: () -> Unit,
    onStartOver: () -> Unit,
) {
    val analysis = state.result.analysis
    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(
            "${analysis.people.size} people · ${analysis.totalAppearances} appearances",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
        )

        Image(
            bitmap = state.result.collage.asImageBitmap(),
            contentDescription = "Generated collage",
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp)),
        )

        analysis.people.forEach { person ->
            ListItem(
                headlineContent = { Text(person.label) },
                supportingContent = {
                    val word = if (person.appearanceCount == 1) "appearance" else "appearances"
                    Text("${person.appearanceCount} $word")
                },
                leadingContent = {
                    Image(
                        bitmap = person.shot.asImageBitmap(),
                        contentDescription = person.label,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(52.dp)
                            .clip(RoundedCornerShape(10.dp)),
                    )
                },
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = onSave) { Text("Save") }
            Button(onClick = onShare) { Text("Share") }
        }
        TextButton(onClick = onStartOver) { Text("Process another video") }
        Spacer(Modifier.height(24.dp))
    }
}

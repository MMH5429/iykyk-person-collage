package com.iykyk.collage

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.MaterialTheme
import androidx.lifecycle.viewmodel.compose.viewModel
import com.iykyk.collage.ui.AppScreen
import com.iykyk.collage.ui.CollageViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                val collageViewModel: CollageViewModel = viewModel()
                AppScreen(viewModel = collageViewModel, onShare = ::startActivity)
            }
        }
    }
}

package com.example.deepsigntid

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

data class LandmarkPoint(val x: Float, val y: Float)

data class LandmarkOverlay(
    val poseLandmarks: List<LandmarkPoint> = emptyList(),
    val leftHandLandmarks: List<LandmarkPoint> = emptyList(),
    val rightHandLandmarks: List<LandmarkPoint> = emptyList(),
    val imageWidth: Int = 1,
    val imageHeight: Int = 1
)

data class AppState(
    val predictions: List<Prediction> = emptyList(),
    val sentence: List<String> = emptyList(),
    val signState: String = "idle",
    val landmarks: LandmarkOverlay = LandmarkOverlay(),
    val debugInfo: String = "",
    val selection: DigitSelectionState = DigitSelectionState()
)

class MainViewModel : ViewModel() {

    private val _state = MutableStateFlow(AppState())
    val state: StateFlow<AppState> = _state

    var signPredictor: SignPredictor? = null

    fun updatePredictions(predictions: List<Prediction>) {
        _state.value = _state.value.copy(predictions = predictions)
    }

    fun updateSignState(state: String) {
        _state.value = _state.value.copy(signState = state)
    }

    fun updateLandmarks(overlay: LandmarkOverlay) {
        _state.value = _state.value.copy(landmarks = overlay)
    }

    fun updateDebugInfo(info: String) {
        _state.value = _state.value.copy(debugInfo = info)
    }

    fun updateSelectionState(selection: DigitSelectionState) {
        _state.value = _state.value.copy(selection = selection)
    }

    fun addWordToSentence(word: String) {
        val current = _state.value.sentence.toMutableList()
        current.add(word)
        _state.value = _state.value.copy(sentence = current)
    }

    fun clearSentence() {
        _state.value = _state.value.copy(sentence = emptyList())
    }

    fun removeLastWord() {
        val current = _state.value.sentence.toMutableList()
        if (current.isNotEmpty()) current.removeLast()
        _state.value = _state.value.copy(sentence = current)
    }
}

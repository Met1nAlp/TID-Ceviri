package com.example.deepsigntid

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.sqrt

data class DigitSelectionState(
    val active: Boolean = false,
    val remainingMs: Long = 0L,
    val candidates: List<Prediction> = emptyList(),
    val lastEvent: String = "idle",
    val lastReason: String = "",
    val lastDigitValue: Int? = null,
    val lastConfidence: Float = 0f,
    val stableDigit: Int? = null,
    val stableVotes: Int = 0,
    val requiredStableFrames: Int = 3,
    val lastSelected: Prediction? = null,
)

data class DigitSelectionOutcome(
    val event: String,
    val prediction: Prediction? = null,
    val digitValue: Int? = null,
    val confidence: Float = 0f,
)

class DigitSelectionPredictor(private val context: Context) {

    private var module: Module? = null

    private val classNames = listOf("digit_1", "digit_2", "digit_3", "other_digit")
    private val confidenceThreshold = 0.55f   // 0.80 -> 0.55: 2 işaretini yakalamak artık çok daha kolay
    private val selectionTimeoutMs = 3_000L
    private val selectionArmDelayMs = 500L
    private val selectionInterruptGraceMs = maxOf(selectionArmDelayMs, 2_000L)
    private val stableFrames = 2              // 3 -> 2: Üst üste 2 kare bilmesi yeterli
    private val voteHistorySize = 5
    private val voteHistory = ArrayDeque<Int>(voteHistorySize + 1)

    private var active = false
    private var candidates: List<Prediction> = emptyList()
    private var startedAtMs = 0L
    private var expiresAtMs = 0L
    private var lastEvent = "idle"
    private var lastReason = ""
    private var lastDigitValue: Int? = null
    private var lastConfidence = 0f
    private var lastSelected: Prediction? = null

    init {
        loadModel()
    }

    fun isActive(): Boolean = active

    fun isArming(): Boolean = active && SystemClock.elapsedRealtime() < startedAtMs + selectionArmDelayMs

    fun isInterruptGuardActive(): Boolean =
        active && SystemClock.elapsedRealtime() < startedAtMs + selectionInterruptGraceMs

    fun hasDigitEvidence(): Boolean = voteHistory.isNotEmpty() || lastDigitValue != null

    fun startSelection(candidates: List<Prediction>) {
        this.candidates = candidates.take(3)
        this.active = this.candidates.isNotEmpty()
        this.startedAtMs = SystemClock.elapsedRealtime()
        this.expiresAtMs = startedAtMs + selectionTimeoutMs
        voteHistory.clear()
        lastEvent = if (active) "armed" else "idle"
        lastReason = if (active) "awaiting_digit" else "no_candidates"
        lastDigitValue = null
        lastConfidence = 0f
        lastSelected = null
    }

    fun cancel(reason: String = "cancelled") {
        active = false
        voteHistory.clear()
        lastEvent = "cancelled"
        lastReason = reason
    }

    fun processLandmarks(landmarks: FloatArray): DigitSelectionOutcome? {
        if (!active) {
            return null
        }

        val now = SystemClock.elapsedRealtime()
        if (now >= expiresAtMs) {
            active = false
            voteHistory.clear()
            lastEvent = "timeout"
            lastReason = "selection_timeout"
            return DigitSelectionOutcome(event = "timeout")
        }

        if (now < startedAtMs + selectionArmDelayMs) {
            lastEvent = "waiting"
            lastReason = "selection_arm_delay"
            lastDigitValue = null
            lastConfidence = 0f
            return null
        }

        val features = extractCanonicalHandFeatures(landmarks)
        if (features == null) {
            lastEvent = "waiting"
            lastReason = "no_hand_detected"
            lastDigitValue = null
            lastConfidence = 0f
            return null
        }

        val probs = runInference(features.values)
        if (probs == null) {
            lastEvent = "waiting"
            lastReason = "model_unavailable"
            return null
        }

        val topIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val topLabel = classNames.getOrElse(topIndex) { "unknown" }
        val confidence = probs[topIndex]
        val digitValue = when (topLabel) {
            "digit_1" -> 1
            "digit_2" -> 2
            "digit_3" -> 3
            else -> null
        }

        lastDigitValue = digitValue
        lastConfidence = confidence

        if (digitValue == null || confidence < confidenceThreshold) {
            lastEvent = "waiting"
            lastReason = "low_confidence_digit"
            return null
        }

        if (voteHistory.size >= voteHistorySize) {
            voteHistory.removeFirst()
        }
        voteHistory.addLast(digitValue)
        lastEvent = "digit_seen"
        lastReason = "digit_$digitValue"

        val recentVotes = voteHistory.takeLast(stableFrames)
        if (recentVotes.size < stableFrames || recentVotes.distinct().size != 1) {
            return null
        }

        val candidateIndex = digitValue - 1
        if (candidateIndex !in candidates.indices) {
            lastEvent = "waiting"
            lastReason = "digit_without_candidate"
            return null
        }

        val chosenPrediction = candidates[candidateIndex]
        active = false
        voteHistory.clear()
        lastEvent = "selected"
        lastReason = "stable_digit_match"
        lastSelected = chosenPrediction

        return DigitSelectionOutcome(
            event = "selected",
            prediction = chosenPrediction,
            digitValue = digitValue,
            confidence = confidence,
        )
    }

    fun getState(): DigitSelectionState {
        val remainingMs = if (active) {
            (expiresAtMs - SystemClock.elapsedRealtime()).coerceAtLeast(0L)
        } else {
            0L
        }

        val stableDigit = voteHistory.groupingBy { it }
            .eachCount()
            .maxByOrNull { it.value }
            ?.key

        return DigitSelectionState(
            active = active,
            remainingMs = remainingMs,
            candidates = candidates,
            lastEvent = lastEvent,
            lastReason = lastReason,
            lastDigitValue = lastDigitValue,
            lastConfidence = lastConfidence * 100f,
            stableDigit = stableDigit,
            stableVotes = voteHistory.size,
            requiredStableFrames = stableFrames,
            lastSelected = lastSelected,
        )
    }

    private fun loadModel() {
        try {
            val modelFile = assetFilePath("digit_selection_mobile.ptl")
            module = LiteModuleLoader.load(modelFile)
            Log.d("DigitSelection", "Digit model loaded (${File(modelFile).length() / 1024} KB)")
        } catch (e: Exception) {
            Log.e("DigitSelection", "Digit model error: ${e.message}")
        }
    }

    private data class CanonicalHandFeatures(
        val values: FloatArray,
    )

    private fun extractCanonicalHandFeatures(landmarks: FloatArray): CanonicalHandFeatures? {
        if (landmarks.size < 258) {
            return null
        }

        val leftStart = 132
        val rightStart = 195
        val handSize = 63

        val leftHand = landmarks.copyOfRange(leftStart, leftStart + handSize)
        val rightHand = landmarks.copyOfRange(rightStart, rightStart + handSize)

        val leftMagnitude = handMagnitude(leftHand)
        val rightMagnitude = handMagnitude(rightHand)

        val selected = when {
            leftMagnitude <= 0.1f && rightMagnitude <= 0.1f -> return null
            rightMagnitude > leftMagnitude -> Pair(rightHand, "Right")
            else -> Pair(leftHand, "Left")
        }

        val features = selected.first.copyOf()
        val handedness = selected.second

        val wristX = features[0]
        val wristY = features[1]
        val wristZ = features[2]

        for (i in features.indices step 3) {
            features[i] -= wristX
            features[i + 1] -= wristY
            features[i + 2] -= wristZ
        }

        var scale = 0f
        for (i in features.indices step 3) {
            val norm = sqrt(features[i] * features[i] + features[i + 1] * features[i + 1])
            if (norm > scale) {
                scale = norm
            }
        }
        if (scale < 1e-6f) {
            scale = 1f
        }

        for (i in features.indices) {
            features[i] /= scale
        }

        if (handedness == "Right") {
            for (i in features.indices step 3) {
                features[i] *= -1f
            }
        }

        val mean = features.average().toFloat()
        var sumSq = 0f
        for (value in features) {
            val diff = value - mean
            sumSq += diff * diff
        }
        val std = sqrt(sumSq / features.size) + 1e-8f
        for (i in features.indices) {
            features[i] = (features[i] - mean) / std
        }

        return CanonicalHandFeatures(values = features)
    }

    private fun handMagnitude(hand: FloatArray): Float {
        var sum = 0f
        for (value in hand) {
            sum += abs(value)
        }
        return sum
    }

    private fun runInference(features: FloatArray): FloatArray? {
        val mod = module ?: return null
        return try {
            val inputTensor = Tensor.fromBlob(features, longArrayOf(1, features.size.toLong()))
            val output = mod.forward(IValue.from(inputTensor)).toTensor().dataAsFloatArray
            softmax(output)
        } catch (e: Exception) {
            Log.e("DigitSelection", "Inference error: ${e.message}")
            null
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expValues = FloatArray(logits.size)
        var sumExp = 0f

        for (i in logits.indices) {
            val expValue = kotlin.math.exp((logits[i] - maxLogit).toDouble()).toFloat()
            expValues[i] = expValue
            sumExp += expValue
        }

        if (sumExp <= 0f) {
            return FloatArray(logits.size) { 1f / logits.size }
        }

        return FloatArray(logits.size) { idx -> expValues[idx] / sumExp }
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (!file.exists() || file.length() == 0L) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }
}

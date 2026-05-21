package com.example.deepsigntid

import android.content.Context
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.sqrt

data class Prediction(
    val labelTr: String,
    val labelEn: String,
    val confidence: Float,
    val classId: Int,
)

class SignPredictor(private val context: Context) {

    private var module: Module? = null
    private var labelsTr: List<String> = emptyList()
    private var labelsEn: List<String> = emptyList()
    private val digitSelectionPredictor = DigitSelectionPredictor(context)

    private val seqLength = 48
    private val featureSize = 258
    private val maxRawSignFrames = 72

    // Segmentation thresholds aligned with the desktop/web path.
    private val motionThreshold = 0.008f
    private val idleThreshold = 0.006f
    private val minSignFrames = 15
    private val minDecisionFrames = minSignFrames
    private val idleFramesToStop = 7
    private val startFrames = 2
    private val minConfidence = 40f

    // Temperature scaling + margin filter aligned with the desktop path.
    private val temperature = 1.5f
    private val marginThreshold = 0.15f
    private val minHandFramesDivisor = 8

    // Pre-buffer helps keep the beginning of the sign.
    private val preBufferSize = 8
    private val preBuffer = ArrayDeque<FloatArray>(preBufferSize + 2)

    // Voting over the last few predictions.
    private val voteHistorySize = 3
    private val predictionHistory = mutableListOf<Int>()

    // Cooldown avoids immediate double-triggering.
    private val cooldownFrames = 20
    private val selectionCooldownFrames = 12
    private val selectionInterruptRequiredFrames = 4
    private val selectionInterruptMotionThreshold = 0.0115f
    private val trailingIdleKeepFrames = 2
    private val poseMotionIndices = intArrayOf(11, 12, 13, 14, 15, 16)
    private var cooldownCounter = 0
    private var selectionInterruptFrames = 0

    private var state = "idle"
    private val signFrames = mutableListOf<FloatArray>()
    private var idleFrames = 0
    private var signingFrames = 0
    private var prevLandmarks: FloatArray? = null
    private var frameDebugCount = 0

    var onPrediction: ((List<Prediction>) -> Unit)? = null
    var onStateChange: ((String) -> Unit)? = null
    var onDebug: ((String) -> Unit)? = null
    var onSelectionStateChange: ((DigitSelectionState) -> Unit)? = null
    var onSelectionConfirmed: ((Prediction) -> Unit)? = null

    init {
        loadModel()
        loadLabels()
        emitSelectionState()
    }

    fun forcePredict() {
        if (signFrames.size >= 5) {
            onDebug?.invoke("Manuel tahmin (${signFrames.size} frame)")
            predictSign()
        } else {
            onDebug?.invoke("Yeterli frame yok: ${signFrames.size}/5")
        }
    }

    fun cancelSelection(reason: String = "manual_selection") {
        if (!digitSelectionPredictor.isActive()) {
            return
        }
        digitSelectionPredictor.cancel(reason)
        selectionInterruptFrames = 0
        cooldownCounter = maxOf(cooldownCounter, selectionCooldownFrames)
        preBuffer.clear()
        emitSelectionState()
    }

    fun processLandmarks(landmarks: FloatArray) {
        if (landmarks.size != featureSize) return
        frameDebugCount++
        val motion = computeMotion(landmarks)

        if (digitSelectionPredictor.isActive()) {
            val selectionOutcome = digitSelectionPredictor.processLandmarks(landmarks)
            emitSelectionState()

            when (selectionOutcome?.event) {
                "selected" -> {
                    selectionOutcome.prediction?.let { prediction ->
                        onSelectionConfirmed?.invoke(prediction)
                        onDebug?.invoke("Secildi: ${prediction.labelTr}")
                    }
                    selectionInterruptFrames = 0
                    cooldownCounter = maxOf(cooldownCounter, selectionCooldownFrames)
                    preBuffer.clear()
                }
                "timeout" -> {
                    selectionInterruptFrames = 0
                    onDebug?.invoke("Secim zamani doldu")
                    cooldownCounter = maxOf(cooldownCounter, selectionCooldownFrames)
                    preBuffer.clear()
                }
            }

            if (selectionOutcome != null) {
                prevLandmarks = landmarks.copyOf()
                return
            }

            if (digitSelectionPredictor.isInterruptGuardActive()) {
                selectionInterruptFrames = 0
                prevLandmarks = landmarks.copyOf()
                return
            }

            if (digitSelectionPredictor.hasDigitEvidence()) {
                selectionInterruptFrames = 0
                prevLandmarks = landmarks.copyOf()
                return
            }

            if (motion > selectionInterruptMotionThreshold) {
                selectionInterruptFrames++
            } else {
                selectionInterruptFrames = 0
            }

            if (selectionInterruptFrames < selectionInterruptRequiredFrames) {
                prevLandmarks = landmarks.copyOf()
                return
            }

            digitSelectionPredictor.cancel("new_sign_started")
            selectionInterruptFrames = 0
            emitSelectionState()
            onDebug?.invoke("Secim iptal edildi, yeni isaret algilandi")
        }

        preBuffer.addLast(landmarks.copyOf())
        if (preBuffer.size > preBufferSize) {
            preBuffer.removeFirst()
        }

        if (cooldownCounter > 0) {
            cooldownCounter--
            prevLandmarks = landmarks.copyOf()
            return
        }

        if (!hasHandLandmarks(landmarks) && frameDebugCount % 90 == 0) {
            onDebug?.invoke("Eller net gorunmuyor")
        }

        if (frameDebugCount % 15 == 0) {
            val msg = "motion=${"%.4f".format(motion)} state=$state frames=${signFrames.size} idle=$idleFrames"
            Log.d("SignPredictor", msg)
            onDebug?.invoke(msg)
        }

        when (state) {
            "idle" -> {
                if (motion > motionThreshold) {
                    signingFrames++
                    if (signingFrames >= startFrames) {
                        state = "signing"
                        signFrames.clear()
                        predictionHistory.clear()
                        for (bufferedFrame in preBuffer) {
                            signFrames.add(bufferedFrame)
                        }
                        idleFrames = 0
                        signingFrames = 0
                        onStateChange?.invoke("signing")
                        onDebug?.invoke("Isaret basladi! motion=${"%.4f".format(motion)} (+${preBuffer.size} pre-buffer)")
                    }
                } else {
                    signingFrames = 0
                }
            }

            "signing" -> {
                signFrames.add(landmarks.copyOf())

                if (motion < idleThreshold) {
                    idleFrames++
                    if (idleFrames >= idleFramesToStop) {
                        if (signFrames.size >= minDecisionFrames) {
                            onDebug?.invoke("Isaret bitti (${signFrames.size} frame), tahmin yapiliyor...")
                            predictSign()
                            cooldownCounter = cooldownFrames
                        } else {
                            onDebug?.invoke("Cok kisa isaret (${signFrames.size}/$minSignFrames), atlandi")
                        }
                        state = "idle"
                        signFrames.clear()
                        idleFrames = 0
                        onStateChange?.invoke("idle")
                    }
                } else {
                    if (idleFrames > 0) {
                        onDebug?.invoke("Hareket devam ediyor (idle sifirlandi)")
                    }
                    idleFrames = 0
                }

                if (signFrames.size >= maxRawSignFrames) {
                    onDebug?.invoke("Maks ham frame ($maxRawSignFrames), tahmin yapiliyor...")
                    predictSign()
                    cooldownCounter = cooldownFrames
                    state = "idle"
                    signFrames.clear()
                    idleFrames = 0
                    onStateChange?.invoke("idle")
                }
            }
        }

        prevLandmarks = landmarks.copyOf()
    }

    private fun loadModel() {
        try {
            val modelFile = assetFilePath("best_model_mobile.ptl")
            module = LiteModuleLoader.load(modelFile)
            Log.d("SignPredictor", "Model yuklendi (${File(modelFile).length() / 1024} KB)")
        } catch (e: Exception) {
            Log.e("SignPredictor", "Model hatasi: ${e.message}")
        }
    }

    private fun loadLabels() {
        try {
            labelsTr = context.assets.open("labels_tr.txt").bufferedReader().readLines()
            labelsEn = context.assets.open("labels_en.txt").bufferedReader().readLines()
            Log.d("SignPredictor", "Labels: ${labelsTr.size} sinif")
        } catch (e: Exception) {
            Log.e("SignPredictor", "Label hatasi: ${e.message}")
        }
    }

    private fun computeMotion(landmarks: FloatArray): Float {
        val prev = prevLandmarks ?: return 0f

        val handStart = 132
        val leftHandEnd = 195
        val rightHandEnd = minOf(258, landmarks.size)

        var leftSum = 0f
        for (i in handStart until leftHandEnd) {
            leftSum += abs(landmarks[i] - prev[i])
        }
        val leftMotion = leftSum / (leftHandEnd - handStart)

        var rightSum = 0f
        for (i in leftHandEnd until rightHandEnd) {
            rightSum += abs(landmarks[i] - prev[i])
        }
        val rightMotion = rightSum / (rightHandEnd - leftHandEnd)
        val handMotion = maxOf(leftMotion, rightMotion)

        var poseSum = 0f
        var poseCount = 0
        for (landmarkIndex in poseMotionIndices) {
            val base = landmarkIndex * 4
            poseSum += abs(landmarks[base] - prev[base])
            poseSum += abs(landmarks[base + 1] - prev[base + 1])
            poseSum += abs(landmarks[base + 2] - prev[base + 2])
            poseCount += 3
        }
        val poseMotion = if (poseCount > 0) poseSum / poseCount else 0f

        return maxOf(handMotion, poseMotion * 0.6f)
    }

    private fun predictSign() {
        val mod = module ?: run {
            onDebug?.invoke("Model yuklenmemis")
            return
        }
        if (signFrames.isEmpty()) return

        try {
            val trailingIdleTrim = maxOf(0, idleFrames - trailingIdleKeepFrames)
            val effectiveFrames = if (trailingIdleTrim > 0 && signFrames.size - trailingIdleTrim >= 1) {
                signFrames.dropLast(trailingIdleTrim)
            } else {
                signFrames.toList()
            }

            val n = effectiveFrames.size
            val validFrames = effectiveFrames.count { frame ->
                var magnitude = 0f
                for (value in frame) {
                    magnitude += abs(value)
                }
                (magnitude / frame.size) > 0.01f
            }
            if (validFrames < n * 0.5f) {
                onDebug?.invoke("Cok fazla bos frame ($validFrames/$n), tahmin atlandi")
                predictionHistory.clear()
                return
            }

            val handFrames = effectiveFrames.count { hasHandLandmarks(it) }
            val minRequiredHandFrames = maxOf(3, n / minHandFramesDivisor)

            if (handFrames < minRequiredHandFrames) {
                onDebug?.invoke("El landmark zayif ($handFrames/$n frame), tahmin atlandi")
                predictionHistory.clear()
                return
            }

            val inputData = FloatArray(seqLength * featureSize)
            for (i in 0 until seqLength) {
                val idx = i.toFloat() * (n - 1) / (seqLength - 1)
                val lower = idx.toInt().coerceIn(0, n - 1)
                val upper = (lower + 1).coerceIn(0, n - 1)
                val weight = idx - lower

                for (featureIndex in 0 until featureSize) {
                    val value = if (lower == upper) {
                        effectiveFrames[lower][featureIndex]
                    } else {
                        (1f - weight) * effectiveFrames[lower][featureIndex] +
                            weight * effectiveFrames[upper][featureIndex]
                    }
                    inputData[i * featureSize + featureIndex] = value
                }
            }

            val mean = inputData.average().toFloat()
            var sumSq = 0.0
            for (value in inputData) {
                val diff = (value - mean).toDouble()
                sumSq += diff * diff
            }
            val std = (sqrt(sumSq / inputData.size) + 1e-8).toFloat()
            for (i in inputData.indices) {
                inputData[i] = (inputData[i] - mean) / std
            }

            val inputTensor = Tensor.fromBlob(
                inputData,
                longArrayOf(1, seqLength.toLong(), featureSize.toLong()),
            )

            val output = mod.forward(IValue.from(inputTensor)).toTensor()
            val scores = output.dataAsFloatArray
            val probs = temperatureSoftmax(scores, temperature)

            val topConfidence = (probs.maxOrNull() ?: 0f) * 100f
            val sorted = probs.sortedDescending()
            val margin = if (sorted.size >= 2) sorted[0] - sorted[1] else 1f
            val lowConfidence = topConfidence < minConfidence
            val ambiguous = margin < marginThreshold

            val topClassId = probs.indices.maxByOrNull { probs[it] } ?: 0
            predictionHistory.add(topClassId)
            if (predictionHistory.size > voteHistorySize) {
                predictionHistory.removeAt(0)
            }

            val votedClassId = if (predictionHistory.size >= 2) {
                predictionHistory.groupBy { it }
                    .maxByOrNull { it.value.size }
                    ?.key ?: topClassId
            } else {
                topClassId
            }

            val finalClassId = if (
                votedClassId != topClassId &&
                predictionHistory.count { it == votedClassId } >= 2
            ) {
                onDebug?.invoke(
                    "Voting: ${labelsTr.getOrElse(topClassId) { "?" }} -> ${labelsTr.getOrElse(votedClassId) { "?" }} " +
                        "(${predictionHistory.count { it == votedClassId }}/${predictionHistory.size} oy)",
                )
                votedClassId
            } else {
                topClassId
            }

            val orderedClassIds = mutableListOf(finalClassId)
            probs.indices
                .filter { it != finalClassId }
                .sortedByDescending { probs[it] }
                .take(2)
                .forEach { orderedClassIds.add(it) }

            val top3 = orderedClassIds.map { idx ->
                Prediction(
                    labelTr = labelsTr.getOrElse(idx) { "Sinif $idx" },
                    labelEn = labelsEn.getOrElse(idx) { "Class $idx" },
                    confidence = probs[idx] * 100f,
                    classId = idx,
                )
            }

            onPrediction?.invoke(top3)
            selectionInterruptFrames = 0
            digitSelectionPredictor.startSelection(top3)
            emitSelectionState()
            val debugReason = when {
                lowConfidence && ambiguous -> {
                    "Dusuk guven ve belirsiz tahmin, yine de top-3 gosteriliyor"
                }
                lowConfidence -> {
                    "Dusuk guvenli tahmin, top-3 gosteriliyor"
                }
                ambiguous -> {
                    "Belirsiz tahmin, top-3 gosteriliyor"
                }
                else -> {
                    "1-2-3 ile secim bekleniyor"
                }
            }
            onDebug?.invoke(debugReason)
            Log.d(
                "SignPredictor",
                "Tahmin: ${top3.firstOrNull()?.labelTr} %${top3.firstOrNull()?.confidence?.toInt()}",
            )
        } catch (e: Exception) {
            val msg = "Tahmin hatasi: ${e.javaClass.simpleName}: ${e.message}"
            Log.e("SignPredictor", msg)
            onDebug?.invoke(msg)
        }
    }

    private fun temperatureSoftmax(scores: FloatArray, temperature: Float): FloatArray {
        val maxScore = scores.maxOrNull() ?: 0f
        val expScores = FloatArray(scores.size)
        var sumExp = 0f

        for (i in scores.indices) {
            val expValue = kotlin.math.exp(((scores[i] - maxScore) / temperature).toDouble()).toFloat()
            expScores[i] = expValue
            sumExp += expValue
        }

        if (sumExp <= 0f) {
            return FloatArray(scores.size) { 1f / scores.size }
        }

        return FloatArray(scores.size) { index -> expScores[index] / sumExp }
    }

    private fun emitSelectionState() {
        onSelectionStateChange?.invoke(digitSelectionPredictor.getState())
    }

    private fun hasHandLandmarks(frame: FloatArray): Boolean {
        var handMagnitude = 0f
        for (i in 132 until featureSize) {
            handMagnitude += abs(frame[i])
        }
        return handMagnitude > 0.1f
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

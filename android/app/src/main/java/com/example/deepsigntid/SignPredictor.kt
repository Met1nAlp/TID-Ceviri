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

data class Prediction(
    val labelTr: String,
    val labelEn: String,
    val confidence: Float,
    val classId: Int
)

class SignPredictor(private val context: Context) {

    private var module: Module? = null
    private var labelsTr: List<String> = emptyList()
    private var labelsEn: List<String> = emptyList()

    private val SEQ_LENGTH = 48
    private val FEATURE_SIZE = 258

    // Segmentasyon eşikleri — eğitim pipeline'ı ile uyumlu
    private val MOTION_THRESHOLD    = 0.015f  // hareket başlangıç eşiği
    private val IDLE_THRESHOLD      = 0.005f  // durma eşiği
    private val MIN_SIGN_FRAMES     = 15       // min geçerli frame (~0.5s)
    private val IDLE_FRAMES_TO_STOP = 8        // dur sayacı
    private val START_FRAMES        = 3        // kaç frame hareketle başla
    private val MIN_CONFIDENCE      = 25f      // bu altındaki tahmin gösterilmez (%)

    // Durum makinesi
    private var state = "idle"
    private val signFrames = mutableListOf<FloatArray>()
    private var idleFrames = 0
    private var signingFrames = 0
    private var prevLandmarks: FloatArray? = null
    private var frameDebugCount = 0

    var onPrediction: ((List<Prediction>) -> Unit)? = null
    var onStateChange: ((String) -> Unit)? = null
    var onDebug: ((String) -> Unit)? = null

    fun forcePredict() {
        if (signFrames.size >= 5) {
            onDebug?.invoke("🔵 Manuel tahmin (${signFrames.size} frame)")
            predictSign()
        } else {
            onDebug?.invoke("⚠️ Yeterli frame yok: ${signFrames.size}/5")
        }
    }

    init {
        loadModel()
        loadLabels()
    }

    private fun loadModel() {
        try {
            val modelFile = assetFilePath("best_model_mobile.ptl")
            module = LiteModuleLoader.load(modelFile)
            val msg = "✅ Model yüklendi (${File(modelFile).length() / 1024} KB)"
            Log.d("SignPredictor", msg)
            onDebug?.invoke(msg)
        } catch (e: Exception) {
            val msg = "❌ Model hatası: ${e.javaClass.simpleName}: ${e.message}"
            Log.e("SignPredictor", msg)
            onDebug?.invoke(msg)
        }
    }

    private fun loadLabels() {
        try {
            labelsTr = context.assets.open("labels_tr.txt").bufferedReader().readLines()
            labelsEn = context.assets.open("labels_en.txt").bufferedReader().readLines()
            Log.d("SignPredictor", "Labels: ${labelsTr.size} sınıf")
        } catch (e: Exception) {
            Log.e("SignPredictor", "Label hatası: ${e.message}")
        }
    }

    fun processLandmarks(landmarks: FloatArray) {
        if (landmarks.size != FEATURE_SIZE) return
        frameDebugCount++

        val motion = computeMotion(landmarks)

        if (frameDebugCount % 30 == 0) {
            val msg = "motion=${"%.4f".format(motion)} state=$state frames=${signFrames.size}"
            Log.d("SignPredictor", msg)
            onDebug?.invoke(msg)
        }

        when (state) {
            "idle" -> {
                signFrames.clear()
                signFrames.add(landmarks.copyOf())

                if (motion > MOTION_THRESHOLD) {
                    signingFrames++
                    if (signingFrames >= START_FRAMES) {
                        state = "signing"
                        signFrames.clear()
                        idleFrames = 0
                        signingFrames = 0
                        onStateChange?.invoke("signing")
                        onDebug?.invoke("✋ İşaret başladı! motion=${"%.4f".format(motion)}")
                    }
                } else {
                    signingFrames = 0
                }
            }

            "signing" -> {
                signFrames.add(landmarks.copyOf())

                if (motion < IDLE_THRESHOLD) {
                    idleFrames++
                    if (idleFrames >= IDLE_FRAMES_TO_STOP) {
                        if (signFrames.size >= MIN_SIGN_FRAMES) {
                            onDebug?.invoke("⏹ İşaret bitti (${signFrames.size} frame), tahmin yapılıyor...")
                            predictSign()
                        } else {
                            onDebug?.invoke("⚠️ Çok kısa işaret (${signFrames.size}/${MIN_SIGN_FRAMES}), atlandı")
                        }
                        state = "idle"
                        signFrames.clear()
                        idleFrames = 0
                        onStateChange?.invoke("idle")
                    }
                } else {
                    idleFrames = 0
                }

                // Maks frame'e ulaşınca zorla tahmin et (orijinal eşik: SEQ_LENGTH)
                if (signFrames.size >= SEQ_LENGTH) {
                    onDebug?.invoke("⏹ Maks frame ($SEQ_LENGTH), tahmin yapılıyor...")
                    predictSign()
                    state = "idle"
                    signFrames.clear()
                    idleFrames = 0
                    onStateChange?.invoke("idle")
                }
            }
        }

        prevLandmarks = landmarks.copyOf()
    }

    private fun computeMotion(landmarks: FloatArray): Float {
        val prev = prevLandmarks ?: return 0f
        var sum = 0f
        val handStart = 132
        val handEnd = minOf(258, landmarks.size)
        for (i in handStart until handEnd) {
            sum += abs(landmarks[i] - prev[i])
        }
        return sum / (handEnd - handStart)
    }

    /**
     * Eğitim pipeline'ı (preprocess.py) ile birebir aynı:
     *  - Her iki durum için (kısa / uzun) lineer interpolasyon ile SEQ_LENGTH'e çek
     *  - Tek pencere yaklaşımı (ensemble kaldırıldı — sinyali bozuyordu)
     */
    private fun predictSign() {
        val mod = module ?: run {
            onDebug?.invoke("❌ Model yüklenmemiş!")
            return
        }
        if (signFrames.isEmpty()) return

        try {
            val n = signFrames.size

            // Lineer interpolasyon — preprocess.py _normalize_sequence_length ile aynı
            val inputData = FloatArray(SEQ_LENGTH * FEATURE_SIZE)
            for (i in 0 until SEQ_LENGTH) {
                val idx  = i.toFloat() * (n - 1) / (SEQ_LENGTH - 1)
                val lower = idx.toInt().coerceIn(0, n - 1)
                val upper = (lower + 1).coerceIn(0, n - 1)
                val weight = idx - lower

                for (f in 0 until FEATURE_SIZE) {
                    val value = if (lower == upper) {
                        signFrames[lower][f]
                    } else {
                        (1f - weight) * signFrames[lower][f] + weight * signFrames[upper][f]
                    }
                    inputData[i * FEATURE_SIZE + f] = value
                }
            }

            // Normalize: mean/std — dataset.py satır 74-76 ile aynı
            val mean = inputData.average().toFloat()
            val std = run {
                var sumSq = 0.0
                for (v in inputData) sumSq += (v - mean).toDouble() * (v - mean)
                (Math.sqrt(sumSq / inputData.size) + 1e-8).toFloat()
            }
            for (i in inputData.indices) inputData[i] = (inputData[i] - mean) / std

            val inputTensor = Tensor.fromBlob(
                inputData,
                longArrayOf(1, SEQ_LENGTH.toLong(), FEATURE_SIZE.toLong())
            )

            val output = mod.forward(IValue.from(inputTensor)).toTensor()
            val scores = output.dataAsFloatArray

            // Softmax
            val maxScore = scores.max()!!
            val expScores = FloatArray(scores.size) { Math.exp((scores[it] - maxScore).toDouble()).toFloat() }
            val sumExp = expScores.sum()
            val probs = FloatArray(scores.size) { expScores[it] / sumExp }

            // Düşük güven filtresi
            val topConfidence = probs.max()!! * 100f
            if (topConfidence < MIN_CONFIDENCE) {
                onDebug?.invoke("⚠️ Güven düşük (${topConfidence.toInt()}%), gösterilmiyor")
                return
            }

            // Top-3
            val top3 = probs.mapIndexed { idx, prob -> idx to prob }
                .sortedByDescending { it.second }
                .take(3)
                .map { (idx, prob) ->
                    Prediction(
                        labelTr = labelsTr.getOrElse(idx) { "Sınıf $idx" },
                        labelEn = labelsEn.getOrElse(idx) { "Class $idx" },
                        confidence = prob * 100f,
                        classId = idx
                    )
                }

            onPrediction?.invoke(top3)
            Log.d("SignPredictor", "Tahmin: ${top3.firstOrNull()?.labelTr} %${top3.firstOrNull()?.confidence?.toInt()}")

        } catch (e: Exception) {
            val msg = "❌ Tahmin hatası: ${e.javaClass.simpleName}: ${e.message}"
            Log.e("SignPredictor", msg)
            onDebug?.invoke(msg)
        }
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

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

    // Segmentasyon esikleri — config.py ile ayni (web ile tutarli)
    private val MOTION_THRESHOLD    = 0.008f   // web: 0.008
    private val IDLE_THRESHOLD      = 0.006f   // web: 0.006
    private val MIN_SIGN_FRAMES     = 15       // web: 15
    private val IDLE_FRAMES_TO_STOP = 10       // web: 10
    private val START_FRAMES        = 2        // web: 2
    private val MIN_CONFIDENCE      = 40f      // web CONFIDENCE_THRESHOLD=0.4 ile ayni

    // Temperature Scaling — web (pytorch_predictor.py) ile ayni
    // T > 1 -> model daha az emin, belirsiz siniflar ayrilir
    private val TEMPERATURE         = 1.5f

    // Margin filter — web ile ayni: top1-top2 < %15 ise 'belirsiz'
    private val MARGIN_THRESHOLD    = 0.15f    // web: 0.15
    private val MIN_HAND_FRAMES_DIVISOR = 8

    // Pre-buffer: isaret baslamadan onceki frame'leri yakala
    private val PRE_BUFFER_SIZE     = 8        // son 8 frame hafizada tut
    private val preBuffer = ArrayDeque<FloatArray>(PRE_BUFFER_SIZE + 2)

    // Tahmin oylama: son tahminleri tut, en sik tekrarlanani goster
    private val VOTE_HISTORY_SIZE   = 3
    private val predictionHistory = mutableListOf<Int>()  // son N class id

    // Cooldown: tahmin sonrasi bekleme (cift tetiklemeyi onle)
    private val COOLDOWN_FRAMES     = 20       // ~0.7s bekleme
    private var cooldownCounter     = 0

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
            onDebug?.invoke("Manuel tahmin (${signFrames.size} frame)")
            predictSign()
        } else {
            onDebug?.invoke("Yeterli frame yok: ${signFrames.size}/5")
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

    fun processLandmarks(landmarks: FloatArray) {
        if (landmarks.size != FEATURE_SIZE) return
        frameDebugCount++

        // Pre-buffer guncelle (her zaman son N frame'i tut)
        preBuffer.addLast(landmarks.copyOf())
        if (preBuffer.size > PRE_BUFFER_SIZE) {
            preBuffer.removeFirst()
        }

        // Cooldown kontrolu
        if (cooldownCounter > 0) {
            cooldownCounter--
            prevLandmarks = landmarks.copyOf()
            return
        }

        // El algilama kontrolu
        if (!hasHandLandmarks(landmarks)) {
            if (frameDebugCount % 90 == 0) {
                onDebug?.invoke("Eller net gorunmuyor")
            }
        }

        val motion = computeMotion(landmarks)

        if (frameDebugCount % 15 == 0) {
            val msg = "motion=${"%.4f".format(motion)} state=$state frames=${signFrames.size} idle=$idleFrames"
            Log.d("SignPredictor", msg)
            onDebug?.invoke(msg)
        }

        when (state) {
            "idle" -> {
                if (motion > MOTION_THRESHOLD) {
                    signingFrames++
                    if (signingFrames >= START_FRAMES) {
                        state = "signing"
                        signFrames.clear()
                        predictionHistory.clear()
                        // Pre-buffer'daki frame'leri ekle (isaretin basini yakala)
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

                if (motion < IDLE_THRESHOLD) {
                    idleFrames++
                    if (idleFrames >= IDLE_FRAMES_TO_STOP) {
                        if (signFrames.size >= MIN_SIGN_FRAMES) {
                            onDebug?.invoke("Isaret bitti (${signFrames.size} frame), tahmin yapiliyor...")
                            predictSign()
                            cooldownCounter = COOLDOWN_FRAMES  // cooldown baslat
                        } else {
                            onDebug?.invoke("Cok kisa isaret (${signFrames.size}/${MIN_SIGN_FRAMES}), atlandi")
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

                // Maks frame'e ulasinca zorla tahmin et
                if (signFrames.size >= SEQ_LENGTH) {
                    onDebug?.invoke("Maks frame ($SEQ_LENGTH), tahmin yapiliyor...")
                    predictSign()
                    cooldownCounter = COOLDOWN_FRAMES
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
        
        // Tek el bile hareket ediyorsa algıla
        // Sol el (132-194) ve sağ el (195-257) ayrı ayrı kontrol et
        val handStart = 132
        val leftHandEnd = 195
        val rightHandEnd = minOf(258, landmarks.size)
        
        // Sol el hareketi
        var leftSum = 0f
        for (i in handStart until leftHandEnd) {
            leftSum += abs(landmarks[i] - prev[i])
        }
        val leftMotion = leftSum / (leftHandEnd - handStart)
        
        // Sağ el hareketi
        var rightSum = 0f
        for (i in leftHandEnd until rightHandEnd) {
            rightSum += abs(landmarks[i] - prev[i])
        }
        val rightMotion = rightSum / (rightHandEnd - leftHandEnd)
        
        // En yüksek hareketi al (tek el yeterli)
        return maxOf(leftMotion, rightMotion)
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
            val handFrames = signFrames.count { hasHandLandmarks(it) }
            val minRequiredHandFrames = maxOf(3, n / MIN_HAND_FRAMES_DIVISOR)

            if (handFrames < minRequiredHandFrames) {
                onDebug?.invoke("El landmark zayif ($handFrames/$n frame), tahmin atlandi")
                predictionHistory.clear()
                return
            }

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

            // Temperature Scaling — logit'leri T'ye bol, sonra softmax uygula
            // Web (pytorch_predictor.py) ile ayni yaklasim
            val maxScore = scores.max()!!
            val expScores = FloatArray(scores.size) {
                Math.exp(((scores[it] - maxScore) / TEMPERATURE).toDouble()).toFloat()
            }
            val sumExp = expScores.sum()
            val probs = FloatArray(scores.size) { expScores[it] / sumExp }

            // Dusuk guven filtresi (web: CONFIDENCE_THRESHOLD=0.4)
            val topConfidence = probs.max()!! * 100f
            if (topConfidence < MIN_CONFIDENCE) {
                onDebug?.invoke("Guven dusuk (${topConfidence.toInt()}%), atlandi")
                return
            }

            // Margin filter — web ile ayni: top1 - top2 < %15 ise belirsiz say
            val sorted = probs.sortedDescending()
            val margin = sorted[0] - sorted[1]
            if (margin < MARGIN_THRESHOLD) {
                onDebug?.invoke("Belirsiz tahmin (margin: ${(margin*100).toInt()}% < ${(MARGIN_THRESHOLD*100).toInt()}%), atlandi")
                return
            }

            val topClassId = probs.indices.maxByOrNull { probs[it] } ?: 0

            // Tahmin gecmisine ekle (voting icin)
            predictionHistory.add(topClassId)
            if (predictionHistory.size > VOTE_HISTORY_SIZE) {
                predictionHistory.removeAt(0)
            }

            // Voting: son 3 tahminde en sik tekrarlanan sinifi bul
            val votedClassId = if (predictionHistory.size >= 2) {
                predictionHistory.groupBy { it }
                    .maxByOrNull { it.value.size }?.key ?: topClassId
            } else {
                topClassId
            }

            // Eger voting sonucu farkli ve tekrar sayisi >= 2 ise voted'i kullan
            val finalClassId = if (votedClassId != topClassId &&
                predictionHistory.count { it == votedClassId } >= 2) {
                onDebug?.invoke("Voting: ${labelsTr.getOrElse(topClassId){"?"}} -> ${labelsTr.getOrElse(votedClassId){"?"}} (${predictionHistory.count { it == votedClassId }}/${predictionHistory.size} oy)")
                votedClassId
            } else {
                topClassId
            }

            // Top-3 (finalClassId'yi gercekten ilk siraya koy)
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

    private fun hasHandLandmarks(frame: FloatArray): Boolean {
        var handMagnitude = 0f
        for (i in 132 until FEATURE_SIZE) {
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

package com.example.deepsigntid

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech
import java.util.Locale
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.deepsigntid.ui.theme.DeepSignTIDTheme
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

class MainActivity : ComponentActivity() {

    private val viewModel: MainViewModel by viewModels()
    private lateinit var cameraExecutor: ExecutorService
    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private var signPredictor: SignPredictor? = null
    private val frameCount = AtomicInteger(0)
    private var tts: TextToSpeech? = null

    private val requestPermissionLauncher = registerForActivityResult(

        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) setupMediaPipe()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        cameraExecutor = Executors.newSingleThreadExecutor()

        // TTS Init
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale("tr", "TR")
            }
        }

        // On-device tahmin
        signPredictor = try {
            SignPredictor(this).apply {
                onPrediction = { predictions -> viewModel.updatePredictions(predictions) }
                onStateChange = { state -> viewModel.updateSignState(state) }
                onDebug = { msg -> viewModel.updateDebugInfo(msg) }
            }.also { viewModel.signPredictor = it }
        } catch (e: Exception) {
            Log.e("MainActivity", "SignPredictor hata: ${e.message}")
            null
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            setupMediaPipe()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            DeepSignTIDTheme {
                MainScreen(
                    viewModel = viewModel,
                    onStartCamera = { startCamera(it) },
                    onSpeak = { text -> tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null) }
                )
            }
        }
    }

    private fun setupMediaPipe() {
        Thread {
            try {
                poseLandmarker = PoseLandmarker.createFromOptions(this,
                    PoseLandmarker.PoseLandmarkerOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_heavy.task").build())
                        .setRunningMode(RunningMode.VIDEO)
                        .setNumPoses(1)
                        .setMinPoseDetectionConfidence(0.3f)  // Eğitimle aynı (0.3)
                        .setMinPosePresenceConfidence(0.3f)   // Eğitimle aynı (0.3)
                        .setMinTrackingConfidence(0.3f)       // Video mode için takip güveni
                        .build())

                handLandmarker = HandLandmarker.createFromOptions(this,
                    HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
                        .setRunningMode(RunningMode.VIDEO)
                        .setNumHands(2)
                        .setMinHandDetectionConfidence(0.3f)  // Eğitimle aynı (0.3)
                        .setMinHandPresenceConfidence(0.3f)   // Eğitimle aynı (0.3)
                        .setMinTrackingConfidence(0.3f)       // Video mode için
                        .build())

                runOnUiThread { viewModel.updateDebugInfo("Hazir") }
            } catch (e: Exception) {
                Log.e("MainActivity", "MediaPipe hata: ${e.message}")
            }
        }.start()
    }

    private fun startCamera(previewView: PreviewView) {
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(cameraExecutor) { proxy -> processFrame(proxy) } }
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageAnalyzer)
            } catch (e: Exception) { Log.e("Camera", "Hata", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val currentFrame = frameCount.incrementAndGet()
        try {
            val poseLmkr = poseLandmarker
            val handLmkr = handLandmarker
            if (poseLmkr == null || handLmkr == null) { imageProxy.close(); return }

            // Bitmap + rotasyon
            val rawBitmap = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            val rotatedBitmap = if (rotation != 0) {
                val matrix = Matrix(); matrix.postRotate(rotation.toFloat())
                Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            } else rawBitmap

            // Yatay aynalama — web'deki cv2.flip(frame, 1) ile ayni
            // Web boyle yapiyor ve calisiyor, egitim verisiyle uyumlu
            val flipMatrix = Matrix(); flipMatrix.setScale(-1f, 1f, rotatedBitmap.width / 2f, 0f)
            val bitmap = Bitmap.createBitmap(rotatedBitmap, 0, 0, rotatedBitmap.width, rotatedBitmap.height, flipMatrix, true)

            val mpImage = BitmapImageBuilder(bitmap).build()
            
            // Generate timestamp for video tracking
            val timestampMs = imageProxy.imageInfo.timestamp / 1_000_000L

            // Landmark algila
            val result = detectLandmarks(poseLmkr, handLmkr, mpImage, bitmap.width, bitmap.height, timestampMs)

            if (result != null) {
                val (landmarks, overlay) = result

                // On-device tahmin (SignPredictor)
                signPredictor?.processLandmarks(landmarks)

                // Overlay guncelle (her 2. frame)
                if (currentFrame % 2 == 0) {
                    runOnUiThread { viewModel.updateLandmarks(overlay) }
                }
            }
        } catch (e: Exception) {
            Log.e("Frame", e.message ?: "error")
        }
        imageProxy.close()
    }

    /**
     * Landmark cikarmasi — EGITIM PIPELINE'I ILE BIREBIR ESLESIYOR:
     * - Pose: 33 x (x, y, z, visibility) = 132 feature → index 0-131
     * - Left hand ("Left" label): 21 x (x, y, z) = 63 feature → index 132-194
     * - Right hand ("Right" label): 21 x (x, y, z) = 63 feature → index 195-257
     * - TAKAS YOK — egitimde de takas yok
     */
    private fun detectLandmarks(
        poseLmkr: PoseLandmarker, handLmkr: HandLandmarker,
        mpImage: com.google.mediapipe.framework.image.MPImage, imgW: Int, imgH: Int, timestampMs: Long
    ): Pair<FloatArray, LandmarkOverlay>? {
        val output = FloatArray(258) // tum sifir (el bulunamazsa 0 kalir)
        val poseP = mutableListOf<LandmarkPoint>()
        val leftP = mutableListOf<LandmarkPoint>()
        val rightP = mutableListOf<LandmarkPoint>()

        // ── POSE ──
        try {
            val pr = poseLmkr.detectForVideo(mpImage, timestampMs)
            if (pr.landmarks().isEmpty()) return null // pose yoksa atla
            val pose = pr.landmarks()[0]
            for (i in 0 until minOf(pose.size, 33)) {
                output[i * 4 + 0] = pose[i].x()
                output[i * 4 + 1] = pose[i].y()
                output[i * 4 + 2] = pose[i].z()
                output[i * 4 + 3] = if (pose[i].visibility().isPresent) pose[i].visibility().get() else 1.0f
                poseP.add(LandmarkPoint(pose[i].x(), pose[i].y()))
            }
        } catch (_: Exception) { return null }

        // ── ELLER ──
        // Egitimle AYNI: "Left" → index 132, "Right" → index 195
        // TAKAS YOK!
        try {
            val hr = handLmkr.detectForVideo(mpImage, timestampMs)
            if (hr.landmarks().isNotEmpty()) {
                for (i in hr.landmarks().indices) {
                    if (i >= hr.handedness().size) break
                    val pts = hr.landmarks()[i]
                    val label = hr.handedness()[i][0].categoryName()

                    val coords = FloatArray(63)
                    val points = mutableListOf<LandmarkPoint>()
                    for (j in 0 until minOf(pts.size, 21)) {
                        coords[j * 3 + 0] = pts[j].x()
                        coords[j * 3 + 1] = pts[j].y()
                        coords[j * 3 + 2] = pts[j].z()
                        points.add(LandmarkPoint(pts[j].x(), pts[j].y()))
                    }

                    // Egitimle birebir: Left→132, Right→195
                    if (label == "Left") {
                        coords.copyInto(output, 132)
                        leftP.addAll(points)
                    } else {
                        coords.copyInto(output, 195)
                        rightP.addAll(points)
                    }
                }
            }
        } catch (_: Exception) {}

        return Pair(output, LandmarkOverlay(poseP, leftP, rightP, imgW, imgH))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        poseLandmarker?.close()
        handLandmarker?.close()
        tts?.stop()
        tts?.shutdown()
    }
}

// ─── Baglanti cizgileri ───
val POSE_CONNECTIONS = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 7, 0 to 4, 4 to 5, 5 to 6, 6 to 8,
    9 to 10, 11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16,
    11 to 23, 12 to 24, 23 to 24, 23 to 25, 24 to 26, 25 to 27, 26 to 28
)
val HAND_CONNECTIONS = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8,
    0 to 9, 9 to 10, 10 to 11, 11 to 12, 0 to 13, 13 to 14, 14 to 15, 15 to 16,
    0 to 17, 17 to 18, 18 to 19, 19 to 20, 5 to 9, 9 to 13, 13 to 17
)

fun mapLandmarkToCanvas(lmX: Float, lmY: Float, vw: Float, vh: Float, iw: Int, ih: Int): Offset {
    val imgAspect = iw.toFloat() / ih; val viewAspect = vw / vh
    val scale: Float; val ox: Float; val oy: Float
    if (imgAspect > viewAspect) { scale = vh / ih; ox = (iw * scale - vw) / 2f; oy = 0f }
    else { scale = vw / iw; ox = 0f; oy = (ih * scale - vh) / 2f }
    // Bitmap zaten flip'li, preview da flip'li → mirror gerekmez
    return Offset(lmX * iw * scale - ox, lmY * ih * scale - oy)
}

@Composable
fun MainScreen(viewModel: MainViewModel, onStartCamera: (PreviewView) -> Unit, onSpeak: (String) -> Unit) {
    val state by viewModel.state.collectAsState()

    Column(modifier = Modifier.fillMaxSize().background(Color(0xFF0A0A0A))) {
        // Header
        Box(modifier = Modifier.fillMaxWidth().background(Color(0xFF1A1A2E))
            .padding(top = 40.dp, start = 16.dp, end = 16.dp, bottom = 12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("DeepSign TID", color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.Bold)
                    Text("Türk İşaret Dili Tanıma", color = Color(0xFF888888), fontSize = 12.sp)
                }
                Box(modifier = Modifier.background(
                    if (state.signState == "signing") Color(0xFF4CAF50) else Color(0xFF2D2D4E),
                    RoundedCornerShape(20.dp)).padding(horizontal = 12.dp, vertical = 6.dp)) {
                    Text(if (state.signState == "signing") "KAYIT" else "Bekliyor",
                        color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
            }
        }

        // Kamera + overlay (büyük)
        Box(modifier = Modifier.fillMaxWidth().weight(1f)) {
            AndroidView(
                factory = { ctx -> PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }.also { onStartCamera(it) } },
                modifier = Modifier.fillMaxSize())
            Canvas(modifier = Modifier.fillMaxSize()) {
                val vw = size.width; val vh = size.height; val lm = state.landmarks
                fun pt(p: LandmarkPoint) = mapLandmarkToCanvas(p.x, p.y, vw, vh, lm.imageWidth, lm.imageHeight)
                for ((a, b) in POSE_CONNECTIONS) {
                    if (a < lm.poseLandmarks.size && b < lm.poseLandmarks.size)
                        drawLine(Color(0xAAFF7744), pt(lm.poseLandmarks[a]), pt(lm.poseLandmarks[b]), 3f)
                }
                lm.poseLandmarks.forEach { drawCircle(Color(0xFFFF6633), 4f, pt(it)) }
                for ((a, b) in HAND_CONNECTIONS) {
                    if (a < lm.leftHandLandmarks.size && b < lm.leftHandLandmarks.size)
                        drawLine(Color(0xFF44FF44), pt(lm.leftHandLandmarks[a]), pt(lm.leftHandLandmarks[b]), 3f)
                }
                lm.leftHandLandmarks.forEach { drawCircle(Color(0xFF00FF00), 5f, pt(it)) }
                for ((a, b) in HAND_CONNECTIONS) {
                    if (a < lm.rightHandLandmarks.size && b < lm.rightHandLandmarks.size)
                        drawLine(Color(0xFF4488FF), pt(lm.rightHandLandmarks[a]), pt(lm.rightHandLandmarks[b]), 3f)
                }
                lm.rightHandLandmarks.forEach { drawCircle(Color(0xFF3377FF), 5f, pt(it)) }
            }
            if (state.debugInfo.isNotEmpty()) {
                Text(state.debugInfo, color = Color.Yellow, fontSize = 10.sp,
                    modifier = Modifier.align(Alignment.BottomStart).background(Color(0x88000000)).padding(4.dp))
            }
        }

        // Alt panel: tahminler + cümle
        Column(modifier = Modifier.fillMaxWidth().background(Color(0xFF111111)).padding(10.dp)) {
            // Tahminler - 3 kutu yan yana
            Text("Tahminler", color = Color(0xFF666666), fontSize = 11.sp, modifier = Modifier.padding(bottom = 6.dp))
            if (state.predictions.isEmpty()) {
                Box(modifier = Modifier.fillMaxWidth().background(Color(0xFF1E1E1E), RoundedCornerShape(10.dp))
                    .padding(14.dp), contentAlignment = Alignment.Center) {
                    Text("İşaret yapın...", color = Color(0xFF444444), fontSize = 13.sp)
                }
            } else {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    state.predictions.forEachIndexed { i, p ->
                        PredictionCard(p, isTop = i == 0, onAdd = { viewModel.addWordToSentence(p.labelTr) }, modifier = Modifier.weight(1f))
                    }
                    repeat(3 - state.predictions.size) {
                        Box(modifier = Modifier.weight(1f).background(Color(0xFF1A1A1A), RoundedCornerShape(10.dp)).padding(12.dp))
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Cümle
            Text("Cümle", color = Color(0xFF666666), fontSize = 11.sp, modifier = Modifier.padding(bottom = 4.dp))
            Box(modifier = Modifier.fillMaxWidth().background(Color(0xFF1E1E1E), RoundedCornerShape(10.dp))
                .padding(12.dp).heightIn(min = 40.dp)) {
                Text(
                    if (state.sentence.isEmpty()) "Cümle burada görünecek..." else state.sentence.joinToString(" "),
                    color = if (state.sentence.isEmpty()) Color(0xFF444444) else Color.White,
                    fontSize = 16.sp, fontWeight = FontWeight.Medium
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedButton(onClick = { viewModel.removeLastWord() }, modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFFFF9800), containerColor = Color.Transparent)
                ) { Text("Geri Al", fontSize = 12.sp) }
                Button(onClick = { viewModel.clearSentence() }, modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFF44336))
                ) { Text("Temizle", fontSize = 12.sp) }
                Button(onClick = {
                    val textToSpeak = if (state.sentence.isNotEmpty()) state.sentence.joinToString(" ") else "Söylenecek kelime yok"
                    onSpeak(textToSpeak)
                }, modifier = Modifier.weight(1.2f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4CAF50))
                ) { Text("Seslendir", fontSize = 12.sp) }
            }
        }
    }
}

@Composable
fun PredictionCard(prediction: Prediction, isTop: Boolean, onAdd: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier = modifier
            .background(if (isTop) Color(0xFF1A2744) else Color(0xFF1E1E1E), RoundedCornerShape(10.dp))
            .clickable { onAdd() }
            .padding(10.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "${prediction.confidence.toInt()}%",
                color = if (isTop) Color(0xFF4CAF50) else Color(0xFF666666),
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = prediction.labelTr,
                color = Color.White,
                fontSize = if (isTop) 15.sp else 13.sp,
                fontWeight = if (isTop) FontWeight.Bold else FontWeight.Normal,
                maxLines = 2,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )
        }
    }
}
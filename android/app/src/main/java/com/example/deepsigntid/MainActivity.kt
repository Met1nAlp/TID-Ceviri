package com.example.deepsigntid

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.speech.tts.TextToSpeech
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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.deepsigntid.ui.theme.DeepSignTIDTheme
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.Locale
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
        ActivityResultContracts.RequestPermission(),
    ) { isGranted ->
        if (isGranted) setupMediaPipe()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        cameraExecutor = Executors.newSingleThreadExecutor()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale("tr", "TR")
                tts?.setSpeechRate(0.75f)
                tts?.setPitch(0.96f)
            }
        }

        signPredictor = try {
            SignPredictor(this).apply {
                onPrediction = { predictions -> viewModel.updatePredictions(predictions) }
                onStateChange = { state -> viewModel.updateSignState(state) }
                onDebug = { message -> viewModel.updateDebugInfo(message) }
                onSelectionStateChange = { selection -> viewModel.updateSelectionState(selection) }
                onSelectionConfirmed = { prediction ->
                    viewModel.addWordToSentence(prediction.labelTr)
                    viewModel.updateDebugInfo("Seçildi: ${prediction.labelTr}")
                }
            }.also { viewModel.signPredictor = it }
        } catch (e: Exception) {
            Log.e("MainActivity", "SignPredictor hata: ${e.message}")
            null
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            setupMediaPipe()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            DeepSignTIDTheme {
                MainScreen(
                    viewModel = viewModel,
                    onStartCamera = { startCamera(it) },
                    onSpeak = { text -> tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null) },
                    onPredictionTap = { prediction ->
                        viewModel.signPredictor?.cancelSelection("manual_selection")
                        viewModel.addWordToSentence(prediction.labelTr)
                    },
                )
            }
        }
    }

    private fun setupMediaPipe() {
        Thread {
            try {
                poseLandmarker = PoseLandmarker.createFromOptions(
                    this,
                    PoseLandmarker.PoseLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath("pose_landmarker_heavy.task")
                                .build(),
                        )
                        .setRunningMode(RunningMode.VIDEO)
                        .setNumPoses(1)
                        .setMinPoseDetectionConfidence(0.3f)
                        .setMinPosePresenceConfidence(0.3f)
                        .setMinTrackingConfidence(0.3f)
                        .build(),
                )

                handLandmarker = HandLandmarker.createFromOptions(
                    this,
                    HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(
                            BaseOptions.builder()
                                .setModelAssetPath("hand_landmarker.task")
                                .build(),
                        )
                        .setRunningMode(RunningMode.VIDEO)
                        .setNumHands(2)
                        .setMinHandDetectionConfidence(0.3f)
                        .setMinHandPresenceConfidence(0.3f)
                        .setMinTrackingConfidence(0.3f)
                        .build(),
                )

                runOnUiThread { viewModel.updateDebugInfo("Hazır") }
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
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(cameraExecutor) { proxy -> processFrame(proxy) }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalyzer,
                )
            } catch (e: Exception) {
                Log.e("Camera", "Kamera hatası", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val currentFrame = frameCount.incrementAndGet()
        try {
            val poseLmkr = poseLandmarker
            val handLmkr = handLandmarker
            if (poseLmkr == null || handLmkr == null) {
                imageProxy.close()
                return
            }

            val rawBitmap = imageProxy.toBitmap()
            val rotation = imageProxy.imageInfo.rotationDegrees
            val rotatedBitmap = if (rotation != 0) {
                val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                Bitmap.createBitmap(
                    rawBitmap,
                    0,
                    0,
                    rawBitmap.width,
                    rawBitmap.height,
                    matrix,
                    true,
                )
            } else {
                rawBitmap
            }

            val flipMatrix = Matrix().apply {
                setScale(-1f, 1f, rotatedBitmap.width / 2f, 0f)
            }
            val bitmap = Bitmap.createBitmap(
                rotatedBitmap,
                0,
                0,
                rotatedBitmap.width,
                rotatedBitmap.height,
                flipMatrix,
                true,
            )

            val mpImage = BitmapImageBuilder(bitmap).build()
            val timestampMs = imageProxy.imageInfo.timestamp / 1_000_000L

            val result = detectLandmarks(
                poseLmkr = poseLmkr,
                handLmkr = handLmkr,
                mpImage = mpImage,
                imgW = bitmap.width,
                imgH = bitmap.height,
                timestampMs = timestampMs,
            )

            if (result != null) {
                val (landmarks, overlay) = result
                signPredictor?.processLandmarks(landmarks)

                if (currentFrame % 2 == 0) {
                    runOnUiThread {
                        viewModel.updateLandmarks(overlay)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("Frame", e.message ?: "error")
        }
        imageProxy.close()
    }

    private fun detectLandmarks(
        poseLmkr: PoseLandmarker,
        handLmkr: HandLandmarker,
        mpImage: com.google.mediapipe.framework.image.MPImage,
        imgW: Int,
        imgH: Int,
        timestampMs: Long,
    ): Pair<FloatArray, LandmarkOverlay>? {
        val output = FloatArray(258)
        val posePoints = mutableListOf<LandmarkPoint>()
        val leftPoints = mutableListOf<LandmarkPoint>()
        val rightPoints = mutableListOf<LandmarkPoint>()

        try {
            val poseResult = poseLmkr.detectForVideo(mpImage, timestampMs)
            if (poseResult.landmarks().isNotEmpty()) {
                val pose = poseResult.landmarks()[0]
                for (i in 0 until minOf(pose.size, 33)) {
                    output[i * 4] = pose[i].x()
                    output[i * 4 + 1] = pose[i].y()
                    output[i * 4 + 2] = pose[i].z()
                    output[i * 4 + 3] = if (pose[i].visibility().isPresent) {
                        pose[i].visibility().get()
                    } else {
                        1.0f
                    }
                    posePoints.add(LandmarkPoint(pose[i].x(), pose[i].y()))
                }
            }
        } catch (_: Exception) {
            // Keep zero-filled pose slots when pose detection fails.
        }

        try {
            val handResult = handLmkr.detectForVideo(mpImage, timestampMs)
            if (handResult.landmarks().isNotEmpty()) {
                for (i in handResult.landmarks().indices) {
                    if (i >= handResult.handedness().size) break

                    val points = handResult.landmarks()[i]
                    val label = handResult.handedness()[i][0].categoryName()
                    val coords = FloatArray(63)
                    val overlayPoints = mutableListOf<LandmarkPoint>()

                    for (j in 0 until minOf(points.size, 21)) {
                        coords[j * 3] = points[j].x()
                        coords[j * 3 + 1] = points[j].y()
                        coords[j * 3 + 2] = points[j].z()
                        overlayPoints.add(LandmarkPoint(points[j].x(), points[j].y()))
                    }

                    if (label == "Left") {
                        coords.copyInto(output, 132)
                        leftPoints.addAll(overlayPoints)
                    } else {
                        coords.copyInto(output, 195)
                        rightPoints.addAll(overlayPoints)
                    }
                }
            }
        } catch (_: Exception) {
            // Keep zero-filled hand slots when hand detection fails.
        }

        return Pair(
            output,
            LandmarkOverlay(
                poseLandmarks = posePoints,
                leftHandLandmarks = leftPoints,
                rightHandLandmarks = rightPoints,
                imageWidth = imgW,
                imageHeight = imgH,
            ),
        )
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

val POSE_CONNECTIONS = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 7, 0 to 4, 4 to 5, 5 to 6, 6 to 8,
    9 to 10, 11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16,
    11 to 23, 12 to 24, 23 to 24, 23 to 25, 24 to 26, 25 to 27, 26 to 28,
)

val HAND_CONNECTIONS = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4, 0 to 5, 5 to 6, 6 to 7, 7 to 8,
    0 to 9, 9 to 10, 10 to 11, 11 to 12, 0 to 13, 13 to 14, 14 to 15, 15 to 16,
    0 to 17, 17 to 18, 18 to 19, 19 to 20, 5 to 9, 9 to 13, 13 to 17,
)

fun mapLandmarkToCanvas(
    lmX: Float,
    lmY: Float,
    viewWidth: Float,
    viewHeight: Float,
    imageWidth: Int,
    imageHeight: Int,
): Offset {
    val imageAspect = imageWidth.toFloat() / imageHeight
    val viewAspect = viewWidth / viewHeight

    val scale: Float
    val offsetX: Float
    val offsetY: Float

    if (imageAspect > viewAspect) {
        scale = viewHeight / imageHeight
        offsetX = (imageWidth * scale - viewWidth) / 2f
        offsetY = 0f
    } else {
        scale = viewWidth / imageWidth
        offsetX = 0f
        offsetY = (imageHeight * scale - viewHeight) / 2f
    }

    return Offset(
        x = lmX * imageWidth * scale - offsetX,
        y = lmY * imageHeight * scale - offsetY,
    )
}

@Composable
fun MainScreen(
    viewModel: MainViewModel,
    onStartCamera: (PreviewView) -> Unit,
    onSpeak: (String) -> Unit,
    onPredictionTap: (Prediction) -> Unit,
) {
    val state by viewModel.state.collectAsState()
    val selection = state.selection

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0A0A0A)),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF1A1A2E))
                .padding(top = 40.dp, start = 16.dp, end = 16.dp, bottom = 12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "DeepSign TID",
                        color = Color.White,
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        text = "Türk İşaret Dili Tanıma",
                        color = Color(0xFF888888),
                        fontSize = 12.sp,
                    )
                }

                val headerLabel = when {
                    selection.active -> "SEÇİM"
                    state.signState == "signing" -> "KAYIT"
                    else -> "Bekliyor"
                }
                val headerColor = when {
                    selection.active -> Color(0xFF00897B)
                    state.signState == "signing" -> Color(0xFF4CAF50)
                    else -> Color(0xFF2D2D4E)
                }

                Box(
                    modifier = Modifier
                        .background(headerColor, RoundedCornerShape(20.dp))
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                ) {
                    Text(
                        text = headerLabel,
                        color = Color.White,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
        ) {
            AndroidView(
                factory = { context ->
                    PreviewView(context).apply {
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                    }.also { onStartCamera(it) }
                },
                modifier = Modifier.fillMaxSize(),
            )

            Canvas(modifier = Modifier.fillMaxSize()) {
                val viewWidth = size.width
                val viewHeight = size.height
                val landmarks = state.landmarks

                fun point(item: LandmarkPoint): Offset {
                    return mapLandmarkToCanvas(
                        lmX = item.x,
                        lmY = item.y,
                        viewWidth = viewWidth,
                        viewHeight = viewHeight,
                        imageWidth = landmarks.imageWidth,
                        imageHeight = landmarks.imageHeight,
                    )
                }

                for ((a, b) in POSE_CONNECTIONS) {
                    if (a < landmarks.poseLandmarks.size && b < landmarks.poseLandmarks.size) {
                        drawLine(
                            color = Color(0xAAFF7744),
                            start = point(landmarks.poseLandmarks[a]),
                            end = point(landmarks.poseLandmarks[b]),
                            strokeWidth = 3f,
                        )
                    }
                }
                landmarks.poseLandmarks.forEach { drawCircle(Color(0xFFFF6633), 4f, point(it)) }

                for ((a, b) in HAND_CONNECTIONS) {
                    if (a < landmarks.leftHandLandmarks.size && b < landmarks.leftHandLandmarks.size) {
                        drawLine(
                            color = Color(0xFF44FF44),
                            start = point(landmarks.leftHandLandmarks[a]),
                            end = point(landmarks.leftHandLandmarks[b]),
                            strokeWidth = 3f,
                        )
                    }
                }
                landmarks.leftHandLandmarks.forEach { drawCircle(Color(0xFF00FF00), 5f, point(it)) }

                for ((a, b) in HAND_CONNECTIONS) {
                    if (a < landmarks.rightHandLandmarks.size && b < landmarks.rightHandLandmarks.size) {
                        drawLine(
                            color = Color(0xFF4488FF),
                            start = point(landmarks.rightHandLandmarks[a]),
                            end = point(landmarks.rightHandLandmarks[b]),
                            strokeWidth = 3f,
                        )
                    }
                }
                landmarks.rightHandLandmarks.forEach { drawCircle(Color(0xFF3377FF), 5f, point(it)) }
            }

            if (state.debugInfo.isNotEmpty()) {
                Text(
                    text = state.debugInfo,
                    color = Color.Yellow,
                    fontSize = 10.sp,
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .background(Color(0x88000000))
                        .padding(4.dp),
                )
            }
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF111111))
                .padding(10.dp),
        ) {
            Text(
                text = "Tahminler",
                color = Color(0xFF666666),
                fontSize = 11.sp,
                modifier = Modifier.padding(bottom = 6.dp),
            )

            if (selection.active) {
                SelectionPanel(selection = selection)
                Spacer(modifier = Modifier.height(8.dp))
            }

            if (state.predictions.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF1E1E1E), RoundedCornerShape(10.dp))
                        .padding(14.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "İşaret yapın...",
                        color = Color(0xFF444444),
                        fontSize = 13.sp,
                    )
                }
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    state.predictions.forEachIndexed { index, prediction ->
                        PredictionCard(
                            prediction = prediction,
                            isTop = index == 0,
                            selectionNumber = index + 1,
                            onAdd = { onPredictionTap(prediction) },
                            modifier = Modifier.weight(1f),
                        )
                    }

                    repeat(3 - state.predictions.size) {
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .background(Color(0xFF1A1A1A), RoundedCornerShape(10.dp))
                                .padding(12.dp),
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Cümle",
                color = Color(0xFF666666),
                fontSize = 11.sp,
                modifier = Modifier.padding(bottom = 4.dp),
            )
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF1E1E1E), RoundedCornerShape(10.dp))
                    .padding(12.dp)
                    .heightIn(min = 40.dp),
            ) {
                Text(
                    text = if (state.sentence.isEmpty()) {
                        "Cümle burada görünecek..."
                    } else {
                        state.sentence.joinToString(" ")
                    },
                    color = if (state.sentence.isEmpty()) Color(0xFF444444) else Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium,
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                OutlinedButton(
                    onClick = { viewModel.removeLastWord() },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = Color(0xFFFF9800),
                        containerColor = Color.Transparent,
                    ),
                ) {
                    Text(text = "Geri Al", fontSize = 12.sp)
                }

                Button(
                    onClick = { viewModel.clearSentence() },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFF44336)),
                ) {
                    Text(text = "Temizle", fontSize = 12.sp)
                }

                Button(
                    onClick = {
                        val textToSpeak = if (state.sentence.isNotEmpty()) {
                            state.sentence.joinToString(" ")
                        } else {
                            "Söylenecek kelime yok"
                        }
                        onSpeak(textToSpeak)
                    },
                    modifier = Modifier.weight(1.2f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4CAF50)),
                ) {
                    Text(text = "Seslendir", fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
fun SelectionPanel(selection: DigitSelectionState) {
    val remainingSeconds = selection.remainingMs / 1000f
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = Color(0xFF0F2B2A),
        shape = RoundedCornerShape(12.dp),
        tonalElevation = 0.dp,
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "1-2-3 ile onayla",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f),
                )
                Text(
                    text = "${"%.1f".format(remainingSeconds)} sn",
                    color = Color(0xFF9EE8DD),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = "Birinci, ikinci veya üçüncü tahmini elinle seçebilirsin.",
                color = Color(0xFFC7E8E3),
                fontSize = 12.sp,
            )

            if (selection.lastDigitValue != null) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Son sayı: ${selection.lastDigitValue}  •  Güven: ${selection.lastConfidence.toInt()}%",
                    color = Color(0xFF9EE8DD),
                    fontSize = 12.sp,
                )
            }

            if (selection.stableDigit != null) {
                Text(
                    text = "Kararlı sayı: ${selection.stableDigit} (${selection.stableVotes}/${selection.requiredStableFrames})",
                    color = Color(0xFF9EE8DD),
                    fontSize = 12.sp,
                )
            }
        }
    }
}

@Composable
fun PredictionCard(
    prediction: Prediction,
    isTop: Boolean,
    selectionNumber: Int,
    onAdd: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .background(
                if (isTop) Color(0xFF1A2744) else Color(0xFF1E1E1E),
                RoundedCornerShape(10.dp),
            )
            .clickable { onAdd() }
            .padding(10.dp),
    ) {
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .background(Color(0xFF233A62), RoundedCornerShape(50))
                .padding(horizontal = 8.dp, vertical = 3.dp),
        ) {
            Text(
                text = selectionNumber.toString(),
                color = Color.White,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
            )
        }

        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "${prediction.confidence.toInt()}%",
                color = if (isTop) Color(0xFF4CAF50) else Color(0xFF666666),
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = prediction.labelTr,
                color = Color.White,
                fontSize = if (isTop) 15.sp else 13.sp,
                fontWeight = if (isTop) FontWeight.Bold else FontWeight.Normal,
                maxLines = 2,
                textAlign = TextAlign.Center,
            )
        }
    }
}

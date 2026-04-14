package com.example.deepsigntid.network

import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.GET

data class PredictionResponse(
    val predictions: List<ServerPrediction>,
    val sign_state: String
)

data class ServerPrediction(
    val label_tr: String,
    val label_en: String,
    val confidence: Float
)

interface ApiService {
    @Multipart
    @POST("/predict_frame")
    suspend fun predictFrame(
        @Part frame: MultipartBody.Part
    ): Response<PredictionResponse>

    @Multipart
    @POST("/predict_sign")
    suspend fun predictSign(
        @Part frames: List<MultipartBody.Part>
    ): Response<PredictionResponse>

    @GET("/ping")
    suspend fun ping(): Response<Map<String, String>>
}

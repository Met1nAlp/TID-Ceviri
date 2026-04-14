package com.example.deepsigntid.network

import com.google.gson.GsonBuilder
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitClient {
    // Varsayılan: aynı WiFi ağındaki bilgisayar
    // Kullanıcı bunu değiştirebilir
    var baseUrl: String = "http://10.5.49.209:5000"
        private set

    private var retrofit: Retrofit? = null

    private val client = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .build()

    fun updateBaseUrl(url: String) {
        baseUrl = url
        retrofit = null // Force rebuild
    }

    fun getApiService(): ApiService {
        if (retrofit == null) {
            val gson = GsonBuilder().create()
            retrofit = Retrofit.Builder()
                .baseUrl(baseUrl)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build()
        }
        return retrofit!!.create(ApiService::class.java)
    }
}

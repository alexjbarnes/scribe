package com.alexb151.verba

import android.app.Application

class VerbaApp : Application() {
    companion object {
        var nativeLoadError: String? = null
    }

    override fun onCreate() {
        super.onCreate()
        try {
            System.loadLibrary("verba_rs_lib")
        } catch (e: Throwable) {
            nativeLoadError = "${e::class.java.name}: ${e.message}"
            android.util.Log.e("VerbaApp", "Failed to load native library", e)
        }
    }
}

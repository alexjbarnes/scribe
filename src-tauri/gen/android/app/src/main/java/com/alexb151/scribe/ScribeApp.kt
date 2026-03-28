package com.alexb151.scribe

import android.app.Application

class ScribeApp : Application() {
    companion object {
        var nativeLoadError: String? = null
    }

    override fun onCreate() {
        super.onCreate()
        try {
            System.loadLibrary("scribe_rs_lib")
        } catch (e: Throwable) {
            nativeLoadError = "${e::class.java.name}: ${e.message}"
            android.util.Log.e("ScribeApp", "Failed to load native library", e)
        }
    }
}

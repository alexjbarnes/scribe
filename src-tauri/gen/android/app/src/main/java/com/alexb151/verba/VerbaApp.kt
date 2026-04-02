package com.alexb151.verba

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.widget.Toast

class VerbaApp : Application() {
    companion object {
        var instance: VerbaApp? = null
        var nativeLoadError: String? = null

        @JvmStatic
        fun showToast(msg: String) {
            val app = instance ?: return
            Handler(Looper.getMainLooper()).post {
                Toast.makeText(app, msg, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        try {
            System.loadLibrary("verba_rs_lib")
        } catch (e: Throwable) {
            nativeLoadError = "${e::class.java.name}: ${e.message}"
            android.util.Log.e("VerbaApp", "Failed to load native library", e)
        }
    }
}

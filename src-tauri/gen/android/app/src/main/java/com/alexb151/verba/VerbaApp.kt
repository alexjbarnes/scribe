package com.alexb151.verba

import android.app.Application
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
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

        private var audioFocusRequest: AudioFocusRequest? = null

        @JvmStatic
        fun requestAudioFocus(): Boolean {
            val app = instance ?: return false
            val am = app.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return false
            val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .build()
            val result = am.requestAudioFocus(req)
            return if (result == AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                audioFocusRequest = req
                true
            } else {
                false
            }
        }

        @JvmStatic
        fun abandonAudioFocus(): Boolean {
            val app = instance ?: return false
            val am = app.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return false
            val req = audioFocusRequest ?: return false
            audioFocusRequest = null
            return am.abandonAudioFocusRequest(req) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
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

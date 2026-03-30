package com.alexb151.verba

import android.os.Bundle
import android.widget.TextView
import android.widget.ScrollView
import android.util.TypedValue
import androidx.activity.enableEdgeToEdge

class MainActivity : TauriActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    val err = VerbaApp.nativeLoadError
    if (err != null) {
      super.onCreate(savedInstanceState)
      val sv = ScrollView(this)
      val tv = TextView(this)
      tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
      tv.setPadding(32, 100, 32, 32)
      tv.text = "Native library failed to load:\n\n$err"
      sv.addView(tv)
      setContentView(sv)
      return
    }
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
  }
}

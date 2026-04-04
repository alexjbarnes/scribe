package com.alexb151.verba

import android.accessibilityservice.AccessibilityService
import android.content.BroadcastReceiver
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.PixelFormat
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.content.SharedPreferences
import android.animation.ValueAnimator
import android.widget.Toast
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.util.TypedValue
import java.util.concurrent.Executors
import kotlin.math.abs
import org.json.JSONArray

class VerbaAccessibilityService : AccessibilityService() {

    companion object {
        private const val LOG_ERROR = 0
        private const val LOG_WARN = 1
        private const val LOG_INFO = 2
        private const val LOG_DEBUG = 3

        @JvmStatic
        private external fun nativeInit(dataDir: String): Boolean
        @JvmStatic
        private external fun nativePreloadModel()
        @JvmStatic
        private external fun nativeStartRecording(): Boolean
        @JvmStatic
        private external fun nativeStopAndTranscribe(): String?
        @JvmStatic
        private external fun nativeStopAndTranscribeRaw(): String?
        @JvmStatic
        private external fun nativeMatchSnippet(text: String): String?
        @JvmStatic
        private external fun nativeListSnippets(): String?
        @JvmStatic
        private external fun nativeAddSnippetTrigger(id: String, trigger: String)
        @JvmStatic
        private external fun nativeDestroy()
        @JvmStatic
        private external fun nativeLog(level: Int, msg: String)

        private fun logD(msg: String) { try { nativeLog(LOG_DEBUG, msg) } catch (_: Exception) {} }
        private fun logI(msg: String) { try { nativeLog(LOG_INFO, msg) } catch (_: Exception) {} }
        private fun logW(msg: String) { try { nativeLog(LOG_WARN, msg) } catch (_: Exception) {} }
        private fun logE(msg: String) { try { nativeLog(LOG_ERROR, msg) } catch (_: Exception) {} }
    }

    @Volatile private var initialized = false
    @Volatile private var recording = false
    @Volatile private var processing = false
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var windowManager: WindowManager? = null
    private var overlayView: View? = null
    private var micButton: ImageView? = null
    private var isOverlayShowing = false
    private var focusedNode: AccessibilityNodeInfo? = null

    private var snippetPickerView: View? = null

    // Drag + long-press state
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDragging = false
    @Volatile private var snippetMode = false
    private var longPressRunnable: Runnable? = null
    private var longPressFired = false
    private val longPressDelayMs = 500L

    private var ringView: RoundedRingView? = null
    private var ringAnimator: ValueAnimator? = null

    // Debounced hide to avoid flickering when keyboard opens
    private var pendingHide: Runnable? = null

    // After showing on VIEW_FOCUSED, verify keyboard appeared within this
    // window. If it didn't, the focus was phantom (e.g. Maps transition).
    private var pendingKeyboardCheck: Runnable? = null
    private val KEYBOARD_VERIFY_MS = 1500L

    // Saved position
    private var savedX = -1
    private var savedY = -1

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                Intent.ACTION_SCREEN_OFF -> hideOverlay()
                Intent.ACTION_USER_PRESENT -> if (focusedNode != null) showOverlay()
            }
        }
    }
    private val prefs: SharedPreferences by lazy {
        getSharedPreferences("verba_overlay", MODE_PRIVATE)
    }

    override fun onCreate() {
        super.onCreate()
        logI("onCreate")
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

        val dataDir = applicationContext.dataDir.absolutePath
        logI("dataDir=$dataDir")
        executor.execute {
            logI("nativeInit starting")
            initialized = nativeInit(dataDir)
            if (initialized) {
                logI("nativeInit succeeded, preloading model")
                mainHandler.post { toast("Loading model...") }
                nativePreloadModel()
                logI("model preloaded")
                mainHandler.post {
                    toast("Model ready")
                    micButton?.alpha = 1.0f
                }
            } else {
                logE("nativeInit failed")
                mainHandler.post {
                    toast("No model found. Open the app to download one.")
                    micButton?.alpha = 0.4f
                }
            }
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        logI("onServiceConnected")
        createOverlay()
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_USER_PRESENT)
        }
        registerReceiver(screenReceiver, filter)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return

        val typeName = when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_FOCUSED -> "VIEW_FOCUSED"
            AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> "TEXT_CHANGED"
            AccessibilityEvent.TYPE_VIEW_TEXT_SELECTION_CHANGED -> "TEXT_SEL_CHANGED"
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> "WINDOW_STATE"
            AccessibilityEvent.TYPE_WINDOWS_CHANGED -> "WINDOWS_CHANGED"
            else -> return
        }

        when (event.eventType) {
            AccessibilityEvent.TYPE_WINDOWS_CHANGED -> {
                // Window list changed (keyboard appeared/disappeared, dialog
                // opened/closed). If overlay is showing and keyboard gone,
                // the user likely left the text field.
                if (isOverlayShowing && !recording && !isKeyboardVisible()) {
                    logD("overlay[$typeName]: keyboard no longer visible, scheduling hide")
                    scheduleHide()
                }
            }
            AccessibilityEvent.TYPE_VIEW_FOCUSED -> {
                val source = event.source ?: return
                val editable = isEditableTextField(source)
                logD("overlay[$typeName]: class=${source.className} editable=$editable focused=${source.isFocused} sel=${source.textSelectionStart}..${source.textSelectionEnd}")
                if (editable) {
                    logI("overlay[$typeName]: SHOW (editable field focused)")
                    cancelPendingHide()
                    cancelKeyboardVerification()
                    focusedNode?.recycle()
                    focusedNode = source
                    showOverlay()
                    // If keyboard isn't visible yet, schedule a verification.
                    // Real focus: keyboard appears within ~300ms, cancels check.
                    // Phantom focus (Maps transition): keyboard never comes, hide.
                    if (!isKeyboardVisible()) {
                        scheduleKeyboardVerification()
                    }
                } else {
                    source.recycle()
                    if (!recording) {
                        logD("overlay[$typeName]: non-editable focused, scheduling hide")
                        scheduleHide()
                    }
                }
            }
            AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED,
            AccessibilityEvent.TYPE_VIEW_TEXT_SELECTION_CHANGED -> {
                // Update tracked node if overlay is showing.
                // Never trigger show from these (browser address bars
                // fire TEXT_SEL_CHANGED on resume without user interaction).
                val source = event.source ?: return
                val editable = isEditableTextField(source)
                logD("overlay[$typeName]: class=${source.className} editable=$editable focused=${source.isFocused} sel=${source.textSelectionStart}..${source.textSelectionEnd}")
                if (editable && isOverlayShowing) {
                    focusedNode?.recycle()
                    focusedNode = source
                } else {
                    source.recycle()
                }
            }
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                val pkg = event.packageName?.toString() ?: ""
                val cls = event.className?.toString() ?: ""
                logD("overlay[$typeName]: pkg=$pkg class=$cls")

                // Ignore our own overlay
                if (pkg == packageName) {
                    logD("overlay[$typeName]: ignoring own package")
                    return
                }

                // Keyboard appeared: strong signal for text input.
                if (cls.contains("SoftInputWindow") || cls.contains("InputMethodService")) {
                    logD("overlay[$typeName]: keyboard appeared")
                    cancelPendingHide()
                    cancelKeyboardVerification()
                    if (!isOverlayShowing) {
                        val editable = findFocusedEditText()
                        if (editable != null) {
                            logI("overlay[$typeName]: SHOW (keyboard + focused editable: ${editable.className})")
                            focusedNode?.recycle()
                            focusedNode = editable
                        } else {
                            logI("overlay[$typeName]: SHOW (keyboard opened, no editable found)")
                        }
                        showOverlay()
                    }
                    return
                }

                // Ignore system UI packages (notifications, spell check, etc.)
                if (pkg.startsWith("com.android.systemui")
                    || pkg.startsWith("com.google.android.ext.services")
                    || pkg.startsWith("com.google.android.inputmethod")) {
                    logD("overlay[$typeName]: ignoring system package")
                    return
                }

                // Real app-level window change: only hide if it looks like an
                // activity switch (not a popup/dialog within the current app).
                if (recording) {
                    logD("overlay[$typeName]: recording active, not hiding")
                    return
                }

                if (cls.contains("Activity")) {
                    if (!isOverlayShowing) {
                        val editable = findFocusedEditText()
                        if (editable != null) {
                            logI("overlay[$typeName]: SHOW (activity + focused editable: ${editable.className})")
                            cancelPendingHide()
                            cancelKeyboardVerification()
                            focusedNode?.recycle()
                            focusedNode = editable
                            showOverlay()
                            return
                        }
                        logD("overlay[$typeName]: no focused editable found")
                    }
                    scheduleHide()
                } else {
                    // Non-activity window (dialog, popup, Maps navigation
                    // overlay, etc). If showing, verify keyboard is still up.
                    if (isOverlayShowing) {
                        logD("overlay[$typeName]: non-activity window ($cls), scheduling hide")
                        scheduleHide()
                    } else {
                        logD("overlay[$typeName]: ignoring non-activity window ($cls)")
                    }
                }
            }
        }
    }

    private fun scheduleHide() {
        if (recording) {
            logD("overlay[scheduleHide]: recording active, not scheduling")
            return
        }
        cancelPendingHide()
        pendingHide = Runnable {
            if (recording) {
                logD("overlay[scheduleHide]: recording active, skipping hide")
                return@Runnable
            }
            if (isKeyboardVisible()) {
                logD("overlay[scheduleHide]: keyboard still visible, keeping overlay")
            } else {
                logI("overlay[scheduleHide]: keyboard not visible, HIDE")
                hideOverlay()
                focusedNode?.recycle()
                focusedNode = null
            }
        }
        mainHandler.postDelayed(pendingHide!!, 500)
    }

    private fun cancelPendingHide() {
        pendingHide?.let { mainHandler.removeCallbacks(it) }
        pendingHide = null
    }

    /** True when the soft keyboard window is on screen. */
    private fun isKeyboardVisible(): Boolean {
        return try {
            windows.any { it.type == AccessibilityWindowInfo.TYPE_INPUT_METHOD }
        } catch (e: Exception) {
            false
        }
    }

    /**
     * After showing on VIEW_FOCUSED, verify keyboard appeared.
     * Real focus triggers keyboard within ~300ms. Phantom focus (Maps
     * transition) does not. If keyboard never shows, hide the overlay.
     */
    private fun scheduleKeyboardVerification() {
        cancelKeyboardVerification()
        pendingKeyboardCheck = Runnable {
            if (isOverlayShowing && !recording && !isKeyboardVisible()) {
                logI("overlay[keyboardCheck]: keyboard never appeared, HIDE (phantom focus)")
                hideOverlay()
                focusedNode?.recycle()
                focusedNode = null
            }
        }
        mainHandler.postDelayed(pendingKeyboardCheck!!, KEYBOARD_VERIFY_MS)
    }

    private fun cancelKeyboardVerification() {
        pendingKeyboardCheck?.let { mainHandler.removeCallbacks(it) }
        pendingKeyboardCheck = null
    }

    override fun onInterrupt() {
        logW("onInterrupt called")
    }

    private fun isEditableTextField(node: AccessibilityNodeInfo): Boolean {
        if (node.isEditable) return true
        val className = node.className?.toString() ?: return false
        return className.contains("EditText") || className.contains("AutoCompleteTextView")
    }

    private fun createOverlay() {
        if (overlayView != null) return

        val density = resources.displayMetrics.density
        val buttonSize = (48 * density).toInt()

        val ringSize = (56 * density).toInt()
        val container = FrameLayout(this)

        // Recording ring (hidden by default, sits behind button)
        val ring = RoundedRingView(this).apply {
            visibility = View.INVISIBLE
        }
        val ringLp = FrameLayout.LayoutParams(ringSize, ringSize).apply {
            gravity = Gravity.CENTER
        }
        container.addView(ring, ringLp)
        ringView = ring

        val button = ImageView(this).apply {
            setImageResource(R.drawable.ic_overlay)
            setBackgroundResource(R.drawable.mic_button_bg)
            scaleType = ImageView.ScaleType.CENTER_CROP
            clipToOutline = true
            outlineProvider = object : android.view.ViewOutlineProvider() {
                override fun getOutline(view: android.view.View, outline: android.graphics.Outline) {
                    val r = (12 * density).toInt()
                    outline.setRoundRect(0, 0, view.width, view.height, r.toFloat())
                }
            }
            elevation = 0f
            // Start dimmed until init completes
            alpha = if (initialized) 1.0f else 0.5f
        }
        val btnLp = FrameLayout.LayoutParams(buttonSize, buttonSize).apply {
            gravity = Gravity.CENTER
        }
        container.addView(button, btnLp)
        micButton = button

        container.setOnTouchListener { _, event ->
            handleTouch(event)
        }

        overlayView = container
    }

    private fun handleTouch(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.rawX
                lastTouchY = event.rawY
                isDragging = false
                longPressFired = false
                micButton?.scaleX = 0.85f
                micButton?.scaleY = 0.85f

                // Schedule long-press → snippet mode
                longPressRunnable = Runnable {
                    if (!isDragging && !recording && !processing) {
                        longPressFired = true
                        startSnippetRecording()
                    }
                }
                mainHandler.postDelayed(longPressRunnable!!, longPressDelayMs)
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = event.rawX - lastTouchX
                val dy = event.rawY - lastTouchY
                if (abs(dx) > 8 || abs(dy) > 8 || isDragging) {
                    isDragging = true
                    cancelLongPress()
                    micButton?.scaleX = 1.0f
                    micButton?.scaleY = 1.0f
                    val params = overlayView?.layoutParams as? WindowManager.LayoutParams
                    if (params != null) {
                        params.x -= dx.toInt()
                        params.y -= dy.toInt()
                        windowManager?.updateViewLayout(overlayView, params)
                    }
                    lastTouchX = event.rawX
                    lastTouchY = event.rawY
                }
                return true
            }
            MotionEvent.ACTION_UP -> {
                cancelLongPress()
                micButton?.scaleX = 1.0f
                micButton?.scaleY = 1.0f

                if (longPressFired) {
                    // This is the release after the long-press that just started
                    // snippet recording — let it continue. Next tap stops it.
                    longPressFired = false
                } else if (!isDragging) {
                    onMicClick()
                } else {
                    val params = overlayView?.layoutParams as? WindowManager.LayoutParams
                    if (params != null) {
                        savedX = params.x
                        savedY = params.y
                        prefs.edit().putInt("overlay_x", savedX).putInt("overlay_y", savedY).apply()
                    }
                }
                return true
            }
        }
        return false
    }

    private fun cancelLongPress() {
        longPressRunnable?.let { mainHandler.removeCallbacks(it) }
        longPressRunnable = null
    }

    private fun showOverlay() {
        if (isOverlayShowing || overlayView == null) {
            logD("showOverlay: skip (showing=$isOverlayShowing, view=${overlayView != null})")
            return
        }

        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP_MR1) {
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_SYSTEM_ALERT
        }

        val density = resources.displayMetrics.density
        val defaultX = (16 * density).toInt()
        val defaultY = (80 * density).toInt()

        if (savedX < 0) {
            savedX = prefs.getInt("overlay_x", defaultX)
            savedY = prefs.getInt("overlay_y", defaultY)
        }

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.END
            x = savedX
            y = savedY
        }

        try {
            windowManager?.addView(overlayView, params)
            isOverlayShowing = true
            logI("showOverlay: added at x=$savedX y=$savedY")
        } catch (e: Exception) {
            logE("showOverlay: failed to add view: $e")
        }
    }

    private fun hideOverlay() {
        if (!isOverlayShowing || overlayView == null) return
        logI("hideOverlay")
        cancelKeyboardVerification()
        dismissSnippetPicker()

        try {
            windowManager?.removeView(overlayView)
        } catch (e: Exception) {
            logE("hideOverlay: failed to remove view: $e")
        }
        isOverlayShowing = false
    }

    private fun onMicClick() {
        logI("onMicClick: initialized=$initialized recording=$recording processing=$processing snippetMode=$snippetMode")
        if (!initialized) {
            toast("Model still loading...")
            return
        }
        if (processing) {
            logD("onMicClick: ignoring, transcription in progress")
            return
        }

        if (recording) {
            if (snippetMode) stopAndMatchSnippet() else stopAndTranscribe()
        } else {
            startRecording()
        }
    }

    private fun startRecordingRing(snippet: Boolean = false) {
        ringView?.let { ring ->
            ring.setColor(if (snippet) 0xFFFF9944.toInt() else 0xFFFF4444.toInt())
            ring.visibility = View.VISIBLE
            ringAnimator?.cancel()
            ringAnimator = ValueAnimator.ofFloat(1f, 0f).apply {
                duration = 1500
                repeatCount = ValueAnimator.INFINITE
                interpolator = android.view.animation.LinearInterpolator()
                addUpdateListener { ring.setPhase(it.animatedValue as Float) }
                start()
            }
        }
    }

    private fun showProcessingRing() {
        ringView?.let { ring ->
            ring.setColor(0xFF4488FF.toInt())
        }
    }

    private fun flashCompleteRing() {
        ringAnimator?.cancel()
        ringAnimator = null
        ringView?.let { ring ->
            ring.setColor(0xFF44BB44.toInt())
            ring.setSolid()
            mainHandler.postDelayed({
                ring.visibility = View.INVISIBLE
            }, 1500)
        }
    }

    private fun hideRing() {
        ringAnimator?.cancel()
        ringAnimator = null
        ringView?.visibility = View.INVISIBLE
    }

    private fun startRecording() {
        logI("startRecording")
        recording = true
        hapticFeedback()
        startRecordingRing()

        executor.execute {
            val started = nativeStartRecording()
            logI("nativeStartRecording returned $started")
            if (!started) {
                mainHandler.post {
                    recording = false
                    hideRing()
                    toast("Failed to start recording")
                }
            }
        }
    }

    private fun stopAndTranscribe() {
        logI("stopAndTranscribe")
        recording = false
        processing = true
        hapticFeedback()
        showProcessingRing()

        executor.execute {
            val text = nativeStopAndTranscribe()
            logI("nativeStopAndTranscribe returned ${text?.length ?: 0} chars")
            mainHandler.post {
                processing = false
                if (!text.isNullOrEmpty()) {
                    logI("injecting text: \"${text.take(80)}\"")
                    flashCompleteRing()
                    injectText(text)
                } else {
                    logW("no text returned from transcription")
                    hideRing()
                    toast("No speech detected")
                }
            }
        }
    }

    private fun startSnippetRecording() {
        if (!initialized || processing || recording) return
        logI("startSnippetRecording (long-press)")
        snippetMode = true
        recording = true
        hapticFeedback()
        startRecordingRing(snippet = true)

        executor.execute {
            val started = nativeStartRecording()
            logI("snippet nativeStartRecording returned $started")
            if (!started) {
                mainHandler.post {
                    snippetMode = false
                    recording = false
                    hideRing()
                    toast("Failed to start recording")
                }
            }
        }
    }

    private fun stopAndMatchSnippet() {
        logI("stopAndMatchSnippet")
        recording = false
        processing = true
        hapticFeedback()
        showProcessingRing()

        executor.execute {
            val text = nativeStopAndTranscribeRaw()
            logI("snippet transcribed (raw): ${text?.take(80) ?: "(null)"}")

            if (text.isNullOrEmpty()) {
                mainHandler.post {
                    snippetMode = false
                    processing = false
                    hideRing()
                    showSnippetPicker(null)
                }
                return@execute
            }

            val body = try { nativeMatchSnippet(text) } catch (e: Exception) {
                logE("nativeMatchSnippet failed: $e")
                null
            }
            mainHandler.post {
                snippetMode = false
                processing = false
                if (body != null) {
                    logI("snippet matched, injecting body (${body.length} chars)")
                    flashCompleteRing()
                    injectText(body)
                } else {
                    hideRing()
                    showSnippetPicker(text)
                }
            }
        }
    }

    private fun showSnippetPicker(triggerText: String?) {
        dismissSnippetPicker()

        val json = try { nativeListSnippets() } catch (e: Exception) {
            logE("nativeListSnippets failed: $e")
            null
        }
        if (json.isNullOrEmpty()) {
            toast("No snippets configured")
            return
        }

        val snippets = try { JSONArray(json) } catch (e: Exception) {
            logE("Failed to parse snippets JSON: $e")
            toast("No snippets configured")
            return
        }
        if (snippets.length() == 0) {
            toast("No snippets configured")
            return
        }

        // Design tokens matching the Tauri frontend
        val colSurfaceContainerHigh = Color.parseColor("#20201f")
        val colSurfaceContainerHighest = Color.parseColor("#262626")
        val colOnSurface = Color.parseColor("#ffffff")
        val colOnSurfaceVariant = Color.parseColor("#adaaaa")
        val colOutlineVariant = Color.parseColor("#484847")
        val colPrimary = Color.parseColor("#9ba8ff")

        val density = resources.displayMetrics.density
        val dp = { value: Int -> (value * density).toInt() }

        val panelBg = GradientDrawable().apply {
            setColor(colSurfaceContainerHigh)
            cornerRadius = 16 * density
            setStroke(1, Color.argb(51, 0x48, 0x48, 0x47)) // outline-variant/20
        }

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = panelBg
            setPadding(dp(24), dp(24), dp(24), dp(24))
            elevation = 8 * density
        }

        // Title
        val title = TextView(this).apply {
            text = if (triggerText != null) "No snippet matched" else "Select snippet"
            setTextColor(colOnSurface)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            paint.isFakeBoldText = true
        }
        container.addView(title)

        // Subtitle with heard text
        if (triggerText != null) {
            val subtitle = TextView(this).apply {
                text = "Heard: \"${triggerText.take(40)}\""
                setTextColor(colOnSurfaceVariant)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { topMargin = dp(2) }
            }
            container.addView(subtitle)
        }

        // Instruction
        val instruction = TextView(this).apply {
            text = "Select a snippet to paste${if (triggerText != null) " and learn this trigger" else ""}:"
            setTextColor(colOnSurfaceVariant)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(12); bottomMargin = dp(12) }
        }
        container.addView(instruction)

        // Scrollable snippet list
        val scroll = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                height = (dp(48) * snippets.length().coerceAtMost(5))
            }
        }
        val list = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        for (i in 0 until snippets.length()) {
            val obj = snippets.getJSONObject(i)
            val snippetId = obj.getString("id")
            val body = obj.getString("body")
            val triggers = obj.getJSONArray("triggers")
            val triggerPreview = if (triggers.length() > 0) triggers.getString(0) else ""

            val itemBg = GradientDrawable().apply {
                setColor(Color.TRANSPARENT)
                cornerRadius = 8 * density
            }

            val row = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = itemBg
                setPadding(dp(12), dp(8), dp(12), dp(8))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = dp(2) }
                isClickable = true
                isFocusable = true
                setOnClickListener {
                    logI("Picker: selected snippet id=$snippetId")
                    dismissSnippetPicker()
                    flashCompleteRing()
                    injectText(body)
                    if (triggerText != null) {
                        executor.execute {
                            logI("Picker: self-heal adding trigger \"$triggerText\" to $snippetId")
                            nativeAddSnippetTrigger(snippetId, triggerText)
                        }
                    }
                }
                setOnTouchListener { v, ev ->
                    when (ev.action) {
                        MotionEvent.ACTION_DOWN -> {
                            (v.background as? GradientDrawable)?.setColor(colSurfaceContainerHighest)
                            false
                        }
                        MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                            (v.background as? GradientDrawable)?.setColor(Color.TRANSPARENT)
                            false
                        }
                        else -> false
                    }
                }
            }

            val triggerLabel = TextView(this).apply {
                text = triggerPreview
                setTextColor(colPrimary)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                typeface = android.graphics.Typeface.MONOSPACE
                maxLines = 1
            }
            val bodyPreview = TextView(this).apply {
                text = body.take(60) + if (body.length > 60) "..." else ""
                setTextColor(colOnSurfaceVariant)
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                maxLines = 1
            }
            row.addView(triggerLabel)
            row.addView(bodyPreview)
            list.addView(row)
        }

        scroll.addView(list)
        container.addView(scroll)

        // Dismiss button
        val dismissBtn = TextView(this).apply {
            text = "Dismiss"
            setTextColor(colOnSurfaceVariant)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            paint.isFakeBoldText = true
            setPadding(dp(16), dp(8), dp(16), dp(8))
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = dp(12)
                gravity = Gravity.END
            }
            val btnBg = GradientDrawable().apply {
                setColor(Color.TRANSPARENT)
                cornerRadius = 8 * density
            }
            background = btnBg
            setOnClickListener { dismissSnippetPicker() }
        }
        container.addView(dismissBtn)

        // Dismiss on outside touch
        container.setOnTouchListener { _, ev ->
            if (ev.action == MotionEvent.ACTION_OUTSIDE) {
                dismissSnippetPicker()
                true
            } else false
        }

        // Add as overlay
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP_MR1) {
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_SYSTEM_ALERT
        }

        val maxWidth = (resources.displayMetrics.widthPixels * 0.85).toInt()
        val params = WindowManager.LayoutParams(
            maxWidth,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.CENTER
        }

        try {
            windowManager?.addView(container, params)
            snippetPickerView = container
            logI("Picker: shown with ${snippets.length()} snippets")
        } catch (e: Exception) {
            logE("Picker: failed to show: $e")
        }
    }

    private fun dismissSnippetPicker() {
        snippetPickerView?.let { view ->
            try {
                windowManager?.removeView(view)
            } catch (_: Exception) {}
            snippetPickerView = null
        }
    }

    private fun injectText(text: String) {
        val liveFocus = findFocusedEditText()
        val node = liveFocus ?: focusedNode
        if (node == null) {
            logW("injectText: no focused node found, copying to clipboard")
            copyToClipboard(text)
            toast("Copied to clipboard (no text field focused)")
            return
        }
        logD("injectText: node=${node.className} editable=${node.isEditable} fromLive=${liveFocus != null}")

        // Read existing text for context. If the field is showing hint text
        // (placeholder), treat it as empty to avoid contaminating the output.
        val selStart = node.textSelectionStart
        val rawText = node.text?.toString() ?: ""
        val isHint = node.isShowingHintText || (node.hintText != null && rawText == node.hintText.toString())
        val currentText = if (isHint) "" else rawText
        val before = if (selStart > 0) currentText.substring(0, selStart.coerceAtMost(currentText.length)) else ""
        val selEnd = node.textSelectionEnd
        val after = if (selEnd >= 0 && !isHint) currentText.substring(selEnd.coerceAtMost(currentText.length)) else ""
        val adjusted = adjustForContext(text, before, after)
        logD("injectText: sel=$selStart..$selEnd hint=$isHint before=${before.takeLast(20)} after=${after.take(20)} -> $adjusted")

        // Primary: ACTION_SET_TEXT (doesn't touch clipboard)
        try {
            val fullText = if (isHint || selStart < 0) {
                adjusted
            } else {
                before + adjusted + after
            }
            val args = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, fullText)
            }
            if (node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) {
                logI("injectText: ACTION_SET_TEXT succeeded")
                val newCursor = if (isHint || selStart < 0) adjusted.length else before.length + adjusted.length
                val selArgs = Bundle().apply {
                    putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_START_INT, newCursor)
                    putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_END_INT, newCursor)
                }
                node.performAction(AccessibilityNodeInfo.ACTION_SET_SELECTION, selArgs)
                return
            }
            logW("injectText: ACTION_SET_TEXT failed, trying ACTION_PASTE")
        } catch (e: Exception) {
            logE("injectText: ACTION_SET_TEXT threw: $e")
        }

        // Fallback: clipboard paste (for apps that don't support SET_TEXT)
        try {
            val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            val oldClip = clipboard.primaryClip
            clipboard.setPrimaryClip(ClipData.newPlainText("verba", adjusted))
            val pasted = node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
            logI("injectText: ACTION_PASTE returned $pasted")
            // Always restore previous clipboard
            if (oldClip != null) {
                mainHandler.postDelayed({ clipboard.setPrimaryClip(oldClip) }, 300)
            }
            if (pasted) return
            logW("injectText: ACTION_PASTE also failed")
            copyToClipboard(adjusted)
            toast("Copied to clipboard (paste failed)")
        } catch (e: Exception) {
            logE("injectText: ACTION_PASTE threw: $e")
            copyToClipboard(adjusted)
            toast("Copied to clipboard")
        }
    }

    /**
     * Adjust transcribed text based on surrounding context in the text field.
     * - Mid-sentence: lowercase first letter, drop trailing period
     * - Start of field / after sentence-ending punctuation: keep as-is
     * - Add leading space if text before doesn't end with whitespace
     */
    private fun adjustForContext(text: String, before: String, after: String): String {
        if (text.isEmpty()) return text

        var result = text

        // Should we capitalize?
        val trimmedBefore = before.trimEnd()
        val atSentenceStart = before.isEmpty()
                || trimmedBefore.endsWith('.')
                || trimmedBefore.endsWith('!')
                || trimmedBefore.endsWith('?')
                || before.endsWith('\n')

        if (!atSentenceStart && result[0].isUpperCase()) {
            result = result[0].lowercase() + result.substring(1)
        }

        // Remove trailing period if there's text after cursor
        if (after.trimStart().isNotEmpty() && result.endsWith('.')) {
            result = result.dropLast(1)
        }

        // Add leading space if needed
        if (before.isNotEmpty()
            && !before.endsWith(' ')
            && !before.endsWith('\n')
            && !before.endsWith('\t')
        ) {
            result = " $result"
        }

        return result
    }

    private fun copyToClipboard(text: String) {
        try {
            val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.setPrimaryClip(ClipData.newPlainText("verba", text))
        } catch (e: Exception) {
            logE("copyToClipboard failed: $e")
        }
    }

    private fun findFocusedEditText(): AccessibilityNodeInfo? {
        return try {
            val root = rootInActiveWindow ?: return null
            val focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
                ?: return null
            if (isEditableTextField(focused)) {
                // Accept if OS reports it focused, or if it has a cursor
                // position (WebView fields may not report isFocused but do
                // have a valid selection when the user taps in).
                if (focused.isFocused || focused.textSelectionStart >= 0) {
                    return focused
                }
            }
            focused.recycle()
            null
        } catch (e: Exception) {
            logE("findFocusedEditText: $e")
            null
        }
    }

    private fun isHapticEnabled(): Boolean {
        // Check both possible config locations (Tauri app_data_dir and IME dataDir)
        val candidates = listOf(
            java.io.File(applicationContext.dataDir, "app_data/config.toml"),
            java.io.File(applicationContext.dataDir, "config.toml")
        )
        val configFile = candidates.firstOrNull { it.exists() } ?: return true
        return try {
            !configFile.readText().contains("haptic_feedback = false")
        } catch (_: Exception) { true }
    }

    private fun hapticFeedback() {
        if (!isHapticEnabled()) return
        try {
            val vibrator = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                val vm = getSystemService(VIBRATOR_MANAGER_SERVICE) as? android.os.VibratorManager
                vm?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(VIBRATOR_SERVICE) as? android.os.Vibrator
            }
            if (vibrator == null || !vibrator.hasVibrator()) {
                logW("haptic: no vibrator")
                return
            }

            val effect = android.os.VibrationEffect.createPredefined(
                android.os.VibrationEffect.EFFECT_CLICK
            )

            // USAGE_ACCESSIBILITY is exempt from haptic-disabled setting
            // and background UID suppression -- exactly right for an
            // accessibility service overlay.
            val attrs = android.os.VibrationAttributes.createForUsage(
                android.os.VibrationAttributes.USAGE_ACCESSIBILITY
            )
            vibrator.vibrate(effect, attrs)
            logD("haptic: fired click (USAGE_ACCESSIBILITY)")
        } catch (e: Exception) {
            logW("haptic failed: $e")
        }
    }

    private fun toast(msg: String) {
        Toast.makeText(applicationContext, msg, Toast.LENGTH_SHORT).show()
    }

    private inner class RoundedRingView(context: android.content.Context) : View(context) {
        private val dp = resources.displayMetrics.density
        private val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 5f * dp
            color = 0xFFFF4444.toInt()
            strokeCap = android.graphics.Paint.Cap.ROUND
        }
        private val path = android.graphics.Path()
        private val measure = android.graphics.PathMeasure()
        private var totalLength = 0f

        override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
            val inset = paint.strokeWidth / 2f
            val cr = 13f * dp
            path.reset()
            path.addRoundRect(
                inset, inset, w - inset, h - inset,
                cr, cr, android.graphics.Path.Direction.CW
            )
            measure.setPath(path, true)
            totalLength = measure.length
        }

        fun setColor(color: Int) {
            paint.color = color
            invalidate()
        }

        fun setPhase(phase: Float) {
            if (totalLength <= 0f) return
            val dash = totalLength * 0.75f
            val gap = totalLength * 0.25f
            paint.pathEffect = android.graphics.DashPathEffect(
                floatArrayOf(dash, gap), phase * totalLength
            )
            invalidate()
        }

        fun setSolid() {
            paint.pathEffect = null
            invalidate()
        }

        override fun onDraw(canvas: android.graphics.Canvas) {
            if (totalLength > 0f) canvas.drawPath(path, paint)
        }
    }

    override fun onDestroy() {
        logI("onDestroy")
        dismissSnippetPicker()
        try { unregisterReceiver(screenReceiver) } catch (_: Exception) {}
        if (isOverlayShowing && overlayView != null) {
            try { windowManager?.removeView(overlayView) } catch (_: Exception) {}
        }
        focusedNode?.recycle()
        if (initialized) {
            executor.execute { nativeDestroy() }
        }
        executor.shutdown()
        super.onDestroy()
    }
}

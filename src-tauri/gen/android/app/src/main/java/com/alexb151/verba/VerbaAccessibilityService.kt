package com.alexb151.verba

import android.accessibilityservice.AccessibilityService
import android.content.ClipData
import android.content.ClipboardManager
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
import android.widget.FrameLayout
import android.widget.ImageView
import android.content.SharedPreferences
import android.animation.ValueAnimator
import android.widget.Toast
import java.util.concurrent.Executors
import kotlin.math.abs

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
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var windowManager: WindowManager? = null
    private var overlayView: View? = null
    private var micButton: ImageView? = null
    private var isOverlayShowing = false
    private var focusedNode: AccessibilityNodeInfo? = null

    // Drag state
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDragging = false

    private var ringView: RoundedRingView? = null
    private var ringAnimator: ValueAnimator? = null

    // Debounced hide to avoid flickering when keyboard opens
    private var pendingHide: Runnable? = null
    private var lastEditTextSeenMs = 0L

    // Saved position
    private var savedX = -1
    private var savedY = -1
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
                mainHandler.post { toast("Verba: loading model...") }
                nativePreloadModel()
                logI("model preloaded")
                mainHandler.post {
                    toast("Verba: ready")
                    micButton?.alpha = 1.0f
                }
            } else {
                logE("nativeInit failed")
                mainHandler.post {
                    toast("Verba: no model found. Open the app to download one.")
                    micButton?.alpha = 0.4f
                }
            }
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        logI("onServiceConnected")
        createOverlay()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return

        when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_FOCUSED -> {
                val source = event.source ?: return
                if (isEditableTextField(source)) {
                    cancelPendingHide()
                    focusedNode?.recycle()
                    focusedNode = source
                    lastEditTextSeenMs = System.currentTimeMillis()
                    showOverlay()
                } else {
                    source.recycle()
                    if (!recording) {
                        scheduleHide()
                    }
                }
            }
            AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED,
            AccessibilityEvent.TYPE_VIEW_TEXT_SELECTION_CHANGED -> {
                val source = event.source ?: return
                if (isEditableTextField(source)) {
                    if (!isOverlayShowing) {
                        cancelPendingHide()
                        focusedNode?.recycle()
                        focusedNode = source
                        lastEditTextSeenMs = System.currentTimeMillis()
                        showOverlay()
                    } else {
                        source.recycle()
                    }
                } else {
                    source.recycle()
                }
            }
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                if (recording) return
                // When an app comes to foreground, check if an editable field
                // already has focus. This catches the case where the cursor
                // was already in a text box before we switched to the app.
                if (!isOverlayShowing) {
                    val editable = findFocusedEditText()
                    if (editable != null) {
                        cancelPendingHide()
                        focusedNode?.recycle()
                        focusedNode = editable
                        lastEditTextSeenMs = System.currentTimeMillis()
                        showOverlay()
                        return
                    }
                }
                scheduleHide()
            }
        }
    }

    private fun scheduleHide() {
        cancelPendingHide()
        pendingHide = Runnable {
            val focused = findFocusedEditText()
            if (focused != null) {
                focusedNode?.recycle()
                focusedNode = focused
                lastEditTextSeenMs = System.currentTimeMillis()
            } else if (focusedNode?.refresh() == true && isEditableTextField(focusedNode!!)) {
                lastEditTextSeenMs = System.currentTimeMillis()
            } else {
                val rootEditable = findEditableInRoot()
                if (rootEditable != null) {
                    focusedNode?.recycle()
                    focusedNode = rootEditable
                    lastEditTextSeenMs = System.currentTimeMillis()
                    return@Runnable
                }

                val elapsed = System.currentTimeMillis() - lastEditTextSeenMs
                if (elapsed < 2000 && isOverlayShowing) {
                    mainHandler.postDelayed(pendingHide!!, 500)
                    return@Runnable
                }
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
            visibility = View.GONE
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
            elevation = 6 * density
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
                micButton?.scaleX = 0.85f
                micButton?.scaleY = 0.85f
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = event.rawX - lastTouchX
                val dy = event.rawY - lastTouchY
                if (abs(dx) > 8 || abs(dy) > 8 || isDragging) {
                    isDragging = true
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
                micButton?.scaleX = 1.0f
                micButton?.scaleY = 1.0f
                if (!isDragging) {
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

        try {
            windowManager?.removeView(overlayView)
        } catch (e: Exception) {
            logE("hideOverlay: failed to remove view: $e")
        }
        isOverlayShowing = false
    }

    private fun onMicClick() {
        logI("onMicClick: initialized=$initialized recording=$recording")
        if (!initialized) {
            toast("Verba: model still loading...")
            return
        }

        if (recording) {
            stopAndTranscribe()
        } else {
            startRecording()
        }
    }

    private fun startRecordingRing() {
        ringView?.let { ring ->
            ring.visibility = View.VISIBLE
            ringAnimator?.cancel()
            ringAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
                duration = 1500
                repeatCount = ValueAnimator.INFINITE
                interpolator = android.view.animation.LinearInterpolator()
                addUpdateListener { ring.setPhase(it.animatedValue as Float) }
                start()
            }
        }
    }

    private fun stopRecordingRing() {
        ringAnimator?.cancel()
        ringAnimator = null
        ringView?.visibility = View.GONE
    }

    private fun startRecording() {
        logI("startRecording")
        recording = true
        startRecordingRing()

        executor.execute {
            val started = nativeStartRecording()
            logI("nativeStartRecording returned $started")
            if (!started) {
                mainHandler.post {
                    recording = false
                    stopRecordingRing()
                    toast("Verba: failed to start recording")
                }
            }
        }
    }

    private fun stopAndTranscribe() {
        logI("stopAndTranscribe")
        recording = false
        stopRecordingRing()
        micButton?.alpha = 0.5f

        executor.execute {
            val text = nativeStopAndTranscribe()
            logI("nativeStopAndTranscribe returned ${text?.length ?: 0} chars")
            mainHandler.post {
                micButton?.alpha = 1.0f
                if (!text.isNullOrEmpty()) {
                    logI("injecting text: \"${text.take(80)}\"")
                    injectText(text)
                } else {
                    logW("no text returned from transcription")
                    toast("Verba: no speech detected")
                }
            }
        }
    }

    private fun injectText(text: String) {
        val liveFocus = findFocusedEditText()
        val node = liveFocus ?: focusedNode
        if (node == null) {
            logW("injectText: no focused node found, copying to clipboard")
            copyToClipboard(text)
            toast("Verba: copied to clipboard (no text field focused)")
            return
        }
        logD("injectText: node=${node.className} editable=${node.isEditable} fromLive=${liveFocus != null}")

        try {
            val currentText = node.text?.toString() ?: ""
            var selStart = node.textSelectionStart
            var selEnd = node.textSelectionEnd
            logD("injectText: currentLen=${currentText.length} sel=$selStart..$selEnd")

            if (selStart < 0) selStart = currentText.length
            if (selEnd < 0) selEnd = selStart

            val before = currentText.substring(0, selStart.coerceAtMost(currentText.length))
            val after = currentText.substring(selEnd.coerceAtMost(currentText.length))
            val adjusted = adjustForContext(text, before, after)
            logD("injectText: context before=${before.takeLast(20)} after=${after.take(20)} → $adjusted")
            val newText = before + adjusted + after

            val args = Bundle().apply {
                putCharSequence(
                    AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                    newText
                )
            }
            if (node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) {
                logI("injectText: ACTION_SET_TEXT succeeded")
                val newCursor = before.length + adjusted.length
                val selArgs = Bundle().apply {
                    putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_START_INT, newCursor)
                    putInt(AccessibilityNodeInfo.ACTION_ARGUMENT_SELECTION_END_INT, newCursor)
                }
                node.performAction(AccessibilityNodeInfo.ACTION_SET_SELECTION, selArgs)
                return
            }
            logW("injectText: ACTION_SET_TEXT returned false, trying clipboard paste")
        } catch (e: Exception) {
            logE("injectText: ACTION_SET_TEXT threw: $e")
        }

        // Fallback: clipboard paste
        try {
            val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            val oldClip = clipboard.primaryClip
            clipboard.setPrimaryClip(ClipData.newPlainText("verba", text))
            val pasted = node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
            logI("injectText: ACTION_PASTE returned $pasted")
            if (!pasted) {
                logW("injectText: paste failed, text left on clipboard")
                toast("Verba: copied to clipboard (paste failed)")
            }
            if (oldClip != null) {
                mainHandler.postDelayed({ clipboard.setPrimaryClip(oldClip) }, 500)
            }
        } catch (e: Exception) {
            logE("injectText: clipboard paste threw: $e")
            copyToClipboard(text)
            toast("Verba: copied to clipboard")
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
                || trimmedBefore.endsWith('\n')

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
                return focused
            }
            val editable = findEditableChild(focused)
            if (editable != null) {
                focused.recycle()
                return editable
            }
            focused.recycle()
            null
        } catch (e: Exception) {
            logE("findFocusedEditText: $e")
            null
        }
    }

    private fun findEditableChild(node: AccessibilityNodeInfo): AccessibilityNodeInfo? {
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            if (isEditableTextField(child)) return child
            val found = findEditableChild(child)
            child.recycle()
            if (found != null) return found
        }
        return null
    }

    private fun findEditableInRoot(): AccessibilityNodeInfo? {
        return try {
            val root = rootInActiveWindow ?: return null
            findEditableChild(root)
        } catch (e: Exception) {
            logE("findEditableInRoot: $e")
            null
        }
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
    }

    private inner class RoundedRingView(context: android.content.Context) : View(context) {
        private val dp = resources.displayMetrics.density
        private val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            style = android.graphics.Paint.Style.STROKE
            strokeWidth = 3f * dp
            color = 0xFFFF4444.toInt()
            strokeCap = android.graphics.Paint.Cap.ROUND
        }
        private val path = android.graphics.Path()
        private val measure = android.graphics.PathMeasure()
        private var totalLength = 0f

        override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
            val inset = paint.strokeWidth / 2f
            val cr = 15f * dp
            path.reset()
            path.addRoundRect(
                inset, inset, w - inset, h - inset,
                cr, cr, android.graphics.Path.Direction.CW
            )
            measure.setPath(path, true)
            totalLength = measure.length
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

        override fun onDraw(canvas: android.graphics.Canvas) {
            if (totalLength > 0f) canvas.drawPath(path, paint)
        }
    }

    override fun onDestroy() {
        logI("onDestroy")
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

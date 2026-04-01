package com.alexb151.verba

import android.content.Intent
import androidx.core.content.FileProvider
import java.io.File

object ShareHelper {
    @JvmStatic
    fun shareFile(json: String) {
        val context = VerbaApp.instance ?: throw Exception("App context not available")

        // Write to cache dir which FileProvider already serves via cache-path
        val file = File(context.cacheDir, "history_export.json")
        file.writeText(json)

        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/json"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }

        val chooser = Intent.createChooser(intent, "Share history").apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(chooser)
    }
}

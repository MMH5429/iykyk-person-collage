package com.iykyk.collage.collage

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import com.iykyk.collage.core.collage.GridSpec
import com.iykyk.collage.core.model.AnalysisResult
import com.iykyk.collage.core.model.Person
import kotlin.math.max
import kotlin.math.min

/**
 * Draws the finished collage at Instagram Story resolution.
 *
 * Tiles are centre-cropped from generously-cropped source shots, so faces stay large and
 * sharp without the tight-bounding-box look the assignment warns against.
 */
class CollageRenderer {

    fun render(analysis: AnalysisResult): Bitmap {
        val bitmap = Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        drawBackground(canvas)
        drawHeader(canvas, analysis)

        val people = analysis.people
        val cells = GridSpec.forCount(people.size)
        if (cells.isNotEmpty()) {
            val columns = GridSpec.columnsFor(people.size)
            val rows = GridSpec.rowsFor(people.size)

            val gridHeight = HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
            val cellWidth = (WIDTH - MARGIN * 2 - GUTTER * (columns - 1)) / columns
            val cellHeight = (gridHeight - GUTTER * (rows - 1)) / rows

            people.forEachIndexed { index, person ->
                val cell = cells[index]
                val left = MARGIN + cell.column * (cellWidth + GUTTER)
                val width = cellWidth * cell.columnSpan + GUTTER * (cell.columnSpan - 1)
                val top = HEADER_HEIGHT + cell.row * (cellHeight + GUTTER)
                drawTile(canvas, person, RectF(left, top, left + width, top + cellHeight))
            }
        }

        drawFooter(canvas, analysis)
        return bitmap
    }

    private fun drawBackground(canvas: Canvas) {
        val paint = Paint().apply {
            shader = LinearGradient(
                0f, 0f, WIDTH.toFloat(), HEIGHT.toFloat(),
                intArrayOf(BG_TOP, BG_BOTTOM), null, Shader.TileMode.CLAMP,
            )
        }
        canvas.drawRect(0f, 0f, WIDTH.toFloat(), HEIGHT.toFloat(), paint)
    }

    private fun drawHeader(canvas: Canvas, analysis: AnalysisResult) {
        val title = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 68f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val subtitle = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = ACCENT
            textSize = 38f
        }
        val peopleWord = if (analysis.people.size == 1) "person" else "people"
        val appearanceWord = if (analysis.totalAppearances == 1) "appearance" else "appearances"

        canvas.drawText("Who's in this video", MARGIN, 112f, title)
        canvas.drawText(
            "${analysis.people.size} $peopleWord  ·  ${analysis.totalAppearances} $appearanceWord",
            MARGIN, 170f, subtitle,
        )
    }

    private fun drawTile(canvas: Canvas, person: Person, bounds: RectF) {
        val restorePoint = canvas.save()
        canvas.clipPath(Path().apply { addRoundRect(bounds, CORNER, CORNER, Path.Direction.CW) })

        canvas.drawRect(bounds, Paint().apply { color = TILE_BG })
        canvas.drawBitmap(person.shot, centreCropSource(person.shot, bounds), bounds, IMAGE_PAINT)

        // A soft scrim so the label stays readable over any photo.
        canvas.drawRect(
            RectF(bounds.left, bounds.bottom - SCRIM_HEIGHT, bounds.right, bounds.bottom),
            Paint().apply {
                shader = LinearGradient(
                    0f, bounds.bottom - SCRIM_HEIGHT, 0f, bounds.bottom,
                    intArrayOf(Color.TRANSPARENT, SCRIM), null, Shader.TileMode.CLAMP,
                )
            },
        )

        val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 36f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        canvas.drawText(person.label, bounds.left + 26f, bounds.bottom - 32f, label)

        canvas.restoreToCount(restorePoint)

        // Border, drawn outside the clip so it is not shaved in half.
        canvas.drawRoundRect(bounds, CORNER, CORNER, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f
            color = BORDER
        })

        drawBadge(canvas, person.appearanceCount, bounds)
    }

    /** The "x4" appearance-count badge — the number the assignment asks to be shown. */
    private fun drawBadge(canvas: Canvas, count: Int, bounds: RectF) {
        val text = "×$count"
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.BLACK
            textSize = 34f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val textWidth = textPaint.measureText(text)
        val badge = RectF(
            bounds.right - textWidth - BADGE_PADDING * 2 - 18f,
            bounds.top + 18f,
            bounds.right - 18f,
            bounds.top + 18f + BADGE_HEIGHT,
        )
        canvas.drawRoundRect(
            badge, BADGE_HEIGHT / 2f, BADGE_HEIGHT / 2f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply { color = ACCENT },
        )
        canvas.drawText(text, badge.left + BADGE_PADDING, badge.bottom - 16f, textPaint)
    }

    private fun drawFooter(canvas: Canvas, analysis: AnalysisResult) {
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = FOOTER_TEXT
            textSize = 30f
        }
        canvas.drawText(analysis.sourceName, MARGIN, HEIGHT - 48f, paint)
    }

    /**
     * Source rect that fills [bounds] without distortion, keeping the centre of the shot —
     * which is where the face is, because the crop was built around it.
     */
    private fun centreCropSource(bitmap: Bitmap, bounds: RectF): Rect {
        val targetRatio = bounds.width() / bounds.height()
        val sourceRatio = bitmap.width.toFloat() / bitmap.height

        return if (sourceRatio > targetRatio) {
            val w = (bitmap.height * targetRatio).toInt().coerceAtLeast(1)
            val x = max(0, (bitmap.width - w) / 2)
            Rect(x, 0, min(bitmap.width, x + w), bitmap.height)
        } else {
            val h = (bitmap.width / targetRatio).toInt().coerceAtLeast(1)
            val y = max(0, (bitmap.height - h) / 2)
            Rect(0, y, bitmap.width, min(bitmap.height, y + h))
        }
    }

    companion object {
        const val WIDTH = 1080
        const val HEIGHT = 1920

        private const val MARGIN = 44f
        private const val GUTTER = 20f
        private const val CORNER = 34f
        private const val HEADER_HEIGHT = 220f
        private const val FOOTER_HEIGHT = 100f
        private const val SCRIM_HEIGHT = 120f
        private const val BADGE_HEIGHT = 54f
        private const val BADGE_PADDING = 20f

        private val IMAGE_PAINT = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

        private val BG_TOP = Color.rgb(18, 20, 30)
        private val BG_BOTTOM = Color.rgb(38, 26, 54)
        private val TILE_BG = Color.rgb(28, 30, 42)
        private val BORDER = Color.argb(56, 255, 255, 255)
        private val SCRIM = Color.argb(190, 0, 0, 0)
        private val ACCENT = Color.rgb(126, 231, 195)
        private val FOOTER_TEXT = Color.argb(140, 255, 255, 255)
    }
}

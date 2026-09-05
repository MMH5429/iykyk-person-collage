package com.iykyk.collage.core.track

import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.quality.FaceQualityScorer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TrackletBuilderTest {

    private val config = PipelineConfig.Default
    private val builder = TrackletBuilder(config, FaceQualityScorer(config))

    /** A clearly-visible face at [box] on frame [i]; 8 fps means 125 ms per frame. */
    private fun obs(
        i: Int,
        box: BoxF,
        trackingId: Int? = null,
        sharpness: Float = 500f,
    ) = FaceObservation(
        frameIndex = i,
        timestampMs = i * 125L,
        trackingId = trackingId,
        box = box,
        leftEye = null,
        rightEye = null,
        headEulerX = 0f,
        headEulerY = 0f,
        headEulerZ = 0f,
        leftEyeOpenProb = 0.9f,
        rightEyeOpenProb = 0.9f,
        smileProb = 0.5f,
        rawSharpness = sharpness,
        frameWidth = 540,
        frameHeight = 960,
    )

    // Two well-separated, fully in-frame faces in a 540x960 analysis frame.
    private val left = BoxF(60f, 400f, 260f, 600f)
    private val right = BoxF(290f, 400f, 490f, 600f)

    /** Wraps per-frame detections into the frame-indexed structure the builder takes. */
    private fun frames(vararg f: List<FaceObservation>) = f.toList()

    @Test
    fun `a single continuous run yields one tracklet`() {
        val result = builder.build(frames(
            listOf(obs(0, left)),
            listOf(obs(1, left)),
            listOf(obs(2, left)),
            listOf(obs(3, left)),
        ))
        assertEquals(1, result.size)
        assertEquals(4, result[0].visibleFrameCount)
    }

    @Test
    fun `a long absence splits one run into two appearances`() {
        // Frames 0-3 present, then a 1.25 s gap, then frames 13-16 present again.
        val result = builder.build(frames(
            listOf(obs(0, left)), listOf(obs(1, left)), listOf(obs(2, left)), listOf(obs(3, left)),
            emptyList(), emptyList(), emptyList(), emptyList(), emptyList(),
            emptyList(), emptyList(), emptyList(), emptyList(),
            listOf(obs(13, left)), listOf(obs(14, left)), listOf(obs(15, left)), listOf(obs(16, left)),
        ))
        assertEquals(2, result.size)
    }

    @Test
    fun `a single dropped frame does not split an appearance`() {
        // One 250 ms hole, inside the 400 ms gap tolerance.
        val result = builder.build(frames(
            listOf(obs(0, left)), listOf(obs(1, left)),
            emptyList(),
            listOf(obs(3, left)), listOf(obs(4, left)),
        ))
        assertEquals(1, result.size)
        assertEquals(4, result[0].visibleFrameCount)
    }

    @Test
    fun `a changed tracking id does not split an appearance`() {
        // ML Kit reissues the id mid-run; the boxes still overlap, so it is one appearance.
        val result = builder.build(frames(
            listOf(obs(0, left, trackingId = 7)),
            listOf(obs(1, left, trackingId = 7)),
            listOf(obs(2, left, trackingId = 42)),
            listOf(obs(3, left, trackingId = 42)),
        ))
        assertEquals(1, result.size)
    }

    @Test
    fun `two people on screen together yield two overlapping tracklets`() {
        val result = builder.build(frames(
            listOf(obs(0, left), obs(0, right)),
            listOf(obs(1, left), obs(1, right)),
            listOf(obs(2, left), obs(2, right)),
        ))
        assertEquals(2, result.size)
        assertTrue(result[0].overlapsInTime(result[1]))
    }

    @Test
    fun `crossing faces are not swapped between tracks`() {
        // Two faces that stay well separated must never be associated with each other.
        val result = builder.build(frames(
            listOf(obs(0, left), obs(0, right)),
            listOf(obs(1, right), obs(1, left)),  // detection order deliberately reversed
            listOf(obs(2, left), obs(2, right)),
        ))
        assertEquals(2, result.size)
        result.forEach { t ->
            val xs = t.observations.map { it.observation.box.centerX }
            // Every observation in a track sits on the same side of the frame.
            assertTrue(xs.all { it < 275f } || xs.all { it > 275f })
        }
    }

    @Test
    fun `a scene cut ends an appearance even when the face stays in place`() {
        // The montage case: a hard cut to a different person standing in the same spot.
        // IoU alone would chain these into one long appearance.
        // Each half must clear the minimum appearance duration, so four frames a side.
        val cuts = listOf(false, false, false, false, true, false, false, false)
        val result = builder.build(frames(
            listOf(obs(0, left)), listOf(obs(1, left)), listOf(obs(2, left)), listOf(obs(3, left)),
            listOf(obs(4, left)), listOf(obs(5, left)), listOf(obs(6, left)), listOf(obs(7, left)),
        ), cuts)
        assertEquals(2, result.size)
    }

    @Test
    fun `no cut flags means association behaves as before`() {
        val result = builder.build(frames(
            listOf(obs(0, left)), listOf(obs(1, left)), listOf(obs(2, left)), listOf(obs(3, left)),
        ), emptyList())
        assertEquals(1, result.size)
    }

    @Test
    fun `a one frame flicker is not an appearance`() {
        val result = builder.build(frames(listOf(obs(0, left))))
        assertEquals(0, result.size)
    }

    @Test
    fun `a whip pan of blurred frames counts for nobody`() {
        // Present in every frame, but never sharp enough to be clearly visible.
        val result = builder.build(frames(
            listOf(obs(0, left, sharpness = 4f)),
            listOf(obs(1, left, sharpness = 3f)),
            listOf(obs(2, left, sharpness = 5f)),
            listOf(obs(3, left, sharpness = 4f)),
        ))
        assertEquals(0, result.size)
    }

    @Test
    fun `blurred frames on the way in do not extend the visible span`() {
        val result = builder.build(frames(
            listOf(obs(0, left, sharpness = 4f)),   // blurred approach
            listOf(obs(1, left, sharpness = 4f)),
            listOf(obs(2, left, sharpness = 500f)), // becomes clearly visible here
            listOf(obs(3, left, sharpness = 500f)),
            listOf(obs(4, left, sharpness = 500f)),
        ))
        assertEquals(1, result.size)
        assertEquals(250L, result[0].visibleStartMs)
        assertEquals(3, result[0].visibleFrameCount)
    }

    @Test
    fun `the best shot is the highest scoring visible observation`() {
        val result = builder.build(frames(
            listOf(obs(0, left, sharpness = 100f)),
            listOf(obs(1, left, sharpness = 900f)),
            listOf(obs(2, left, sharpness = 200f)),
        ))
        assertEquals(1, result.size)
        assertEquals(125L, result[0].best!!.observation.timestampMs)
    }
}

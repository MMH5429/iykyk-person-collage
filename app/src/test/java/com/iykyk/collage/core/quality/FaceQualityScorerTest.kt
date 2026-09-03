package com.iykyk.collage.core.quality

import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceQualityScorerTest {

    private val config = PipelineConfig.Default
    private val scorer = FaceQualityScorer(config)

    /** A large, sharp, frontal, centred, eyes-open, smiling face. */
    private fun goodFace(
        box: BoxF = BoxF(170f, 400f, 370f, 600f),
        yaw: Float = 0f,
        sharpness: Float = 500f,
        eyeOpen: Float = 0.95f,
    ) = FaceObservation(
        frameIndex = 0,
        timestampMs = 0,
        trackingId = 1,
        box = box,
        leftEye = null,
        rightEye = null,
        headEulerX = 0f,
        headEulerY = yaw,
        headEulerZ = 0f,
        leftEyeOpenProb = eyeOpen,
        rightEyeOpenProb = eyeOpen,
        smileProb = 0.8f,
        rawSharpness = sharpness,
        frameWidth = 540,
        frameHeight = 960,
    )

    @Test
    fun `a good face clears every gate`() {
        val s = scorer.score(goodFace())
        assertTrue(scorer.isClearlyVisible(s))
    }

    @Test
    fun `sharpness saturates at one`() {
        val s = scorer.score(goodFace(sharpness = 10_000f))
        assertEquals(1f, s.sharpness, 1e-4f)
    }

    @Test
    fun `a motion blurred face fails the visibility gate`() {
        // A whip-pan pass: everything else fine, but almost no high-frequency detail.
        val s = scorer.score(goodFace(sharpness = 5f))
        assertFalse(scorer.isClearlyVisible(s))
    }

    @Test
    fun `frontality falls as yaw grows`() {
        val frontal = scorer.score(goodFace(yaw = 0f)).frontality
        val turned = scorer.score(goodFace(yaw = 30f)).frontality
        val profile = scorer.score(goodFace(yaw = 100f)).frontality
        assertTrue(frontal > turned)
        assertTrue(turned > profile)
        assertEquals(1f, frontal, 1e-4f)
        assertEquals(0.4f, profile, 1e-4f)
    }

    @Test
    fun `a face clipped by the frame edge scores low completeness`() {
        val clipped = scorer.score(goodFace(box = BoxF(-60f, 400f, 140f, 600f)))
        assertTrue(clipped.completeness < 0.75f)
        assertFalse(scorer.isClearlyVisible(clipped))
    }

    @Test
    fun `closed eyes lower the score but do not fail the gate`() {
        // Eyes-open is a preference for the representative shot, not a visibility gate:
        // a blinking person is still visibly present.
        val closed = scorer.score(goodFace(eyeOpen = 0.02f))
        assertTrue(scorer.isClearlyVisible(closed))
        assertTrue(closed.weightedScore(config) < scorer.score(goodFace()).weightedScore(config))
    }

    @Test
    fun `a tiny background face fails the size gate`() {
        val tiny = scorer.score(goodFace(box = BoxF(10f, 10f, 40f, 40f)))
        assertFalse(scorer.isClearlyVisible(tiny))
    }

    @Test
    fun `missing classification probabilities are treated as neutral`() {
        val noProbs = goodFace().copy(
            leftEyeOpenProb = null, rightEyeOpenProb = null, smileProb = null
        )
        val s = scorer.score(noProbs)
        assertEquals(0.5f, s.eyesOpen, 1e-4f)
        assertEquals(0.5f, s.expression, 1e-4f)
    }

    @Test
    fun `weights sum to one`() {
        val c = PipelineConfig.Default
        val total = c.wFrontality + c.wSharpness + c.wEyesOpen +
            c.wFaceSize + c.wCompleteness + c.wExpression
        assertEquals(1f, total, 1e-5f)
    }
}

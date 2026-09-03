package com.iykyk.collage.core.track

import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.ScoredObservation
import com.iykyk.collage.core.model.Tracklet
import com.iykyk.collage.core.quality.FaceQualityScorer
import com.iykyk.collage.core.quality.weightedScore

/**
 * Groups per-frame detections into tracklets by greedy IoU association.
 *
 * Each surviving tracklet is one appearance. Detections that are present but not clearly
 * visible (blurred, turned away, clipped) still keep a track alive, so a person is not
 * double-counted while they pass through a blur — but they do not contribute to the
 * appearance's visible span.
 *
 * Association is by geometry rather than ML Kit tracking ids, because ML Kit reissues ids
 * part-way through a continuous appearance. The id is only a tie-breaking bonus.
 */
class TrackletBuilder(
    private val config: PipelineConfig,
    private val scorer: FaceQualityScorer,
) {

    private inner class OpenTrack(val id: Int) {
        val observations = mutableListOf<ScoredObservation>()
        var lastTimestampMs: Long = Long.MIN_VALUE
        lateinit var lastObservation: FaceObservation

        fun append(obs: FaceObservation) {
            val signals = scorer.score(obs)
            observations += ScoredObservation(
                observation = obs,
                signals = signals,
                clearlyVisible = scorer.isClearlyVisible(signals),
                score = signals.weightedScore(config),
            )
            lastObservation = obs
            lastTimestampMs = obs.timestampMs
        }
    }

    /**
     * @param observationsByFrame detections per sampled frame, in presentation order.
     * @return tracklets that qualify as appearances, ordered by when they became visible.
     */
    fun build(observationsByFrame: List<List<FaceObservation>>): List<Tracklet> {
        val open = mutableListOf<OpenTrack>()
        val closed = mutableListOf<OpenTrack>()
        var nextId = 0

        for (frame in observationsByFrame) {
            val frameTime = frame.firstOrNull()?.timestampMs

            // Retire tracks that have been unseen for longer than the gap tolerance.
            if (frameTime != null) {
                val expired = open.filter { frameTime - it.lastTimestampMs > config.gapToleranceMs }
                closed += expired
                open -= expired.toSet()
            }

            val unmatchedTracks = open.toMutableList()
            val unmatchedObs = frame.toMutableList()

            // Greedy: repeatedly take the globally best (track, detection) pair.
            while (unmatchedTracks.isNotEmpty() && unmatchedObs.isNotEmpty()) {
                var bestTrack: OpenTrack? = null
                var bestObs: FaceObservation? = null
                var bestScore = 0f

                for (track in unmatchedTracks) {
                    for (candidate in unmatchedObs) {
                        val score = associationScore(track.lastObservation, candidate)
                        if (score > bestScore) {
                            bestScore = score
                            bestTrack = track
                            bestObs = candidate
                        }
                    }
                }
                if (bestTrack == null || bestObs == null || bestScore < config.minAssociationIou) break

                bestTrack.append(bestObs)
                unmatchedTracks -= bestTrack
                unmatchedObs -= bestObs
            }

            // Anything left over starts a new track.
            for (candidate in unmatchedObs) {
                val track = OpenTrack(nextId++)
                track.append(candidate)
                open += track
            }
        }

        closed += open
        return closed
            .map { Tracklet(it.id, it.observations.toList()) }
            .filter { it.isAppearance(config) }
            .sortedBy { it.visibleStartMs }
    }

    /**
     * IoU between a track's last box and a candidate detection, plus a small bonus when ML
     * Kit agrees on the tracking id. The bonus breaks ties; it never creates a match on its
     * own, so a reissued id cannot split a track and a reused id cannot merge two.
     */
    private fun associationScore(last: FaceObservation, candidate: FaceObservation): Float {
        val iou = last.box.iou(candidate.box)
        if (iou <= 0f) return 0f
        val idsAgree = last.trackingId != null && last.trackingId == candidate.trackingId
        return iou + if (idsAgree) config.trackingIdBonus else 0f
    }
}

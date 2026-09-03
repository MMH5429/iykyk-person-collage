package com.iykyk.collage.pipeline

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.samples.SampleVideos
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class PersonCollagePipelineTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun runOnSample(index: Int): List<PipelineState> = runBlocking {
        val sample = SampleVideos.ensureExtracted(context)[index]
        PersonCollagePipeline(context, PipelineConfig.Default)
            .run(sample.uri, sample.label)
            .toList()
    }

    private fun analysisOf(index: Int) =
        runOnSample(index).filterIsInstance<PipelineState.Done>().single().result.analysis

    @Test
    fun sampleOneCompletesAndFindsPeople() {
        val states = runOnSample(0)
        val done = states.filterIsInstance<PipelineState.Done>().singleOrNull()
        assertTrue("pipeline did not complete: ${states.last()}", done != null)

        val analysis = done!!.result.analysis
        assertTrue("found ${analysis.people.size} people", analysis.people.isNotEmpty())
        analysis.people.forEach { person ->
            assertTrue("a person had no appearances", person.appearanceCount >= 1)
            assertTrue("shot bitmap is empty", person.shot.width > 0 && person.shot.height > 0)
        }
    }

    @Test
    fun progressIsMonotonicAndCoversEveryStage() {
        val states = runOnSample(0)
        val fractions = states.filterIsInstance<PipelineState.Analysing>().map { it.fraction }
        assertTrue("no progress reported", fractions.isNotEmpty())
        assertEquals(fractions.sorted(), fractions)
        assertTrue(fractions.all { it in 0f..1f })

        assertTrue(states.any { it is PipelineState.Preparing })
        assertTrue(states.any { it is PipelineState.GroupingPeople })
        assertTrue(states.any { it is PipelineState.SelectingShots })
    }

    @Test
    fun peopleAreOrderedByFirstAppearanceAndLabelledSequentially() {
        val analysis = analysisOf(0)
        val firstSeen = analysis.people.map { it.appearances.first().startMs }
        assertEquals(firstSeen.sorted(), firstSeen)
        assertEquals(
            analysis.people.indices.map { "Person ${it + 1}" },
            analysis.people.map { it.label },
        )
    }

    @Test
    fun eachAppearanceIsANonEmptySegment() {
        val analysis = analysisOf(0)
        analysis.people.forEach { person ->
            person.appearances.forEach { assertTrue(it.endMs >= it.startMs) }
        }
    }

    @Test
    fun coOccurringPeopleAreNeverMerged() {
        // The cannot-link guarantee, checked on real data: no person may have two
        // appearances that overlap in time with each other.
        val analysis = analysisOf(0)
        analysis.people.forEach { person ->
            person.appearances.sortedBy { it.startMs }.zipWithNext { a, b ->
                assertTrue("${person.label} is in two places at once", b.startMs > a.endMs)
            }
        }
    }

    @Test
    fun allThreeSamplesProcessWithoutError() {
        (0..2).forEach { i ->
            val last = runOnSample(i).last()
            assertTrue("sample ${i + 1} failed: $last", last is PipelineState.Done)
        }
    }
}

# Video-based Unique-Person Collage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An Android app that processes a portrait video fully on-device, groups appearances of the same person, counts each person's appearances, picks a strong representative shot per person, and renders a saveable/shareable collage.

**Architecture:** A `MediaCodec` decoder emits downsampled NV21 frames; ML Kit detects faces; each detection is quality-scored; detections are associated across frames by IoU into **tracklets** (one tracklet = one appearance); each tracklet is embedded once with FaceNet-512 averaged over its best crops; tracklets are grouped into people by constrained agglomerative clustering with a *cannot-link* rule for time-overlapping tracklets. All grouping/counting/scoring logic is pure Kotlin over plain data so it is JVM-unit-testable and tunable offline.

**Tech Stack:** Kotlin, Jetpack Compose, minSdk 26, ML Kit Face Detection 16.1.7, TensorFlow Lite 2.17.0 (FaceNet-512, Inception-ResNet-v1, 160×160 → 512-d), Coroutines 1.11.0, JUnit 4.13.2.

**Spec:** `docs/superpowers/specs/2026-09-03-video-person-collage-design.md`

## Global Constraints

- Kotlin only. **minSdk 26**, target/compile SDK = whatever the Studio wizard picks (do not lower it).
- Jetpack Compose for UI. Collage itself is rendered to a `Bitmap` via `android.graphics.Canvas`, never with Compose.
- Everything on-device. **No network calls anywhere in the app.** No backend.
- Application ID and root package: `com.iykyk.collage`.
- Nothing about the three sample clips may be hardcoded. Bundled samples must flow through the *identical* code path as a user-picked video.
- All pipeline work runs off the main thread; the UI stays responsive and cancellable.
- Every "pure" class listed below must have **zero Android imports** so it runs in `src/test/` on the JVM.
- Tuned constants (thresholds, weights) live in one `PipelineConfig` object with named fields — never as inline magic numbers.
- Chosen similarity threshold τ must end up documented in the README (assignment requirement).
- Deadline: Sunday 6 September 2026, 23:59 IST.

### Pinned dependency versions (verified against live Maven metadata 2026-09-03)

| Dependency | Version |
|---|---|
| `com.google.mlkit:face-detection` | 16.1.7 |
| `org.tensorflow:tensorflow-lite` | 2.17.0 |
| `org.jetbrains.kotlinx:kotlinx-coroutines-android` | 1.11.0 |
| `org.jetbrains.kotlinx:kotlinx-coroutines-play-services` | 1.11.0 |
| `org.jetbrains.kotlinx:kotlinx-serialization-json` | 1.11.0 |
| `junit:junit` | 4.13.2 |
| `androidx.test.ext:junit` | 1.3.0 |
| AGP / Kotlin / Compose BOM | whatever the Studio wizard generates — **do not hand-edit** |

### Assets already downloaded and verified

Located in the session scratchpad `C:\Users\mrina\AppData\Local\Temp\claude\D--IFYKYK\a563bd06-2ca9-48b8-9d45-20f1aee26a8a\scratchpad\`:

- `facenet_512.tflite` (24,394,880 bytes) — Inception-ResNet-v1, input tensor `input_1`, 160×160×3 float, 512-d output
- `s1.mp4` (4,705,478 bytes), `s2.mp4` (4,776,144), `s3.mp4` (4,754,137)

---

## File Structure

```
app/src/main/java/com/iykyk/collage/
  MainActivity.kt                      # Compose entry point
  ui/AppScreen.kt                      # pick / processing / result states
  ui/CollageViewModel.kt               # holds pipeline state, launches work
  core/PipelineConfig.kt               # ALL tuned constants
  core/model/Geometry.kt               # BoxF, PointF2, IoU, intersection  (pure)
  core/model/FaceObservation.kt        # one detected face in one frame     (pure)
  core/model/Tracklet.kt               # ScoredObservation, Tracklet        (pure)
  core/model/Person.kt                 # PersonResult, CollageResult        (pure)
  core/quality/Sharpness.kt            # variance of Laplacian over luma    (pure)
  core/quality/FaceQualityScorer.kt    # six signals + gates + score        (pure)
  core/track/TrackletBuilder.kt        # greedy IoU association             (pure)
  core/cluster/CosineDistance.kt       # normalisation + distance           (pure)
  core/cluster/ConstrainedClusterer.kt # avg-linkage + cannot-link          (pure)
  core/frame/Nv21.kt                   # YUV_420_888 -> downsampled NV21    (pure)
  core/frame/RotationMapper.kt         # upright <-> raw coordinate mapping (pure)
  core/collage/GridSpec.kt             # person count -> grid layout        (pure)
  video/FrameSource.kt                 # interface + Nv21Frame
  video/MediaCodecFrameSource.kt       # sequential decode -> ImageReader
  video/FullResFrameGrabber.kt         # MMR seek for representative shots
  detect/MlKitFaceDetector.kt          # ML Kit -> FaceObservation
  embed/FaceAligner.kt                 # eye-based similarity transform
  embed/FaceEmbedder.kt                # TFLite FaceNet-512
  pipeline/PersonCollagePipeline.kt    # orchestration, progress Flow
  pipeline/PipelineState.kt            # sealed progress states
  pipeline/AnalysisDump.kt             # debug JSON export for tuning
  collage/CollageRenderer.kt           # Canvas -> 1080x1920 bitmap
  export/CollageExporter.kt            # MediaStore save + share sheet
  samples/SampleVideos.kt              # assets/ -> cache, same code path

app/src/main/assets/
  facenet_512.tflite
  samples/sample1.mp4 sample2.mp4 sample3.mp4

app/src/test/java/com/iykyk/collage/     # pure JVM tests (the tuning harness lives here)
app/src/androidTest/java/com/iykyk/collage/  # decode + embed on-device tests
```

**Why these boundaries:** everything under `core/` is pure Kotlin with no Android types, which is what makes the 50%-weighted accuracy logic testable and tunable without a device. Android-touching code (`video/`, `detect/`, `embed/`, `collage/`, `export/`) is thin adapter layer around it.

---

## Task 1: Project skeleton, dependencies, assets

**Files:**
- Create: the whole Gradle project via the Android Studio wizard (do not hand-write Gradle files)
- Modify: `gradle/libs.versions.toml`, `app/build.gradle.kts`, `app/src/main/AndroidManifest.xml`
- Create: `app/src/main/assets/facenet_512.tflite`, `app/src/main/assets/samples/sample{1,2,3}.mp4`

**Interfaces:**
- Consumes: nothing
- Produces: a buildable `:app` module with all dependencies resolved and assets bundled

- [ ] **Step 1: Install Android Studio and set the JDK**

Install the current Android Studio. Then **Settings → Build, Execution, Deployment → Build Tools → Gradle → Gradle JDK = the bundled JBR (21)**.

The system JDK on this machine is **23**, which AGP does not support. Do not select it.

- [ ] **Step 2: Generate the project with the wizard**

New Project → **Empty Activity** (the Compose one) with:
- Name: `IYKYK Collage`
- Package name: `com.iykyk.collage`
- Language: Kotlin, Build configuration language: **Kotlin DSL + version catalog**
- Minimum SDK: **API 26**

The wizard refuses a non-empty directory, and `D:\IFYKYK` already contains `.git/` and `docs/`. So generate into `D:\IFYKYK-scaffold`, then move it in:

```bash
cd /d && mv IFYKYK-scaffold/* IFYKYK-scaffold/.gitignore IFYKYK/ 2>/dev/null; rmdir IFYKYK-scaffold
cd /d/IFYKYK && ls
```

Merge the wizard's `.gitignore` into the existing one rather than letting it clobber `docs/`.

- [ ] **Step 3: Add dependencies to the version catalog**

In `gradle/libs.versions.toml`, add to `[versions]`:

```toml
mlkitFaceDetection = "16.1.7"
tensorflowLite = "2.17.0"
coroutines = "1.11.0"
serializationJson = "1.11.0"
junit4 = "4.13.2"
androidxTestJunit = "1.3.0"
```

and to `[libraries]`:

```toml
mlkit-face-detection = { group = "com.google.mlkit", name = "face-detection", version.ref = "mlkitFaceDetection" }
tensorflow-lite = { group = "org.tensorflow", name = "tensorflow-lite", version.ref = "tensorflowLite" }
kotlinx-coroutines-android = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-android", version.ref = "coroutines" }
kotlinx-coroutines-play-services = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-play-services", version.ref = "coroutines" }
kotlinx-coroutines-test = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-test", version.ref = "coroutines" }
kotlinx-serialization-json = { group = "org.jetbrains.kotlinx", name = "kotlinx-serialization-json", version.ref = "serializationJson" }
```

Keep every wizard-generated entry untouched.

- [ ] **Step 4: Wire the dependencies and asset packing in `app/build.gradle.kts`**

Add the Kotlin serialization plugin to the `plugins { }` block:

```kotlin
kotlin("plugin.serialization") version "2.4.10"
```

Inside `android { }`:

```kotlin
androidResources {
    // TFLite models must not be compressed or the interpreter cannot mmap them.
    noCompress += "tflite"
}
packaging {
    resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
}
```

Inside `dependencies { }`:

```kotlin
implementation(libs.mlkit.face.detection)
implementation(libs.tensorflow.lite)
implementation(libs.kotlinx.coroutines.android)
implementation(libs.kotlinx.coroutines.play.services)
implementation(libs.kotlinx.serialization.json)
testImplementation(libs.kotlinx.coroutines.test)
```

- [ ] **Step 5: Declare the legacy write permission**

In `app/src/main/AndroidManifest.xml`, inside `<manifest>` before `<application>`:

```xml
<uses-permission
    android:name="android.permission.WRITE_EXTERNAL_STORAGE"
    android:maxSdkVersion="28" />
```

API 29+ writes through `MediaStore` with no permission; 26–28 needs this.

- [ ] **Step 6: Copy in the assets**

```bash
SCRATCH="/c/Users/mrina/AppData/Local/Temp/claude/D--IFYKYK/a563bd06-2ca9-48b8-9d45-20f1aee26a8a/scratchpad"
mkdir -p /d/IFYKYK/app/src/main/assets/samples
cp "$SCRATCH/facenet_512.tflite" /d/IFYKYK/app/src/main/assets/facenet_512.tflite
cp "$SCRATCH/s1.mp4" /d/IFYKYK/app/src/main/assets/samples/sample1.mp4
cp "$SCRATCH/s2.mp4" /d/IFYKYK/app/src/main/assets/samples/sample2.mp4
cp "$SCRATCH/s3.mp4" /d/IFYKYK/app/src/main/assets/samples/sample3.mp4
ls -l /d/IFYKYK/app/src/main/assets /d/IFYKYK/app/src/main/assets/samples
```

Expected: `facenet_512.tflite` at 24,394,880 bytes and three `.mp4` files at ~4.7 MB each.

- [ ] **Step 7: Build and verify**

```bash
cd /d/IFYKYK && ./gradlew :app:assembleDebug
```

Expected: `BUILD SUCCESSFUL`. Install and launch on the physical device; the wizard's default screen appears.

- [ ] **Step 8: Commit**

```bash
cd /d/IFYKYK
git add -A
git commit -m "chore: scaffold Compose app, dependencies, FaceNet model and sample assets"
```

---

## Task 2: Geometry and quality primitives (pure JVM)

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/core/model/Geometry.kt`
- Create: `app/src/main/java/com/iykyk/collage/core/quality/Sharpness.kt`
- Test: `app/src/test/java/com/iykyk/collage/core/model/GeometryTest.kt`
- Test: `app/src/test/java/com/iykyk/collage/core/quality/SharpnessTest.kt`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `BoxF(left, top, right, bottom)` with `width`, `height`, `area`, `centerX`, `centerY`, `intersect(BoxF): BoxF?`, `iou(BoxF): Float`, `clampTo(w, h): BoxF`, `expand(factor): BoxF`
  - `PointF2(x, y)`
  - `laplacianVarianceFromLuma(luma: ByteArray, rowStride: Int, region: BoxF): Float`

- [ ] **Step 1: Write the failing geometry test**

`app/src/test/java/com/iykyk/collage/core/model/GeometryTest.kt`:

```kotlin
package com.iykyk.collage.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GeometryTest {

    @Test
    fun `identical boxes have iou of one`() {
        val a = BoxF(0f, 0f, 10f, 10f)
        assertEquals(1f, a.iou(a), 1e-4f)
    }

    @Test
    fun `disjoint boxes have iou of zero`() {
        val a = BoxF(0f, 0f, 10f, 10f)
        val b = BoxF(20f, 20f, 30f, 30f)
        assertEquals(0f, a.iou(b), 1e-4f)
        assertNull(a.intersect(b))
    }

    @Test
    fun `half overlapping boxes have iou of one third`() {
        // Union 150, intersection 50.
        val a = BoxF(0f, 0f, 10f, 10f)
        val b = BoxF(5f, 0f, 15f, 10f)
        assertEquals(1f / 3f, a.iou(b), 1e-4f)
    }

    @Test
    fun `expand grows the box about its centre`() {
        val e = BoxF(10f, 10f, 20f, 20f).expand(2f)
        assertEquals(BoxF(5f, 5f, 25f, 25f), e)
    }

    @Test
    fun `clampTo keeps the box inside the frame`() {
        val c = BoxF(-5f, -5f, 50f, 50f).clampTo(40, 30)
        assertEquals(BoxF(0f, 0f, 40f, 30f), c)
    }
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.model.GeometryTest"
```

Expected: compilation failure — `Unresolved reference: BoxF`.

- [ ] **Step 3: Implement the geometry types**

`app/src/main/java/com/iykyk/collage/core/model/Geometry.kt`:

```kotlin
package com.iykyk.collage.core.model

import kotlin.math.max
import kotlin.math.min

/** An axis-aligned rectangle in frame coordinates. Pure data — no Android types. */
data class BoxF(val left: Float, val top: Float, val right: Float, val bottom: Float) {

    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val area: Float get() = max(0f, width) * max(0f, height)
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f

    fun intersect(other: BoxF): BoxF? {
        val l = max(left, other.left)
        val t = max(top, other.top)
        val r = min(right, other.right)
        val b = min(bottom, other.bottom)
        return if (l < r && t < b) BoxF(l, t, r, b) else null
    }

    /** Intersection over union. 0 when disjoint, 1 when identical. */
    fun iou(other: BoxF): Float {
        val inter = intersect(other)?.area ?: return 0f
        val union = area + other.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** Scales the box about its own centre. factor 2 doubles each side. */
    fun expand(factor: Float): BoxF {
        val halfW = width * factor / 2f
        val halfH = height * factor / 2f
        return BoxF(centerX - halfW, centerY - halfH, centerX + halfW, centerY + halfH)
    }

    fun clampTo(frameWidth: Int, frameHeight: Int): BoxF = BoxF(
        left = left.coerceIn(0f, frameWidth.toFloat()),
        top = top.coerceIn(0f, frameHeight.toFloat()),
        right = right.coerceIn(0f, frameWidth.toFloat()),
        bottom = bottom.coerceIn(0f, frameHeight.toFloat()),
    )
}

data class PointF2(val x: Float, val y: Float)
```

- [ ] **Step 4: Run the geometry test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.model.GeometryTest"
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Write the failing sharpness test**

`app/src/test/java/com/iykyk/collage/core/quality/SharpnessTest.kt`:

```kotlin
package com.iykyk.collage.core.quality

import com.iykyk.collage.core.model.BoxF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SharpnessTest {

    private val width = 32
    private val height = 32
    private val region = BoxF(0f, 0f, 32f, 32f)

    private fun luma(pixel: (x: Int, y: Int) -> Int): ByteArray {
        val out = ByteArray(width * height)
        for (y in 0 until height) for (x in 0 until width) {
            out[y * width + x] = pixel(x, y).toByte()
        }
        return out
    }

    @Test
    fun `flat image has near zero laplacian variance`() {
        val flat = luma { _, _ -> 128 }
        assertEquals(0f, laplacianVarianceFromLuma(flat, width, region), 1e-3f)
    }

    @Test
    fun `checkerboard has high laplacian variance`() {
        val checker = luma { x, y -> if ((x + y) % 2 == 0) 0 else 255 }
        assertTrue(laplacianVarianceFromLuma(checker, width, region) > 1000f)
    }

    @Test
    fun `blurred edge scores lower than hard edge`() {
        val hard = luma { x, _ -> if (x < 16) 0 else 255 }
        val soft = luma { x, _ -> ((x.toFloat() / width) * 255f).toInt() }
        assertTrue(
            laplacianVarianceFromLuma(soft, width, region) <
                laplacianVarianceFromLuma(hard, width, region)
        )
    }

    @Test
    fun `region restricts the measurement`() {
        // Sharp only on the left half; measuring the right half must be flat.
        val half = luma { x, y -> if (x < 16) (if ((x + y) % 2 == 0) 0 else 255) else 128 }
        val right = BoxF(17f, 1f, 31f, 31f)
        assertEquals(0f, laplacianVarianceFromLuma(half, width, right), 1e-3f)
    }

    @Test
    fun `degenerate region returns zero`() {
        val flat = luma { _, _ -> 128 }
        assertEquals(0f, laplacianVarianceFromLuma(flat, width, BoxF(5f, 5f, 6f, 6f)), 1e-3f)
    }
}
```

- [ ] **Step 6: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.quality.SharpnessTest"
```

Expected: `Unresolved reference: laplacianVarianceFromLuma`.

- [ ] **Step 7: Implement sharpness**

`app/src/main/java/com/iykyk/collage/core/quality/Sharpness.kt`:

```kotlin
package com.iykyk.collage.core.quality

import com.iykyk.collage.core.model.BoxF
import kotlin.math.roundToInt

/**
 * Variance of the 4-neighbour Laplacian over [region] of a single-channel luma plane.
 *
 * Low variance means few high-frequency details, i.e. a soft or motion-blurred face.
 * This is what excludes whip-pan passes: a blurred face scores near zero and never
 * counts as "clearly visible".
 *
 * Operates directly on the Y plane of YUV_420_888, so no colour conversion is needed.
 */
fun laplacianVarianceFromLuma(luma: ByteArray, rowStride: Int, region: BoxF): Float {
    // Interior pixels only — the kernel needs one pixel of margin on every side.
    val x0 = region.left.roundToInt().coerceAtLeast(1)
    val y0 = region.top.roundToInt().coerceAtLeast(1)
    val x1 = region.right.roundToInt().coerceAtMost(rowStride - 2)
    val y1 = (region.bottom.roundToInt()).coerceAtMost(luma.size / rowStride - 2)
    if (x1 - x0 < 2 || y1 - y0 < 2) return 0f

    var sum = 0.0
    var sumSq = 0.0
    var n = 0

    for (y in y0..y1) {
        val row = y * rowStride
        val up = row - rowStride
        val down = row + rowStride
        for (x in x0..x1) {
            val c = luma[row + x].toInt() and 0xFF
            val l = luma[row + x - 1].toInt() and 0xFF
            val r = luma[row + x + 1].toInt() and 0xFF
            val u = luma[up + x].toInt() and 0xFF
            val d = luma[down + x].toInt() and 0xFF
            val lap = (l + r + u + d - 4 * c).toDouble()
            sum += lap
            sumSq += lap * lap
            n++
        }
    }
    if (n == 0) return 0f
    val mean = sum / n
    return ((sumSq / n) - mean * mean).toFloat().coerceAtLeast(0f)
}
```

- [ ] **Step 8: Run the sharpness test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.quality.SharpnessTest"
```

Expected: PASS, 5 tests.

- [ ] **Step 9: Commit the primitives so far**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/core app/src/test/java/com/iykyk/collage/core
git commit -m "feat(core): add geometry primitives and Laplacian sharpness measure"
```

The quality scorer that consumes these lands in Task 2b, immediately below, because it needs `FaceObservation` and `PipelineConfig` defined first.

---

## Task 2b: FaceObservation, PipelineConfig, and the quality scorer

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/core/model/FaceObservation.kt`
- Create: `app/src/main/java/com/iykyk/collage/core/PipelineConfig.kt`
- Create: `app/src/main/java/com/iykyk/collage/core/quality/FaceQualityScorer.kt`
- Test: `app/src/test/java/com/iykyk/collage/core/quality/FaceQualityScorerTest.kt`

**Interfaces:**
- Consumes: `BoxF`, `PointF2`, `laplacianVarianceFromLuma` from Task 2
- Produces: `FaceObservation`, `QualitySignals`, `PipelineConfig`, `FaceQualityScorer`

- [ ] **Step 1: Define `FaceObservation` and `QualitySignals`**

`app/src/main/java/com/iykyk/collage/core/model/FaceObservation.kt`:

```kotlin
package com.iykyk.collage.core.model

/**
 * One face detected in one sampled frame, in *upright* analysis-frame coordinates.
 *
 * "Upright" means the video's rotation metadata has already been applied, so
 * [box] and the landmarks are in the same space a viewer sees.
 */
data class FaceObservation(
    val frameIndex: Int,
    val timestampMs: Long,
    /** ML Kit tracking id when available; used only as an association hint. */
    val trackingId: Int?,
    val box: BoxF,
    val leftEye: PointF2?,
    val rightEye: PointF2?,
    /** Head pose in degrees: X = pitch (nod), Y = yaw (turn), Z = roll (tilt). */
    val headEulerX: Float,
    val headEulerY: Float,
    val headEulerZ: Float,
    val leftEyeOpenProb: Float?,
    val rightEyeOpenProb: Float?,
    val smileProb: Float?,
    /** Raw variance-of-Laplacian over the face region; normalised by the scorer. */
    val rawSharpness: Float,
    val frameWidth: Int,
    val frameHeight: Int,
)

/** Six independent quality signals, each normalised to [0, 1]. */
data class QualitySignals(
    val sharpness: Float,
    val frontality: Float,
    val eyesOpen: Float,
    val expression: Float,
    val faceSize: Float,
    val completeness: Float,
)
```

- [ ] **Step 2: Define `PipelineConfig`**

`app/src/main/java/com/iykyk/collage/core/PipelineConfig.kt`:

```kotlin
package com.iykyk.collage.core

/**
 * Every tuned constant in the pipeline, in one place.
 *
 * Defaults here are starting points. Task 14 sweeps them against Sample 1's
 * published ground truth (5 people x 4 appearances) and the winning values are
 * written back here and documented in the README.
 */
data class PipelineConfig(
    // --- Sampling -------------------------------------------------------
    /** Frames analysed per second of video. */
    val sampleFps: Int = 8,
    /** Long edge of the analysis frame in pixels; the decoder subsamples to roughly this. */
    val analysisLongEdge: Int = 540,

    // --- Quality normalisation ------------------------------------------
    /** Raw Laplacian variance that maps to a sharpness score of 1.0. */
    val sharpnessSaturation: Float = 350f,
    val maxYawDeg: Float = 60f,
    val maxPitchDeg: Float = 45f,
    val maxRollDeg: Float = 45f,
    /** Face-box area fraction of the frame that maps to a faceSize score of 1.0. */
    val faceAreaSaturation: Float = 0.06f,
    /** Distance from the frame edge, as a fraction of the short edge, below which a face reads as clipped. */
    val edgeMarginFraction: Float = 0.01f,

    // --- "Clearly visible" gates ----------------------------------------
    val minSharpness: Float = 0.20f,
    val minFrontality: Float = 0.35f,
    val minFaceSize: Float = 0.10f,
    val minCompleteness: Float = 0.85f,

    // --- Representative-shot weights (must sum to 1) --------------------
    val wFrontality: Float = 0.25f,
    val wSharpness: Float = 0.25f,
    val wEyesOpen: Float = 0.20f,
    val wFaceSize: Float = 0.10f,
    val wCompleteness: Float = 0.10f,
    val wExpression: Float = 0.10f,

    // --- Tracklet association -------------------------------------------
    /** How long a track may go unseen before it is closed. */
    val gapToleranceMs: Long = 400,
    /** Minimum IoU for a detection to join an open track. */
    val minAssociationIou: Float = 0.20f,
    /** Bonus added to the association score when ML Kit tracking ids agree. */
    val trackingIdBonus: Float = 0.15f,
    /** A tracklet must hold at least this many clearly-visible frames to be an appearance. */
    val minVisibleFrames: Int = 2,
    /** ...and span at least this long. */
    val minVisibleDurationMs: Long = 200,

    // --- Embedding -------------------------------------------------------
    /** Best-quality crops averaged into one embedding per tracklet. */
    val cropsPerTracklet: Int = 5,

    // --- Clustering ------------------------------------------------------
    /** Cosine-distance cut for agglomerative clustering. THE similarity threshold. */
    val clusterThreshold: Float = 0.55f,
    /** Tracklets closer than this in embedding *and* adjacent in time are one appearance. */
    val sameAppearanceDistance: Float = 0.25f,

    // --- Representative crop ---------------------------------------------
    /** Crop side as a multiple of the face box; deliberately generous, never tight. */
    val representativeCropFactor: Float = 2.2f,
) {
    companion object {
        val Default = PipelineConfig()
    }
}
```

- [ ] **Step 3: Write the failing quality-scorer test**

`app/src/test/java/com/iykyk/collage/core/quality/FaceQualityScorerTest.kt`:

```kotlin
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
        val profile = scorer.score(goodFace(yaw = 75f)).frontality
        assertTrue(frontal > turned)
        assertTrue(turned > profile)
        assertEquals(1f, frontal, 1e-4f)
        assertEquals(0f, profile, 1e-4f)
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
```

- [ ] **Step 4: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.quality.FaceQualityScorerTest"
```

Expected: `Unresolved reference: FaceQualityScorer`.

- [ ] **Step 5: Implement the scorer**

`app/src/main/java/com/iykyk/collage/core/quality/FaceQualityScorer.kt`:

```kotlin
package com.iykyk.collage.core.quality

import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.QualitySignals
import kotlin.math.abs
import kotlin.math.min

/** Weighted sum used to rank candidate shots for a person. */
fun QualitySignals.weightedScore(c: PipelineConfig): Float =
    c.wFrontality * frontality +
        c.wSharpness * sharpness +
        c.wEyesOpen * eyesOpen +
        c.wFaceSize * faceSize +
        c.wCompleteness * completeness +
        c.wExpression * expression

/**
 * Turns a raw detection into six normalised quality signals, and decides whether the
 * face is "clearly visible" in the sense the assignment defines an appearance.
 *
 * Pure: no Android types, so the whole gating policy is unit-testable and tunable offline.
 */
class FaceQualityScorer(private val config: PipelineConfig) {

    fun score(obs: FaceObservation): QualitySignals = QualitySignals(
        sharpness = (obs.rawSharpness / config.sharpnessSaturation).coerceIn(0f, 1f),
        frontality = frontality(obs),
        eyesOpen = eyesOpen(obs),
        expression = obs.smileProb ?: NEUTRAL,
        faceSize = faceSize(obs),
        completeness = completeness(obs),
    )

    /**
     * A face counts as clearly visible when it is sharp, reasonably front-facing, big
     * enough to identify, and not cut off by the frame edge.
     *
     * Eyes-open and expression are deliberately excluded: a person mid-blink is still
     * present. Those signals only influence which shot is chosen to represent them.
     */
    fun isClearlyVisible(s: QualitySignals): Boolean =
        s.sharpness >= config.minSharpness &&
            s.frontality >= config.minFrontality &&
            s.faceSize >= config.minFaceSize &&
            s.completeness >= config.minCompleteness

    private fun frontality(obs: FaceObservation): Float {
        val yaw = (abs(obs.headEulerY) / config.maxYawDeg).coerceIn(0f, 1f)
        val pitch = (abs(obs.headEulerX) / config.maxPitchDeg).coerceIn(0f, 1f)
        val roll = (abs(obs.headEulerZ) / config.maxRollDeg).coerceIn(0f, 1f)
        // Yaw dominates: turning away hides identity far faster than nodding or tilting.
        return (1f - (0.60f * yaw + 0.25f * pitch + 0.15f * roll)).coerceIn(0f, 1f)
    }

    private fun eyesOpen(obs: FaceObservation): Float {
        val l = obs.leftEyeOpenProb
        val r = obs.rightEyeOpenProb
        if (l == null && r == null) return NEUTRAL
        return min(l ?: NEUTRAL, r ?: NEUTRAL)
    }

    private fun faceSize(obs: FaceObservation): Float {
        val frameArea = (obs.frameWidth.toFloat() * obs.frameHeight.toFloat())
        if (frameArea <= 0f) return 0f
        return ((obs.box.area / frameArea) / config.faceAreaSaturation).coerceIn(0f, 1f)
    }

    /**
     * How much of the face box lies inside the frame, with an extra penalty for sitting
     * right on the edge — a face touching the border is usually mid-exit and partly cut.
     */
    private fun completeness(obs: FaceObservation): Float {
        val frame = BoxF(0f, 0f, obs.frameWidth.toFloat(), obs.frameHeight.toFloat())
        val visible = obs.box.intersect(frame)?.area ?: return 0f
        if (obs.box.area <= 0f) return 0f
        val inside = (visible / obs.box.area).coerceIn(0f, 1f)

        val margin = min(obs.frameWidth, obs.frameHeight) * config.edgeMarginFraction
        val touchesEdge = obs.box.left <= margin ||
            obs.box.top <= margin ||
            obs.box.right >= obs.frameWidth - margin ||
            obs.box.bottom >= obs.frameHeight - margin
        return if (touchesEdge) inside * EDGE_PENALTY else inside
    }

    private companion object {
        const val NEUTRAL = 0.5f
        const val EDGE_PENALTY = 0.6f
    }
}
```

- [ ] **Step 6: Run the scorer test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.quality.FaceQualityScorerTest"
```

Expected: PASS, 9 tests.

- [ ] **Step 7: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/core app/src/test/java/com/iykyk/collage/core
git commit -m "feat(core): add face observation model, pipeline config and quality scorer"
```

---

## Task 3: TrackletBuilder — appearances by IoU association (pure JVM)

This is the single highest-leverage task in the plan: a tracklet *is* an appearance, so this class decides the appearance counts that carry 50% of the grade.

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/core/model/Tracklet.kt`
- Create: `app/src/main/java/com/iykyk/collage/core/track/TrackletBuilder.kt`
- Test: `app/src/test/java/com/iykyk/collage/core/track/TrackletBuilderTest.kt`

**Interfaces:**
- Consumes: `BoxF`, `FaceObservation`, `QualitySignals`, `weightedScore`, `FaceQualityScorer`, `PipelineConfig`
- Produces:
  - `ScoredObservation(observation, signals, clearlyVisible, score)`
  - `Tracklet(id, observations)` with `visibleStartMs`, `visibleEndMs`, `visibleFrameCount`, `best`, `overlapsInTime(other)`, `isAppearance(config)`
  - `TrackletBuilder(config, scorer).build(observationsByFrame: List<List<FaceObservation>>): List<Tracklet>`

**Why association rather than ML Kit tracking ids:** ML Kit reissues tracking ids part-way through a continuous appearance, which would inflate the count. Greedy IoU association across consecutive sampled frames is immune to that, uses the tracking id only as a tie-breaking bonus, and — unlike ML Kit's tracker — is fully testable on the JVM with synthetic sequences.

- [ ] **Step 1: Define the tracklet model**

`app/src/main/java/com/iykyk/collage/core/model/Tracklet.kt`:

```kotlin
package com.iykyk.collage.core.model

import com.iykyk.collage.core.PipelineConfig

/** A detection with its quality assessment attached. */
data class ScoredObservation(
    val observation: FaceObservation,
    val signals: QualitySignals,
    val clearlyVisible: Boolean,
    val score: Float,
)

/**
 * One continuous run of a single face across consecutive sampled frames.
 *
 * A tracklet that passes [isAppearance] is exactly one "appearance" as the assignment
 * defines it: a continuous segment during which the person is clearly visible.
 */
data class Tracklet(
    val id: Int,
    val observations: List<ScoredObservation>,
) {
    private val visible: List<ScoredObservation> = observations.filter { it.clearlyVisible }

    /** Bounds of the *clearly visible* span — the appearance proper, not the blurry approach. */
    val visibleStartMs: Long get() = visible.firstOrNull()?.observation?.timestampMs ?: -1L
    val visibleEndMs: Long get() = visible.lastOrNull()?.observation?.timestampMs ?: -1L
    val visibleFrameCount: Int get() = visible.size
    val visibleDurationMs: Long
        get() = if (visible.isEmpty()) 0L else visibleEndMs - visibleStartMs

    /** The best candidate shot in this tracklet, ranked by weighted quality. */
    val best: ScoredObservation? get() = visible.maxByOrNull { it.score }

    fun isAppearance(config: PipelineConfig): Boolean =
        visibleFrameCount >= config.minVisibleFrames &&
            visibleDurationMs >= config.minVisibleDurationMs

    /**
     * True when both tracklets are clearly visible at the same moment.
     *
     * Two people on screen together cannot be the same person, so this drives the
     * cannot-link constraint in clustering.
     */
    fun overlapsInTime(other: Tracklet): Boolean {
        if (visibleFrameCount == 0 || other.visibleFrameCount == 0) return false
        return visibleStartMs <= other.visibleEndMs && other.visibleStartMs <= visibleEndMs
    }
}
```

- [ ] **Step 2: Write the failing tracklet-builder test**

`app/src/test/java/com/iykyk/collage/core/track/TrackletBuilderTest.kt`:

```kotlin
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

    private val left = BoxF(100f, 400f, 300f, 600f)
    private val right = BoxF(1000f, 400f, 1200f, 600f)

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
            assertTrue(xs.all { it < 540f } || xs.all { it > 540f })
        }
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
```

- [ ] **Step 3: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.track.TrackletBuilderTest"
```

Expected: `Unresolved reference: TrackletBuilder`.

- [ ] **Step 4: Implement the builder**

`app/src/main/java/com/iykyk/collage/core/track/TrackletBuilder.kt`:

```kotlin
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
 * Each surviving tracklet is one appearance. Detections that are present but not
 * clearly visible (blurred, turned away, clipped) still keep a track alive so a person
 * is not double-counted while they pass through a blur, but they do not contribute to
 * the appearance's visible span.
 */
class TrackletBuilder(
    private val config: PipelineConfig,
    private val scorer: FaceQualityScorer,
) {

    private class OpenTrack(val id: Int) {
        val observations = mutableListOf<ScoredObservation>()
        var lastTimestampMs: Long = Long.MIN_VALUE
        lateinit var lastObservation: FaceObservation
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
     * IoU between a track's last box and a candidate detection, plus a small bonus when
     * ML Kit agrees on the tracking id. The bonus breaks ties; it never creates a match
     * on its own, so a reissued id cannot split a track and a reused id cannot merge two.
     */
    private fun associationScore(last: FaceObservation, candidate: FaceObservation): Float {
        val iou = last.box.iou(candidate.box)
        if (iou <= 0f) return 0f
        val idsAgree = last.trackingId != null && last.trackingId == candidate.trackingId
        return iou + if (idsAgree) config.trackingIdBonus else 0f
    }

    private fun OpenTrack.append(obs: FaceObservation) {
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
```

- [ ] **Step 5: Run the test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.track.TrackletBuilderTest"
```

Expected: PASS, 10 tests.

- [ ] **Step 6: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/core app/src/test/java/com/iykyk/collage/core
git commit -m "feat(core): build appearances as tracklets via greedy IoU association"
```

---

## Task 4: Constrained agglomerative clustering (pure JVM)

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/core/cluster/CosineDistance.kt`
- Create: `app/src/main/java/com/iykyk/collage/core/cluster/ConstrainedClusterer.kt`
- Test: `app/src/test/java/com/iykyk/collage/core/cluster/ConstrainedClustererTest.kt`

**Interfaces:**
- Consumes: nothing from earlier tasks (deliberately standalone — it takes plain vectors)
- Produces:
  - `FloatArray.l2Normalized(): FloatArray`
  - `cosineDistance(a: FloatArray, b: FloatArray): Float` — `1 - dot` for unit vectors, range [0, 2]
  - `ConstrainedClusterer(threshold: Float).cluster(vectors: List<FloatArray>, cannotLink: Set<Pair<Int, Int>>): List<List<Int>>` — returns member index lists, each sorted ascending, ordered by first member

- [ ] **Step 1: Write the failing test**

`app/src/test/java/com/iykyk/collage/core/cluster/ConstrainedClustererTest.kt`:

```kotlin
package com.iykyk.collage.core.cluster

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

class ConstrainedClustererTest {

    /** A unit vector in the plane at [deg] degrees, embedded in a 4-d space. */
    private fun v(deg: Double): FloatArray {
        val r = Math.toRadians(deg)
        return floatArrayOf(cos(r).toFloat(), sin(r).toFloat(), 0f, 0f).l2Normalized()
    }

    private fun clusterOf(result: List<List<Int>>, index: Int): List<Int> =
        result.first { index in it }

    @Test
    fun `cosine distance is zero for identical unit vectors and one for orthogonal`() {
        assertEquals(0f, cosineDistance(v(0.0), v(0.0)), 1e-4f)
        assertEquals(1f, cosineDistance(v(0.0), v(90.0)), 1e-4f)
    }

    @Test
    fun `l2 normalisation gives unit length`() {
        val n = floatArrayOf(3f, 4f, 0f, 0f).l2Normalized()
        assertEquals(1f, n.fold(0f) { acc, x -> acc + x * x }, 1e-5f)
    }

    @Test
    fun `two tight groups become two clusters`() {
        // 0,1 near 0 degrees; 2,3 near 90 degrees.
        val result = ConstrainedClusterer(threshold = 0.2f)
            .cluster(listOf(v(0.0), v(4.0), v(90.0), v(94.0)), emptySet())
        assertEquals(2, result.size)
        assertEquals(listOf(0, 1), clusterOf(result, 0))
        assertEquals(listOf(2, 3), clusterOf(result, 2))
    }

    @Test
    fun `everything merges when the threshold is loose`() {
        val result = ConstrainedClusterer(threshold = 1.9f)
            .cluster(listOf(v(0.0), v(45.0), v(90.0)), emptySet())
        assertEquals(1, result.size)
    }

    @Test
    fun `nothing merges when the threshold is tight`() {
        val result = ConstrainedClusterer(threshold = 0.0001f)
            .cluster(listOf(v(0.0), v(10.0), v(20.0)), emptySet())
        assertEquals(3, result.size)
    }

    @Test
    fun `a cannot link pair is never merged`() {
        // Identical vectors, but they were on screen at the same time.
        val result = ConstrainedClusterer(threshold = 0.5f)
            .cluster(listOf(v(0.0), v(0.0)), setOf(0 to 1))
        assertEquals(2, result.size)
    }

    @Test
    fun `cannot link propagates through merges`() {
        // A, B, C are all identical. A cannot link with C.
        // A+B may merge, but C must then stay out of that cluster.
        val result = ConstrainedClusterer(threshold = 0.5f)
            .cluster(listOf(v(0.0), v(0.0), v(0.0)), setOf(0 to 2))
        assertEquals(2, result.size)
        assertTrue(2 !in clusterOf(result, 0))
    }

    @Test
    fun `two people alternating on screen stay separate`() {
        // The Sample 1 shape: person A and person B each appear twice, sharing the
        // frame once. Their embeddings are close-ish but the overlap forbids merging.
        val vectors = listOf(v(0.0), v(30.0), v(3.0), v(33.0))  // A1, B1, A2, B2
        val cannotLink = setOf(0 to 1, 2 to 3)                  // A1|B1 and A2|B2 co-occur
        val result = ConstrainedClusterer(threshold = 0.15f).cluster(vectors, cannotLink)
        assertEquals(2, result.size)
        assertEquals(listOf(0, 2), clusterOf(result, 0))
        assertEquals(listOf(1, 3), clusterOf(result, 1))
    }

    @Test
    fun `average linkage is used rather than single linkage`() {
        // A chain 0 - 1 - 2 where neighbours are close but the ends are far apart.
        // Single linkage would merge all three; average linkage must not.
        val result = ConstrainedClusterer(threshold = 0.02f)
            .cluster(listOf(v(0.0), v(10.0), v(20.0)), emptySet())
        assertTrue(result.size >= 2)
    }

    @Test
    fun `empty input yields no clusters`() {
        assertEquals(0, ConstrainedClusterer(0.5f).cluster(emptyList(), emptySet()).size)
    }

    @Test
    fun `a single vector yields one cluster`() {
        assertEquals(listOf(listOf(0)), ConstrainedClusterer(0.5f).cluster(listOf(v(0.0)), emptySet()))
    }
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.cluster.ConstrainedClustererTest"
```

Expected: `Unresolved reference: ConstrainedClusterer`.

- [ ] **Step 3: Implement the distance helpers**

`app/src/main/java/com/iykyk/collage/core/cluster/CosineDistance.kt`:

```kotlin
package com.iykyk.collage.core.cluster

import kotlin.math.sqrt

/** Returns a copy scaled to unit length. A zero vector is returned unchanged. */
fun FloatArray.l2Normalized(): FloatArray {
    var sumSq = 0f
    for (x in this) sumSq += x * x
    val norm = sqrt(sumSq)
    if (norm <= 1e-12f) return copyOf()
    return FloatArray(size) { this[it] / norm }
}

/**
 * Cosine distance in [0, 2]. Both vectors are assumed L2-normalised, which is what
 * [FaceEmbedder] guarantees, so this reduces to `1 - dot`.
 */
fun cosineDistance(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "embedding dimensions differ: ${a.size} vs ${b.size}" }
    var dot = 0f
    for (i in a.indices) dot += a[i] * b[i]
    return 1f - dot
}
```

- [ ] **Step 4: Implement the clusterer**

`app/src/main/java/com/iykyk/collage/core/cluster/ConstrainedClusterer.kt`:

```kotlin
package com.iykyk.collage.core.cluster

/**
 * Average-linkage agglomerative clustering over embeddings, with hard cannot-link
 * constraints.
 *
 * The constraint is what makes co-occurring people resolve correctly: two tracklets
 * that are clearly visible at the same moment are, by definition, different people, so
 * no similarity score is allowed to merge them. Constraints propagate — once a cluster
 * contains a member that cannot link with x, the whole cluster cannot absorb x.
 *
 * Input sizes here are tens of tracklets, so the straightforward O(n^3) loop is far
 * cheaper than the bookkeeping needed to avoid it.
 */
class ConstrainedClusterer(private val threshold: Float) {

    /**
     * @param vectors one L2-normalised embedding per item.
     * @param cannotLink unordered index pairs that must never share a cluster.
     * @return member indices per cluster, each sorted ascending, ordered by first member.
     */
    fun cluster(vectors: List<FloatArray>, cannotLink: Set<Pair<Int, Int>>): List<List<Int>> {
        if (vectors.isEmpty()) return emptyList()

        val forbidden = HashSet<Long>(cannotLink.size * 2)
        for ((a, b) in cannotLink) {
            forbidden += key(a, b)
            forbidden += key(b, a)
        }

        val clusters = vectors.indices.map { mutableListOf(it) }.toMutableList()

        while (clusters.size > 1) {
            var bestI = -1
            var bestJ = -1
            var bestDistance = Float.MAX_VALUE

            for (i in 0 until clusters.size - 1) {
                for (j in i + 1 until clusters.size) {
                    if (isForbidden(clusters[i], clusters[j], forbidden)) continue
                    val d = averageLinkage(clusters[i], clusters[j], vectors)
                    if (d < bestDistance) {
                        bestDistance = d
                        bestI = i
                        bestJ = j
                    }
                }
            }

            if (bestI < 0 || bestDistance > threshold) break

            clusters[bestI].addAll(clusters[bestJ])
            clusters.removeAt(bestJ)
        }

        return clusters
            .map { it.sorted() }
            .sortedBy { it.first() }
    }

    private fun isForbidden(a: List<Int>, b: List<Int>, forbidden: Set<Long>): Boolean {
        for (x in a) for (y in b) if (key(x, y) in forbidden) return true
        return false
    }

    private fun averageLinkage(a: List<Int>, b: List<Int>, vectors: List<FloatArray>): Float {
        var total = 0f
        for (x in a) for (y in b) total += cosineDistance(vectors[x], vectors[y])
        return total / (a.size * b.size)
    }

    private fun key(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
}
```

- [ ] **Step 5: Run the test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.cluster.ConstrainedClustererTest"
```

Expected: PASS, 11 tests.

- [ ] **Step 6: Run the whole unit-test suite**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest
```

Expected: all tests from Tasks 2, 2b, 3 and 4 pass together.

- [ ] **Step 7: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/core app/src/test/java/com/iykyk/collage/core
git commit -m "feat(core): add constrained average-linkage clustering for identity grouping"
```

---

## Task 5: NV21 downsampling and rotation mapping (pure JVM)

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/core/frame/Nv21.kt`
- Create: `app/src/main/java/com/iykyk/collage/core/frame/RotationMapper.kt`
- Test: `app/src/test/java/com/iykyk/collage/core/frame/Nv21Test.kt`
- Test: `app/src/test/java/com/iykyk/collage/core/frame/RotationMapperTest.kt`

**Interfaces:**
- Consumes: `BoxF`
- Produces:
  - `data class PlaneData(buffer: ByteArray, rowStride: Int, pixelStride: Int)`
  - `downsampleToNv21(y, u, v: PlaneData, srcWidth, srcHeight, step: Int): Nv21Buffer`
  - `data class Nv21Buffer(bytes: ByteArray, width: Int, height: Int)` with `lumaRowStride = width`
  - `chooseStep(srcWidth, srcHeight, targetLongEdge): Int`
  - `RotationMapper.uprightSize(w, h, rotationDeg): Pair<Int, Int>`
  - `RotationMapper.uprightToRaw(box: BoxF, rawW, rawH, rotationDeg): BoxF`

**Why this exists:** ML Kit at full 1080p costs 60–100 ms per frame, which is too slow for 240 frames. Subsampling the YUV planes by an integer step is cheap (it touches only 1/step² of the pixels) and brings detection down to 25–40 ms per frame. The rotation mapper exists because ML Kit reports boxes in *upright* coordinates while our pixel buffers are in *raw* coordinates; cropping needs the inverse mapping.

- [ ] **Step 1: Write the failing NV21 test**

`app/src/test/java/com/iykyk/collage/core/frame/Nv21Test.kt`:

```kotlin
package com.iykyk.collage.core.frame

import org.junit.Assert.assertEquals
import org.junit.Test

class Nv21Test {

    /** A y-plane where every pixel encodes its own x coordinate, for easy assertions. */
    private fun yPlane(w: Int, h: Int, stride: Int = w) = PlaneData(
        buffer = ByteArray(stride * h) { i -> ((i % stride) % 256).toByte() },
        rowStride = stride,
        pixelStride = 1,
    )

    private fun uvPlane(w: Int, h: Int, fill: Int) = PlaneData(
        buffer = ByteArray((w / 2) * (h / 2)) { fill.toByte() },
        rowStride = w / 2,
        pixelStride = 1,
    )

    @Test
    fun `chooseStep picks the smallest step that fits the target`() {
        assertEquals(1, chooseStep(540, 960, targetLongEdge = 960))
        assertEquals(2, chooseStep(1080, 1920, targetLongEdge = 960))
        assertEquals(2, chooseStep(1080, 1920, targetLongEdge = 540))  // step 2 -> 960 long edge
        assertEquals(4, chooseStep(2160, 3840, targetLongEdge = 960))
        assertEquals(1, chooseStep(320, 240, targetLongEdge = 960))    // never upscales
    }

    @Test
    fun `step of one preserves dimensions and luma`() {
        val out = downsampleToNv21(yPlane(8, 4), uvPlane(8, 4, 128), uvPlane(8, 4, 200), 8, 4, step = 1)
        assertEquals(8, out.width)
        assertEquals(4, out.height)
        assertEquals(8 * 4 + 8 * 4 / 2, out.bytes.size)
        // First row of luma is 0,1,2,...,7
        for (x in 0 until 8) assertEquals(x, out.bytes[x].toInt() and 0xFF)
    }

    @Test
    fun `step of two halves the dimensions and samples every other pixel`() {
        val out = downsampleToNv21(yPlane(8, 4), uvPlane(8, 4, 128), uvPlane(8, 4, 200), 8, 4, step = 2)
        assertEquals(4, out.width)
        assertEquals(2, out.height)
        // Luma row is now 0,2,4,6
        for (i in 0 until 4) assertEquals(i * 2, out.bytes[i].toInt() and 0xFF)
    }

    @Test
    fun `output size matches the nv21 contract`() {
        val out = downsampleToNv21(yPlane(1080, 1920), uvPlane(1080, 1920, 128), uvPlane(1080, 1920, 128), 1080, 1920, step = 2)
        assertEquals(540, out.width)
        assertEquals(960, out.height)
        assertEquals(540 * 960 * 3 / 2, out.bytes.size)
    }

    @Test
    fun `chroma is interleaved as v then u`() {
        val out = downsampleToNv21(yPlane(4, 4), uvPlane(4, 4, 111), uvPlane(4, 4, 222), 4, 4, step = 1)
        val chromaStart = 4 * 4
        assertEquals(222, out.bytes[chromaStart].toInt() and 0xFF)      // V first
        assertEquals(111, out.bytes[chromaStart + 1].toInt() and 0xFF)  // then U
    }

    @Test
    fun `row stride padding is respected`() {
        // A plane padded to stride 16 for a 8-wide image must still read the right pixels.
        val out = downsampleToNv21(yPlane(8, 4, stride = 16), uvPlane(8, 4, 128), uvPlane(8, 4, 128), 8, 4, step = 1)
        for (x in 0 until 8) assertEquals(x, out.bytes[x].toInt() and 0xFF)
    }
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.frame.Nv21Test"
```

Expected: `Unresolved reference: downsampleToNv21`.

- [ ] **Step 3: Implement the downsampler**

`app/src/main/java/com/iykyk/collage/core/frame/Nv21.kt`:

```kotlin
package com.iykyk.collage.core.frame

/** One plane of a YUV_420_888 image, copied out of its (possibly padded) direct buffer. */
data class PlaneData(
    val buffer: ByteArray,
    val rowStride: Int,
    val pixelStride: Int,
) {
    override fun equals(other: Any?): Boolean =
        other is PlaneData && buffer.contentEquals(other.buffer) &&
            rowStride == other.rowStride && pixelStride == other.pixelStride

    override fun hashCode(): Int =
        (buffer.contentHashCode() * 31 + rowStride) * 31 + pixelStride
}

/** A tightly packed NV21 image: full-resolution luma followed by interleaved V,U. */
data class Nv21Buffer(val bytes: ByteArray, val width: Int, val height: Int) {
    /** Luma is packed, so the row stride is simply the width. */
    val lumaRowStride: Int get() = width

    override fun equals(other: Any?): Boolean =
        other is Nv21Buffer && bytes.contentEquals(other.bytes) &&
            width == other.width && height == other.height

    override fun hashCode(): Int = (bytes.contentHashCode() * 31 + width) * 31 + height
}

/**
 * Smallest integer subsampling step whose result still covers [targetLongEdge].
 *
 * Integer stepping keeps the inner loop to a pointer add, and never upsamples.
 */
fun chooseStep(srcWidth: Int, srcHeight: Int, targetLongEdge: Int): Int {
    val longEdge = maxOf(srcWidth, srcHeight)
    var step = 1
    while (longEdge / (step * 2) >= targetLongEdge && step < MAX_STEP) step *= 2
    return step
}

/**
 * Nearest-neighbour subsample of a YUV_420_888 image into a packed NV21 buffer.
 *
 * NV21 is what `InputImage.fromByteBuffer` wants, and it keeps the luma plane contiguous
 * so sharpness can be measured on it directly with no colour conversion.
 *
 * Output dimensions are forced even, as the 4:2:0 chroma layout requires.
 */
fun downsampleToNv21(
    y: PlaneData,
    u: PlaneData,
    v: PlaneData,
    srcWidth: Int,
    srcHeight: Int,
    step: Int,
): Nv21Buffer {
    require(step >= 1) { "step must be positive" }
    val outWidth = (srcWidth / step) and 1.inv()
    val outHeight = (srcHeight / step) and 1.inv()
    require(outWidth >= 2 && outHeight >= 2) { "downsampled frame is degenerate" }

    val lumaSize = outWidth * outHeight
    val out = ByteArray(lumaSize * 3 / 2)

    // Luma.
    var o = 0
    for (row in 0 until outHeight) {
        val srcRow = row * step * y.rowStride
        var srcCol = 0
        for (@Suppress("UNUSED_PARAMETER") col in 0 until outWidth) {
            out[o++] = y.buffer[srcRow + srcCol]
            srcCol += step * y.pixelStride
        }
    }

    // Chroma: one V,U pair per 2x2 luma block, so the source is subsampled by step as well.
    val chromaWidth = outWidth / 2
    val chromaHeight = outHeight / 2
    for (row in 0 until chromaHeight) {
        val srcRow = row * step * u.rowStride
        for (col in 0 until chromaWidth) {
            val srcCol = col * step * u.pixelStride
            val vi = row * step * v.rowStride + col * step * v.pixelStride
            out[o++] = v.buffer.getOrElse(vi) { 128 }
            out[o++] = u.buffer.getOrElse(srcRow + srcCol) { 128 }
        }
    }

    return Nv21Buffer(out, outWidth, outHeight)
}

private const val MAX_STEP = 8
```

- [ ] **Step 4: Run the NV21 test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.frame.Nv21Test"
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Write the failing rotation-mapper test**

`app/src/test/java/com/iykyk/collage/core/frame/RotationMapperTest.kt`:

```kotlin
package com.iykyk.collage.core.frame

import com.iykyk.collage.core.model.BoxF
import org.junit.Assert.assertEquals
import org.junit.Test

class RotationMapperTest {

    @Test
    fun `zero and one eighty keep the frame dimensions`() {
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1080, 1920, 0))
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1080, 1920, 180))
    }

    @Test
    fun `ninety and two seventy swap the frame dimensions`() {
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1920, 1080, 90))
        assertEquals(1080 to 1920, RotationMapper.uprightSize(1920, 1080, 270))
    }

    @Test
    fun `zero rotation is the identity`() {
        val box = BoxF(10f, 20f, 30f, 40f)
        assertEquals(box, RotationMapper.uprightToRaw(box, rawW = 100, rawH = 200, rotationDeg = 0))
    }

    @Test
    fun `ninety degrees maps the upright top left to the raw bottom left`() {
        // Raw 200x100 rotated 90 CW gives an upright 100x200 frame.
        // Upright (0,0)-(10,20) comes from raw (0, 90)-(20, 100).
        val mapped = RotationMapper.uprightToRaw(BoxF(0f, 0f, 10f, 20f), rawW = 200, rawH = 100, rotationDeg = 90)
        assertEquals(BoxF(0f, 90f, 20f, 100f), mapped)
    }

    @Test
    fun `one eighty flips both axes`() {
        val mapped = RotationMapper.uprightToRaw(BoxF(10f, 20f, 30f, 40f), rawW = 100, rawH = 200, rotationDeg = 180)
        assertEquals(BoxF(70f, 160f, 90f, 180f), mapped)
    }

    @Test
    fun `two seventy is the inverse of ninety`() {
        val mapped = RotationMapper.uprightToRaw(BoxF(0f, 0f, 10f, 20f), rawW = 200, rawH = 100, rotationDeg = 270)
        assertEquals(BoxF(180f, 0f, 200f, 10f), mapped)
    }

    @Test
    fun `mapping a full frame box covers the whole raw frame`() {
        for (deg in listOf(0, 90, 180, 270)) {
            val (uw, uh) = RotationMapper.uprightSize(200, 100, deg)
            val mapped = RotationMapper.uprightToRaw(
                BoxF(0f, 0f, uw.toFloat(), uh.toFloat()), rawW = 200, rawH = 100, rotationDeg = deg
            )
            assertEquals("rotation $deg", BoxF(0f, 0f, 200f, 100f), mapped)
        }
    }
}
```

- [ ] **Step 6: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.frame.RotationMapperTest"
```

Expected: `Unresolved reference: RotationMapper`.

- [ ] **Step 7: Implement the rotation mapper**

`app/src/main/java/com/iykyk/collage/core/frame/RotationMapper.kt`:

```kotlin
package com.iykyk.collage.core.frame

import com.iykyk.collage.core.model.BoxF
import kotlin.math.max
import kotlin.math.min

/**
 * Converts between the *raw* pixel space of a decoded frame and the *upright* space a
 * viewer sees after the container's rotation metadata is applied.
 *
 * ML Kit reports face boxes in upright coordinates, but crops have to be taken from the
 * raw buffer, so every crop needs this inverse mapping.
 */
object RotationMapper {

    /** Dimensions of the frame after [rotationDeg] of clockwise rotation is applied. */
    fun uprightSize(rawW: Int, rawH: Int, rotationDeg: Int): Pair<Int, Int> =
        when (normalise(rotationDeg)) {
            90, 270 -> rawH to rawW
            else -> rawW to rawH
        }

    /**
     * Maps a box given in upright coordinates back into raw buffer coordinates.
     *
     * [rawW] and [rawH] are the dimensions of the undecoded frame, before rotation.
     */
    fun uprightToRaw(box: BoxF, rawW: Int, rawH: Int, rotationDeg: Int): BoxF {
        val corners = when (normalise(rotationDeg)) {
            0 -> listOf(box.left to box.top, box.right to box.bottom)

            // Upright (x, y) came from raw (y, rawH - x).
            90 -> listOf(
                box.top to (rawH - box.left),
                box.bottom to (rawH - box.right),
            )

            180 -> listOf(
                (rawW - box.left) to (rawH - box.top),
                (rawW - box.right) to (rawH - box.bottom),
            )

            // Upright (x, y) came from raw (rawW - y, x).
            else -> listOf(
                (rawW - box.top) to box.left,
                (rawW - box.bottom) to box.right,
            )
        }
        val (x0, y0) = corners[0]
        val (x1, y1) = corners[1]
        return BoxF(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    }

    private fun normalise(deg: Int): Int = ((deg % 360) + 360) % 360
}
```

- [ ] **Step 8: Run the rotation test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.frame.RotationMapperTest"
```

Expected: PASS, 7 tests.

- [ ] **Step 9: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/core/frame app/src/test/java/com/iykyk/collage/core/frame
git commit -m "feat(core): add NV21 subsampling and rotation coordinate mapping"
```

---

## Task 6: MediaCodec frame source

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/video/FrameSource.kt`
- Create: `app/src/main/java/com/iykyk/collage/video/MediaCodecFrameSource.kt`
- Test: `app/src/androidTest/java/com/iykyk/collage/video/MediaCodecFrameSourceTest.kt`
- Create: `app/src/main/java/com/iykyk/collage/samples/SampleVideos.kt`

**Interfaces:**
- Consumes: `Nv21Buffer`, `PlaneData`, `downsampleToNv21`, `chooseStep`, `PipelineConfig`
- Produces:
  - `data class AnalysisFrame(index: Int, timestampMs: Long, nv21: Nv21Buffer, rotationDegrees: Int)`
  - `interface FrameSource { val durationMs: Long; fun frames(): Flow<AnalysisFrame> }`
  - `MediaCodecFrameSource(context, uri, config) : FrameSource`
  - `SampleVideos.ensureExtracted(context): List<SampleVideo>` where `SampleVideo(label: String, uri: Uri)`

**Note on ordering:** this task's verification is an on-device instrumented test, so it needs the physical device connected. Everything before this point runs on the JVM.

- [ ] **Step 1: Define the frame-source contract**

`app/src/main/java/com/iykyk/collage/video/FrameSource.kt`:

```kotlin
package com.iykyk.collage.video

import com.iykyk.collage.core.frame.Nv21Buffer
import kotlinx.coroutines.flow.Flow

/**
 * One sampled frame ready for analysis.
 *
 * [nv21] is in *raw* orientation; [rotationDegrees] is what must be applied to make it
 * upright. ML Kit is given both, so it reports boxes in upright coordinates.
 */
data class AnalysisFrame(
    val index: Int,
    val timestampMs: Long,
    val nv21: Nv21Buffer,
    val rotationDegrees: Int,
)

/**
 * Supplies sampled frames of a video in presentation order.
 *
 * Implementations must be cold: collecting [frames] starts a fresh decode, and
 * cancelling the collector releases the codec.
 */
interface FrameSource {
    /** Total clip duration, used to drive determinate progress. */
    val durationMs: Long

    fun frames(): Flow<AnalysisFrame>
}
```

- [ ] **Step 2: Write the failing instrumented test**

`app/src/androidTest/java/com/iykyk/collage/video/MediaCodecFrameSourceTest.kt`:

```kotlin
package com.iykyk.collage.video

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.samples.SampleVideos
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MediaCodecFrameSourceTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    private fun sampleSource(): MediaCodecFrameSource {
        val sample = SampleVideos.ensureExtracted(context).first()
        return MediaCodecFrameSource(context, sample.uri, config)
    }

    @Test
    fun reportsDurationOfRoughlyThirtySeconds() {
        val duration = sampleSource().durationMs
        assertTrue("duration was $duration", duration in 25_000..35_000)
    }

    @Test
    fun emitsRoughlyTheConfiguredNumberOfFrames() = runBlocking {
        val frames = sampleSource().frames().toList()
        val expected = (30 * config.sampleFps)
        assertTrue(
            "expected about $expected frames, got ${frames.size}",
            frames.size in (expected * 0.7).toInt()..(expected * 1.3).toInt()
        )
    }

    @Test
    fun timestampsIncreaseMonotonically() = runBlocking {
        val stamps = sampleSource().frames().toList().map { it.timestampMs }
        assertEquals(stamps.sorted(), stamps)
        assertEquals(stamps.distinct().size, stamps.size)
    }

    @Test
    fun downsampledFramesRespectTheAnalysisTarget() = runBlocking {
        val frame = sampleSource().frames().take(1).toList().single()
        val longEdge = maxOf(frame.nv21.width, frame.nv21.height)
        assertTrue("long edge was $longEdge", longEdge <= config.analysisLongEdge * 2)
        assertEquals(frame.nv21.width * frame.nv21.height * 3 / 2, frame.nv21.bytes.size)
    }

    @Test
    fun portraitClipReportsAnUprightPortraitFrame() = runBlocking {
        val frame = sampleSource().frames().take(1).toList().single()
        val (w, h) = com.iykyk.collage.core.frame.RotationMapper.uprightSize(
            frame.nv21.width, frame.nv21.height, frame.rotationDegrees
        )
        assertTrue("upright frame $w x $h is not portrait", h > w)
    }

    @Test
    fun cancellingEarlyStopsDecoding() = runBlocking {
        // take(3) cancels the flow; this must return promptly rather than decode the clip.
        val frames = sampleSource().frames().take(3).toList()
        assertEquals(3, frames.size)
    }
}
```

- [ ] **Step 3: Implement sample extraction**

`app/src/main/java/com/iykyk/collage/samples/SampleVideos.kt`:

```kotlin
package com.iykyk.collage.samples

import android.content.Context
import android.net.Uri
import androidx.core.net.toUri
import java.io.File

data class SampleVideo(val label: String, val uri: Uri)

/**
 * Copies the bundled sample clips out of assets on first use.
 *
 * These are a convenience for demoing; they are handed to the pipeline as ordinary
 * content URIs and travel the identical code path as a video the user picks.
 */
object SampleVideos {

    private val ASSET_NAMES = listOf(
        "Sample 1" to "samples/sample1.mp4",
        "Sample 2" to "samples/sample2.mp4",
        "Sample 3" to "samples/sample3.mp4",
    )

    fun ensureExtracted(context: Context): List<SampleVideo> {
        val dir = File(context.cacheDir, "samples").apply { mkdirs() }
        return ASSET_NAMES.map { (label, assetPath) ->
            val out = File(dir, assetPath.substringAfterLast('/'))
            if (!out.exists() || out.length() == 0L) {
                context.assets.open(assetPath).use { input ->
                    out.outputStream().use { output -> input.copyTo(output) }
                }
            }
            SampleVideo(label, out.toUri())
        }
    }
}
```

- [ ] **Step 4: Implement the MediaCodec frame source**

`app/src/main/java/com/iykyk/collage/video/MediaCodecFrameSource.kt`:

```kotlin
package com.iykyk.collage.video

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.frame.PlaneData
import com.iykyk.collage.core.frame.chooseStep
import com.iykyk.collage.core.frame.downsampleToNv21
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.Dispatchers
import android.media.ImageReader
import android.graphics.ImageFormat
import java.nio.ByteBuffer

/**
 * Decodes a video once, front to back, emitting one downsampled NV21 frame per sampling
 * interval.
 *
 * Sequential decode rather than seeking: `MediaMetadataRetriever.getFrameAtTime` costs
 * 50-150 ms per seek, so 240 samples would take 15-30 s. Walking the stream once costs a
 * few seconds and keeps the progress bar honest.
 */
class MediaCodecFrameSource(
    private val context: Context,
    private val uri: Uri,
    private val config: PipelineConfig,
) : FrameSource {

    override val durationMs: Long by lazy {
        MediaMetadataRetriever().use { mmr ->
            mmr.setDataSource(context, uri)
            mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        }
    }

    override fun frames(): Flow<AnalysisFrame> = callbackFlow {
        val extractor = MediaExtractor().apply { setDataSource(context, uri) }
        val trackIndex = (0 until extractor.trackCount).firstOrNull { i ->
            extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
        } ?: run {
            close(IllegalArgumentException("no video track in $uri"))
            extractor.release()
            return@callbackFlow
        }

        extractor.selectTrack(trackIndex)
        val format = extractor.getTrackFormat(trackIndex)
        val rawWidth = format.getInteger(MediaFormat.KEY_WIDTH)
        val rawHeight = format.getInteger(MediaFormat.KEY_HEIGHT)
        val rotation = if (format.containsKey(KEY_ROTATION)) format.getInteger(KEY_ROTATION) else 0
        val step = chooseStep(rawWidth, rawHeight, config.analysisLongEdge)
        val sampleIntervalUs = 1_000_000L / config.sampleFps

        val reader = ImageReader.newInstance(rawWidth, rawHeight, ImageFormat.YUV_420_888, MAX_IMAGES)
        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, reader.surface, null, 0)
        codec.start()

        var emitted = 0
        var nextSampleUs = 0L
        var sawEos = false
        val bufferInfo = MediaCodec.BufferInfo()

        try {
            while (!sawEos && !isClosedForSend) {
                // Feed the decoder.
                val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                if (inputIndex >= 0) {
                    val input = codec.getInputBuffer(inputIndex)!!
                    val size = extractor.readSampleData(input, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(
                            inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                        )
                    } else {
                        codec.queueInputBuffer(inputIndex, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }

                // Drain it.
                when (val outputIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER,
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED,
                    MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit

                    else -> {
                        if (outputIndex >= 0) {
                            val presentationUs = bufferInfo.presentationTimeUs
                            val wanted = presentationUs >= nextSampleUs
                            // Only frames we actually want are rendered to the reader.
                            codec.releaseOutputBuffer(outputIndex, wanted)

                            if (wanted) {
                                nextSampleUs = presentationUs + sampleIntervalUs
                                reader.acquireNextImage()?.use { image ->
                                    val nv21 = downsampleToNv21(
                                        y = image.planes[0].toPlaneData(),
                                        u = image.planes[1].toPlaneData(),
                                        v = image.planes[2].toPlaneData(),
                                        srcWidth = image.width,
                                        srcHeight = image.height,
                                        step = step,
                                    )
                                    trySend(
                                        AnalysisFrame(
                                            index = emitted++,
                                            timestampMs = presentationUs / 1000,
                                            nv21 = nv21,
                                            rotationDegrees = rotation,
                                        )
                                    )
                                }
                            }
                        }
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            sawEos = true
                        }
                    }
                }
            }
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            runCatching { reader.close() }
            runCatching { extractor.release() }
        }
        close()
        awaitClose { }
    }.buffer(FRAME_BUFFER).flowOn(Dispatchers.IO)

    /** Copies a plane out of its direct buffer so it survives `Image.close()`. */
    private fun android.media.Image.Plane.toPlaneData(): PlaneData {
        val source: ByteBuffer = buffer
        val bytes = ByteArray(source.remaining())
        source.get(bytes)
        return PlaneData(bytes, rowStride, pixelStride)
    }

    private companion object {
        const val TIMEOUT_US = 10_000L
        const val MAX_IMAGES = 3
        const val FRAME_BUFFER = 4
        const val KEY_ROTATION = "rotation-degrees"
    }
}
```

- [ ] **Step 5: Run the instrumented test on the device**

Connect the physical device with USB debugging on, then:

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.video.MediaCodecFrameSourceTest"
```

Expected: PASS, 6 tests. If `acquireNextImage` returns null on some frames the reader is
starved — raise `MAX_IMAGES` to 4 before changing anything else.

- [ ] **Step 6: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/video app/src/main/java/com/iykyk/collage/samples app/src/androidTest
git commit -m "feat(video): sequential MediaCodec frame source with NV21 downsampling"
```

---

## Task 7: ML Kit face detection and crop extraction

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/detect/MlKitFaceDetector.kt`
- Create: `app/src/main/java/com/iykyk/collage/detect/CropExtractor.kt`
- Test: `app/src/androidTest/java/com/iykyk/collage/detect/MlKitFaceDetectorTest.kt`

**Interfaces:**
- Consumes: `AnalysisFrame`, `FaceObservation`, `BoxF`, `PointF2`, `RotationMapper`, `laplacianVarianceFromLuma`, `PipelineConfig`
- Produces:
  - `MlKitFaceDetector(config)` with `suspend fun detect(frame: AnalysisFrame): List<FaceObservation>` and `fun close()`
  - `CropExtractor.faceCrop(frame: AnalysisFrame, obs: FaceObservation, expandFactor: Float): Bitmap` — a generous, upright RGB crop around the face

**Key detail:** sharpness is measured on the luma plane over the face region *in raw coordinates*, which is why `RotationMapper` exists. No colour conversion is needed for the quality pass at all.

- [ ] **Step 1: Write the failing instrumented test**

`app/src/androidTest/java/com/iykyk/collage/detect/MlKitFaceDetectorTest.kt`:

```kotlin
package com.iykyk.collage.detect

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.samples.SampleVideos
import com.iykyk.collage.video.AnalysisFrame
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MlKitFaceDetectorTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    private fun firstFrames(n: Int): List<AnalysisFrame> = runBlocking {
        val sample = SampleVideos.ensureExtracted(context).first()
        MediaCodecFrameSource(context, sample.uri, config).frames().take(n).toList()
    }

    @Test
    fun detectsAtLeastOneFaceSomewhereInTheOpeningSeconds() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            val total = firstFrames(24).sumOf { detector.detect(it).size }
            assertTrue("no faces found in the first 3 seconds", total > 0)
        } finally {
            detector.close()
        }
    }

    @Test
    fun observationsAreInUprightCoordinatesAndInsideTheFrame() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            val observations = firstFrames(24).flatMap { detector.detect(it) }
            assertTrue(observations.isNotEmpty())
            observations.forEach { o ->
                assertTrue("frame must be portrait upright", o.frameHeight > o.frameWidth)
                assertTrue(o.box.centerX in 0f..o.frameWidth.toFloat())
                assertTrue(o.box.centerY in 0f..o.frameHeight.toFloat())
            }
        } finally {
            detector.close()
        }
    }

    @Test
    fun sharpnessIsPopulatedAndVaries() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            val values = firstFrames(40).flatMap { detector.detect(it) }.map { it.rawSharpness }
            assertTrue(values.isNotEmpty())
            assertTrue("sharpness never populated", values.any { it > 0f })
            assertTrue("sharpness is constant, region mapping is probably wrong",
                values.distinct().size > 1)
        } finally {
            detector.close()
        }
    }

    @Test
    fun cropIsGenerousRatherThanTightToTheBox() = runBlocking {
        val detector = MlKitFaceDetector(config)
        try {
            val frames = firstFrames(24)
            val (frame, obs) = frames.firstNotNullOf { f ->
                detector.detect(f).firstOrNull()?.let { f to it }
            }
            val crop = CropExtractor.faceCrop(frame, obs, expandFactor = 2.2f)
            assertTrue("crop is empty", crop.width > 0 && crop.height > 0)
            assertTrue(
                "crop ${crop.width}px is not wider than the ${obs.box.width}px face box",
                crop.width > obs.box.width * 1.5f
            )
        } finally {
            detector.close()
        }
    }
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.detect.MlKitFaceDetectorTest"
```

Expected: compilation failure — `Unresolved reference: MlKitFaceDetector`.

- [ ] **Step 3: Implement the detector**

`app/src/main/java/com/iykyk/collage/detect/MlKitFaceDetector.kt`:

```kotlin
package com.iykyk.collage.detect

import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.frame.RotationMapper
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.PointF2
import com.iykyk.collage.core.quality.laplacianVarianceFromLuma
import com.iykyk.collage.video.AnalysisFrame
import kotlinx.coroutines.tasks.await
import java.nio.ByteBuffer

/**
 * Wraps ML Kit face detection and turns each detected face into a plain
 * [FaceObservation] in upright coordinates.
 *
 * Tracking is enabled so the tracklet builder gets an extra association hint, but the
 * builder never depends on it: ML Kit reissues ids mid-appearance, so IoU association is
 * the real mechanism and the id is only a tie-breaker.
 */
class MlKitFaceDetector(private val config: PipelineConfig) {

    private val detector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(MIN_FACE_SIZE)
            .enableTracking()
            .build()
    )

    suspend fun detect(frame: AnalysisFrame): List<FaceObservation> {
        val image = InputImage.fromByteBuffer(
            ByteBuffer.wrap(frame.nv21.bytes),
            frame.nv21.width,
            frame.nv21.height,
            frame.rotationDegrees,
            InputImage.IMAGE_FORMAT_NV21,
        )
        val faces = detector.process(image).await()
        if (faces.isEmpty()) return emptyList()

        val (uprightW, uprightH) = RotationMapper.uprightSize(
            frame.nv21.width, frame.nv21.height, frame.rotationDegrees
        )

        return faces.map { face -> face.toObservation(frame, uprightW, uprightH) }
    }

    private fun Face.toObservation(
        frame: AnalysisFrame,
        uprightW: Int,
        uprightH: Int,
    ): FaceObservation {
        val box = BoxF(
            boundingBox.left.toFloat(),
            boundingBox.top.toFloat(),
            boundingBox.right.toFloat(),
            boundingBox.bottom.toFloat(),
        )

        // Sharpness is measured on the luma plane, which is in raw orientation.
        val rawRegion = RotationMapper
            .uprightToRaw(box, frame.nv21.width, frame.nv21.height, frame.rotationDegrees)
            .clampTo(frame.nv21.width, frame.nv21.height)

        return FaceObservation(
            frameIndex = frame.index,
            timestampMs = frame.timestampMs,
            trackingId = trackingId,
            box = box.clampTo(uprightW, uprightH),
            leftEye = getLandmark(FaceLandmark.LEFT_EYE)?.position?.let { PointF2(it.x, it.y) },
            rightEye = getLandmark(FaceLandmark.RIGHT_EYE)?.position?.let { PointF2(it.x, it.y) },
            headEulerX = headEulerAngleX,
            headEulerY = headEulerAngleY,
            headEulerZ = headEulerAngleZ,
            leftEyeOpenProb = leftEyeOpenProbability,
            rightEyeOpenProb = rightEyeOpenProbability,
            smileProb = smilingProbability,
            rawSharpness = laplacianVarianceFromLuma(
                luma = frame.nv21.bytes,
                rowStride = frame.nv21.lumaRowStride,
                region = rawRegion,
            ),
            frameWidth = uprightW,
            frameHeight = uprightH,
        )
    }

    fun close() = detector.close()

    private companion object {
        /** Faces smaller than this fraction of the frame are background, not subjects. */
        const val MIN_FACE_SIZE = 0.08f
    }
}
```

- [ ] **Step 4: Implement crop extraction**

`app/src/main/java/com/iykyk/collage/detect/CropExtractor.kt`:

```kotlin
package com.iykyk.collage.detect

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import com.iykyk.collage.core.frame.RotationMapper
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.video.AnalysisFrame
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Cuts an upright RGB crop around a detected face.
 *
 * Only the crop region is colour-converted, so this stays cheap even though it runs for
 * every candidate shot. The crop is deliberately generous — the assignment calls out
 * tight bounding-box crops as a source of low-resolution, ugly tiles.
 */
object CropExtractor {

    fun faceCrop(frame: AnalysisFrame, obs: FaceObservation, expandFactor: Float): Bitmap {
        // Square, generous, centred on the face, in upright coordinates.
        val side = max(obs.box.width, obs.box.height) * expandFactor
        val upright = BoxF(
            obs.box.centerX - side / 2f,
            obs.box.centerY - side / 2f,
            obs.box.centerX + side / 2f,
            obs.box.centerY + side / 2f,
        ).clampTo(obs.frameWidth, obs.frameHeight)

        val raw = RotationMapper
            .uprightToRaw(upright, frame.nv21.width, frame.nv21.height, frame.rotationDegrees)
            .clampTo(frame.nv21.width, frame.nv21.height)

        val rect = Rect(
            raw.left.roundToInt(),
            raw.top.roundToInt(),
            raw.right.roundToInt(),
            raw.bottom.roundToInt(),
        )
        require(rect.width() > 1 && rect.height() > 1) { "degenerate crop rect $rect" }

        val yuv = YuvImage(frame.nv21.bytes, ImageFormat.NV21, frame.nv21.width, frame.nv21.height, null)
        val jpeg = ByteArrayOutputStream().also { yuv.compressToJpeg(rect, JPEG_QUALITY, it) }.toByteArray()
        val rawBitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)

        if (frame.rotationDegrees % 360 == 0) return rawBitmap
        val matrix = Matrix().apply { postRotate(frame.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
            .also { if (it !== rawBitmap) rawBitmap.recycle() }
    }

    /** High enough that the embedding is unaffected; the tile itself comes from full res. */
    private const val JPEG_QUALITY = 95
}
```

- [ ] **Step 5: Run the instrumented test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.detect.MlKitFaceDetectorTest"
```

Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/detect app/src/androidTest
git commit -m "feat(detect): ML Kit face detection with luma sharpness and generous crops"
```

---

## Task 8: Face alignment and FaceNet-512 embedding

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/embed/FaceAligner.kt`
- Create: `app/src/main/java/com/iykyk/collage/embed/FaceEmbedder.kt`
- Test: `app/src/androidTest/java/com/iykyk/collage/embed/FaceEmbedderTest.kt`

**Interfaces:**
- Consumes: `FaceObservation`, `PointF2`, `l2Normalized`, `cosineDistance`, `CropExtractor`, `AnalysisFrame`
- Produces:
  - `interface FaceAligner { fun align(crop: Bitmap, obs: FaceObservation, cropOrigin: PointF2, size: Int): Bitmap }`
  - `EyeAligner` (similarity transform from the eye landmarks) and `PlainResizeAligner` (fallback)
  - `FaceEmbedder(context)` with `val inputSize: Int`, `val dimensions: Int`, `fun embed(face: Bitmap): FloatArray` (L2-normalised), `fun embedAveraged(faces: List<Bitmap>): FloatArray`, `fun close()`

**Model facts (verified):** `facenet_512.tflite` is Inception-ResNet-v1 with input tensor `input_1`, shape `[1, 160, 160, 3]` float32, output `[1, 512]`. Input is per-image standardised (prewhitened), not scaled to `[0,1]` — using the wrong normalisation silently destroys accuracy, so it is asserted in the test.

- [ ] **Step 1: Write the failing instrumented test**

`app/src/androidTest/java/com/iykyk/collage/embed/FaceEmbedderTest.kt`:

```kotlin
package com.iykyk.collage.embed

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.cluster.cosineDistance
import com.iykyk.collage.detect.CropExtractor
import com.iykyk.collage.detect.MlKitFaceDetector
import com.iykyk.collage.samples.SampleVideos
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class FaceEmbedderTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val config = PipelineConfig.Default

    @Test
    fun modelReportsTheExpectedShape() {
        FaceEmbedder(context).use { embedder ->
            assertEquals(160, embedder.inputSize)
            assertEquals(512, embedder.dimensions)
        }
    }

    @Test
    fun embeddingsAreUnitLength() {
        FaceEmbedder(context).use { embedder ->
            val bitmap = android.graphics.Bitmap.createBitmap(160, 160, android.graphics.Bitmap.Config.ARGB_8888)
            val v = embedder.embed(bitmap)
            assertEquals(512, v.size)
            assertEquals(1f, v.fold(0f) { a, x -> a + x * x }, 1e-3f)
        }
    }

    /**
     * The real signal test: two crops of the same person, two frames apart, must be much
     * closer than crops of two different people sharing a frame. If this fails, the input
     * normalisation or the alignment is wrong, not the threshold.
     */
    @Test
    fun sameFaceIsCloserThanDifferentFaces() = runBlocking {
        val sample = SampleVideos.ensureExtracted(context).first()
        val frames = MediaCodecFrameSource(context, sample.uri, config).frames().take(200).toList()
        val detector = MlKitFaceDetector(config)
        FaceEmbedder(context).use { embedder ->
            try {
                // A frame containing two distinct faces at once.
                val twoFaceFrame = frames.firstNotNullOfOrNull { f ->
                    val obs = detector.detect(f)
                    if (obs.size >= 2) f to obs else null
                }
                assertTrue("no frame with two simultaneous faces found", twoFaceFrame != null)
                val (frame, observations) = twoFaceFrame!!

                val a1 = embedder.embed(CropExtractor.faceCrop(frame, observations[0], 1.6f))
                val b1 = embedder.embed(CropExtractor.faceCrop(frame, observations[1], 1.6f))

                // The same first face one sampled frame later.
                val nextFrame = frames[frames.indexOf(frame) + 1]
                val nextObs = detector.detect(nextFrame)
                    .minByOrNull { abs(it.box.centerX - observations[0].box.centerX) }!!
                val a2 = embedder.embed(CropExtractor.faceCrop(nextFrame, nextObs, 1.6f))

                val sameDistance = cosineDistance(a1, a2)
                val differentDistance = cosineDistance(a1, b1)
                assertTrue(
                    "same-person distance $sameDistance was not clearly below " +
                        "different-person distance $differentDistance",
                    sameDistance < differentDistance - 0.15f
                )
            } finally {
                detector.close()
            }
        }
    }

    @Test
    fun averagingProducesAUnitVector() {
        FaceEmbedder(context).use { embedder ->
            val bitmaps = List(3) {
                android.graphics.Bitmap.createBitmap(160, 160, android.graphics.Bitmap.Config.ARGB_8888)
            }
            val v = embedder.embedAveraged(bitmaps)
            assertEquals(1f, v.fold(0f) { a, x -> a + x * x }, 1e-3f)
        }
    }
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.embed.FaceEmbedderTest"
```

Expected: `Unresolved reference: FaceEmbedder`.

- [ ] **Step 3: Implement the aligners**

`app/src/main/java/com/iykyk/collage/embed/FaceAligner.kt`:

```kotlin
package com.iykyk.collage.embed

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import com.iykyk.collage.core.model.PointF2

/**
 * Normalises a face crop into the model's square input.
 *
 * Two implementations exist so the tuning harness can measure which actually helps
 * rather than assuming: alignment usually improves embeddings, but it also amplifies
 * landmark error on small faces.
 */
interface FaceAligner {
    /**
     * @param crop a generous face crop, already upright.
     * @param leftEye eye landmark in *crop-local* pixel coordinates, or null.
     * @param rightEye as above.
     * @param size the model's square input edge.
     */
    fun align(crop: Bitmap, leftEye: PointF2?, rightEye: PointF2?, size: Int): Bitmap
}

/** Straight resize. Robust, and the fallback whenever eye landmarks are missing. */
class PlainResizeAligner : FaceAligner {
    override fun align(crop: Bitmap, leftEye: PointF2?, rightEye: PointF2?, size: Int): Bitmap =
        Bitmap.createScaledBitmap(crop, size, size, true)
}

/**
 * Similarity transform placing both eyes at fixed canonical positions, which removes
 * in-plane rotation and scale variation before the model sees the face.
 */
class EyeAligner(private val fallback: FaceAligner = PlainResizeAligner()) : FaceAligner {

    override fun align(crop: Bitmap, leftEye: PointF2?, rightEye: PointF2?, size: Int): Bitmap {
        if (leftEye == null || rightEye == null) return fallback.align(crop, null, null, size)

        val source = floatArrayOf(leftEye.x, leftEye.y, rightEye.x, rightEye.y)
        val target = floatArrayOf(
            LEFT_EYE_X * size, EYE_Y * size,
            RIGHT_EYE_X * size, EYE_Y * size,
        )

        val matrix = Matrix()
        // Two point pairs give a similarity transform: rotation, uniform scale, translation.
        if (!matrix.setPolyToPoly(source, 0, target, 0, 2)) {
            return fallback.align(crop, null, null, size)
        }

        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(crop, matrix, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
        return out
    }

    private companion object {
        // Canonical FaceNet eye placement: eyes on the upper third, symmetric about centre.
        const val LEFT_EYE_X = 0.34f
        const val RIGHT_EYE_X = 0.66f
        const val EYE_Y = 0.38f
    }
}
```

- [ ] **Step 4: Implement the embedder**

`app/src/main/java/com/iykyk/collage/embed/FaceEmbedder.kt`:

```kotlin
package com.iykyk.collage.embed

import android.content.Context
import android.graphics.Bitmap
import com.iykyk.collage.core.cluster.l2Normalized
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.sqrt

/**
 * FaceNet-512 (Inception-ResNet-v1) running on-device through TFLite.
 *
 * Input and output shapes are read from the model at load time rather than hardcoded, so
 * swapping in a different embedding model needs no code change here.
 */
class FaceEmbedder(context: Context, threads: Int = DEFAULT_THREADS) : Closeable {

    private val interpreter: Interpreter = Interpreter(
        loadModel(context),
        Interpreter.Options().apply {
            setNumThreads(threads)
            setUseXNNPACK(true)
        }
    )

    /** Square input edge the model expects, e.g. 160. */
    val inputSize: Int = interpreter.getInputTensor(0).shape()[1]

    /** Embedding dimensionality, e.g. 512. */
    val dimensions: Int = interpreter.getOutputTensor(0).shape()[1]

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(inputSize * inputSize * CHANNELS * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())

    private val pixels = IntArray(inputSize * inputSize)

    /** Returns the L2-normalised embedding of a single aligned face. */
    fun embed(face: Bitmap): FloatArray {
        val scaled = if (face.width == inputSize && face.height == inputSize) {
            face
        } else {
            Bitmap.createScaledBitmap(face, inputSize, inputSize, true)
        }

        writeStandardised(scaled)
        if (scaled !== face) scaled.recycle()

        val output = Array(1) { FloatArray(dimensions) }
        interpreter.run(inputBuffer, output)
        return output[0].l2Normalized()
    }

    /**
     * Mean of several embeddings, renormalised.
     *
     * Averaging a tracklet's best crops is what makes identity robust: a single bad frame
     * can no longer invent a person.
     */
    fun embedAveraged(faces: List<Bitmap>): FloatArray {
        require(faces.isNotEmpty()) { "cannot average zero embeddings" }
        val sum = FloatArray(dimensions)
        for (face in faces) {
            val v = embed(face)
            for (i in sum.indices) sum[i] += v[i]
        }
        return sum.l2Normalized()
    }

    /**
     * FaceNet expects per-image standardisation — (pixel - mean) / adjusted stddev —
     * not a [0,1] or [-1,1] rescale. Getting this wrong degrades similarity silently.
     */
    private fun writeStandardised(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        var sum = 0.0
        var sumSq = 0.0
        val n = pixels.size * CHANNELS
        for (p in pixels) {
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            sum += r + g + b
            sumSq += (r * r + g * g + b * b).toDouble()
        }
        val mean = sum / n
        val variance = (sumSq / n) - mean * mean
        // The 1/sqrt(N) floor mirrors TensorFlow's per_image_standardization.
        val std = max(sqrt(max(variance, 0.0)), 1.0 / sqrt(n.toDouble())).toFloat()
        val meanF = mean.toFloat()

        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.putFloat((((p shr 16) and 0xFF) - meanF) / std)
            inputBuffer.putFloat((((p shr 8) and 0xFF) - meanF) / std)
            inputBuffer.putFloat(((p and 0xFF) - meanF) / std)
        }
        inputBuffer.rewind()
    }

    override fun close() = interpreter.close()

    private companion object {
        const val MODEL_ASSET = "facenet_512.tflite"
        const val CHANNELS = 3
        const val DEFAULT_THREADS = 4

        fun loadModel(context: Context): ByteBuffer =
            context.assets.openFd(MODEL_ASSET).use { fd ->
                fd.createInputStream().use { stream ->
                    stream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength,
                    )
                }
            }
    }
}
```

- [ ] **Step 5: Run the instrumented test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.embed.FaceEmbedderTest"
```

Expected: PASS, 4 tests. The `sameFaceIsCloserThanDifferentFaces` case is the one that
matters — if it fails, suspect the standardisation in `writeStandardised` or a
`noCompress` misconfiguration on the `.tflite` asset before touching any threshold.

- [ ] **Step 6: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/embed app/src/androidTest
git commit -m "feat(embed): FaceNet-512 embeddings with eye-based alignment"
```

---

## Task 9: Pipeline orchestration — the integration milestone

This is where the app first works end to end. After this task, processing a real clip produces real people and real counts.

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/core/model/Person.kt`
- Create: `app/src/main/java/com/iykyk/collage/pipeline/PipelineState.kt`
- Create: `app/src/main/java/com/iykyk/collage/pipeline/PersonCollagePipeline.kt`
- Create: `app/src/main/java/com/iykyk/collage/video/FullResFrameGrabber.kt`
- Test: `app/src/androidTest/java/com/iykyk/collage/pipeline/PersonCollagePipelineTest.kt`

**Interfaces:**
- Consumes: everything from Tasks 2–8
- Produces:
  - `data class Appearance(startMs: Long, endMs: Long)`
  - `data class Person(id: Int, label: String, appearances: List<Appearance>, shot: Bitmap)` with `appearanceCount`
  - `data class AnalysisResult(people: List<Person>, sourceName: String, durationMs: Long)`
  - `sealed interface PipelineState` — `Idle`, `Preparing`, `Analysing(fraction)`, `GroupingPeople`, `SelectingShots`, `BuildingCollage`, `Done(CollageResult)`, `Failed(message)`
  - `PersonCollagePipeline(context, config).run(uri, sourceName): Flow<PipelineState>`
  - `FullResFrameGrabber(context, uri).grab(timestampMs): Bitmap?` and `close()`

- [ ] **Step 1: Define the result model**

`app/src/main/java/com/iykyk/collage/core/model/Person.kt`:

```kotlin
package com.iykyk.collage.core.model

import android.graphics.Bitmap

/** One continuous visible segment of a person. */
data class Appearance(val startMs: Long, val endMs: Long)

/**
 * A unique person found in the video, with every segment they appeared in and the single
 * best shot chosen to represent them.
 */
data class Person(
    val id: Int,
    val label: String,
    val appearances: List<Appearance>,
    val shot: Bitmap,
) {
    val appearanceCount: Int get() = appearances.size
}

data class AnalysisResult(
    val people: List<Person>,
    val sourceName: String,
    val durationMs: Long,
) {
    val totalAppearances: Int get() = people.sumOf { it.appearanceCount }
}
```

- [ ] **Step 2: Define the progress states**

`app/src/main/java/com/iykyk/collage/pipeline/PipelineState.kt`:

```kotlin
package com.iykyk.collage.pipeline

import android.graphics.Bitmap
import com.iykyk.collage.core.model.AnalysisResult

data class CollageResult(val analysis: AnalysisResult, val collage: Bitmap)

/** Progress of one processing run, surfaced straight to the UI. */
sealed interface PipelineState {
    data object Idle : PipelineState
    data object Preparing : PipelineState

    /** [fraction] in 0..1 across the clip's duration. */
    data class Analysing(val fraction: Float) : PipelineState
    data object GroupingPeople : PipelineState
    data object SelectingShots : PipelineState
    data object BuildingCollage : PipelineState

    data class Done(val result: CollageResult) : PipelineState
    data class Failed(val message: String) : PipelineState

    /** Human-readable stage name for the progress UI. */
    val label: String
        get() = when (this) {
            Idle -> ""
            Preparing -> "Preparing video"
            is Analysing -> "Finding faces"
            GroupingPeople -> "Grouping people"
            SelectingShots -> "Choosing best shots"
            BuildingCollage -> "Building collage"
            is Done -> "Done"
            is Failed -> "Failed"
        }
}
```

- [ ] **Step 3: Implement the full-resolution frame grabber**

`app/src/main/java/com/iykyk/collage/video/FullResFrameGrabber.kt`:

```kotlin
package com.iykyk.collage.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import java.io.Closeable

/**
 * Pulls single frames at full resolution by seeking.
 *
 * Seeking is slow, which is why analysis uses a sequential decoder instead — but the
 * representative shots are only a handful of frames, so a few seeks are the cheapest way
 * to get full-quality pixels for the collage tiles.
 */
class FullResFrameGrabber(context: Context, uri: Uri) : Closeable {

    private val retriever = MediaMetadataRetriever().apply { setDataSource(context, uri) }

    fun grab(timestampMs: Long): Bitmap? = runCatching {
        retriever.getFrameAtTime(
            timestampMs * 1_000L,
            MediaMetadataRetriever.OPTION_CLOSEST,
        )
    }.getOrNull()

    override fun close() = retriever.release()
}
```

- [ ] **Step 4: Write the failing end-to-end instrumented test**

`app/src/androidTest/java/com/iykyk/collage/pipeline/PersonCollagePipelineTest.kt`:

```kotlin
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
        val analysis = runOnSample(0).filterIsInstance<PipelineState.Done>().single().result.analysis
        val firstSeen = analysis.people.map { it.appearances.first().startMs }
        assertEquals(firstSeen.sorted(), firstSeen)
        assertEquals(analysis.people.indices.map { "Person ${it + 1}" }, analysis.people.map { it.label })
    }

    @Test
    fun eachAppearanceIsANonEmptyNonOverlappingSegment() {
        val analysis = runOnSample(0).filterIsInstance<PipelineState.Done>().single().result.analysis
        analysis.people.forEach { person ->
            person.appearances.zipWithNext { a, b ->
                assertTrue("appearances overlap for ${person.label}", a.endMs <= b.startMs)
            }
            person.appearances.forEach { assertTrue(it.endMs >= it.startMs) }
        }
    }

    @Test
    fun coOccurringPeopleAreNeverMerged() {
        // The cannot-link guarantee, checked on real data: no person may have two
        // appearances that overlap in time with each other.
        val analysis = runOnSample(0).filterIsInstance<PipelineState.Done>().single().result.analysis
        analysis.people.forEach { person ->
            val sorted = person.appearances.sortedBy { it.startMs }
            sorted.zipWithNext { a, b ->
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
```

- [ ] **Step 5: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.pipeline.PersonCollagePipelineTest"
```

Expected: `Unresolved reference: PersonCollagePipeline`.

- [ ] **Step 6: Implement the pipeline**

`app/src/main/java/com/iykyk/collage/pipeline/PersonCollagePipeline.kt`:

```kotlin
package com.iykyk.collage.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.iykyk.collage.collage.CollageRenderer
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.cluster.ConstrainedClusterer
import com.iykyk.collage.core.model.Appearance
import com.iykyk.collage.core.model.AnalysisResult
import com.iykyk.collage.core.model.BoxF
import com.iykyk.collage.core.model.FaceObservation
import com.iykyk.collage.core.model.Person
import com.iykyk.collage.core.model.PointF2
import com.iykyk.collage.core.model.ScoredObservation
import com.iykyk.collage.core.model.Tracklet
import com.iykyk.collage.core.quality.FaceQualityScorer
import com.iykyk.collage.core.track.TrackletBuilder
import com.iykyk.collage.detect.CropExtractor
import com.iykyk.collage.detect.MlKitFaceDetector
import com.iykyk.collage.embed.EyeAligner
import com.iykyk.collage.embed.FaceEmbedder
import com.iykyk.collage.video.AnalysisFrame
import com.iykyk.collage.video.FullResFrameGrabber
import com.iykyk.collage.video.MediaCodecFrameSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Runs the whole analysis for one video and emits progress.
 *
 * Stage order matters: tracklets must exist before anything is embedded (embedding once
 * per appearance rather than once per frame is what keeps this fast and stable), and
 * embeddings must exist before clustering can group appearances into people.
 */
class PersonCollagePipeline(
    private val context: Context,
    private val config: PipelineConfig,
) {

    private val scorer = FaceQualityScorer(config)
    private val aligner = EyeAligner()

    fun run(uri: Uri, sourceName: String): Flow<PipelineState> = flow {
        emit(PipelineState.Preparing)

        val source = MediaCodecFrameSource(context, uri, config)
        val durationMs = source.durationMs.coerceAtLeast(1L)

        // --- Stage 1: decode, detect, and keep the best crops per live track ----------
        val detector = MlKitFaceDetector(config)
        val observationsByFrame = mutableListOf<List<FaceObservation>>()
        val frameCache = HashMap<Int, AnalysisFrame>()

        try {
            source.frames().collect { frame ->
                currentCoroutineContext().ensureActive()
                val observations = detector.detect(frame)
                observationsByFrame += observations
                if (observations.isNotEmpty()) frameCache[frame.index] = frame
                emit(PipelineState.Analysing((frame.timestampMs.toFloat() / durationMs).coerceIn(0f, 1f)))
            }
        } finally {
            detector.close()
        }
        emit(PipelineState.Analysing(1f))

        // --- Stage 2: appearances ------------------------------------------------------
        emit(PipelineState.GroupingPeople)
        val tracklets = TrackletBuilder(config, scorer).build(observationsByFrame)
        if (tracklets.isEmpty()) {
            emit(PipelineState.Failed("No clearly visible faces were found in this video."))
            return@flow
        }

        // --- Stage 3: one averaged embedding per appearance ----------------------------
        val embeddings: List<FloatArray>
        FaceEmbedder(context).use { embedder ->
            embeddings = tracklets.map { tracklet ->
                currentCoroutineContext().ensureActive()
                val crops = tracklet.bestCrops(frameCache, embedder.inputSize)
                embedder.embedAveraged(crops).also { crops.forEach(Bitmap::recycle) }
            }
        }

        // --- Stage 4: group appearances into people ------------------------------------
        val cannotLink = buildCannotLink(tracklets)
        val clusters = ConstrainedClusterer(config.clusterThreshold).cluster(embeddings, cannotLink)

        // --- Stage 5: representative shots ---------------------------------------------
        emit(PipelineState.SelectingShots)
        val people = FullResFrameGrabber(context, uri).use { grabber ->
            clusters
                .map { members -> members.map { tracklets[it] } }
                .sortedBy { group -> group.minOf { it.visibleStartMs } }
                .mapIndexed { index, group ->
                    currentCoroutineContext().ensureActive()
                    Person(
                        id = index,
                        label = "Person ${index + 1}",
                        appearances = group
                            .map { Appearance(it.visibleStartMs, it.visibleEndMs) }
                            .sortedBy { it.startMs },
                        shot = representativeShot(group, grabber, frameCache),
                    )
                }
        }

        val analysis = AnalysisResult(people, sourceName, durationMs)

        // --- Stage 6: collage ------------------------------------------------------------
        emit(PipelineState.BuildingCollage)
        val collage = CollageRenderer().render(analysis)
        emit(PipelineState.Done(CollageResult(analysis, collage)))
    }.flowOn(Dispatchers.Default)

    /** Tracklets that are clearly visible at the same moment cannot be the same person. */
    private fun buildCannotLink(tracklets: List<Tracklet>): Set<Pair<Int, Int>> = buildSet {
        for (i in tracklets.indices) {
            for (j in i + 1 until tracklets.size) {
                if (tracklets[i].overlapsInTime(tracklets[j])) add(i to j)
            }
        }
    }

    /** The top-scoring crops of one appearance, aligned to the model's input. */
    private fun Tracklet.bestCrops(
        frameCache: Map<Int, AnalysisFrame>,
        inputSize: Int,
    ): List<Bitmap> = observations
        .filter { it.clearlyVisible }
        .sortedByDescending { it.score }
        .take(config.cropsPerTracklet)
        .mapNotNull { scored -> alignedCrop(scored, frameCache, inputSize) }
        .ifEmpty {
            // Should not happen for an admitted tracklet, but never hand an empty list on.
            listOfNotNull(observations.maxByOrNull { it.score }
                ?.let { alignedCrop(it, frameCache, inputSize) })
        }

    private fun alignedCrop(
        scored: ScoredObservation,
        frameCache: Map<Int, AnalysisFrame>,
        inputSize: Int,
    ): Bitmap? {
        val frame = frameCache[scored.observation.frameIndex] ?: return null
        val obs = scored.observation
        val expand = EMBED_CROP_FACTOR
        val crop = runCatching { CropExtractor.faceCrop(frame, obs, expand) }.getOrNull() ?: return null

        // Landmarks are in upright frame coordinates; the aligner needs crop-local ones.
        val side = max(obs.box.width, obs.box.height) * expand
        val originX = obs.box.centerX - side / 2f
        val originY = obs.box.centerY - side / 2f
        val scale = crop.width / side
        fun local(p: PointF2?) = p?.let { PointF2((it.x - originX) * scale, (it.y - originY) * scale) }

        return aligner.align(crop, local(obs.leftEye), local(obs.rightEye), inputSize)
            .also { if (it !== crop) crop.recycle() }
    }

    /**
     * The best shot across all of a person's appearances, taken from a *full-resolution*
     * re-decode and cropped generously — never tight to the face box.
     */
    private fun representativeShot(
        group: List<Tracklet>,
        grabber: FullResFrameGrabber,
        frameCache: Map<Int, AnalysisFrame>,
    ): Bitmap {
        val best = group.mapNotNull { it.best }.maxBy { it.score }
        val obs = best.observation

        val fullFrame = grabber.grab(obs.timestampMs)
        if (fullFrame != null) {
            // Analysis coordinates are downsampled; scale up to the full-res frame.
            val scale = fullFrame.width.toFloat() / obs.frameWidth
            val side = max(obs.box.width, obs.box.height) * config.representativeCropFactor * scale
            val cx = obs.box.centerX * scale
            val cy = obs.box.centerY * scale
            val rect = BoxF(cx - side / 2f, cy - side / 2f, cx + side / 2f, cy + side / 2f)
                .clampTo(fullFrame.width, fullFrame.height)

            val w = (rect.right - rect.left).roundToInt()
            val h = (rect.bottom - rect.top).roundToInt()
            if (w > 1 && h > 1) {
                return Bitmap.createBitmap(fullFrame, rect.left.roundToInt(), rect.top.roundToInt(), w, h)
                    .also { if (it !== fullFrame) fullFrame.recycle() }
            }
            fullFrame.recycle()
        }

        // Fall back to the analysis-resolution crop if the seek failed.
        val frame = frameCache.getValue(obs.frameIndex)
        return CropExtractor.faceCrop(frame, obs, config.representativeCropFactor)
    }

    private companion object {
        /** Tighter than the collage crop: the model wants the face to fill the input. */
        const val EMBED_CROP_FACTOR = 1.6f
    }
}
```

- [ ] **Step 7: Add a placeholder renderer so this compiles**

`CollageRenderer` is built properly in Task 11. To keep this task independently
runnable, create a minimal version now and replace its body there:

`app/src/main/java/com/iykyk/collage/collage/CollageRenderer.kt`:

```kotlin
package com.iykyk.collage.collage

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import com.iykyk.collage.core.model.AnalysisResult

/** Placeholder implementation; Task 11 replaces the body with the real layout. */
class CollageRenderer {
    fun render(analysis: AnalysisResult): Bitmap =
        Bitmap.createBitmap(WIDTH, HEIGHT, Bitmap.Config.ARGB_8888).also {
            Canvas(it).drawColor(Color.BLACK)
        }

    companion object {
        const val WIDTH = 1080
        const val HEIGHT = 1920
    }
}
```

- [ ] **Step 8: Run the end-to-end test**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.pipeline.PersonCollagePipelineTest"
```

Expected: PASS, 6 tests. Note the person and appearance counts for Sample 1 in the test
output — Task 14 tunes against them. Do **not** adjust thresholds here.

- [ ] **Step 9: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage app/src/androidTest
git commit -m "feat(pipeline): end-to-end analysis producing people, appearances and shots"
```

---

## Task 10: Debug analysis dump for offline tuning

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/pipeline/AnalysisDump.kt`
- Modify: `app/src/main/java/com/iykyk/collage/pipeline/PersonCollagePipeline.kt` (emit the dump)
- Test: `app/src/test/java/com/iykyk/collage/pipeline/AnalysisDumpTest.kt`

**Interfaces:**
- Consumes: `Tracklet`, `PipelineConfig`
- Produces:
  - `@Serializable data class TrackletRecord(index, startMs, endMs, visibleFrames, bestScore, bestTimestampMs, embedding: List<Float>)`
  - `@Serializable data class AnalysisDump(sourceName, durationMs, config: Map<String, Float>, tracklets: List<TrackletRecord>, cannotLink: List<List<Int>>)`
  - `AnalysisDump.toJson(): String`, `AnalysisDump.fromJson(String): AnalysisDump`
  - `PersonCollagePipeline.lastDump: AnalysisDump?`

**Why:** re-running the device pipeline for every threshold experiment costs minutes. Dumping tracklet embeddings once lets the JVM harness in Task 14 sweep thresholds in seconds against Sample 1's published ground truth.

- [ ] **Step 1: Write the failing round-trip test**

`app/src/test/java/com/iykyk/collage/pipeline/AnalysisDumpTest.kt`:

```kotlin
package com.iykyk.collage.pipeline

import org.junit.Assert.assertEquals
import org.junit.Test

class AnalysisDumpTest {

    private val dump = AnalysisDump(
        sourceName = "Sample 1",
        durationMs = 30_000,
        tracklets = listOf(
            TrackletRecord(0, 100, 900, 7, 0.82f, 400, listOf(0.1f, 0.2f, 0.3f)),
            TrackletRecord(1, 200, 800, 5, 0.71f, 500, listOf(0.4f, 0.5f, 0.6f)),
        ),
        cannotLink = listOf(listOf(0, 1)),
    )

    @Test
    fun `json round trips exactly`() {
        assertEquals(dump, AnalysisDump.fromJson(dump.toJson()))
    }

    @Test
    fun `embeddings survive the round trip`() {
        val back = AnalysisDump.fromJson(dump.toJson())
        assertEquals(listOf(0.1f, 0.2f, 0.3f), back.tracklets[0].embedding)
    }

    @Test
    fun `cannot link pairs survive the round trip`() {
        assertEquals(listOf(listOf(0, 1)), AnalysisDump.fromJson(dump.toJson()).cannotLink)
    }
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.pipeline.AnalysisDumpTest"
```

Expected: `Unresolved reference: AnalysisDump`.

- [ ] **Step 3: Implement the dump**

`app/src/main/java/com/iykyk/collage/pipeline/AnalysisDump.kt`:

```kotlin
package com.iykyk.collage.pipeline

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/** One appearance, flattened for offline threshold sweeps. */
@Serializable
data class TrackletRecord(
    val index: Int,
    val startMs: Long,
    val endMs: Long,
    val visibleFrames: Int,
    val bestScore: Float,
    val bestTimestampMs: Long,
    val embedding: List<Float>,
)

/**
 * Everything the tuning harness needs to replay clustering offline: the appearances, their
 * embeddings, and the cannot-link pairs the video geometry implies.
 */
@Serializable
data class AnalysisDump(
    val sourceName: String,
    val durationMs: Long,
    val tracklets: List<TrackletRecord>,
    /** Index pairs that must not merge, as two-element lists. */
    val cannotLink: List<List<Int>>,
) {
    fun toJson(): String = JSON.encodeToString(this)

    companion object {
        private val JSON = Json { prettyPrint = false; ignoreUnknownKeys = true }
        fun fromJson(text: String): AnalysisDump = JSON.decodeFromString(text)
    }
}
```

- [ ] **Step 4: Populate the dump from the pipeline**

In `PersonCollagePipeline`, add the property and fill it just after clustering inputs are
ready (immediately before Stage 4's `ConstrainedClusterer` call):

```kotlin
    /** Populated after every run; Task 14's harness reads this via the debug export. */
    @Volatile
    var lastDump: AnalysisDump? = null
        private set
```

```kotlin
        lastDump = AnalysisDump(
            sourceName = sourceName,
            durationMs = durationMs,
            tracklets = tracklets.mapIndexed { i, t ->
                TrackletRecord(
                    index = i,
                    startMs = t.visibleStartMs,
                    endMs = t.visibleEndMs,
                    visibleFrames = t.visibleFrameCount,
                    bestScore = t.best?.score ?: 0f,
                    bestTimestampMs = t.best?.observation?.timestampMs ?: 0L,
                    embedding = embeddings[i].toList(),
                )
            },
            cannotLink = cannotLink.map { listOf(it.first, it.second) },
        )
```

- [ ] **Step 5: Run the test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.pipeline.AnalysisDumpTest"
```

Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/pipeline app/src/test/java/com/iykyk/collage/pipeline
git commit -m "feat(pipeline): serialisable analysis dump for offline threshold tuning"
```

---

## Task 11: Grid layout and collage rendering

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/core/collage/GridSpec.kt`
- Modify: `app/src/main/java/com/iykyk/collage/collage/CollageRenderer.kt` (replace the Task 9 placeholder)
- Test: `app/src/test/java/com/iykyk/collage/core/collage/GridSpecTest.kt`
- Test: `app/src/androidTest/java/com/iykyk/collage/collage/CollageRendererTest.kt`

**Interfaces:**
- Consumes: `AnalysisResult`, `Person`
- Produces:
  - `data class Cell(row: Int, column: Int, columnSpan: Int)`
  - `GridSpec.forCount(n: Int): List<Cell>` — pure, portrait-biased, last row centred by spanning
  - `CollageRenderer().render(analysis: AnalysisResult): Bitmap` — 1080×1920

- [ ] **Step 1: Write the failing grid test**

`app/src/test/java/com/iykyk/collage/core/collage/GridSpecTest.kt`:

```kotlin
package com.iykyk.collage.core.collage

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GridSpecTest {

    @Test
    fun `one person fills a single cell`() {
        assertEquals(listOf(Cell(0, 0, 1)), GridSpec.forCount(1))
    }

    @Test
    fun `two people stack in one column on a portrait canvas`() {
        val cells = GridSpec.forCount(2)
        assertEquals(2, cells.size)
        assertEquals(listOf(0, 1), cells.map { it.row })
        assertTrue("both should be in column 0", cells.all { it.column == 0 })
    }

    @Test
    fun `four people form a two by two grid`() {
        val cells = GridSpec.forCount(4)
        assertEquals(4, cells.size)
        assertEquals(setOf(0, 1), cells.map { it.row }.toSet())
        assertEquals(setOf(0, 1), cells.map { it.column }.toSet())
    }

    @Test
    fun `five people use two columns with the last row spanning`() {
        val cells = GridSpec.forCount(5)
        assertEquals(5, cells.size)
        assertEquals(3, cells.maxOf { it.row } + 1)
        // The odd one out spans the full width rather than leaving a hole.
        assertEquals(2, cells.last().columnSpan)
    }

    @Test
    fun `every count from one to twelve produces exactly that many cells`() {
        for (n in 1..12) assertEquals("count $n", n, GridSpec.forCount(n).size)
    }

    @Test
    fun `no two cells occupy the same position`() {
        for (n in 1..12) {
            val positions = GridSpec.forCount(n).map { it.row to it.column }
            assertEquals("count $n has overlapping cells", positions.size, positions.distinct().size)
        }
    }

    @Test
    fun `larger groups stay at most three columns wide`() {
        for (n in 1..12) {
            assertTrue("count $n", GridSpec.forCount(n).all { it.column + it.columnSpan <= 3 })
        }
    }

    @Test
    fun `zero people yields no cells`() {
        assertEquals(emptyList<Cell>(), GridSpec.forCount(0))
    }
}
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.collage.GridSpecTest"
```

Expected: `Unresolved reference: GridSpec`.

- [ ] **Step 3: Implement the grid**

`app/src/main/java/com/iykyk/collage/core/collage/GridSpec.kt`:

```kotlin
package com.iykyk.collage.core.collage

/** A tile position in the collage grid. [columnSpan] lets a short final row stay centred. */
data class Cell(val row: Int, val column: Int, val columnSpan: Int)

/**
 * Chooses a grid for a given number of people on a 9:16 canvas.
 *
 * Portrait-biased: at most three columns, so tiles stay tall enough to show a face well.
 * A final row with leftover space spans its tiles rather than leaving an empty hole.
 */
object GridSpec {

    fun forCount(count: Int): List<Cell> {
        if (count <= 0) return emptyList()

        val columns = when {
            count <= 2 -> 1
            count <= 6 -> 2
            else -> 3
        }

        val cells = mutableListOf<Cell>()
        var remaining = count
        var row = 0
        while (remaining > 0) {
            val inThisRow = minOf(columns, remaining)
            // Widen tiles so a short row still fills the canvas.
            val span = columns / inThisRow
            for (i in 0 until inThisRow) {
                cells += Cell(row = row, column = i * span, columnSpan = span)
            }
            remaining -= inThisRow
            row++
        }
        return cells
    }

    /** Number of grid columns used for [count] people. */
    fun columnsFor(count: Int): Int = when {
        count <= 2 -> 1
        count <= 6 -> 2
        else -> 3
    }

    /** Number of grid rows used for [count] people. */
    fun rowsFor(count: Int): Int =
        if (count <= 0) 0 else (count + columnsFor(count) - 1) / columnsFor(count)
}
```

- [ ] **Step 4: Run the grid test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.core.collage.GridSpecTest"
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Write the failing renderer test**

`app/src/androidTest/java/com/iykyk/collage/collage/CollageRendererTest.kt`:

```kotlin
package com.iykyk.collage.collage

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.iykyk.collage.core.model.AnalysisResult
import com.iykyk.collage.core.model.Appearance
import com.iykyk.collage.core.model.Person
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CollageRendererTest {

    private fun swatch(color: Int) = Bitmap.createBitmap(400, 500, Bitmap.Config.ARGB_8888)
        .also { Canvas(it).drawColor(color) }

    private fun analysis(count: Int) = AnalysisResult(
        people = (0 until count).map { i ->
            Person(
                id = i,
                label = "Person ${i + 1}",
                appearances = List(i + 1) { k -> Appearance(k * 1000L, k * 1000L + 500) },
                shot = swatch(Color.rgb(40 + i * 30, 90, 200 - i * 15)),
            )
        },
        sourceName = "Sample 1",
        durationMs = 30_000,
    )

    @Test
    fun rendersAtStoryResolution() {
        val bmp = CollageRenderer().render(analysis(5))
        assertEquals(1080, bmp.width)
        assertEquals(1920, bmp.height)
    }

    @Test
    fun everyPersonCountFromOneToNineRendersWithoutError() {
        for (n in 1..9) {
            val bmp = CollageRenderer().render(analysis(n))
            assertTrue("count $n produced an empty bitmap", bmp.width > 0)
            bmp.recycle()
        }
    }

    @Test
    fun tilePixelsActuallyAppearInTheOutput() {
        // Each person's swatch colour must be present somewhere, i.e. tiles were drawn.
        val bmp = CollageRenderer().render(analysis(4))
        val pixels = IntArray(bmp.width * bmp.height)
        bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val distinct = pixels.toHashSet()
        assertTrue("only ${distinct.size} distinct colours; tiles were probably not drawn",
            distinct.size > 50)
    }

    @Test
    fun aspectRatioIsInstagramStory() {
        val bmp = CollageRenderer().render(analysis(3))
        assertEquals(9f / 16f, bmp.width.toFloat() / bmp.height, 1e-3f)
    }
}
```

- [ ] **Step 6: Implement the renderer**

Replace `app/src/main/java/com/iykyk/collage/collage/CollageRenderer.kt` entirely:

```kotlin
package com.iykyk.collage.collage

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.Typeface
import com.iykyk.collage.core.collage.Cell
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

            val gridTop = HEADER_HEIGHT
            val gridHeight = HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
            val cellWidth = (WIDTH - MARGIN * 2 - GUTTER * (columns - 1)) / columns.toFloat()
            val cellHeight = (gridHeight - MARGIN - GUTTER * (rows - 1)) / rows.toFloat()

            people.forEachIndexed { index, person ->
                val cell = cells[index]
                val left = MARGIN + cell.column * (cellWidth + GUTTER)
                val width = cellWidth * cell.columnSpan + GUTTER * (cell.columnSpan - 1)
                val top = gridTop + cell.row * (cellHeight + GUTTER)
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
            typeface = Typeface.DEFAULT
        }
        val peopleWord = if (analysis.people.size == 1) "person" else "people"
        val appearanceWord = if (analysis.totalAppearances == 1) "appearance" else "appearances"

        canvas.drawText("Who's in this video", MARGIN, 110f, title)
        canvas.drawText(
            "${analysis.people.size} $peopleWord  ·  ${analysis.totalAppearances} $appearanceWord",
            MARGIN, 168f, subtitle,
        )
    }

    private fun drawTile(canvas: Canvas, person: Person, bounds: RectF) {
        val clip = canvas.save()
        val path = android.graphics.Path().apply {
            addRoundRect(bounds, CORNER, CORNER, android.graphics.Path.Direction.CW)
        }
        canvas.clipPath(path)

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
            textSize = 34f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        canvas.drawText(person.label, bounds.left + 24f, bounds.bottom - 30f, label)

        canvas.restoreToCount(clip)

        // Border, drawn outside the clip so it is not cut in half.
        canvas.drawRoundRect(bounds, CORNER, CORNER, Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f
            color = BORDER
        })

        drawBadge(canvas, person.appearanceCount, bounds)
    }

    /** The "×4" appearance-count badge — the number the assignment asks to be shown. */
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
        canvas.drawRoundRect(badge, BADGE_HEIGHT / 2f, BADGE_HEIGHT / 2f,
            Paint(Paint.ANTI_ALIAS_FLAG).apply { color = ACCENT })
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
            val w = (bitmap.height * targetRatio).toInt()
            val x = max(0, (bitmap.width - w) / 2)
            Rect(x, 0, min(bitmap.width, x + w), bitmap.height)
        } else {
            val h = (bitmap.width / targetRatio).toInt()
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
```

- [ ] **Step 7: Run the renderer test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.collage.CollageRendererTest"
```

Expected: PASS, 4 tests.

- [ ] **Step 8: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage app/src/test app/src/androidTest
git commit -m "feat(collage): adaptive story-format collage with appearance-count badges"
```

---

## Task 12: Saving to the gallery and sharing

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/export/CollageExporter.kt`
- Create: `app/src/main/res/xml/file_paths.xml`
- Modify: `app/src/main/AndroidManifest.xml` (FileProvider)
- Test: `app/src/androidTest/java/com/iykyk/collage/export/CollageExporterTest.kt`

**Interfaces:**
- Consumes: a `Bitmap` and a display name
- Produces:
  - `CollageExporter(context)` with `suspend fun saveToGallery(bitmap, displayName): Result<Uri>` and `suspend fun shareIntent(bitmap, displayName): Intent`

- [ ] **Step 1: Declare the FileProvider**

`app/src/main/res/xml/file_paths.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<paths>
    <cache-path name="shared_collages" path="shared/" />
</paths>
```

In `app/src/main/AndroidManifest.xml`, inside `<application>`:

```xml
<provider
    android:name="androidx.core.content.FileProvider"
    android:authorities="${applicationId}.fileprovider"
    android:exported="false"
    android:grantUriPermissions="true">
    <meta-data
        android:name="android.support.FILE_PROVIDER_PATHS"
        android:resource="@xml/file_paths" />
</provider>
```

- [ ] **Step 2: Write the failing instrumented test**

`app/src/androidTest/java/com/iykyk/collage/export/CollageExporterTest.kt`:

```kotlin
package com.iykyk.collage.export

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CollageExporterTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val exporter = CollageExporter(context)

    private fun bitmap() = Bitmap.createBitmap(1080, 1920, Bitmap.Config.ARGB_8888)
        .also { Canvas(it).drawColor(Color.MAGENTA) }

    @Test
    fun savingToTheGalleryReturnsAReadableUri() = runBlocking {
        val result = exporter.saveToGallery(bitmap(), "test-collage")
        assertTrue("save failed: ${result.exceptionOrNull()}", result.isSuccess)

        val uri = result.getOrThrow()
        context.contentResolver.openInputStream(uri).use { stream ->
            assertNotNull(stream)
            assertTrue("saved image is empty", (stream!!.available()) > 0)
        }
    }

    @Test
    fun shareIntentIsASendIntentWithAnImageStream() = runBlocking {
        val intent = exporter.shareIntent(bitmap(), "test-collage")
        assertEquals(Intent.ACTION_SEND, intent.action)
        assertEquals("image/jpeg", intent.type)
        assertNotNull(intent.getParcelableExtra<android.net.Uri>(Intent.EXTRA_STREAM))
        assertTrue(
            "read permission not granted to the receiving app",
            intent.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION != 0
        )
    }

    @Test
    fun sharedFileIsReadableThroughTheProvider() = runBlocking {
        val intent = exporter.shareIntent(bitmap(), "test-collage")
        val uri = intent.getParcelableExtra<android.net.Uri>(Intent.EXTRA_STREAM)!!
        assertEquals("content", uri.scheme)
        context.contentResolver.openInputStream(uri).use { assertNotNull(it) }
    }
}
```

- [ ] **Step 3: Run it and confirm it fails**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.export.CollageExporterTest"
```

Expected: `Unresolved reference: CollageExporter`.

- [ ] **Step 4: Implement the exporter**

`app/src/main/java/com/iykyk/collage/export/CollageExporter.kt`:

```kotlin
package com.iykyk.collage.export

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.core.content.FileProvider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Writes the collage to the device gallery and hands it to the system share sheet.
 *
 * Two different mechanisms on purpose: saving goes through [MediaStore] so the image shows
 * up in the gallery, while sharing goes through a [FileProvider] on the cache directory so
 * no permission or gallery entry is needed just to send it somewhere.
 */
class CollageExporter(private val context: Context) {

    suspend fun saveToGallery(bitmap: Bitmap, displayName: String): Result<Uri> =
        withContext(Dispatchers.IO) {
            runCatching {
                val fileName = "$displayName-${System.currentTimeMillis()}.jpg"
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                    put(MediaStore.Images.Media.MIME_TYPE, MIME_TYPE)
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$ALBUM")
                        put(MediaStore.Images.Media.IS_PENDING, 1)
                    }
                }

                val resolver = context.contentResolver
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: error("MediaStore refused to create an entry")

                resolver.openOutputStream(uri)?.use { out ->
                    if (!bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)) {
                        error("failed to encode the collage")
                    }
                } ?: error("could not open an output stream for $uri")

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                }
                uri
            }
        }

    suspend fun shareIntent(bitmap: Bitmap, displayName: String): Intent =
        withContext(Dispatchers.IO) {
            val dir = File(context.cacheDir, "shared").apply { mkdirs() }
            val file = File(dir, "$displayName.jpg")
            file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, it) }

            val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
            Intent(Intent.ACTION_SEND).apply {
                type = MIME_TYPE
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        }

    private companion object {
        const val MIME_TYPE = "image/jpeg"
        const val JPEG_QUALITY = 95
        const val ALBUM = "IYKYK Collages"
    }
}
```

- [ ] **Step 5: Run the test and confirm it passes**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.export.CollageExporterTest"
```

Expected: PASS, 3 tests. On an API 26–28 device the save test needs
`WRITE_EXTERNAL_STORAGE` granted; grant it with
`adb shell pm grant com.iykyk.collage android.permission.WRITE_EXTERNAL_STORAGE`.

- [ ] **Step 6: Commit**

```bash
cd /d/IFYKYK
git add app/src/main app/src/androidTest
git commit -m "feat(export): save the collage to the gallery and share via the system sheet"
```

---

## Task 13: Compose UI and ViewModel

**Files:**
- Create: `app/src/main/java/com/iykyk/collage/ui/CollageViewModel.kt`
- Create: `app/src/main/java/com/iykyk/collage/ui/AppScreen.kt`
- Modify: `app/src/main/java/com/iykyk/collage/MainActivity.kt`

**Interfaces:**
- Consumes: `PersonCollagePipeline`, `PipelineState`, `CollageResult`, `CollageExporter`, `SampleVideos`
- Produces: `CollageViewModel` with `state: StateFlow<PipelineState>`, `message: StateFlow<String?>`, `process(uri, name)`, `cancel()`, `reset()`, `save()`, `share()`; `AppScreen(viewModel, onShare: (Intent) -> Unit)`

- [ ] **Step 1: Implement the ViewModel**

`app/src/main/java/com/iykyk/collage/ui/CollageViewModel.kt`:

```kotlin
package com.iykyk.collage.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.export.CollageExporter
import com.iykyk.collage.pipeline.PersonCollagePipeline
import com.iykyk.collage.pipeline.PipelineState
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch

/**
 * Owns one processing run at a time.
 *
 * The pipeline is a cold Flow on a background dispatcher, so cancelling [job] genuinely
 * stops decoding rather than leaving the codec running behind a discarded UI.
 */
class CollageViewModel(application: Application) : AndroidViewModel(application) {

    private val pipeline = PersonCollagePipeline(application, PipelineConfig.Default)
    private val exporter = CollageExporter(application)

    private val _state = MutableStateFlow<PipelineState>(PipelineState.Idle)
    val state: StateFlow<PipelineState> = _state.asStateFlow()

    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message.asStateFlow()

    private var job: Job? = null

    fun process(uri: Uri, sourceName: String) {
        job?.cancel()
        _message.value = null
        job = viewModelScope.launch {
            pipeline.run(uri, sourceName)
                .catch { cause ->
                    if (cause is CancellationException) throw cause
                    _state.value = PipelineState.Failed(cause.message ?: "Processing failed.")
                }
                .collect { _state.value = it }
        }
    }

    fun cancel() {
        job?.cancel()
        job = null
        _state.value = PipelineState.Idle
    }

    fun reset() {
        cancel()
        _message.value = null
    }

    fun save() {
        val result = currentResult() ?: return
        viewModelScope.launch {
            val outcome = exporter.saveToGallery(result.collage, slug(result))
            _message.value = if (outcome.isSuccess) {
                "Saved to your gallery"
            } else {
                "Could not save: ${outcome.exceptionOrNull()?.message}"
            }
        }
    }

    fun share(onIntent: (Intent) -> Unit) {
        val result = currentResult() ?: return
        viewModelScope.launch {
            onIntent(Intent.createChooser(exporter.shareIntent(result.collage, slug(result)), "Share collage"))
        }
    }

    fun consumeMessage() { _message.value = null }

    private fun currentResult() = (_state.value as? PipelineState.Done)?.result

    private fun slug(result: com.iykyk.collage.pipeline.CollageResult) =
        result.analysis.sourceName.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-')
}
```

- [ ] **Step 2: Implement the screen**

`app/src/main/java/com/iykyk/collage/ui/AppScreen.kt`:

```kotlin
package com.iykyk.collage.ui

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.iykyk.collage.pipeline.PipelineState
import com.iykyk.collage.samples.SampleVideos

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScreen(viewModel: CollageViewModel, onShare: (Intent) -> Unit) {
    val state by viewModel.state.collectAsState()
    val message by viewModel.message.collectAsState()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let { viewModel.process(it, "Selected video") } }

    LaunchedEffect(message) {
        message?.let {
            snackbar.showSnackbar(it)
            viewModel.consumeMessage()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = { TopAppBar(title = { Text("Who's in this video") }) },
    ) { padding ->
        Box(
            Modifier.padding(padding).fillMaxSize().padding(20.dp),
            contentAlignment = Alignment.Center,
        ) {
            when (val s = state) {
                is PipelineState.Idle -> PickerContent(
                    onPickVideo = {
                        picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
                    },
                    onPickSample = { index ->
                        val sample = SampleVideos.ensureExtracted(context)[index]
                        viewModel.process(sample.uri, sample.label)
                    },
                )

                is PipelineState.Failed -> FailureContent(s.message) { viewModel.reset() }

                is PipelineState.Done -> ResultContent(
                    state = s,
                    onSave = viewModel::save,
                    onShare = { viewModel.share(onShare) },
                    onStartOver = viewModel::reset,
                )

                else -> ProcessingContent(s) { viewModel.cancel() }
            }
        }
    }
}

@Composable
private fun PickerContent(onPickVideo: () -> Unit, onPickSample: (Int) -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            "Pick a video and I'll find everyone in it.",
            style = MaterialTheme.typography.bodyLarge,
        )
        Spacer(Modifier.height(8.dp))
        Button(onClick = onPickVideo, modifier = Modifier.fillMaxWidth()) {
            Text("Choose a video")
        }
        Text("or try a sample", style = MaterialTheme.typography.labelMedium)
        repeat(3) { i ->
            OutlinedButton(onClick = { onPickSample(i) }, modifier = Modifier.fillMaxWidth()) {
                Text("Sample ${i + 1}")
            }
        }
    }
}

@Composable
private fun ProcessingContent(state: PipelineState, onCancel: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(state.label, style = MaterialTheme.typography.titleMedium)

        if (state is PipelineState.Analysing) {
            LinearProgressIndicator(
                progress = { state.fraction },
                modifier = Modifier.fillMaxWidth(),
            )
            Text("${(state.fraction * 100).toInt()}%", style = MaterialTheme.typography.labelLarge)
        } else {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        TextButton(onClick = onCancel) { Text("Cancel") }
    }
}

@Composable
private fun FailureContent(message: String, onStartOver: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(message, style = MaterialTheme.typography.bodyLarge)
        Button(onClick = onStartOver) { Text("Try another video") }
    }
}

@Composable
private fun ResultContent(
    state: PipelineState.Done,
    onSave: () -> Unit,
    onShare: () -> Unit,
    onStartOver: () -> Unit,
) {
    val analysis = state.result.analysis
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(
            "${analysis.people.size} people · ${analysis.totalAppearances} appearances",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
        )

        Image(
            bitmap = state.result.collage.asImageBitmap(),
            contentDescription = "Generated collage",
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)),
        )

        analysis.people.forEach { person ->
            ListItem(
                headlineContent = { Text(person.label) },
                supportingContent = {
                    Text("${person.appearanceCount} " +
                        if (person.appearanceCount == 1) "appearance" else "appearances")
                },
                leadingContent = {
                    Image(
                        bitmap = person.shot.asImageBitmap(),
                        contentDescription = person.label,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.size(52.dp).clip(RoundedCornerShape(10.dp)),
                    )
                },
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = onSave) { Text("Save") }
            Button(onClick = onShare) { Text("Share") }
        }
        TextButton(onClick = onStartOver) { Text("Process another video") }
        Spacer(Modifier.height(24.dp))
    }
}
```

- [ ] **Step 3: Wire up MainActivity**

Replace the body of `app/src/main/java/com/iykyk/collage/MainActivity.kt`, keeping the
wizard-generated theme name (it is derived from the project name):

```kotlin
package com.iykyk.collage

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import com.iykyk.collage.ui.AppScreen
import com.iykyk.collage.ui.CollageViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            // Keep the theme composable the wizard generated for this project.
            MaterialTheme {
                val viewModel: CollageViewModel = viewModel()
                AppScreen(viewModel = viewModel, onShare = ::startActivity)
            }
        }
    }
}
```

If the wizard generated a theme wrapper such as `IYKYKCollageTheme`, use that instead of
the bare `MaterialTheme` and import it rather than `androidx.compose.material3.MaterialTheme`.

- [ ] **Step 4: Build, install, and drive the app by hand**

```bash
cd /d/IFYKYK && ./gradlew :app:installDebug
```

Walk through, on the device:
1. Tap **Sample 1** → progress advances with a live percentage and a stage label.
2. Tap **Cancel** mid-run → returns to the picker promptly; no ANR.
3. Run Sample 1 to completion → the collage renders with a tile per person and a `×N` badge.
4. Tap **Save** → a "Saved to your gallery" snackbar appears and the image is in the gallery app.
5. Tap **Share** → the system share sheet opens with the image attached.
6. Tap **Choose a video** → the photo picker opens and any video on the device processes.
7. Rotate the device mid-processing → the run survives (the ViewModel outlives the activity).

- [ ] **Step 5: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage
git commit -m "feat(ui): Compose pick, progress and result flow with save and share"
```

---

## Task 14: Tune the threshold against Sample 1's ground truth

The 50%-weighted criterion is settled here. Everything before this was structure; this is calibration.

**Files:**
- Create: `app/src/androidTest/java/com/iykyk/collage/pipeline/DumpAnalysisTest.kt`
- Create: `app/src/test/resources/dumps/` (populated from the device)
- Create: `app/src/test/java/com/iykyk/collage/tuning/ThresholdSweepTest.kt`
- Modify: `app/src/main/java/com/iykyk/collage/core/PipelineConfig.kt` (write the winning values back)

**Interfaces:**
- Consumes: `AnalysisDump`, `ConstrainedClusterer`, `PipelineConfig`
- Produces: a tuned `PipelineConfig.Default` and the numbers the README must quote

**Ground truth (published in the assignment):** Sample 1 contains **5 people, each appearing 4 times, 20 appearances total.** A and B share the frame at 10.1–11.5 s; C and D at 20.2–21.6 s. Samples 2 and 3 have no published counts, so they are sanity-checked by eye only — never fitted to.

- [ ] **Step 1: Add an instrumented test that writes the dumps**

`app/src/androidTest/java/com/iykyk/collage/pipeline/DumpAnalysisTest.kt`:

```kotlin
package com.iykyk.collage.pipeline

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.samples.SampleVideos
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Not a test of behaviour — a data exporter. Running it writes one analysis dump per
 * sample into the app's external files dir, which is then pulled to `src/test/resources`
 * so the JVM sweep in ThresholdSweepTest can replay clustering offline.
 */
@RunWith(AndroidJUnit4::class)
class DumpAnalysisTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun writeDumpsForAllSamples() = runBlocking {
        val outDir = File(context.getExternalFilesDir(null), "dumps").apply { mkdirs() }

        SampleVideos.ensureExtracted(context).forEachIndexed { index, sample ->
            val pipeline = PersonCollagePipeline(context, PipelineConfig.Default)
            pipeline.run(sample.uri, sample.label).collect { }

            val dump = pipeline.lastDump
            assertTrue("no dump produced for ${sample.label}", dump != null)
            File(outDir, "sample${index + 1}.json").writeText(dump!!.toJson())
        }
    }
}
```

- [ ] **Step 2: Produce and pull the dumps**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.pipeline.DumpAnalysisTest"
mkdir -p app/src/test/resources/dumps
adb shell ls /sdcard/Android/data/com.iykyk.collage/files/dumps
adb pull /sdcard/Android/data/com.iykyk.collage/files/dumps/. app/src/test/resources/dumps/
ls -l app/src/test/resources/dumps
```

Expected: `sample1.json`, `sample2.json`, `sample3.json`.

- [ ] **Step 3: Write the sweep**

`app/src/test/java/com/iykyk/collage/tuning/ThresholdSweepTest.kt`:

```kotlin
package com.iykyk.collage.tuning

import com.iykyk.collage.core.PipelineConfig
import com.iykyk.collage.core.cluster.ConstrainedClusterer
import com.iykyk.collage.pipeline.AnalysisDump
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Replays clustering over dumped embeddings across a range of thresholds.
 *
 * This exists because rebuilding and rerunning the APK per experiment costs minutes,
 * while a sweep here costs milliseconds — which is the difference between guessing at the
 * similarity threshold and actually choosing it.
 */
class ThresholdSweepTest {

    private fun load(name: String): AnalysisDump? =
        javaClass.getResourceAsStream("/dumps/$name.json")
            ?.bufferedReader()?.use { AnalysisDump.fromJson(it.readText()) }

    private fun peopleAndAppearances(dump: AnalysisDump, threshold: Float): Pair<Int, List<Int>> {
        val vectors = dump.tracklets.map { it.embedding.toFloatArray() }
        val cannotLink = dump.cannotLink.map { it[0] to it[1] }.toSet()
        val clusters = ConstrainedClusterer(threshold).cluster(vectors, cannotLink)
        return clusters.size to clusters.map { it.size }.sorted()
    }

    @Test
    fun `print the sweep for every sample`() {
        for (name in listOf("sample1", "sample2", "sample3")) {
            val dump = load(name) ?: continue
            println("=== $name: ${dump.tracklets.size} appearances detected ===")
            var t = 0.30f
            while (t <= 1.00f) {
                val (people, sizes) = peopleAndAppearances(dump, t)
                println("  tau=%.2f -> %d people, appearances per person %s".format(t, people, sizes))
                t += 0.05f
            }
        }
    }

    /**
     * The published ground truth for Sample 1. The configured threshold must reproduce it.
     *
     * If the dump does not hold 20 appearances, the problem is upstream in detection or
     * tracklet gating, and no threshold can fix it — fix that first.
     */
    @Test
    fun `configured threshold reproduces sample one ground truth`() {
        val dump = load("sample1")
        assumeTrue("no dump checked in yet; run DumpAnalysisTest first", dump != null)

        assertEquals(
            "expected 20 appearances before clustering, got ${dump!!.tracklets.size}",
            20, dump.tracklets.size,
        )

        val (people, sizes) = peopleAndAppearances(dump, PipelineConfig.Default.clusterThreshold)
        assertEquals("wrong number of unique people", 5, people)
        assertEquals("each person should appear 4 times", listOf(4, 4, 4, 4, 4), sizes)
    }

    @Test
    fun `co-occurring appearances are never grouped together`() {
        val dump = load("sample1") ?: return
        val vectors = dump.tracklets.map { it.embedding.toFloatArray() }
        val cannotLink = dump.cannotLink.map { it[0] to it[1] }.toSet()
        val clusters = ConstrainedClusterer(PipelineConfig.Default.clusterThreshold)
            .cluster(vectors, cannotLink)

        clusters.forEach { cluster ->
            cluster.forEach { a ->
                cluster.forEach { b ->
                    if (a != b) {
                        assertEquals(
                            "appearances $a and $b co-occur but were grouped",
                            false, (a to b) in cannotLink || (b to a) in cannotLink,
                        )
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run the sweep and read the table**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.tuning.ThresholdSweepTest" -i 2>&1 | grep -E "tau=|===|appearances detected"
```

Read the Sample 1 rows and pick the τ that yields **5 people × 4 appearances**. Prefer a
value sitting in the *middle* of a stable plateau rather than at its edge — a threshold
that only works at exactly one value will not generalise to Samples 2 and 3.

- [ ] **Step 5: Deal with a wrong appearance count before touching τ**

If Sample 1's dump does not contain 20 tracklets, no threshold will fix it. Diagnose in
this order, changing one thing at a time and re-running Steps 2 and 4:

| Symptom | Likely cause | Knob |
|---|---|---|
| Too many appearances | one appearance split by a detection dropout | raise `gapToleranceMs` |
| Too many appearances | blurred approach frames admitted as their own segment | raise `minSharpness` |
| Too few appearances | brief genuine appearances rejected | lower `minVisibleFrames` / `minVisibleDurationMs` |
| Too few appearances | sampling too coarse to see short segments | raise `sampleFps` to 10 |
| Faces missed entirely | analysis frame too small, or faces below the size floor | raise `analysisLongEdge`, lower `MIN_FACE_SIZE` |

- [ ] **Step 6: Write the winning values back**

Update `clusterThreshold` (and any gate you changed) in `PipelineConfig`, then confirm:

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest
```

Expected: the whole JVM suite passes, including
`configured threshold reproduces sample one ground truth`.

- [ ] **Step 7: Confirm on the device and eyeball Samples 2 and 3**

```bash
cd /d/IFYKYK && ./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.pipeline.PersonCollagePipelineTest"
```

Then run all three samples in the app. For Samples 2 and 3, check by eye that each tile is
a genuinely different person and that no person appears twice in one collage. Do **not**
tune against them — their counts are unpublished, and fitting to a guess is overfitting.

- [ ] **Step 8: Commit**

```bash
cd /d/IFYKYK
git add app/src/main/java/com/iykyk/collage/core/PipelineConfig.kt app/src/test app/src/androidTest
git commit -m "feat(tuning): calibrate similarity threshold against Sample 1 ground truth"
```

---

## Task 15: README, APK, and the demo recording

**Files:**
- Create: `README.md`
- Create: `.gitignore` entries for build output (verify the wizard's)

**Interfaces:**
- Consumes: the tuned `PipelineConfig`, the sweep output from Task 14
- Produces: the four submission artefacts

- [ ] **Step 1: Write the README**

`README.md` must cover, because the assignment names each explicitly:

1. **What it does** — one paragraph.
2. **Build and setup** — clone, open in Android Studio, Gradle JDK 21, `./gradlew :app:assembleDebug`, minSdk 26. Note that the model and samples are bundled, so there is nothing to download.
3. **Embedding model** — FaceNet-512, Inception-ResNet-v1, TFLite, 160×160×3 float input (per-image standardised), 512-d L2-normalised output, ~24 MB, bundled at `app/src/main/assets/facenet_512.tflite`. State that inference is CPU with XNNPACK, 4 threads.
4. **Similarity threshold** — the tuned τ, stated as a cosine distance, **and how it was chosen**: the offline sweep in `ThresholdSweepTest` against Sample 1's published ground truth of 5 people × 4 appearances, choosing the middle of the stable plateau. Include the sweep table for Sample 1.
5. **How appearances are counted** — the tracklet model, the sharpness gate excluding whip-pans, the gap tolerance, and the cannot-link constraint for co-occurring people. This is the part a reviewer weighting it at 50% will actually read.
6. **Architecture** — the package table from this plan, and the point that `core/` is Android-free and unit-tested.
7. **Testing** — `./gradlew :app:testDebugUnitTest` for the pure logic, `./gradlew :app:connectedDebugAndroidTest` for decode/detect/embed/export on a device.
8. **Known limitations** — honest ones: identity is not tracked across different videos; heavy occlusion or extreme profile can split a person; FaceNet-512 is older than current ArcFace models and was chosen for reliable availability; Samples 2 and 3 have no published ground truth so their counts are unverified.

- [ ] **Step 2: Build the debug APK**

```bash
cd /d/IFYKYK && ./gradlew clean :app:assembleDebug
ls -l app/build/outputs/apk/debug/app-debug.apk
```

Expected: an APK of roughly 60–75 MB (24 MB model + ~14 MB samples + ML Kit).

- [ ] **Step 3: Run the full verification sweep**

```bash
cd /d/IFYKYK && ./gradlew :app:testDebugUnitTest && ./gradlew :app:connectedDebugAndroidTest
```

Expected: everything green. Do not record the demo until it is.

- [ ] **Step 4: Record the demo**

Requirements from the assignment: **≤ 60 seconds, no narration or editing needed, and the collage for each of the three samples must be on screen and legible, held long enough to read.**

Budget it before recording — 60 seconds is tight for three clips:

| Segment | Time |
|---|---|
| Sample 1: tap, processing, counts | ~8 s |
| Sample 1: collage held still | ~7 s |
| Sample 2: tap, processing, counts | ~8 s |
| Sample 2: collage held still | ~7 s |
| Sample 3: tap, processing, counts | ~8 s |
| Sample 3: collage held still | ~7 s |
| Save + share sheet | ~5 s |

If processing takes longer than ~8 s per clip the budget breaks. Either raise
`analysisLongEdge`/`sampleFps` efficiency first, or record processing for Sample 1 only
and jump straight to the collage for 2 and 3 — the hard requirement is that **all three
collages are clearly visible**, not that all three processing runs are.

```bash
adb shell screenrecord --time-limit 60 --bit-rate 8000000 /sdcard/demo.mp4
# ...drive the app on the device, then Ctrl+C...
adb pull /sdcard/demo.mp4 ./demo.mp4
```

Verify before submitting: the video is under 60 s, all three collages appear, and the
appearance counts are readable at normal playback speed.

- [ ] **Step 5: Final commit and tag**

```bash
cd /d/IFYKYK
git add -A
git commit -m "docs: README with build steps, embedding model and similarity threshold"
git log --oneline | head -20
```

- [ ] **Step 6: Confirm every deliverable exists**

- [ ] Git repository with a readable commit history
- [ ] `README.md` covering build steps, embedding model, and the chosen threshold
- [ ] `app/build/outputs/apk/debug/app-debug.apk`
- [ ] `demo.mp4`, ≤ 60 s, showing processing, appearance counts, and all three collages

---

## Appendix: contingencies

**If `MediaCodecFrameSource` misbehaves on the device** (black frames, starved
`ImageReader`, codec exceptions), swap in a seek-based source rather than debugging the
codec under deadline. It satisfies the same `FrameSource` interface, so nothing downstream
changes:

```kotlin
class MmrFrameSource(context: Context, uri: Uri, private val config: PipelineConfig) : FrameSource {
    private val retriever = MediaMetadataRetriever().apply { setDataSource(context, uri) }
    override val durationMs =
        retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L

    override fun frames(): Flow<AnalysisFrame> = flow {
        val stepMs = 1000L / config.sampleFps
        var t = 0L
        var index = 0
        while (t < durationMs) {
            retriever.getFrameAtTime(t * 1000, MediaMetadataRetriever.OPTION_CLOSEST)?.let { bmp ->
                // Convert the bitmap to NV21 here, or refactor the detector to accept Bitmaps.
                emit(bitmapToAnalysisFrame(bmp, index++, t))
            }
            t += stepMs
        }
    }.flowOn(Dispatchers.IO)
}
```

Expect 15–30 s of processing per clip, which breaks the 60-second demo budget — so if this
fallback is needed, record processing for Sample 1 only.

**If accuracy is short on the 50% criterion**, work in this order, re-running the Task 14
sweep after each: verify the appearance count (tracklets) is right *before* touching τ;
then try `PlainResizeAligner` against `EyeAligner`; then raise `cropsPerTracklet`; then
`sampleFps`. Changing τ cannot fix a wrong appearance count.

**If the schedule slips**, cut in this order: the debug JSON dump (Task 10) can be replaced
by reading counts off the screen; the collage polish in Task 11 can fall back to plain
rectangles; the photo-picker path can be dropped so only the bundled samples work. Never
cut Tasks 3, 4, or 14 — they are the 50%.

---

## Self-Review

**Spec coverage.** Every numbered section of the design maps to a task: §3.1 decode → Task 6;
§3.2 detect → Task 7; §3.3 quality gate → Task 2/2b; §3.4 tracklets → Task 3; §3.5 embed →
Task 8; §3.6 cluster → Task 4; §3.7 representative shot → Task 9; §3.8 collage → Task 11;
§3.9 export → Task 12; §4 architecture → the File Structure section, enforced by the
Android-free `core/` rule; §5 UI → Task 13; §6 tuning harness → Task 14; §7 testing →
distributed through every task; §8 deliverables → Task 15.

**Two deliberate refinements to the spec**, both improving on it rather than departing from it:

1. The spec built tracklets from ML Kit tracking ids plus a geometric merge pass. The plan
   instead does greedy IoU association with the tracking id as a bonus. It subsumes both the
   grouping and the merge in one testable pass, and removes the dependency on ML Kit's
   id stability entirely — which the spec listed as its top risk.
2. The spec's separate embedding-aware merge pass is dropped. With IoU association handling
   id churn, and the cannot-link constraint handling co-occurrence, a second merge pass adds
   a tunable knob without a demonstrated need. If a real long-occlusion split shows up
   during Task 14, it can be added then — YAGNI until the data asks for it.

Both are noted here so the divergence from the spec is explicit rather than silent.

**Type consistency.** `PipelineConfig` field names are used identically in Tasks 2b, 3, 4,
9 and 14. `FaceObservation` is constructed only in Task 7 and consumed with the same field
names in Tasks 2b, 3 and 9. `Tracklet.best` returns `ScoredObservation?` and every call
site handles the null. `CollageRenderer.render(AnalysisResult): Bitmap` has the same
signature in its Task 9 placeholder and its Task 11 implementation. `FaceAligner.align`
takes `(Bitmap, PointF2?, PointF2?, Int)` in both implementations and at its one call site
in Task 9.

**Known placeholder, deliberate and scheduled.** `CollageRenderer` in Task 9 Step 7 is a
stub, created only so Task 9 compiles and can be verified independently; Task 11 Step 6
replaces the whole file. This is flagged in both places.

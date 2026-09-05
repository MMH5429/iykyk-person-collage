# Who's in this video — video-based unique-person collage

An Android app that processes a portrait video entirely on-device: it detects faces, groups
appearances of the same person, counts how many times each person appears, picks a strong
representative shot for each, and renders a shareable collage.

No backend, no network calls. Everything runs on the phone.

## Results on the supplied clips

Measured on a Galaxy Note9 (Android 10, API 29), at the calibrated threshold:

| Clip | People found | Appearances found | Ground truth |
|---|---|---|---|
| Sample 1 | **5** | 17 | **5 people**, 20 appearances |
| Sample 2 | 5 | 17 | not published |
| Sample 3 | 5 | 17 | not published |

Sample 1's person count matches the published ground truth exactly, and three of its five
people show exactly the expected four appearances. The appearance total is 17 of 20 — the
shortfall is discussed honestly under [Known limitations](#known-limitations).

The three clips share a cast, and all three independently resolve to five people at the same
threshold, which is a meaningful cross-check rather than a coincidence.

Processing takes roughly **55 seconds per 30-second clip** on that 2018 device, which
notably falls back to a *software* AVC decoder.

---

## Build and setup

**Requirements**

- Android Studio (any recent version), or the Android command-line tools
- **JDK 21** as the Gradle JDK — Android Studio's bundled JBR is fine
- Android SDK **platform 36** and **build-tools 36.1.0**
- A device or emulator running **API 26+**

**Build**

```bash
git clone <this repo>
cd IFYKYK
echo "sdk.dir=/path/to/Android/sdk" > local.properties   # Studio writes this on first sync

./gradlew :app:assembleDebug
```

The debug APK lands at `app/build/outputs/apk/debug/app-debug.apk`.

Nothing needs downloading separately: the embedding model and the three sample clips are
bundled in `app/src/main/assets/`. That makes the debug APK large (~105 MB) — 24 MB of
model, 14 MB of sample video, and the ML Kit and TFLite native libraries for every ABI.

**Toolchain** — AGP 9.4.0, Gradle 9.7.1, Kotlin 2.4.10, compileSdk 36, minSdk 26. AndroidX
is pinned to the newest versions that still target SDK 36, because SDK 37 is not published
yet and the very latest AndroidX releases require it.

---

## Embedding model

**FaceNet-512** — Inception-ResNet-v1, converted to TFLite.

| | |
|---|---|
| File | `app/src/main/assets/facenet_512.tflite` (24 MB) |
| Input | `input_1`, `[1, 160, 160, 3]` float32 |
| Preprocessing | **per-image standardisation**, `(pixel - mean) / adjusted_stddev`, matching TensorFlow's `per_image_standardization` — *not* a `[0,1]` or `[-1,1]` rescale |
| Output | `[1, 512]`, L2-normalised before use |
| Runtime | TensorFlow Lite 2.17.0, CPU with XNNPACK, 4 threads |

Shapes are read from the model at load time rather than hardcoded, so a different embedding
model can be swapped in without code changes. FaceNet-512 was chosen for reliable public
availability; MobileFaceNet would be smaller and uses a more modern loss, but has no
dependable public mirror.

---

## Similarity threshold

**τ = 0.50 cosine distance** (`PipelineConfig.clusterThreshold`).

### How it was chosen

Sample 1 has published ground truth (five people). `DumpAnalysisTest` exports per-appearance
embeddings from a real device run; `ThresholdSweepTest` replays clustering offline in 0.01
steps, so a full sweep takes milliseconds instead of an APK rebuild per experiment.

The number of people found is stable across wide bands, not knife-edge:

| Clip | τ range giving 5 people |
|---|---|
| Sample 1 | **[0.45, 0.64)** ← the only published ground truth |
| Sample 2 | [0.40, 0.52) |
| Sample 3 | [0.47, 0.54) |

0.50 sits inside all three plateaus rather than on any edge. Samples 2 and 3 were used only
as a consistency check, never fitted to — their counts are unpublished, and fitting to a
guess is just overfitting.

Every tuned constant lives in one place, `core/PipelineConfig.kt`, with named fields and the
reasoning attached — no magic numbers scattered through the pipeline.

### Reproducing the calibration

```bash
# 1. Export per-appearance embeddings from a device run of all three clips
adb install -r -t app/build/outputs/apk/debug/app-debug.apk
adb install -r -t app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell "am instrument -w -e class com.iykyk.collage.pipeline.DumpAnalysisTest \
  com.iykyk.collage.test/androidx.test.runner.AndroidJUnitRunner"

# 2. Pull the dumps into the JVM test resources
adb pull /sdcard/Android/data/com.iykyk.collage/files/dumps/. app/src/test/resources/dumps/

# 3. Sweep and read the table
./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.tuning.ThresholdSweepTest" -i \
  | grep -E "tau=|==="
```

Note: `./gradlew connectedAndroidTest` uninstalls the app afterwards, taking its files
directory (and the dumps) with it. Driving the instrumentation directly, as above, keeps
them.

---

## How appearances are counted

This carries the most weight in the brief, so it is worth being precise.

**The unit of the pipeline is the tracklet, not the frame.** A tracklet is one continuous run
of a single face — exactly the brief's definition of an appearance. An identity is a *cluster
of tracklets*, and a person's appearance count is how many tracklets are in their cluster.

1. **Decode** — `MediaCodec` walks the clip once, sampling ~8 fps and subsampling the YUV
   planes to a ~540 px long edge. Sequential decode rather than seeking, because
   `getFrameAtTime` costs 50–150 ms per seek and 240 seeks would take half a minute. Frames
   are delivered through the `ImageReader` callback, so nothing is silently dropped.
2. **Detect** — ML Kit, **FAST mode with tracking disabled**. Both choices are deliberate and
   are explained under [Reproducibility](#reproducibility-a-bug-worth-describing).
3. **Quality-gate** — six normalised signals per detection: sharpness (variance of the
   Laplacian, measured directly on the luma plane), frontality (head yaw/pitch/roll),
   eyes-open, smile, face size, and completeness (how much of the face box lies inside the
   frame).

   A face is **clearly visible** only if sharpness, frontality, size and completeness all
   clear their gates. *This is how whip-pans are excluded* — a motion-blurred face scores
   near zero on sharpness, so it counts for nobody, exactly as the brief requires. No
   special-case motion detection is involved.

   Eyes-open and smile are deliberately **not** gates: someone mid-blink is still present.
   They only influence which shot represents the person.
4. **Detect scene cuts** — consecutive frames are compared by coarse luma histogram. These
   clips are edited montages, and at a hard cut the next person often lands in the same part
   of the screen, so IoU association alone would chain two different people into one
   six-second "appearance". A cut closes every open track. The two populations separate
   cleanly in the measured data — within-shot frames peak at 0.215, the smallest real cut is
   0.225 — so the threshold sits at 0.22.
5. **Associate into tracklets** — greedy IoU matching across consecutive sampled frames, with
   a 400 ms gap tolerance so one dropped detection does not split an appearance in two.
6. **Embed** — the top 5 crops of each tracklet are eye-aligned to 160×160, embedded, and
   **averaged** into one vector per appearance. Averaging stops a single bad frame from
   inventing a phantom person. Crops are also **neighbour-aware**: in side-by-side two-shots
   a generous square crop would otherwise swallow the adjacent face and feed two people into
   one embedding.
7. **Cluster** — average-linkage agglomerative clustering on cosine distance, with a hard
   **cannot-link constraint: two tracklets clearly visible at the same moment can never be
   the same person.** Constraints propagate through merges. This makes co-occurring people
   resolve structurally rather than relying on the threshold getting lucky.

---

## Reproducibility: a bug worth describing

Early on, the same clip analysed twice produced different appearance counts — swinging
between 12 and 19 on Sample 1. Instrumenting the pipeline showed the decoder was innocent:
exactly 188 frames were analysed every run. The variance came from ML Kit.

`enableTracking()` puts the detector into stateful stream mode, which is tuned for a live
camera and adapts to CPU load by skipping work. Disabling it made detection stateless and the
results reproducible. Nothing was lost, because tracklets are built by IoU association and
the tracking id was only ever a tie-breaking bonus.

That change made every frame a full detection, which pushed `PERFORMANCE_MODE_ACCURATE` to
~150 s per clip. The subjects here are large, frontal, foreground faces, so FAST mode handles
them well and brings it back to ~55 s. Appearance counts are now stable run-to-run (17/17/17
across repeated runs), and the clustering outcome is identical.

---

## Representative shots and the collage

The best frame across all of a person's appearances is chosen by a weighted sum of the six
quality signals (frontality 0.25, sharpness 0.25, eyes-open 0.20, face size 0.10,
completeness 0.10, expression 0.10).

That frame's exact timestamp is then **re-decoded at full resolution** and cropped at 2.2× the
face box — deliberately generous, never tight to the bounding box, which the brief calls out
as a source of low-resolution tiles.

The collage renders to a 1080×1920 bitmap (Instagram Story proportions) with an adaptive
grid, rounded tiles, a per-person `×N` appearance-count badge, and a header carrying the
totals. Saving goes through `MediaStore` into `Pictures/IYKYK Collages`; sharing goes through
a `FileProvider` and the standard Android share sheet.

---

## Architecture

Single `:app` module. Compose UI over a `ViewModel`; the pipeline is a cold `Flow` on
`Dispatchers.Default`, so the UI stays responsive and cancelling genuinely stops the decoder.

| Package | Responsibility |
|---|---|
| `core/model/` | `BoxF`, `FaceObservation`, `Tracklet`, `Person` — plain data |
| `core/quality/` | Laplacian sharpness, the six signals, the visibility gates |
| `core/track/` | `TrackletBuilder` — greedy IoU association, scene-cut aware |
| `core/cluster/` | Cosine distance, `ConstrainedClusterer` |
| `core/frame/` | NV21 subsampling, rotation mapping, scene-cut histograms |
| `core/collage/` | `GridSpec` — person count to grid layout |
| `video/` | `FrameSource`, `MediaCodecFrameSource`, `FullResFrameGrabber` |
| `detect/` | ML Kit wrapper, neighbour-aware crop extraction |
| `embed/` | Eye alignment, TFLite `FaceEmbedder` |
| `pipeline/` | Orchestration, progress states, debug analysis dump |
| `collage/` `export/` `ui/` `samples/` | Rendering, save/share, Compose UI, bundled clips |

**Everything under `core/` has zero Android imports.** That is deliberate: it makes the
accuracy-critical logic — gating, association, clustering, counting — unit-testable on the
JVM and tunable offline without a device.

Memory shape is part of the design. Face crops are cut during the decode pass and each frame
is dropped immediately. Retaining whole frames instead costs ~780 KB each, which over a
30-second clip is more than 100 MB of NV21 buffers — enough to hold a mid-range device in
continuous GC.

The three bundled sample clips are a demo convenience only. They are handed to the pipeline
as ordinary URIs and travel the **identical** code path as a video picked from the gallery.

---

## Testing

```bash
./gradlew :app:testDebugUnitTest          # 78 pure-JVM tests, no device needed
./gradlew :app:connectedDebugAndroidTest  # 27 on-device tests
```

The JVM suite covers geometry and IoU, Laplacian sharpness, every quality signal and gate,
tracklet association (gap tolerance, tracking-id churn, blurred whip-pans, crossing faces,
scene cuts), constrained clustering including cannot-link transitivity, NV21 subsampling with
padded row strides, rotation mapping for all four orientations, scene-cut histograms, grid
layout, dump serialisation, and the calibrated threshold against Sample 1's ground truth.

On-device tests cover decoding, detection, embedding discriminability (same face closer than
two different faces sharing a frame), collage rendering, gallery save, and the share intent.

Diagnostics used during development are kept in `androidTest`: `GateDiagnosticsTest` reports
per-gate rejection rates and signal distributions, `TimelineDiagnosticsTest` writes a
per-frame visibility timeline, and `CollageCaptureTest` renders the real collages to disk.

---

## Known limitations

- **17 of 20 appearances on Sample 1.** The person count is exact; three segments are not.
  This was traced rather than guessed. Comparing the per-frame visibility timeline against
  the tracklet spans shows only **two** frames in the whole clip where a clearly-visible face
  is not covered by some tracklet, and no tracklet spans more than 1.4 s, so nothing is being
  merged or dropped by the association step. The shortfall is upstream, and splits two ways:

  - **One face is never detected.** In the C+D two-shot at 20.2–21.6 s, ML Kit reports a
    single face on every frame. Lowering `MIN_FACE_SIZE` from 0.08 to 0.05 did not surface
    the second face and added a spurious sixth identity, so it was reverted.
  - **One segment is deliberately gated out.** Both faces in that two-shot run past the frame
    edge and score 0.656 completeness, under the 0.75 gate. Relaxing the gate to 0.60 does
    recover the appearance — but a 35%-cropped face does not embed reliably, so it forms its
    own singleton and Sample 1 resolves to six people. No threshold repairs it: at 0.60 the
    only tau giving five on Sample 1 is >= 0.64, where Samples 2 and 3 collapse to four and
    three. Correct identity grouping is worth more than one extra appearance, so this was
    reverted too.

  The honest next step is better detection rather than looser gates — `PERFORMANCE_MODE_ACCURATE`
  does find both faces in two-shots, but costs ~150 s per clip instead of ~55 s, which is a
  poor trade for the app's usability. A hybrid that re-runs ACCURATE only on frames where a
  cut boundary suggests a missing subject would likely get the remaining appearances without
  the cost.
- A representative tile can still include a sliver of an adjacent face in tight two-shots.
  The crop excludes the neighbour's centre, but the tile is then centre-cropped to fill.
- Identity is not tracked across different videos; each video is analysed independently.
- Heavy occlusion or a sustained extreme profile can split one person into two identities.
- FaceNet-512 predates current ArcFace-style models and was chosen for availability rather
  than peak accuracy. Embedding crops come from the ~540 px analysis frame; cropping them
  from full resolution was tried and reverted, because the extra full-size bitmaps
  reintroduced the memory pressure the pipeline is shaped to avoid.
- Samples 2 and 3 have no published ground truth, so their counts are unverified.

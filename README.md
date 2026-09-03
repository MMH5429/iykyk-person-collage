# Who's in this video — video-based unique-person collage

An Android app that processes a portrait video entirely on-device: it detects faces,
groups appearances of the same person, counts how many times each person appears, picks a
strong representative shot for each, and renders a shareable collage.

No backend, no network calls. Everything runs on the phone.

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
# Point Gradle at your SDK (Android Studio writes this for you on first sync):
echo "sdk.dir=/path/to/Android/sdk" > local.properties

./gradlew :app:assembleDebug
```

The debug APK lands at `app/build/outputs/apk/debug/app-debug.apk`.

Nothing needs to be downloaded separately: the embedding model and the three sample clips
are bundled in `app/src/main/assets/`. That does make the debug APK large (~105 MB) —
24 MB of model, 14 MB of sample video, and the ML Kit and TFLite native libraries for
every ABI.

**Toolchain versions** — AGP 9.4.0, Gradle 9.7.1, Kotlin 2.4.10, compileSdk 36, minSdk 26.
AndroidX is pinned to the newest versions that still target SDK 36, because SDK 37 is not
published yet and the very latest AndroidX releases require it.

---

## Embedding model

**FaceNet-512** — Inception-ResNet-v1, converted to TFLite.

| | |
|---|---|
| File | `app/src/main/assets/facenet_512.tflite` (24 MB) |
| Input | `input_1`, `[1, 160, 160, 3]` float32 |
| Input preprocessing | **per-image standardisation**, `(pixel - mean) / adjusted_stddev`, matching TensorFlow's `per_image_standardization` — *not* a `[0,1]` or `[-1,1]` rescale |
| Output | `[1, 512]`, L2-normalised before use |
| Runtime | TensorFlow Lite 2.17.0, CPU with XNNPACK, 4 threads |

Input and output shapes are read from the model at load time rather than hardcoded, so a
different embedding model can be swapped in without code changes.

It was chosen for reliable public availability. MobileFaceNet would be smaller and is
trained with a more modern loss, but has no dependable public mirror.

---

## How appearances are counted

This is the part the assignment weights most heavily, so it is worth being precise.

**The unit of the pipeline is the tracklet, not the frame.** A tracklet is one continuous
run of a single tracked face — which is exactly the assignment's definition of an
appearance. An identity is then a *cluster of tracklets*, and a person's appearance count
is simply how many tracklets are in their cluster.

The stages:

1. **Decode** — `MediaCodec` walks the clip once, front to back, sampling ~8 fps and
   subsampling the YUV planes to a ~540 px long edge. Sequential decode rather than
   seeking, because `getFrameAtTime` costs 50–150 ms per seek and 240 seeks would take
   half a minute.
2. **Detect** — ML Kit in accurate mode with landmarks, classification and tracking.
3. **Quality-gate** — every detection is scored on six normalised signals: sharpness
   (variance of the Laplacian, measured directly on the luma plane), frontality (head
   yaw/pitch/roll), eyes-open, smile, face size, and completeness (how far inside the
   frame the box sits).

   A face counts as **clearly visible** only if sharpness, frontality, size and
   completeness all clear their gates. *This is how whip-pans are excluded* — a
   motion-blurred face scores near zero on sharpness, so it counts for nobody, exactly as
   the brief requires. No special-case motion detection is involved.

   Eyes-open and smile are deliberately **not** gates: someone mid-blink is still present.
   Those two only influence which shot gets chosen to represent the person.
4. **Associate into tracklets** — greedy IoU matching across consecutive sampled frames,
   with a 400 ms gap tolerance so one dropped detection doesn't split an appearance in two.
   ML Kit's tracking ID is used only as a small tie-breaking bonus, never as the mechanism:
   ML Kit reissues IDs part-way through a continuous appearance, which would inflate the
   count. A tracklet must hold at least 2 clearly-visible frames spanning 200 ms to be
   admitted as an appearance.
5. **Embed** — the top 5 crops of each tracklet by quality are eye-aligned to 160×160,
   embedded, and **averaged** into one vector per appearance. Averaging is what stops a
   single bad frame from inventing a phantom person.
6. **Cluster** — average-linkage agglomerative clustering on cosine distance, with a hard
   **cannot-link constraint: two tracklets that are clearly visible at the same moment can
   never be the same person.** Constraints propagate through merges.

   This is what makes co-occurring people resolve correctly — in Sample 1, A and B share
   the frame at 10.1–11.5 s and C and D at 20.2–21.6 s — as a structural guarantee rather
   than something the threshold has to get lucky on.

---

## Similarity threshold

**Current value: cosine distance τ = 0.55** (`PipelineConfig.clusterThreshold`).

> ⚠️ **This is the untuned starting value, not a calibrated one.** The calibration step
> below has not been run yet, because it requires a connected Android device and none was
> available on the build machine. Run it before submitting.

Every tuned constant lives in one place, `core/PipelineConfig.kt`, with named fields — no
magic numbers scattered through the pipeline.

### How to calibrate it

Sample 1 has published ground truth: **5 people, 4 appearances each, 20 total.** The
harness replays clustering offline against dumped embeddings, so a full sweep takes
milliseconds instead of an APK rebuild per experiment.

```bash
# 1. Dump per-tracklet embeddings from a real device run of all three samples
./gradlew :app:connectedDebugAndroidTest --tests "com.iykyk.collage.pipeline.DumpAnalysisTest"

# 2. Pull the dumps into the JVM test resources
mkdir -p app/src/test/resources/dumps
adb pull /sdcard/Android/data/com.iykyk.collage/files/dumps/. app/src/test/resources/dumps/

# 3. Sweep tau and read the table
./gradlew :app:testDebugUnitTest --tests "com.iykyk.collage.tuning.ThresholdSweepTest" -i \
  | grep -E "tau=|==="
```

Pick the τ that yields 5 people × 4 appearances, preferring the **middle of a stable
plateau** rather than its edge — a threshold that works at exactly one value will not
generalise to Samples 2 and 3. Write it back into `PipelineConfig` and record the sweep
table here.

**If the dump does not contain 20 appearances, no threshold can fix it** — the appearance
count is decided before clustering. Diagnose upstream first:

| Symptom | Likely cause | Knob |
|---|---|---|
| Too many appearances | one appearance split by a detection dropout | raise `gapToleranceMs` |
| Too many appearances | blurred approach frames admitted as their own segment | raise `minSharpness` |
| Too few appearances | brief genuine appearances rejected | lower `minVisibleFrames` / `minVisibleDurationMs` |
| Too few appearances | sampling too coarse to catch short segments | raise `sampleFps` |
| Faces missed entirely | analysis frame too small, or faces below the size floor | raise `analysisLongEdge`, lower `MIN_FACE_SIZE` |

Samples 2 and 3 have no published counts. They are sanity-checked by eye — never fitted
to, since fitting to a guess is just overfitting.

---

## Representative shots and the collage

The best frame across all of a person's appearances is chosen by a weighted sum of the six
quality signals (frontality 0.25, sharpness 0.25, eyes-open 0.20, face size 0.10,
completeness 0.10, expression 0.10).

That frame's exact timestamp is then **re-decoded at full resolution** and cropped at
2.2× the face box — deliberately generous, never tight to the bounding box, which the
assignment calls out as a source of low-resolution tiles.

The collage renders to a 1080×1920 bitmap (Instagram Story proportions) with an adaptive
grid, rounded tiles, a per-person `×N` appearance-count badge, and a header carrying the
totals. Saving goes through `MediaStore` into `Pictures/IYKYK Collages`; sharing goes
through a `FileProvider` and the standard Android share sheet.

---

## Architecture

Single `:app` module. Compose UI over a `ViewModel`; the pipeline is a cold `Flow` running
on `Dispatchers.Default`, so the UI stays responsive and cancelling genuinely stops the
decoder.

| Package | Responsibility |
|---|---|
| `core/model/` | `BoxF`, `FaceObservation`, `Tracklet`, `Person` — plain data |
| `core/quality/` | Laplacian sharpness, the six signals, the visibility gates |
| `core/track/` | `TrackletBuilder` — greedy IoU association |
| `core/cluster/` | Cosine distance, `ConstrainedClusterer` |
| `core/frame/` | NV21 subsampling, upright↔raw rotation mapping |
| `core/collage/` | `GridSpec` — person count to grid layout |
| `video/` | `FrameSource`, `MediaCodecFrameSource`, `FullResFrameGrabber` |
| `detect/` | ML Kit wrapper, generous crop extraction |
| `embed/` | Eye alignment, TFLite `FaceEmbedder` |
| `pipeline/` | Orchestration, progress states, debug analysis dump |
| `collage/` `export/` `ui/` `samples/` | Rendering, save/share, Compose UI, bundled clips |

**Everything under `core/` has zero Android imports.** That is deliberate: it is what makes
the accuracy-critical logic — gating, association, clustering, counting — unit-testable on
the JVM and tunable offline without a device.

The three bundled sample clips are a demo convenience only. They are handed to the
pipeline as ordinary URIs and travel the **identical** code path as a video picked from the
gallery; nothing about them is special-cased.

---

## Testing

```bash
./gradlew :app:testDebugUnitTest          # 68 pure-JVM tests, no device needed
./gradlew :app:connectedDebugAndroidTest  # decode, detect, embed, render, export — needs a device
```

The JVM suite covers geometry and IoU, Laplacian sharpness, every quality signal and gate,
tracklet association (gap tolerance, tracking-ID churn, blurred whip-pans, crossing faces),
constrained clustering including cannot-link transitivity, NV21 subsampling with padded row
strides, rotation mapping for all four orientations, grid layout, and dump serialisation.

Two tests in `ThresholdSweepTest` skip until the device dumps are checked in.

---

## Known limitations

- **The similarity threshold is not yet calibrated** (see above). This is the single most
  important outstanding item.
- The instrumented tests have not been executed — no device was attached to the build
  machine. The code compiles, but decode, detection, embedding, rendering and export are
  unverified at runtime.
- Identity is not tracked across different videos; each video is analysed independently.
- Heavy occlusion or a sustained extreme profile can split one person into two identities.
- FaceNet-512 predates current ArcFace-style models and was chosen for availability rather
  than peak accuracy.
- Samples 2 and 3 have no published ground truth, so their counts are unverified.

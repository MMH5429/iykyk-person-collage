# Design: Video-based Unique-Person Collage (iykyk Android Assignment)

Date: 2026-09-03
Status: Approved for planning
Deadline: Sunday 6 September 2026, 23:59 IST

## 1. Problem

Build an Android app that processes a portrait video entirely on-device, detects
faces, groups appearances of the same person across the clip, counts each
person's appearances, picks a strong representative shot per person, and renders
a shareable collage.

Grading weights, which drive every trade-off below:

| Criterion | Weight |
|---|---|
| Identity grouping and appearance-count accuracy | 50% |
| Code quality and architecture | 30% |
| Usability, representative-shot quality, collage presentation | 20% |

The three supplied 30-second portrait clips are a test set, not a target. No
result may be hardcoded; the same code path must handle any similar video.

### Definition of an appearance (from the brief)

An appearance is one continuous visible segment: it begins when a person's face
becomes clearly visible and ends when it is no longer clearly visible. Blurred
whip-pan passes count for nobody. Two clearly visible people in one segment
count as one appearance each.

Sample 1 ground truth: five distinct people, four appearances each, 20 total.
A and B share the frame at 10.1-11.5s; C and D share it at 20.2-21.6s. Samples
2 and 3 have no published counts.

## 2. Central design decision

The unit of the pipeline is the **tracklet**, not the frame.

A tracklet is one continuous run of a single tracked face. A tracklet *is* an
appearance. An identity is a cluster of tracklets. This maps one-to-one onto the
brief's definition of counting, and it makes identity far more stable: each
appearance contributes a single embedding averaged over its best frames, rather
than dozens of noisy per-frame embeddings that invent phantom people.

The naive alternative — embed every detected face, cluster all embeddings, count
clusters — loses on both halves of the 50% criterion. It has no concept of a
continuous segment, so it cannot count appearances at all, and its cluster count
is dominated by outlier frames.

## 3. Pipeline

Stages run sequentially inside one coroutine job, off the main thread, emitting
progress as a `Flow` of sealed states.

### 3.1 Decode

Sequential `MediaCodec` decode into an `ImageReader`, walking the clip forward
once. No seeking during analysis.

- Analysis resolution: downscale so the long edge is ~720 px.
- Effective sample rate: ~8 fps (configurable constant). Frames between samples
  are decoded and dropped without conversion.
- Rationale: `MediaMetadataRetriever.getFrameAtTime` costs 50-150 ms per seek.
  At 8 fps over 30 s that is 240 seeks, i.e. 15-30 s of processing. Sequential
  decode does the same work in a few seconds, which both keeps the progress UI
  honest and buys headroom to raise the sample rate if accuracy needs it.
- Isolated behind a `FrameSource` interface, so an `MmrFrameSource` remains
  available as a fallback if a device-specific codec problem appears.

Full-resolution frames are **not** retained. Only the analysis-resolution frame
data needed for quality scoring and embedding survives past each iteration; the
representative shot is re-decoded at full resolution later (3.7).

### 3.2 Detect

ML Kit Face Detection, configured with:

- `PERFORMANCE_MODE_ACCURATE`
- `LANDMARK_MODE_ALL` (eye landmarks drive alignment)
- `CLASSIFICATION_MODE_ALL` (smile probability, eyes-open probability)
- `enableTracking()` in stream mode, giving a `trackingId` stable across
  consecutive frames
- `setMinFaceSize` tuned so distant background faces do not dominate

Contour detection is off; it is incompatible with tracking and unnecessary here.

### 3.3 Quality gate

Every detection is scored on six independent signals, each normalised to [0,1]:

| Signal | Measure |
|---|---|
| Sharpness | Variance of the Laplacian over the grayscale face crop, resized to a fixed 96x96 so the measure is scale-invariant |
| Frontality | Head Euler Y (yaw), X (pitch), Z (roll) folded into a single penalty |
| Eyes open | min(leftEyeOpenProbability, rightEyeOpenProbability) |
| Expression | smilingProbability |
| Face size | Face box area as a fraction of frame area |
| Completeness | How far the face box sits inside the frame bounds; clipped faces score low |

A detection is **clearly visible** when sharpness, frontality, size, and
completeness all clear their individual gates. Detections below the bar are
still fed to the tracker for continuity but do not count toward an appearance.

This is exactly how whip-pans are excluded: a motion-blurred pass fails the
sharpness gate, so it counts for nobody, as the brief requires. No special-case
motion detection is needed.

### 3.4 Build tracklets

1. Group consecutive detections that share a `trackingId`.
2. Close a tracklet when its tracking ID has been absent for longer than a gap
   tolerance (~0.4 s), so a single dropped detection does not split one
   appearance in two.
3. **Geometric merge pass.** ML Kit sometimes issues a fresh tracking ID
   part-way through a continuous appearance. Two adjacent tracklets are merged
   when both hold: they are close in time (within the gap tolerance) and their
   face boxes overlap spatially at the seam. This runs before embedding, since
   it needs only geometry and timing.
4. A tracklet is admitted as an appearance only if it contains at least a
   minimum number of clearly-visible frames and spans a minimum duration. This
   discards single-frame false positives.

A second, **embedding-aware merge pass** runs after 3.5, once every tracklet has
a vector: adjacent tracklets that survived step 3 as separate but are
temporally close and highly similar in embedding are merged and re-embedded.
Splitting the merge into a cheap geometric pass and a later semantic pass keeps
the dependency order clean — geometry needs no model, and the model needs
tracklets to exist.

### 3.5 Embed

Model: **FaceNet-512** (TFLite, 512-d output, ~24 MB), bundled in `assets/` and
documented in the README as required. It is the most reliably obtainable
on-device face embedding model; MobileFaceNet has no dependable public mirror.

Per tracklet:

1. Take the top-K frames by quality score (K ~ 5).
2. Align each face crop: a two-point similarity transform from the detected eye
   landmarks to canonical eye positions in a 160x160 input, correcting in-plane
   rotation and normalising scale.
3. Run the model, L2-normalise each output.
4. Average the K vectors and re-normalise. That single vector represents the
   whole appearance.

Alignment sits behind a `FaceAligner` interface with a second, simpler
implementation (bounding box plus margin, resized, no rotation) so the tuning
harness can measure which actually performs better on the samples rather than
assuming.

### 3.6 Cluster

Agglomerative, average-linkage, over tracklet embeddings, on cosine distance,
cut at a threshold tau.

**Cannot-link constraint: two tracklets that overlap in time can never belong to
the same person.** Constraints propagate through merges. This is what makes
Sample 1's A+B and C+D shared-frame segments resolve into distinct identities
instead of collapsing, and it is a hard structural guarantee rather than
something the threshold has to get lucky on.

tau is a tuned constant, documented in the README as the brief demands.

Appearance count per person = number of tracklets in that person's cluster.

### 3.7 Representative shot

The best frame across all of a person's tracklets, by a weighted sum of the six
quality signals (weights are named constants, documented, and unit-tested).

That frame's exact timestamp is then re-decoded **at full resolution** via a
single `MediaMetadataRetriever` seek — a handful of seeks total, so the cost is
trivial and confined to this one step.

Cropping: a square crop of side ~2.2x the face box, centred on the face and
clamped into the frame. Never tight to the bounding box; the brief calls that
out explicitly as a source of low-resolution, poor-quality tiles.

### 3.8 Collage

Rendered to a 1080x1920 bitmap on an Android `Canvas` — Instagram Story
proportions.

- Adaptive grid: column and row counts derived from the person count by a pure
  `GridSpec` function, favouring portrait-friendly layouts.
- Rounded tiles with gutters, a subtle border, and a per-tile appearance-count
  badge ("x4").
- Header with the person and appearance totals; footer with the source clip name.
- Dark gradient ground, bright tiles.

### 3.9 Export

- **Save:** `MediaStore` into `Pictures/`, JPEG q95. API 29+ needs no
  permission; API 26-28 declares `WRITE_EXTERNAL_STORAGE` with
  `maxSdkVersion="28"`.
- **Share:** written to cache, exposed via `FileProvider`, sent through
  `ACTION_SEND` — the standard Android share sheet.

## 4. Architecture

Single `:app` module. Jetpack Compose UI over a `MainViewModel`. Core logic in
focused packages, each independently understandable and testable:

| Package | Responsibility |
|---|---|
| `video/` | `FrameSource` interface, `MediaCodecFrameSource`, `MmrFrameSource`, full-res frame grab |
| `detect/` | ML Kit wrapper producing plain `FaceDetection` data objects |
| `quality/` | Sharpness, frontality, and the composite quality score |
| `embed/` | `FaceAligner` implementations, `FaceEmbedder` (TFLite) |
| `cluster/` | `TrackletBuilder`, `ConstrainedAgglomerativeClusterer` |
| `collage/` | `GridSpec`, `CollageRenderer` |
| `export/` | `MediaStoreSaver`, `CollageSharer` |
| `pipeline/` | `PersonCollagePipeline` orchestrating the stages, emitting progress |

Threading: the whole pipeline runs on `Dispatchers.Default` with decode on IO.
Nothing touches the main thread except state emission. Cancellation is
cooperative, so backing out of processing stops the work.

Progress is a sealed state: `Idle`, `Preparing`, `Analysing(fraction)`,
`GroupingPeople`, `SelectingShots`, `BuildingCollage`, `Done(result)`,
`Failed(cause)`.

The clustering, tracklet-building, quality-scoring, grid-layout, and
representative-selection logic are pure Kotlin over plain data — no Android
types. They get real unit tests with synthetic inputs. That is the strongest
available lever on the 30% code-quality criterion, and it is what makes the
50% accuracy criterion tunable at all.

## 5. UI

Three states in one flow:

1. **Pick.** A `PickVisualMedia` photo-picker button (no runtime permission
   needed) plus three quick-start buttons for the bundled sample clips. The
   samples are copied out of `assets/` to cache and fed through the *identical*
   code path as any picked video — they are a convenience for the demo, not a
   special case.
2. **Processing.** Stage label plus a determinate progress bar, cancellable.
3. **Result.** Collage preview, a list of people with their appearance counts, a
   headline total, and Save and Share actions.

## 6. Tuning harness

A JVM unit-test harness (`src/test/`) that runs the tracklet-building,
clustering, and counting logic over analysis data dumped from a real device run.

- A debug-only action exports the per-tracklet analysis (embeddings, timings,
  quality scores, box geometry) as JSON.
- The harness sweeps tau and the gate thresholds against Sample 1's known
  ground truth of five people at four appearances each.
- It reports counts and grouping errors per configuration in seconds, instead of
  requiring an APK rebuild per experiment.

The chosen tau and the reasoning go into the README.

## 7. Testing

Unit tests, all pure-JVM:

- `TrackletBuilder`: gap tolerance, ID-change merging, minimum-appearance gate,
  synthetic detection sequences.
- `ConstrainedAgglomerativeClusterer`: correct grouping on synthetic embeddings;
  cannot-link is never violated, including transitively through merges.
- `FaceQualityScorer`: monotonicity of each signal, weighting behaviour.
- `GridSpec`: sane layouts for 1..12 people.
- Representative selection: picks the highest-scoring candidate.

Manual verification: end-to-end on all three sample clips on a physical device.

## 8. Deliverables

- Git repository.
- README: build and setup steps, embedding model used, chosen similarity
  threshold and how it was chosen, architecture overview, known limitations.
- Working debug APK.
- Screen recording (<= 60 s, no narration) showing processing, appearance
  counts, and the finished collage for **each of the three sample clips**, each
  held on screen long enough to read.

## 9. Constraints and environment

- Kotlin, minSdk 26, Compose. Everything on-device; no backend.
- Time box: 12-15 hours.
- Build environment: Android Studio to be installed on this machine; its bundled
  JBR 21 is the JDK to use. The system JDK 23 is not supported by AGP and must
  not be the Gradle JDK.
- A physical Android device is available for running and recording.
- The three sample clips are downloaded and verified.

## 10. Non-goals

Live camera recording (explicitly not evaluated), any backend, processing
history or persistence, batch processing of multiple videos in one run,
face recognition across different videos.

## 11. Risks

| Risk | Mitigation |
|---|---|
| ML Kit tracking IDs unstable mid-appearance | The tracklet merge pass (3.4) explicitly repairs this; tested with synthetic sequences |
| FaceNet-512 not discriminative enough on these faces | Tracklet-level averaged embeddings, the cannot-link constraint, and the tuning harness; the swappable aligner is the next lever |
| `MediaCodec` plumbing is the fiddliest code here | `FrameSource` interface with an `MmrFrameSource` fallback that can be swapped in without touching anything downstream |
| Android Studio install and first Gradle sync eat schedule | Front-loaded as the very first task, before any pipeline work |
| Appearance counts on Samples 2 and 3 are unverifiable | Tune on Sample 1's ground truth only; sanity-check 2 and 3 by eye against the collage, and avoid overfitting to a single clip |

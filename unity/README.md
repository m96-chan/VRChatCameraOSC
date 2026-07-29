# VRChatCameraOSC Avatar Setup (Unity SDK)

Editor wizard for VRChat SDK3 that wires a **Humanoid** avatar's existing
blend shapes and head bone to the standard **VRCFT (Unified Expressions)
`v2/*` parameters** [VRChatCameraOSC](../README.md) sends (issues
[#16](https://github.com/m96-chan/VRChatCameraOSC/issues/16),
[#21](https://github.com/m96-chan/VRChatCameraOSC/issues/21)) — turning it
into a "lite" face-tracking avatar.

> **You may not need this wizard at all.** If your avatar already supports
> **VRCFaceTracking / Unified Expressions** (most commercial
> face-tracking-ready avatars do), just run the tracker: it emits the same
> `v2/` parameters VRCFaceTracking sends — float, bool, and binary forms,
> gated to what the avatar declares — and the avatar works with **no Unity
> work whatsoever** (issue
> [#18](https://github.com/m96-chan/VRChatCameraOSC/issues/18)). This wizard
> is for avatars with plain blend shapes and no face-tracking setup. See
> ["Avatar setup" in the main README](../README.md#avatar-setup-required-for-the-avatar-to-actually-move).

Because the wizard declares standard `v2/*` names, a wizard-made avatar also
works with **VRCFaceTracking itself** (e.g. an iPhone LiveLink module) — not
only with this app.

## What this is — and isn't

VRChat's own OSC receiver already routes `/avatar/parameters/<Name>` into the
matching Animator parameter for any parameter your avatar declares in its VRC
Expression Parameters — that plumbing is built into VRChat, not something this
package reimplements. This is purely an **editor-time setup tool**: given a
Humanoid avatar, it generates/merges the Expression Parameters asset and the
FX/Gesture Animator Controller layers (blend shapes → FX, head pose →
Gesture — [why](#why-head-pose-lives-on-the-gesture-layer-not-fx)) so
VRChat's existing OSC-to-parameter routing has something to drive.

The parameters wired (the webcam-trackable subset of Unified Expressions,
mirroring `src/mapping/unified.rs`):

| Parameter | Range | Default | Drives |
|---|---|---|---|
| `v2/EyeLidLeft` / `v2/EyeLidRight` | `0..1` | **0.75** | blink shape, **inverted** (0 = closed, 0.75 = relaxed open, 1 = wide) |
| `v2/BrowUpLeft` / `v2/BrowUpRight` | `0..1` | 0 | brow-raise shape, **full weight at 0.5** (webcam-tracked raises peak ~0.3–0.5 on the wire — issue #23; values above 0.5 clamp) |
| `v2/JawOpen` | `0..1` | 0 | mouth-open shape |
| `v2/MouthSmileLeft` / `v2/MouthSmileRight` | `0..1` | 0 | smile shape (shared shape OK) |
| `v2/MouthStretchLeft` / `v2/MouthStretchRight` | `0..1` | 0 | mouth-widen shape (shared shape OK) |
| `v2/Head/Yaw` / `v2/Head/Pitch` / `v2/Head/Roll` | `-1..1` | 0 | Humanoid head muscles — **one combined Gesture-layer blend tree** (issues #25/#27) |
| `VCO_GestureLeft` / `VCO_GestureRight` | **Int** `0..7` | 0 | Humanoid finger muscles — one Gesture-layer hand-pose layer per hand; declared only in the **Gestures** hand mode ([issue #8](https://github.com/m96-chan/VRChatCameraOSC/issues/8), [below](#hand-tracking-gestures-or-per-finger-curls)) |
| `VCO_L_ThumbCurl` / `IndexCurl` / `MiddleCurl` / `RingCurl` / `LittleCurl` + `VCO_R_*` | `0..1` | 0 | Humanoid finger stretch muscles (0 = straight, 1 = fully curled) — one nested-blend-tree Gesture layer per hand; declared only in the **Per-finger curls** hand mode (issue #8 phase 3, [below](#hand-tracking-gestures-or-per-finger-curls)) |
| `VCO_L_ArmUpDown` / `VCO_R_ArmUpDown` | `-1..1` | 0 | Humanoid arm muscles (+1 = straight up, -1 = hanging; exactly 0 = hand untracked → passthrough) — one Gesture layer per arm; declared only while the arm toggle is on ([issue #28](https://github.com/m96-chan/VRChatCameraOSC/issues/28), [below](#arm-raise-vco_lr_armupdown)) |

Plus **optional extras** (issue #24) — declared (and costing expression
parameter bits) **only when you wire a blend shape** to them:

| Optional parameter(s) | Drives |
|---|---|
| `v2/CheekPuffLeft` / `Right` | cheek puff (ぷくー) |
| `v2/JawLeft` / `v2/JawRight` | sideways jaw / mouth shift |
| `v2/LipPuckerUpperLeft` / `Right` | pucker (う) |
| `v2/LipFunnelUpperLeft` / `Right` | funnel (お) |
| `v2/MouthFrownLeft` / `Right` | frown / sad mouth |
| `v2/NoseSneerLeft` / `Right` | nose sneer |
| *(eye-wide picker on `v2/EyeLid*`)* | eye-wide shape over the 0.75..1 segment — **no extra parameter bits** |

The `v2/EyeLid*` **0.75 default** matters: with the inverted VRCFT eyelid
semantics, a 0 default would leave the avatar's eyes shut whenever no tracker
is running. Declared at 0.75 the avatar rests with open eyes (issue #21).

It does **not**:
- Create blend shapes — pick from ones your mesh already has.
- Run at avatar runtime. VRChat strips arbitrary `MonoBehaviour`s from
  uploaded avatars, so `v2/Head/*` are driven the same way as the
  blend-shape parameters: through an **additive** Animator layer using
  Humanoid muscle curves (`Head Tilt Left-Right` / `Head Turn Left-Right` /
  `Head Nod Down-Up`), never a script on the avatar.
- Support non-Humanoid (generic) rigs — out of scope for now.

### Migrating from the retired custom10 setup

Avatars set up by a pre-issue-#21 version of this wizard carry the retired
`MouthOpen`/`EyeBlink*`/`Head*` parameters. Just re-run **Apply**: it strips
every legacy parameter and `OSC_*` layer first, then wires the `v2/*` set —
one click migrates in place. **Remove** also cleans both generations.
Re-upload the avatar afterwards.

### Hand tracking: Gestures or Per-finger curls

The wizard's **"Hand tracking mode"** is a 3-way choice — **Off / Gestures
(default) / Per-finger curls**. The two "on" modes are **mutually exclusive
by design**: both drive the same per-hand Fingers muscle group, and
Unity/VRChat compose humanoid Override layers **per masked muscle group**
(the issue #27 lesson) — two layers writing one group fight, last-one-wins
for the whole group. Applying one mode removes the other mode's layers *and*
its parameter declarations, so you only ever pay expression-parameter bits
for the mode you use (2×8 bits for gestures vs. 10×8 bits for curls).

**Gestures mode** — VRChat's native `GestureLeft`/`GestureRight` parameters
are **read-only over OSC**
([vrchat-community/osc#42](https://github.com/vrchat-community/osc/issues/42)),
so the tracker's webcam hand tracking (issue #8) sends its own
`VCO_GestureLeft`/`VCO_GestureRight` **Int** parameters instead — same
standard 0–7 scale (0 Neutral, 1 Fist, 2 HandOpen, 3 FingerPoint, 4 Victory,
5 RockNRoll, 6 HandGun, 7 ThumbsUp). Apply adds one layer per hand to the
Gesture controller (`OSC_VCO_GestureLeft` / `OSC_VCO_GestureRight`): 7
finger-muscle pose states switched by `Equals`-conditioned any-state
transitions (0.1 s cross-fade), each layer masked to that hand's Fingers
body part only.

**The Neutral state is deliberately empty** (no animation): while the
parameter is 0 — tracker off, no hand on camera, or hand at rest — the layer
writes nothing, so **VRChat's own keyboard/controller hand gestures on the
stock Gesture layers underneath keep working exactly as before**. Only a
recognized camera gesture (1–7) overrides the pose, and it releases back to
VRChat's control the moment the tracker returns to 0. Unlike head pose, no
`VRCAnimatorTrackingControl` is involved — fingers aren't IK-held on
Desktop, so plain muscle animation on the Gesture layer just works.

**Per-finger curls mode** (issue #8 phase 3) — the tracker sends ten
`VCO_L/R_{Thumb,Index,Middle,Ring,Little}Curl` floats (0 = straight, 1 =
fully curled) and each hand gets ONE layer
(`OSC_VCO_L_FingerCurls` / `OSC_VCO_R_FingerCurls`, again masked to that
hand's Fingers part only): a nested Simple1D blend tree, depth 5
(Thumb → Index → Middle → Ring → Little), two slots per level at thresholds
0 and 1, bottoming out in 2⁵ = 32 pose clips. Every leaf writes **all 20**
of that hand's finger muscles (the 15 stretch joints at +1 straight / −1
curled; the 5 Spread muscles pinned at neutral 0) — the per-group override
must always carry the whole group. Each finger blends independently and
smoothly (a half-curled param sits halfway between its two slots). Note the
trade-offs vs. gestures mode: 80 parameter bits instead of 16, and **no
empty-Neutral passthrough** — the curl layers always own the fingers while
applied, so VRChat's keyboard/controller gestures are overridden in this
mode.

### Arm raise (`VCO_L/R_ArmUpDown`)

Issue #28 phase 1: raising a hand into the webcam frame should raise the
avatar's **arm**, not just pose the fingers. Leave the wizard's **"Wire arm
raise"** toggle checked (default) and Apply adds one layer per arm to the
Gesture controller (`OSC_VCO_L_ArmUpDown` / `OSC_VCO_R_ArmUpDown`), masked
to that arm's body part only, driven by the tracker's
`VCO_L/R_ArmUpDown` float (**-1** = hanging, **0** = mid /
forward-horizontal, **+1** = straight up):

- The default **Neutral** state is EMPTY (no animation). The tracker sends
  **exactly 0.0 while the hand is untracked**, and the layer only leaves
  Neutral when the parameter escapes a ±0.02 deadband — so with no hand on
  camera the layer writes nothing and **the avatar's own idle/locomotion arm
  animation keeps playing**.
- The **Active** state is a Simple1D blend tree with anchor poses at
  -1 / 0 / +1; every anchor clip writes **all nine** of that arm's muscles
  (shoulder ×2, upper arm ×3, forearm ×2, wrist ×2 — the same per-group
  lesson as the head/fingers). Neutral ⇄ Active transitions cross-fade over
  0.25 s, so gaining/losing hand tracking raises/lowers the arm smoothly.
- The anchor pose values (hang: Arm Down-Up −0.5 with a slightly bent
  forearm; mid: reach forward-horizontal; up: Arm Down-Up 1, straight
  forearm) are **provisional** — tuned on paper, pending live verification
  in VRChat. In particular the `* Front-Back` muscle sign follows the
  "second word = +1" convention the verified head muscles use (so forward =
  negative), which issue #28 flags as the likely sign-flip candidate.
- No `VRCAnimatorTrackingControl`: Desktop arms are animation-driven (the
  avatar's own idle animation moving them proves it). If live testing shows
  IK stealing the arms after all, the fallback is the head-saga recipe
  (TrackingControl + ping-pong exit-time states).
- Like the head, the Gesture playable layer's **first-layer mask** must
  allow the arm body parts (stock `vrc_HandsOnly` denies them) — Apply
  extends its combined `OSC_GestureMask` accordingly (issue #25 behavior,
  now covering Head + arms).

### Why head pose lives on the Gesture layer, not FX

Head pose is wired into the avatar's **Gesture** playable layer, with a
`VRCAnimatorTrackingControl` state behaviour (`Head = Animation`, every other
tracked part `NoChange`) attached to the generated state — not the FX layer
the blend-shape parameters use. Two documented VRChat facts force this
(creators.vrchat.com/avatars/playable-layers/,
creators.vrchat.com/avatars/state-behaviors/):

- At avatar init, the FX playable layer's default mask **"disables all
  humanoid muscles"** — animating transforms/muscles in FX is officially not
  recommended, and a muscle-curve layer placed there is silently inert in the
  VRChat client even though it visibly rotates the head in the Unity editor.
  (This was a real bug in an earlier version of this wizard: head pose worked
  in the editor preview and did nothing in VRChat.)
- The Head bone is IK-driven on Desktop; an Animator layer only wins control
  of it if a `VRCAnimatorTrackingControl` state behaviour sets `Head =
  Animation` ("'Animation' will force that body part to respect values as
  given by the avatar's Animator"). The Gesture layer is VRChat's documented
  home for "animations that need to act on individual body parts while still
  playing the underlying animations for the rest of the body".

**Caveat**: while head-pose wiring is applied, VRChat's own head control
(e.g. Desktop mouse-look) no longer moves the avatar's head — the Animator
(driven by this app's OSC values) has full control instead. Click the wizard's
toggle to **Remove** to restore VRChat's native head control.

If the avatar still had VRChat's **default** Gesture layer (no controller of
its own), Apply creates a real one by copying the VRC SDK's stock
`vrc_AvatarV3HandsLayer` controller (from the SDK's optional "AV3 Demo
Assets" sample) as a starting point, so existing default hand gestures
(fist, point, etc.) aren't lost. If that stock asset isn't in your project
(the sample was never imported), a blank Gesture controller is created
instead and the Apply dialog says so — you'll need to set up hand gestures
manually in that case.

## Install

Distributed as a plain `Assets/`-folder drop-in (`.unitypackage`), not a UPM
package — copy `VRChatCameraOSC/` into your VRChat SDK3 avatar project's
`Assets/` folder (or double-click an exported `VRChatCameraOSC.unitypackage`
and import). No VCC/manifest.json editing needed.

To (re)generate the `.unitypackage` from this folder (end users don't need
the `Tests/` folder, so only `Editor/` is exported):

```bash
Unity -batchmode -quit -projectPath <a VRC SDK3 project> \
  -exportPackage Assets/VRChatCameraOSC/Editor VRChatCameraOSC.unitypackage
```

(`Assets/VRChatCameraOSC` must already exist in that project — copy this
folder in first.)

## Use

1. `VRChatCameraOSC > Avatar Setup Wizard` (menu bar).
2. Drag in your avatar's `VRCAvatarDescriptor`. Must be Humanoid. Selecting an
   avatar automatically runs a best-effort **auto-fill**: it picks the
   `SkinnedMeshRenderer` most likely to be the main face mesh (one named
   exactly "Body", else one containing "body", else whichever has the most
   blend shapes) and guesses each parameter's blend shape by common naming
   convention (VRM `Fcl_*`, `vrc.blink_left`-style, `BrowUp_L`/`_R`, plain
   English like `Smile`/`MouthOpen`). Re-run it anytime with the **"Auto-fill
   from Body mesh"** button — it only fills pickers still on `(skip)`, so it
   never overwrites a manual choice.
3. **Review every picker** — auto-fill is a guess, not a guarantee. For each
   expression parameter, confirm or change the `SkinnedMeshRenderer` + blend
   shape, or set it to `(skip)`. The `v2/EyeLid*` pickers take your **blink /
   eye-close** shape — the wizard generates the inverted mapping (shape at
   100 when the parameter reads 0/closed) for you. Left/Right pairs without
   L/R-split shapes on the mesh can both point at the same shared shape.
4. Leave "wire head pose" checked (default) to also drive the head bone (on
   the Gesture layer — see [above](#why-head-pose-lives-on-the-gesture-layer-not-fx)).
   Note the caveat: while this is wired, VRChat's own head control (e.g.
   Desktop mouse-look) no longer moves the avatar's head.
5. If your avatar has VRChat's native **Eyelids** (Eye Look → Eyelids, auto
   blink/eye-tracking) set to Blendshapes or Bones *and* you wired
   `v2/EyeLidLeft`/`v2/EyeLidRight`, a checkbox appears to disable it — leave
   it checked. Otherwise VRChat's own automatic blink independently
   opens/closes its own eyelid shape (often a *different* shape than the one
   driven by OSC, e.g. a shared `vrc.blink`) on its own timer, fighting with —
   and sometimes fully masking — the OSC-driven blink. Re-enable it yourself
   in Eye Look settings if you ever remove the OSC blink wiring.
6. The button at the bottom is a single **ON/OFF toggle** reflecting whether
   this avatar currently has all parameters wired:
   - **OFF → click → Apply**: removes any retired custom10 leftovers
     ([migration](#migrating-from-the-retired-custom10-setup)); creates (or
     reuses) the avatar's FX Animator Controller, Gesture Animator Controller
     (only if head pose is wired — see above), and Expression Parameters
     asset; adds the missing parameters (with the `v2/EyeLid*` 0.75
     defaults); and adds one `OSC_*` layer per wired parameter (blend shapes
     → FX, head pose → Gesture). Re-running replaces its own
     layers/parameters rather than duplicating them, so it's safe to click
     again after changing your picks.
   - **ON → click → Remove**: strips every parameter and `OSC_*` layer this
     wizard (any version) added from both the FX and Gesture controllers,
     cleanly (including their generated
     BlendTree/AnimationClip/StateMachineBehaviour sub-assets) — a full
     undo, restoring VRChat's native head control, without touching anything
     else on the avatar's controllers or Expression Parameters.
7. Verify **in VRChat** (upload or local test), not just in the editor —
   Humanoid muscle-driven additive layers and Animator direct-blend setups
   behave the same in-editor and in-client, but VRChat's own avatar
   validation/OSC pipeline is the real test.

## Troubleshooting

- **A chest/back-mounted VRCPhysBone (e.g. wings, a tail, hair) flails or
  spins while head tracking is running**: enable **"Is Animated"** on that
  PhysBone component. Per VRChat's own docs, a PhysBone in a bone chain that
  is also being animated (which the head-pose layer's Humanoid retargeting
  can touch, even restricted to the Head via the shared head-only
  `AvatarMask`) must have "Is Animated" on to respect that animation instead
  of fighting it.
- **Eyes stuck closed with the tracker off**: the avatar was probably built
  by a pre-#21 wizard (0-default eyelid params) or the `v2/EyeLid*`
  parameters lost their 0.75 default — re-run Apply and re-upload.
- **Eyes stuck closed / auto-blink fighting OSC blink while tracking**: see
  the Eyelids checkbox note in step 5 above.
- **Hand gestures stopped working after applying head pose**: the wizard
  only creates a new Gesture controller (copying the VRC SDK's stock hands
  layer) if the avatar still had VRChat's *default* Gesture layer. If Apply
  reported it couldn't find the stock `vrc_AvatarV3HandsLayer` asset, either
  import the VRC SDK's "AV3 Demo Assets" sample (Package Manager → VRChat SDK
  - Avatars → Samples) before re-applying, or add the standard hand-gesture
  layers to the generated `..._Gesture.controller` yourself.

## Design notes

- **Single source of truth for the parameters**:
  `VRChatCameraOSC/Editor/OscParameterSpec.cs` mirrors the Unified
  Expressions emission table in the Rust app's `src/mapping/unified.rs`
  (issues #14/#21). Keep the two in sync by hand — there's no automated
  check across the Rust/C# boundary.
- **Idempotent by construction**: every generated Animator layer is named
  `OSC_<ParamName>` (with `/` → `_` — Animator *parameters* keep the
  `v2/...` slash, but asset names must not contain path separators);
  re-running the wizard removes and replaces its own layers instead of
  appending duplicates. Same for Expression Parameters — merge only adds
  parameters that aren't already present by name.
- **`v2/MouthSmile*` vs `v2/MouthStretch*`**: deliberately separate signals
  in the Rust mapping (vertical corner lift vs. horizontal stretch) — wire
  them to different blend shapes if your avatar has both; it's fine to wire
  only one.

## Tests

`VRChatCameraOSC/Tests/Editor/` (Unity Test Framework, EditMode) covers the
Expression Parameters merge logic (including the `v2/EyeLid*` defaults and
custom10 migration) and the Animator layer builder (including the inverted
eyelid blend tree) against synthetic Humanoid/blend-shape data — no real
avatar asset required to run them. Not included in the exported
`.unitypackage` (end users don't need them); only export
`Assets/VRChatCameraOSC/Editor`.

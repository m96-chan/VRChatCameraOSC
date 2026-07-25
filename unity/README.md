# VRChatCameraOSC Avatar Setup (Unity SDK)

Editor wizard for VRChat SDK3 that wires a **Humanoid** avatar's existing
blend shapes and head bone to the 10 OSC parameters
[VRChatCameraOSC](../README.md) sends (issue [#16](https://github.com/m96-chan/VRChatCameraOSC/issues/16)).

## What this is — and isn't

VRChat's own OSC receiver already routes `/avatar/parameters/<Name>` into the
matching Animator parameter for any parameter your avatar declares in its VRC
Expression Parameters — that plumbing is built into VRChat, not something this
package reimplements. This is purely an **editor-time setup tool**: given a
Humanoid avatar, it generates/merges the Expression Parameters asset and the
FX/Gesture Animator Controller layers (blend shapes → FX, head pose →
Gesture — [why](#why-head-pose-lives-on-the-gesture-layer-not-fx)) so
VRChat's existing OSC-to-parameter routing has something to drive.

It does **not**:
- Create blend shapes — pick from ones your mesh already has.
- Run at avatar runtime. VRChat strips arbitrary `MonoBehaviour`s from
  uploaded avatars, so `HeadRoll`/`HeadYaw`/`HeadPitch` are driven the same
  way as the blend-shape parameters: through an **additive** Animator layer
  using Humanoid muscle curves (`Head Tilt Left-Right` / `Head Turn
  Left-Right` / `Head Nod Down-Up`), never a script on the avatar.
- Support non-Humanoid (generic) rigs — out of scope for now.

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

Avatars set up by an older version of this wizard (head pose in FX) are
migrated automatically: re-running Apply removes the old FX-based head
layers as it adds the new Gesture-based ones, and Remove cleans up both
locations regardless of which version applied them — so the wizard's
ON/OFF toggle reads correctly either way.

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
   expression parameter (`MouthOpen`, `EyeBlinkLeft`, `EyeBlinkRight`,
   `BrowUpLeft`, `BrowUpRight`, `MouthSmile`, `MouthWide`), confirm or change
   the `SkinnedMeshRenderer` + blend shape, or set it to `(skip)`. `MouthWide`
   is signed (`-1..1`): a positive shape (wide/grin) and an optional negative
   shape (pucker) — leaving the negative one on `(skip)` is fine if your
   avatar doesn't have one.
4. Leave "wire head pose" checked (default) to also drive the head bone (on
   the Gesture layer — see [above](#why-head-pose-lives-on-the-gesture-layer-not-fx)).
   Note the caveat: while this is wired, VRChat's own head control (e.g.
   Desktop mouse-look) no longer moves the avatar's head.
5. If your avatar has VRChat's native **Eyelids** (Eye Look → Eyelids, auto
   blink/eye-tracking) set to Blendshapes or Bones *and* you wired
   `EyeBlinkLeft`/`EyeBlinkRight`, a checkbox appears to disable it — leave it
   checked. Otherwise VRChat's own automatic blink independently opens/closes
   its own eyelid shape (often a *different* shape than the one driven by
   OSC, e.g. a shared `vrc.blink`) on its own timer, fighting with — and
   sometimes fully masking — the OSC-driven blink. This is exactly what
   caused a real report of "eyes stuck closed" despite `EyeBlinkLeft`/`Right`
   correctly reading near `0` (open) on the OSC side. Re-enable it yourself
   in Eye Look settings if you ever remove the OSC blink wiring.
6. The button at the bottom is a single **ON/OFF toggle** reflecting whether
   this avatar currently has all 10 parameters wired:
   - **OFF → click → Apply**: creates (or reuses) the avatar's FX Animator
     Controller, Gesture Animator Controller (only if head pose is wired —
     see above), and Expression Parameters asset; adds the missing
     parameters; and adds one `OSC_<ParamName>` layer per wired parameter
     (blend shapes → FX, head pose → Gesture). Re-running replaces its own
     layers/parameters rather than duplicating them, so it's safe to click
     again after changing your picks.
   - **ON → click → Remove**: strips every parameter and `OSC_*` layer this
     wizard added from both the FX and Gesture controllers, cleanly
     (including their generated BlendTree/AnimationClip/StateMachineBehaviour
     sub-assets) — a full undo, restoring VRChat's native head control,
     without touching anything else on the avatar's controllers or
     Expression Parameters.
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
- **Eyes stuck closed / auto-blink fighting OSC blink**: see the Eyelids
  checkbox note in step 5 above.
- **Hand gestures stopped working after applying head pose**: the wizard
  only creates a new Gesture controller (copying the VRC SDK's stock hands
  layer) if the avatar still had VRChat's *default* Gesture layer. If Apply
  reported it couldn't find the stock `vrc_AvatarV3HandsLayer` asset, either
  import the VRC SDK's "AV3 Demo Assets" sample (Package Manager → VRChat SDK
  - Avatars → Samples) before re-applying, or add the standard hand-gesture
  layers to the generated `..._Gesture.controller` yourself.

## Design notes

- **Single source of truth for the 10 parameters**:
  `VRChatCameraOSC/Editor/OscParameterSpec.cs` mirrors `PARAM_NAMES`/`PARAM_RANGES`
  in the Rust app's `src/mapping/mod.rs` (issue #14). Keep the two in sync by
  hand — there's no automated check across the Rust/C# boundary.
- **Idempotent by construction**: every generated Animator layer is named
  `OSC_<ParamName>`; re-running the wizard removes and replaces its own
  layers instead of appending duplicates. Same for Expression Parameters —
  merge only adds parameters that aren't already present by name.
- **`MouthSmile` vs `MouthWide`**: deliberately separate signals in the Rust
  mapping (vertical corner lift vs. horizontal stretch/pucker) — wire them to
  different blend shapes if your avatar has both; it's fine to wire only one.

## Tests

`VRChatCameraOSC/Tests/Editor/` (Unity Test Framework, EditMode) covers the
Expression Parameters merge logic and the Animator layer builder against
synthetic Humanoid/blend-shape data — no real avatar asset required to run
them. Not included in the exported `.unitypackage` (end users don't need
them); only export `Assets/VRChatCameraOSC/Editor`.

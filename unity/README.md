# VRChatCameraOSC Avatar Setup (Unity SDK)

Editor wizard for VRChat SDK3 that wires a **Humanoid** avatar's existing
blend shapes and head bone to the 10 OSC parameters
[VRChatCameraOSC](../README.md) sends (issue [#16](https://github.com/m96-chan/VRChatCameraOSC/issues/16)).

## What this is — and isn't

VRChat's own OSC receiver already routes `/avatar/parameters/<Name>` into the
matching Animator parameter for any parameter your avatar declares in its VRC
Expression Parameters — that plumbing is built into VRChat, not something this
package reimplements. This is purely an **editor-time setup tool**: given a
Humanoid avatar, it generates/merges the Expression Parameters asset and FX
Animator Controller layers so VRChat's existing OSC-to-parameter routing has
something to drive.

It does **not**:
- Create blend shapes — pick from ones your mesh already has.
- Run at avatar runtime. VRChat strips arbitrary `MonoBehaviour`s from
  uploaded avatars, so `HeadRoll`/`HeadYaw`/`HeadPitch` are driven the same
  way as the blend-shape parameters: through an **additive** FX layer using
  Humanoid muscle curves (`Head Tilt Left-Right` / `Head Turn Left-Right` /
  `Head Nod Down-Up`), never a script on the avatar.
- Support non-Humanoid (generic) rigs — out of scope for now.

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
4. Leave "wire head pose" checked (default) to also drive the head bone.
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
     Controller and Expression Parameters asset, adds the missing parameters,
     and adds one `OSC_<ParamName>` layer per wired parameter. Re-running
     replaces its own layers/parameters rather than duplicating them, so it's
     safe to click again after changing your picks.
   - **ON → click → Remove**: strips every parameter and `OSC_*` layer this
     wizard added, cleanly (including their generated BlendTree/AnimationClip
     sub-assets) — a full undo without touching anything else on the avatar's
     FX controller or Expression Parameters.
7. Verify **in VRChat** (upload or local test), not just in the editor —
   Humanoid muscle-driven additive layers and Animator direct-blend setups
   behave the same in-editor and in-client, but VRChat's own avatar
   validation/OSC pipeline is the real test.

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

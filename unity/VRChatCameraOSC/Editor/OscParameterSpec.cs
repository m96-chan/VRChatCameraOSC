using System.Collections.Generic;

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// What drives a given OSC parameter once wired into the Animator
    /// Controllers (issue #16).
    /// </summary>
    public enum OscParamKind
    {
        /// <summary>0..1, drives a single blend shape weight (0 at 0, 100 at 1).</summary>
        BlendShape,

        /// <summary>A VRCFT <c>v2/EyeLid*</c> eyelid parameter (0..1,
        /// <b>inverted</b> semantics: 0 = fully closed, ~0.75 = relaxed open,
        /// 1 = wide). Drives a *blink/close* blend shape: weight 100 at 0,
        /// weight 0 at <see cref="OscParameterSpec.EyeLidNeutral"/> and
        /// above. Declared with a non-zero default so the avatar's eyes are
        /// open when no tracker is running (issue #21).</summary>
        EyeLid,

        /// <summary>-1..1, drives the Humanoid head bone via an additive
        /// Gesture-layer (Humanoid muscle curves) — never a runtime script,
        /// see the package README for why.</summary>
        HeadPose,

        /// <summary>Int 0..7 on the standard VRChat gesture scale (0 Neutral,
        /// 1 Fist … 7 ThumbsUp), driving finger-muscle hand poses on the
        /// Gesture layer (issue #8). Declared as an <b>Int</b> expression
        /// parameter, not Float. No blend shape to pick — muscle-driven, so
        /// the wizard's picker UI skips it entirely. Declared only in the
        /// <see cref="HandMode.Gestures"/> hand mode — mutually exclusive
        /// with <see cref="FingerCurl"/> (both drive the same Fingers
        /// muscle group; two layers masking one group fight, issue #27).</summary>
        GestureInt,

        /// <summary>Float 0..1 per-finger curl (issue #8 phase 3): 0 =
        /// straight, 1 = fully curled. Drives that finger's three
        /// "Stretched" joint muscles on the Gesture layer via ONE per-hand
        /// nested blend tree. Declared only in the
        /// <see cref="HandMode.FingerCurls"/> hand mode — mutually
        /// exclusive with <see cref="GestureInt"/>.</summary>
        FingerCurl,

        /// <summary>Float arm-direction/elbow channel (issue #28 phase 2):
        /// <c>VCO_*_ArmUpDown</c> (-1 hanging .. 0 horizontal-or-at-camera ..
        /// +1 overhead) and <c>VCO_*_ArmAcross</c> (+1 across the chest ..
        /// -1 out away from the body) feed the per-arm 2D blend tree;
        /// <c>VCO_*_Elbow</c> (0 straight .. 1 fully bent) feeds the nested
        /// elbow dimension. The tracker decays all of them to exactly 0.0
        /// while the arm is untracked. Declared only when the wizard's arm
        /// toggle is on.</summary>
        ArmFloat,

        /// <summary>Bool <c>VCO_*_ArmTracked</c> (issue #28 phase 2): true
        /// while that arm is tracked — the arm layer's state-machine gate.
        /// Replaces the phase-1 ±0.02 deadband-on-UpDown trick, which
        /// collided with "arm pointing at the camera" legitimately reading
        /// (0,0). Declared as a <b>Bool</b> expression parameter, default
        /// false. Declared only when the wizard's arm toggle is on.</summary>
        ArmTracked,
    }

    /// <summary>
    /// How the wizard wires webcam hand tracking (issue #8). The two "on"
    /// modes are mutually exclusive by construction: gesture poses and
    /// per-finger curls both write the same per-hand Fingers muscle group,
    /// and Unity/VRChat compose humanoid Override layers per masked muscle
    /// GROUP (issue #27) — two layers on one group fight, last-one-wins.
    /// Applying one mode removes the other's layers and declarations.
    /// </summary>
    public enum HandMode
    {
        /// <summary>No hand layers; no hand parameters declared.</summary>
        Off,

        /// <summary>VCO_GestureLeft/Right Int 0-7 pose layers (default).</summary>
        Gestures,

        /// <summary>VCO_*Curl Float per-finger curl layers (10 floats =
        /// 80 expression-parameter bits — noticeably more than the two
        /// gesture ints).</summary>
        FingerCurls,
    }

    /// <summary>One OSC parameter this app sends, and how an avatar should react.</summary>
    public readonly struct OscParamSpec
    {
        public readonly string Name;
        public readonly float Min;
        public readonly float Max;
        /// <summary>Declared VRC Expression Parameter default — the value the
        /// avatar rests at when no tracker is sending (eyes open, face neutral).</summary>
        public readonly float DefaultValue;
        /// <summary>Parameter value at which the driven blend shape reaches
        /// weight 100 (BlendShape kind only; values above it clamp at 100).
        /// Defaults to <see cref="Max"/>. Below-Max values rescale
        /// avatar-side for channels whose webcam-tracked VRCFT values never
        /// approach 1.0 — measured live for the brows (issue #23: a
        /// deliberate raise peaks at ~0.3–0.5 on the wire).</summary>
        public readonly float FullScale;
        /// <summary>Optional parameters are only declared (and only cost VRC
        /// expression-parameter bits) when the user actually wires a blend
        /// shape to them — the core set is always declared (issue #24).</summary>
        public readonly bool Optional;
        public readonly OscParamKind Kind;

        public OscParamSpec(
            string name, float min, float max, float defaultValue, OscParamKind kind,
            float fullScale = 0f, bool optional = false)
        {
            Name = name;
            Min = min;
            Max = max;
            DefaultValue = defaultValue;
            FullScale = fullScale > 0f ? fullScale : max;
            Optional = optional;
            Kind = kind;
        }
    }

    /// <summary>
    /// Single source of truth for the OSC parameters this wizard wires —
    /// since issue #21 these are standard <b>VRCFT Unified Expressions</b>
    /// <c>v2/*</c> parameters (the subset the tracker drives well from a
    /// webcam), so a wizard-made avatar is a "lite" face-tracking avatar
    /// that also works with VRCFaceTracking itself. Mirrors the emission
    /// table in <c>src/mapping/unified.rs</c>; keep the two in sync by hand
    /// — there is no automated check across the Rust/C# boundary.
    /// </summary>
    public static class OscParameterSpec
    {
        /// <summary>VRCFT neutral eyelid value: <c>v2/EyeLid*</c> reads ~0.75
        /// with a relaxed open eye (0 = closed, 1 = wide). Used both as the
        /// declared parameter default and as the blend-tree threshold where
        /// the blink shape reaches weight 0.</summary>
        public const float EyeLidNeutral = 0.75f;

        public static readonly IReadOnlyList<OscParamSpec> All = new[]
        {
            new OscParamSpec("v2/EyeLidLeft", 0f, 1f, EyeLidNeutral, OscParamKind.EyeLid),
            new OscParamSpec("v2/EyeLidRight", 0f, 1f, EyeLidNeutral, OscParamKind.EyeLid),
            new OscParamSpec("v2/BrowUpLeft", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f),
            new OscParamSpec("v2/BrowUpRight", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f),
            new OscParamSpec("v2/JawOpen", 0f, 1f, 0f, OscParamKind.BlendShape),
            new OscParamSpec("v2/MouthSmileLeft", 0f, 1f, 0f, OscParamKind.BlendShape),
            new OscParamSpec("v2/MouthSmileRight", 0f, 1f, 0f, OscParamKind.BlendShape),
            new OscParamSpec("v2/MouthStretchLeft", 0f, 1f, 0f, OscParamKind.BlendShape),
            new OscParamSpec("v2/MouthStretchRight", 0f, 1f, 0f, OscParamKind.BlendShape),
            new OscParamSpec("v2/Head/Yaw", -1f, 1f, 0f, OscParamKind.HeadPose),
            new OscParamSpec("v2/Head/Pitch", -1f, 1f, 0f, OscParamKind.HeadPose),
            new OscParamSpec("v2/Head/Roll", -1f, 1f, 0f, OscParamKind.HeadPose),
            // Hand gestures (issue #8): custom Int transport for the standard
            // VRChat gesture index — the native GestureLeft/Right addresses
            // are read-only over OSC (vrchat-community/osc#42), so the
            // tracker sends these instead. Values keep the standard 0-7
            // scale (0 Neutral, 1 Fist, 2 HandOpen, 3 FingerPoint, 4 Victory,
            // 5 RockNRoll, 6 HandGun, 7 ThumbsUp). Declared only in the
            // Gestures hand mode (mode-gated, not core).
            new OscParamSpec("VCO_GestureLeft", 0f, 7f, 0f, OscParamKind.GestureInt),
            new OscParamSpec("VCO_GestureRight", 0f, 7f, 0f, OscParamKind.GestureInt),
            // Per-finger curls (issue #8 phase 3): 0 = straight, 1 = fully
            // curled. Declared only in the FingerCurls hand mode — 10 floats
            // cost 80 expression-parameter bits, so they must never ride
            // along in gestures mode (declare-only-when-used, issue #24).
            new OscParamSpec("VCO_L_ThumbCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_L_IndexCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_L_MiddleCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_L_RingCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_L_LittleCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_R_ThumbCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_R_IndexCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_R_MiddleCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_R_RingCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            new OscParamSpec("VCO_R_LittleCurl", 0f, 1f, 0f, OscParamKind.FingerCurl),
            // Full-arm tracking (issue #28 phase 2, MediaPipe Pose): per arm
            // one Bool gate + three direction/elbow floats. ArmTracked is
            // the state-machine gate (true while the arm is tracked) — the
            // phase-1 deadband-on-UpDown trick is retired because "arm
            // pointing at the camera" legitimately reads (0,0). UpDown is
            // the upper-arm direction cosine (-1 hanging, 0 horizontal OR at
            // the camera, +1 overhead); Across is +1 toward the opposite
            // shoulder / -1 out away from the body; Elbow is 0 straight ..
            // 1 fully bent. Untracked: the tracker sends ArmTracked=false
            // and decays the floats to exactly 0.0. Declared only when the
            // wizard's arm toggle is on.
            new OscParamSpec("VCO_L_ArmTracked", 0f, 1f, 0f, OscParamKind.ArmTracked),
            new OscParamSpec("VCO_R_ArmTracked", 0f, 1f, 0f, OscParamKind.ArmTracked),
            new OscParamSpec("VCO_L_ArmUpDown", -1f, 1f, 0f, OscParamKind.ArmFloat),
            new OscParamSpec("VCO_R_ArmUpDown", -1f, 1f, 0f, OscParamKind.ArmFloat),
            new OscParamSpec("VCO_L_ArmAcross", -1f, 1f, 0f, OscParamKind.ArmFloat),
            new OscParamSpec("VCO_R_ArmAcross", -1f, 1f, 0f, OscParamKind.ArmFloat),
            new OscParamSpec("VCO_L_Elbow", 0f, 1f, 0f, OscParamKind.ArmFloat),
            new OscParamSpec("VCO_R_Elbow", 0f, 1f, 0f, OscParamKind.ArmFloat),
            // ---- Optional extras (issue #24): declared only when wired, so
            // they cost expression-parameter bits only on avatars that have
            // the shapes. All are ARKit-52-drivable and already emitted by
            // the tracker (src/mapping/unified.rs). ----
            // FullScale < 1 on the channels a webcam demonstrably
            // under-drives (brows measured 0.3–0.5 peak, issue #23; cheek
            // puff reported invisible live, issue #24) — provisional values
            // by analogy, tune against captures as reports come in. Pucker
            // and funnel are strong ARKit channels; they keep full range.
            new OscParamSpec("v2/CheekPuffLeft", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.4f, optional: true),
            new OscParamSpec("v2/CheekPuffRight", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.4f, optional: true),
            new OscParamSpec("v2/JawLeft", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f, optional: true),
            new OscParamSpec("v2/JawRight", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f, optional: true),
            new OscParamSpec("v2/LipPuckerUpperLeft", 0f, 1f, 0f, OscParamKind.BlendShape, optional: true),
            new OscParamSpec("v2/LipPuckerUpperRight", 0f, 1f, 0f, OscParamKind.BlendShape, optional: true),
            new OscParamSpec("v2/LipFunnelUpperLeft", 0f, 1f, 0f, OscParamKind.BlendShape, optional: true),
            new OscParamSpec("v2/LipFunnelUpperRight", 0f, 1f, 0f, OscParamKind.BlendShape, optional: true),
            new OscParamSpec("v2/MouthFrownLeft", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f, optional: true),
            new OscParamSpec("v2/MouthFrownRight", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f, optional: true),
            new OscParamSpec("v2/NoseSneerLeft", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f, optional: true),
            new OscParamSpec("v2/NoseSneerRight", 0f, 1f, 0f, OscParamKind.BlendShape, fullScale: 0.5f, optional: true),
        };

        /// <summary>Kinds whose declaration follows a wizard mode/toggle
        /// (hand mode, arm toggle) instead of being always-core: hand
        /// gestures vs. per-finger curls are mutually exclusive, and every
        /// unused declaration costs expression-parameter bits (issue #24's
        /// declare-only-when-used philosophy).</summary>
        public static bool IsModeGated(OscParamKind kind)
        {
            return kind == OscParamKind.GestureInt
                || kind == OscParamKind.FingerCurl
                || kind == OscParamKind.ArmFloat
                || kind == OscParamKind.ArmTracked;
        }

        /// <summary>The always-declared subset: not Optional (issue #24
        /// wired-only extras) and not mode-gated (hand/arm declarations
        /// follow the wizard's selected mode/toggles).</summary>
        public static System.Collections.Generic.IEnumerable<OscParamSpec> Core
        {
            get
            {
                foreach (var s in All)
                {
                    if (!s.Optional && !IsModeGated(s.Kind))
                    {
                        yield return s;
                    }
                }
            }
        }

        /// <summary>The retired custom10 parameter names (pre-issue-#21).
        /// Applying or removing the wizard also cleans these off an avatar
        /// that was set up by an older version, so re-running the wizard
        /// migrates it instead of leaving dead parameters behind.</summary>
        public static readonly IReadOnlyList<string> LegacyNames = new[]
        {
            "MouthOpen",
            "EyeBlinkLeft",
            "EyeBlinkRight",
            "BrowUpLeft",
            "BrowUpRight",
            "MouthSmile",
            "MouthWide",
            "HeadRoll",
            "HeadYaw",
            "HeadPitch",
        };
    }
}

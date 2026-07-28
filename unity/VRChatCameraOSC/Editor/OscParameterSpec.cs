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
        public readonly OscParamKind Kind;

        public OscParamSpec(
            string name, float min, float max, float defaultValue, OscParamKind kind, float fullScale = 0f)
        {
            Name = name;
            Min = min;
            Max = max;
            DefaultValue = defaultValue;
            FullScale = fullScale > 0f ? fullScale : max;
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
        };

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

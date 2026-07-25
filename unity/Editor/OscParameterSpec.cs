using System.Collections.Generic;

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// What drives a given OSC parameter once wired into the FX Animator
    /// Controller (issue #16).
    /// </summary>
    public enum OscParamKind
    {
        /// <summary>0..1, drives a single blend shape weight (0 at 0, 100 at 1).</summary>
        BlendShape,

        /// <summary>-1..1, drives up to two blend shapes: one for the positive
        /// half of the range, one (optional) for the negative half.</summary>
        SignedBlendShape,

        /// <summary>-1..1, drives the Humanoid head bone via an additive FX
        /// layer (Humanoid muscle curves) — never a runtime script, see the
        /// package README for why.</summary>
        HeadPose,
    }

    /// <summary>One OSC parameter this app sends, and how an avatar should react.</summary>
    public readonly struct OscParamSpec
    {
        public readonly string Name;
        public readonly float Min;
        public readonly float Max;
        public readonly OscParamKind Kind;

        public OscParamSpec(string name, float min, float max, OscParamKind kind)
        {
            Name = name;
            Min = min;
            Max = max;
            Kind = kind;
        }
    }

    /// <summary>
    /// Single source of truth for the 10 OSC parameters VRChatCameraOSC sends —
    /// mirrors <c>PARAM_NAMES</c>/<c>PARAM_RANGES</c> in <c>src/mapping/mod.rs</c>
    /// (issue #14). Keep these two in sync by hand; there is no automated check
    /// across the Rust/C# boundary.
    /// </summary>
    public static class OscParameterSpec
    {
        public static readonly IReadOnlyList<OscParamSpec> All = new[]
        {
            new OscParamSpec("MouthOpen", 0f, 1f, OscParamKind.BlendShape),
            new OscParamSpec("EyeBlinkLeft", 0f, 1f, OscParamKind.BlendShape),
            new OscParamSpec("EyeBlinkRight", 0f, 1f, OscParamKind.BlendShape),
            new OscParamSpec("BrowUpLeft", 0f, 1f, OscParamKind.BlendShape),
            new OscParamSpec("BrowUpRight", 0f, 1f, OscParamKind.BlendShape),
            new OscParamSpec("MouthSmile", 0f, 1f, OscParamKind.BlendShape),
            new OscParamSpec("MouthWide", -1f, 1f, OscParamKind.SignedBlendShape),
            new OscParamSpec("HeadRoll", -1f, 1f, OscParamKind.HeadPose),
            new OscParamSpec("HeadYaw", -1f, 1f, OscParamKind.HeadPose),
            new OscParamSpec("HeadPitch", -1f, 1f, OscParamKind.HeadPose),
        };
    }
}

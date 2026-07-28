using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// Best-effort blend shape auto-suggestion (issue #16 follow-up): most
    /// avatars have a single main face mesh (conventionally named "Body")
    /// with blend shapes following one of a handful of common naming
    /// schemes (VRM's <c>Fcl_*</c>, CATS/community <c>*_L</c>/<c>*_R</c>,
    /// or plain English). This guesses reasonable defaults so the wizard
    /// isn't 100% manual — it always just *pre-fills* the picker, which
    /// stays fully editable/overridable before Apply.
    /// </summary>
    public static class BlendShapeAutoMatcher
    {
        /// <summary>Keyword substrings tried in order (first match wins) per
        /// single-shape parameter, matched against a normalised (lowercase,
        /// spaces/dashes -> underscore) blend shape name.</summary>
        static readonly Dictionary<string, string[]> Keywords = new Dictionary<string, string[]>
        {
            ["v2/JawOpen"] = new[] { "mouth_open", "mouthopen", "open_mouth", "jaw_open", "jawopen", "mouth_a", "fcl_mth_a", "vrc.mouthopen" },
            ["v2/EyeLidLeft"] = new[] { "blink_l", "blinkleft", "eye_close_l", "eyeclose_l", "wink_l", "fcl_eye_close_l" },
            ["v2/EyeLidRight"] = new[] { "blink_r", "blinkright", "eye_close_r", "eyeclose_r", "wink_r", "fcl_eye_close_r" },
            ["v2/BrowUpLeft"] = new[] { "brow_up_l", "browup_l", "browupleft", "fcl_brw_up_l" },
            ["v2/BrowUpRight"] = new[] { "brow_up_r", "browup_r", "browupright", "fcl_brw_up_r" },
            ["v2/MouthSmileLeft"] = new[] { "smile_l", "mouth_smile_l", "smileleft" },
            ["v2/MouthSmileRight"] = new[] { "smile_r", "mouth_smile_r", "smileright" },
            ["v2/MouthStretchLeft"] = new[] { "wide_l", "mouth_wide_l", "stretch_l" },
            ["v2/MouthStretchRight"] = new[] { "wide_r", "mouth_wide_r", "stretch_r" },
            // ---- Optional extras (issue #24) ----
            ["v2/CheekPuffLeft"] = new[] { "cheek_puff_l", "puff_l" },
            ["v2/CheekPuffRight"] = new[] { "cheek_puff_r", "puff_r" },
            ["v2/JawLeft"] = new[] { "jaw_left", "jaw_l", "mouth_left", "mouthleft", "口左" },
            ["v2/JawRight"] = new[] { "jaw_right", "jaw_r", "mouth_right", "mouthright", "口右" },
            ["v2/MouthFrownLeft"] = new[] { "frown_l", "mouth_down_l", "sad_l" },
            ["v2/MouthFrownRight"] = new[] { "frown_r", "mouth_down_r", "sad_r" },
            ["v2/NoseSneerLeft"] = new[] { "sneer_l", "nose_sneer_l" },
            ["v2/NoseSneerRight"] = new[] { "sneer_r", "nose_sneer_r" },
            // Pseudo-params: the optional eye-wide picker on the v2/EyeLid*
            // layers (no OSC parameter of their own — issue #24).
            ["EyeWideLeft"] = new[] { "eye_wide_l", "eyewide_l", "wide_eye_l" },
            ["EyeWideRight"] = new[] { "eye_wide_r", "eyewide_r", "wide_eye_r" },
        };

        /// <summary>Fallback keywords for parameters that often only exist as
        /// one shared (non-L/R-split) shape on simpler avatars — both sides
        /// then drive the same shared shape.</summary>
        static readonly Dictionary<string, string[]> SharedFallback = new Dictionary<string, string[]>
        {
            ["v2/BrowUpLeft"] = new[] { "brow_up", "browup", "fcl_brw_up" },
            ["v2/BrowUpRight"] = new[] { "brow_up", "browup", "fcl_brw_up" },
            ["v2/MouthSmileLeft"] = new[] { "smile", "mouth_smile", "fcl_mth_fun", "happy" },
            ["v2/MouthSmileRight"] = new[] { "smile", "mouth_smile", "fcl_mth_fun", "happy" },
            ["v2/MouthStretchLeft"] = new[] { "wide", "mouth_wide", "grin" },
            ["v2/MouthStretchRight"] = new[] { "wide", "mouth_wide", "grin" },
            ["v2/CheekPuffLeft"] = new[] { "cheek_puff", "cheekpuff", "puff", "fcl_che_fung", "ほっぺ", "頬" },
            ["v2/CheekPuffRight"] = new[] { "cheek_puff", "cheekpuff", "puff", "fcl_che_fung", "ほっぺ", "頬" },
            ["v2/LipPuckerUpperLeft"] = new[] { "pucker", "kiss", "mouth_u", "fcl_mth_u", "う" },
            ["v2/LipPuckerUpperRight"] = new[] { "pucker", "kiss", "mouth_u", "fcl_mth_u", "う" },
            ["v2/LipFunnelUpperLeft"] = new[] { "funnel", "mouth_o", "fcl_mth_o", "お" },
            ["v2/LipFunnelUpperRight"] = new[] { "funnel", "mouth_o", "fcl_mth_o", "お" },
            ["v2/MouthFrownLeft"] = new[] { "frown", "mouth_down", "fcl_mth_down", "sad", "悲" },
            ["v2/MouthFrownRight"] = new[] { "frown", "mouth_down", "fcl_mth_down", "sad", "悲" },
            ["v2/NoseSneerLeft"] = new[] { "sneer", "nose_sneer" },
            ["v2/NoseSneerRight"] = new[] { "sneer", "nose_sneer" },
            ["EyeWideLeft"] = new[] { "eye_wide", "eyewide", "surprised", "びっくり", "見開" },
            ["EyeWideRight"] = new[] { "eye_wide", "eyewide", "surprised", "びっくり", "見開" },
        };

        /// <summary>
        /// Face-region guard (issue #16 live-test regression): generic
        /// keywords like "smile"/"wide" happily matched *eye* shapes
        /// (`eye_smile_1`, `eyelid_inner_wide`) for *mouth* parameters,
        /// leaving the avatar's eyes permanently half-closed with real
        /// blinks stacking on top. A candidate whose name clearly belongs
        /// to another region is rejected outright — better to pre-fill
        /// `(skip)` than to wire the wrong region.
        /// </summary>
        static readonly string[] MouthForbidden = { "eye", "lid", "blink", "brow", "mayu", "cheek", "hoho", "nose" };
        static readonly string[] EyeForbidden = { "mouth", "kuchi", "lip", "brow", "mayu", "cheek", "hoho" };
        static readonly string[] BrowForbidden = { "mouth", "kuchi", "lip", "blink", "cheek", "hoho" };

        static readonly string[] CheekForbidden = { "eye", "lid", "blink", "brow", "mayu", "mouth", "kuchi", "lip", "nose" };
        static readonly string[] NoseForbidden = { "eye", "lid", "blink", "brow", "mayu", "mouth", "kuchi", "lip", "cheek", "hoho" };

        static string[] ForbiddenFor(string paramName)
        {
            if (paramName.StartsWith("v2/Mouth", StringComparison.Ordinal) ||
                paramName.StartsWith("v2/Jaw", StringComparison.Ordinal) ||
                paramName.StartsWith("v2/Lip", StringComparison.Ordinal))
            {
                return MouthForbidden;
            }
            if (paramName.StartsWith("v2/EyeLid", StringComparison.Ordinal) ||
                paramName.StartsWith("EyeWide", StringComparison.Ordinal))
            {
                return EyeForbidden;
            }
            if (paramName.StartsWith("v2/BrowUp", StringComparison.Ordinal))
            {
                return BrowForbidden;
            }
            if (paramName.StartsWith("v2/CheekPuff", StringComparison.Ordinal))
            {
                return CheekForbidden;
            }
            if (paramName.StartsWith("v2/NoseSneer", StringComparison.Ordinal))
            {
                return NoseForbidden;
            }
            return Array.Empty<string>();
        }

        /// <summary>
        /// Picks the renderer most likely to be the main face mesh: one
        /// named exactly "Body" (case-insensitive), else one whose name
        /// contains "body", else the renderer with the most blend shapes.
        /// </summary>
        public static SkinnedMeshRenderer FindBodyRenderer(IReadOnlyList<SkinnedMeshRenderer> renderers)
        {
            if (renderers == null || renderers.Count == 0)
            {
                return null;
            }
            return renderers.FirstOrDefault(r => string.Equals(r.name, "Body", StringComparison.OrdinalIgnoreCase))
                ?? renderers.FirstOrDefault(r => r.name.IndexOf("body", StringComparison.OrdinalIgnoreCase) >= 0)
                ?? renderers.OrderByDescending(r => r.sharedMesh != null ? r.sharedMesh.blendShapeCount : 0).First();
        }

        /// <summary>First blend shape on <paramref name="renderer"/> whose
        /// (normalised) name contains any of <paramref name="keywords"/>, in
        /// keyword order. Null if none match or the renderer has no mesh.</summary>
        public static string FindBlendShape(SkinnedMeshRenderer renderer, IReadOnlyList<string> keywords)
        {
            return FindBlendShape(renderer, keywords, Array.Empty<string>());
        }

        /// <summary>As <see cref="FindBlendShape(SkinnedMeshRenderer, IReadOnlyList{string})"/>,
        /// but a candidate whose normalised name contains any
        /// <paramref name="forbidden"/> substring is skipped (face-region
        /// guard — see <see cref="MouthForbidden"/>).</summary>
        public static string FindBlendShape(
            SkinnedMeshRenderer renderer,
            IReadOnlyList<string> keywords,
            IReadOnlyList<string> forbidden)
        {
            if (renderer == null || renderer.sharedMesh == null || keywords == null)
            {
                return null;
            }
            var mesh = renderer.sharedMesh;
            var names = Enumerable.Range(0, mesh.blendShapeCount)
                .Select(mesh.GetBlendShapeName)
                .Where(n =>
                {
                    var norm = Normalise(n);
                    return forbidden == null || !forbidden.Any(f => norm.Contains(f));
                })
                .ToArray();

            foreach (var keyword in keywords)
            {
                var match = names.FirstOrDefault(n => Normalise(n).Contains(keyword));
                if (match != null)
                {
                    return match;
                }
            }
            return null;
        }

        /// <summary>Best-guess blend shape for a single-shape OSC parameter
        /// (its own keywords, then the shared-shape fallback if any), with
        /// the parameter's face-region guard applied.</summary>
        public static string FindBlendShapeForParam(SkinnedMeshRenderer renderer, string paramName)
        {
            var forbidden = ForbiddenFor(paramName);
            string result = null;
            if (Keywords.TryGetValue(paramName, out var keywords))
            {
                result = FindBlendShape(renderer, keywords, forbidden);
            }
            // Some params (e.g. pucker/funnel) only have shared, non-L/R
            // keywords — the fallback must apply even with no primary entry.
            if (result == null && SharedFallback.TryGetValue(paramName, out var fallback))
            {
                result = FindBlendShape(renderer, fallback, forbidden);
            }
            return result;
        }

        static string Normalise(string name) => name.ToLowerInvariant().Replace(' ', '_').Replace('-', '_');
    }
}

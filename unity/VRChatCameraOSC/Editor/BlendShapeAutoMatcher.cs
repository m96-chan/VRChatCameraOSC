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
            ["MouthOpen"] = new[] { "mouth_open", "mouthopen", "open_mouth", "mouth_a", "fcl_mth_a", "vrc.mouthopen" },
            ["EyeBlinkLeft"] = new[] { "blink_l", "blinkleft", "eye_close_l", "eyeclose_l", "wink_l", "fcl_eye_close_l" },
            ["EyeBlinkRight"] = new[] { "blink_r", "blinkright", "eye_close_r", "eyeclose_r", "wink_r", "fcl_eye_close_r" },
            ["BrowUpLeft"] = new[] { "brow_up_l", "browup_l", "browupleft", "fcl_brw_up_l" },
            ["BrowUpRight"] = new[] { "brow_up_r", "browup_r", "browupright", "fcl_brw_up_r" },
            ["MouthSmile"] = new[] { "smile", "mouth_smile", "fcl_mth_fun", "happy" },
        };

        /// <summary>Fallback keywords for parameters that often only exist as
        /// one shared (non-L/R-split) shape on simpler avatars.</summary>
        static readonly Dictionary<string, string[]> SharedFallback = new Dictionary<string, string[]>
        {
            ["BrowUpLeft"] = new[] { "brow_up", "browup", "fcl_brw_up" },
            ["BrowUpRight"] = new[] { "brow_up", "browup", "fcl_brw_up" },
        };

        public static readonly string[] MouthWidePositiveKeywords = { "wide", "mouth_wide", "grin" };
        public static readonly string[] MouthWideNegativeKeywords = { "pucker", "mouth_pucker", "kiss", "duck" };

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

        static string[] ForbiddenFor(string paramName)
        {
            if (paramName.StartsWith("Mouth", StringComparison.Ordinal))
            {
                return MouthForbidden;
            }
            if (paramName.StartsWith("EyeBlink", StringComparison.Ordinal))
            {
                return EyeForbidden;
            }
            if (paramName.StartsWith("BrowUp", StringComparison.Ordinal))
            {
                return BrowForbidden;
            }
            return Array.Empty<string>();
        }

        /// <summary>Mouth-region-guarded pick for MouthWide's positive
        /// (wide/grin) shape — see <see cref="MouthWidePositiveKeywords"/>.</summary>
        public static string FindMouthWidePositive(SkinnedMeshRenderer renderer)
        {
            return FindBlendShape(renderer, MouthWidePositiveKeywords, MouthForbidden);
        }

        /// <summary>Mouth-region-guarded pick for MouthWide's negative
        /// (pucker/kiss) shape — see <see cref="MouthWideNegativeKeywords"/>.</summary>
        public static string FindMouthWideNegative(SkinnedMeshRenderer renderer)
        {
            return FindBlendShape(renderer, MouthWideNegativeKeywords, MouthForbidden);
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
            if (!Keywords.TryGetValue(paramName, out var keywords))
            {
                return null;
            }
            var forbidden = ForbiddenFor(paramName);
            return FindBlendShape(renderer, keywords, forbidden)
                ?? (SharedFallback.TryGetValue(paramName, out var fallback)
                    ? FindBlendShape(renderer, fallback, forbidden)
                    : null);
        }

        static string Normalise(string name) => name.ToLowerInvariant().Replace(' ', '_').Replace('-', '_');
    }
}

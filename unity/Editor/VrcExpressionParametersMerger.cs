using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// Adds the VRChatCameraOSC Float parameters to a
    /// <see cref="VRCExpressionParameters"/> asset, skipping ones already
    /// present by name so re-running the wizard is idempotent.
    /// </summary>
    public static class VrcExpressionParametersMerger
    {
        /// <returns>How many parameters were newly added.</returns>
        public static int Merge(VRCExpressionParameters asset, IEnumerable<OscParamSpec> specs)
        {
            var existing = new HashSet<string>(
                (asset.parameters ?? new VRCExpressionParameters.Parameter[0])
                    .Where(p => p != null)
                    .Select(p => p.name));

            var toAdd = specs.Where(s => !existing.Contains(s.Name)).ToList();
            if (toAdd.Count == 0)
            {
                return 0;
            }

            Undo.RecordObject(asset, "Add VRChatCameraOSC Expression Parameters");

            var merged = (asset.parameters ?? new VRCExpressionParameters.Parameter[0]).ToList();
            foreach (var spec in toAdd)
            {
                merged.Add(new VRCExpressionParameters.Parameter
                {
                    name = spec.Name,
                    valueType = VRCExpressionParameters.ValueType.Float,
                    defaultValue = 0f,
                    saved = false,
                    networkSynced = true,
                });
            }
            asset.parameters = merged.ToArray();

            EditorUtility.SetDirty(asset);
            return toAdd.Count;
        }

        /// <summary>
        /// Removes the named VRChatCameraOSC parameters (the "OFF" side of
        /// the wizard's apply/remove toggle) — the inverse of
        /// <see cref="Merge"/>. Leaves any other parameter on the asset
        /// untouched.
        /// </summary>
        /// <returns>How many parameters were removed.</returns>
        public static int Remove(VRCExpressionParameters asset, IEnumerable<OscParamSpec> specs)
        {
            var names = new HashSet<string>(specs.Select(s => s.Name));
            var kept = (asset.parameters ?? new VRCExpressionParameters.Parameter[0])
                .Where(p => p == null || !names.Contains(p.name))
                .ToArray();
            var removed = (asset.parameters?.Length ?? 0) - kept.Length;
            if (removed == 0)
            {
                return 0;
            }

            Undo.RecordObject(asset, "Remove VRChatCameraOSC Expression Parameters");
            asset.parameters = kept;
            EditorUtility.SetDirty(asset);
            return removed;
        }

        /// <summary>Whether every VRChatCameraOSC parameter is already present.</summary>
        public static bool IsFullyWired(VRCExpressionParameters asset, IEnumerable<OscParamSpec> specs)
        {
            if (asset == null)
            {
                return false;
            }
            var existing = new HashSet<string>(
                (asset.parameters ?? new VRCExpressionParameters.Parameter[0])
                    .Where(p => p != null)
                    .Select(p => p.name));
            return specs.All(s => existing.Contains(s.Name));
        }
    }
}

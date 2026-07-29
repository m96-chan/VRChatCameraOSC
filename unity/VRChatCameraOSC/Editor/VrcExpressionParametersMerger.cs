using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
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
            return Merge(asset, specs, binaryFace: false, binaryBits: 0);
        }

        /// <summary>
        /// As above, with the issue #29 binary-face mode: when
        /// <paramref name="binaryFace"/> is on, each face float spec
        /// (blend shapes / eyelids / head pose) declares as its VRCFT binary
        /// Bool group (<c>&lt;Name&gt;1/2/4/8...</c> + <c>&lt;Name&gt;Negative</c>
        /// for the signed head axes) instead of one 8-bit Float — 4 bits per
        /// face param instead of 8, the 256-bit-budget relief.
        /// </summary>
        /// <returns>How many parameters were newly added.</returns>
        public static int Merge(VRCExpressionParameters asset, IEnumerable<OscParamSpec> specs, bool binaryFace, int binaryBits)
        {
            var existing = new HashSet<string>(
                (asset.parameters ?? new VRCExpressionParameters.Parameter[0])
                    .Where(p => p != null)
                    .Select(p => p.name));

            var toAdd = new List<VRCExpressionParameters.Parameter>();
            foreach (var spec in specs)
            {
                if (binaryFace && OscParameterSpec.IsFace(spec.Kind))
                {
                    toAdd.AddRange(BinaryParameters(spec, binaryBits).Where(p => !existing.Contains(p.name)));
                }
                else if (!existing.Contains(spec.Name))
                {
                    toAdd.Add(new VRCExpressionParameters.Parameter
                    {
                        name = spec.Name,
                        // Gesture ints (issue #8) are the standard 0-7 VRChat
                        // gesture scale — an Int parameter, so the Animator's
                        // Equals-conditioned pose transitions can key off exact
                        // values. VCO_*_ArmTracked (issue #28 phase 2) is a Bool
                        // — the arm layer's If/IfNot state gate. Everything else
                        // this wizard declares is Float.
                        valueType = spec.Kind == OscParamKind.GestureInt
                            ? VRCExpressionParameters.ValueType.Int
                            : spec.Kind == OscParamKind.ArmTracked
                                ? VRCExpressionParameters.ValueType.Bool
                                : VRCExpressionParameters.ValueType.Float,
                        // Non-zero for v2/EyeLid* (0.75 = relaxed open, VRCFT
                        // semantics): with no tracker sending, the avatar must
                        // rest with open eyes, not shut ones (issue #21).
                        defaultValue = spec.DefaultValue,
                        saved = false,
                        networkSynced = true,
                    });
                }
            }
            if (toAdd.Count == 0)
            {
                return 0;
            }

            Undo.RecordObject(asset, "Add VRChatCameraOSC Expression Parameters");
            asset.parameters = (asset.parameters ?? new VRCExpressionParameters.Parameter[0])
                .Concat(toAdd)
                .ToArray();

            EditorUtility.SetDirty(asset);
            return toAdd.Count;
        }

        /// <summary>The binary Bool declarations of one face float (issue
        /// #29): one Bool per bit named <c>&lt;Name&gt;&lt;2^k&gt;</c>, plus
        /// <c>&lt;Name&gt;Negative</c> for the signed head axes. Bit
        /// defaults encode the spec's rest value on the DECODER's scale
        /// (round(default × (2^N−1)) — the closest decodable step): the
        /// v2/EyeLid* 0.75 neutral becomes 1011b at 4 bits (decodes to
        /// 11/15 ≈ 0.733), so the avatar still rests with open eyes when no
        /// tracker is running (issue #21's regression guard, binary
        /// edition). Everything else rests at all-bits-clear = 0.</summary>
        static IEnumerable<VRCExpressionParameters.Parameter> BinaryParameters(OscParamSpec spec, int binaryBits)
        {
            var steps = (1 << binaryBits) - 1;
            var rest = Mathf.Clamp(Mathf.RoundToInt(spec.DefaultValue * steps), 0, steps);
            for (var k = 0; k < binaryBits; k++)
            {
                yield return new VRCExpressionParameters.Parameter
                {
                    name = spec.Name + (1 << k),
                    valueType = VRCExpressionParameters.ValueType.Bool,
                    defaultValue = ((rest >> k) & 1) == 1 ? 1f : 0f,
                    saved = false,
                    networkSynced = true,
                };
            }
            if (OscParameterSpec.IsSigned(spec))
            {
                yield return new VRCExpressionParameters.Parameter
                {
                    name = spec.Name + "Negative",
                    valueType = VRCExpressionParameters.ValueType.Bool,
                    defaultValue = 0f,
                    saved = false,
                    networkSynced = true,
                };
            }
        }

        /// <summary>
        /// One-call declaration sync for Apply (issue #29): merges the
        /// current declaration set (in the current float/binary form) and
        /// removes every OTHER wizard-owned name — stale float declarations
        /// when switching to binary, stale bit Bools when switching back or
        /// changing the resolution, and deselected optional/mode-gated
        /// params. Foreign (avatar-own) parameters are never touched.
        /// </summary>
        /// <returns>How many parameters were newly added.</returns>
        public static int SyncDeclarations(
            VRCExpressionParameters asset, IEnumerable<OscParamSpec> specs, bool binaryFace, int binaryBits)
        {
            var specList = specs.ToList();
            var added = Merge(asset, specList, binaryFace, binaryBits);
            var current = new HashSet<string>(
                OscParameterSpec.DeclarationNames(specList, binaryFace, binaryBits));
            RemoveByName(asset, OscParameterSpec.AllPossibleNames().Where(n => !current.Contains(n)));
            return added;
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
            return RemoveByName(asset, specs.Select(s => s.Name));
        }

        /// <summary>
        /// Removes parameters by bare name — used to clean the retired
        /// custom10 parameter names (<see cref="OscParameterSpec.LegacyNames"/>)
        /// off an avatar set up by a pre-issue-#21 wizard, so re-applying
        /// migrates instead of accumulating dead parameters.
        /// </summary>
        /// <returns>How many parameters were removed.</returns>
        public static int RemoveByName(VRCExpressionParameters asset, IEnumerable<string> names)
        {
            var nameSet = new HashSet<string>(names);
            var kept = (asset.parameters ?? new VRCExpressionParameters.Parameter[0])
                .Where(p => p == null || !nameSet.Contains(p.name))
                .ToArray();
            var removed = (asset.parameters?.Length ?? 0) - kept.Length;
            if (removed == 0)
            {
                return 0;
            }

            Undo.RecordObject(asset, "Remove legacy VRChatCameraOSC Expression Parameters");
            asset.parameters = kept;
            EditorUtility.SetDirty(asset);
            return removed;
        }

        /// <summary>Synced bits the avatar's OWN parameters (everything not
        /// named in <see cref="OscParameterSpec.All"/>) occupy in VRChat's
        /// 256-bit budget — Bool 1, Int/Float 8, non-synced 0. The wizard's
        /// projected total is these plus the to-declare set's cost (issue
        /// #28 rollout: the full set no longer fits every avatar).</summary>
        public static int ForeignSyncedBits(VRCExpressionParameters asset)
        {
            if (asset == null)
            {
                return 0;
            }
            // Owned names in ANY mode — including the binary bit Bools
            // (issue #29), which must not count as "the avatar's own" when
            // the wizard re-opens on an already-binarized avatar.
            var ours = new HashSet<string>(OscParameterSpec.AllPossibleNames());
            var total = 0;
            foreach (var p in asset.parameters ?? new VRCExpressionParameters.Parameter[0])
            {
                if (p == null || !p.networkSynced || ours.Contains(p.name))
                {
                    continue;
                }
                total += p.valueType == VRCExpressionParameters.ValueType.Bool ? 1 : 8;
            }
            return total;
        }

        /// <summary>Whether every VRChatCameraOSC parameter is already present.</summary>
        public static bool IsFullyWired(VRCExpressionParameters asset, IEnumerable<OscParamSpec> specs)
        {
            return IsFullyWiredNames(asset, specs.Select(s => s.Name));
        }

        /// <summary>Name-level variant: with the binary toggle on, "wired"
        /// means the bit Bool names are present, not the float names — pass
        /// <see cref="OscParameterSpec.DeclarationNames"/> (issue #29).</summary>
        public static bool IsFullyWiredNames(VRCExpressionParameters asset, IEnumerable<string> names)
        {
            if (asset == null)
            {
                return false;
            }
            var existing = new HashSet<string>(
                (asset.parameters ?? new VRCExpressionParameters.Parameter[0])
                    .Where(p => p != null)
                    .Select(p => p.name));
            return names.All(existing.Contains);
        }
    }
}

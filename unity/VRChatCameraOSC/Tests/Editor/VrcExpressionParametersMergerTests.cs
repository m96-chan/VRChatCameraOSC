using System.Linq;
using NUnit.Framework;
using UnityEngine;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    public class VrcExpressionParametersMergerTests
    {
        [Test]
        public void Merge_AddsAllParameters_WhenAssetEmpty()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            var added = VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);

            Assert.AreEqual(OscParameterSpec.All.Count, added);
            Assert.AreEqual(OscParameterSpec.All.Count, asset.parameters.Length);
            CollectionAssert.AreEquivalent(
                OscParameterSpec.All.Select(s => s.Name).ToArray(),
                asset.parameters.Select(p => p.name).ToArray());
            Assert.IsTrue(asset.parameters.All(p => p.valueType == VRCExpressionParameters.ValueType.Float));

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void Merge_DeclaresEyeLidDefaultsAtNeutralOpen_SoEyesAreOpenWithoutATracker()
        {
            // The issue #21 regression this design fixes: v2/EyeLid* semantics
            // are 0 = closed, so a 0 default leaves the avatar's eyes shut
            // whenever no tracker is sending. The declared default must be
            // the VRCFT neutral (0.75); everything else rests at 0.
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);

            foreach (var p in asset.parameters)
            {
                var expected = p.name.StartsWith("v2/EyeLid") ? OscParameterSpec.EyeLidNeutral : 0f;
                Assert.AreEqual(expected, p.defaultValue, 1e-6f, $"default of {p.name}");
            }

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void Merge_IsIdempotent_SkipsExistingByNameAndNeverDuplicates()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new[]
            {
                new VRCExpressionParameters.Parameter
                {
                    name = "v2/JawOpen",
                    valueType = VRCExpressionParameters.ValueType.Float,
                },
            };

            var firstPass = VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);
            Assert.AreEqual(OscParameterSpec.All.Count - 1, firstPass, "should skip the pre-existing v2/JawOpen");
            Assert.AreEqual(OscParameterSpec.All.Count, asset.parameters.Length);

            var secondPass = VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);
            Assert.AreEqual(0, secondPass, "re-running should add nothing new");
            Assert.AreEqual(OscParameterSpec.All.Count, asset.parameters.Length, "must not duplicate");

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void IsFullyWired_ReflectsWhetherAllArePresent()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            Assert.IsFalse(VrcExpressionParametersMerger.IsFullyWired(asset, OscParameterSpec.All));
            Assert.IsFalse(VrcExpressionParametersMerger.IsFullyWired(null, OscParameterSpec.All));

            VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);
            Assert.IsTrue(VrcExpressionParametersMerger.IsFullyWired(asset, OscParameterSpec.All));

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void Remove_IsTheInverseOfMerge_LeavesUnrelatedParametersAlone()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new[]
            {
                new VRCExpressionParameters.Parameter { name = "SomeOtherToggle", valueType = VRCExpressionParameters.ValueType.Bool },
            };

            VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);
            Assert.AreEqual(OscParameterSpec.All.Count + 1, asset.parameters.Length);

            var removed = VrcExpressionParametersMerger.Remove(asset, OscParameterSpec.All);
            Assert.AreEqual(OscParameterSpec.All.Count, removed);
            Assert.AreEqual(1, asset.parameters.Length);
            Assert.AreEqual("SomeOtherToggle", asset.parameters[0].name);
            Assert.IsFalse(VrcExpressionParametersMerger.IsFullyWired(asset, OscParameterSpec.All));

            // Removing again is a harmless no-op.
            Assert.AreEqual(0, VrcExpressionParametersMerger.Remove(asset, OscParameterSpec.All));

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void RemoveByName_StripsLegacyCustom10Parameters_LeavesTheRestAlone()
        {
            // Migration path (issue #21): an avatar set up by the pre-#21
            // wizard carries the retired custom10 names; re-applying must
            // clean them off rather than accumulate dead parameters.
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new[]
            {
                new VRCExpressionParameters.Parameter { name = "MouthOpen", valueType = VRCExpressionParameters.ValueType.Float },
                new VRCExpressionParameters.Parameter { name = "EyeBlinkLeft", valueType = VRCExpressionParameters.ValueType.Float },
                new VRCExpressionParameters.Parameter { name = "SomeOtherToggle", valueType = VRCExpressionParameters.ValueType.Bool },
            };

            var removed = VrcExpressionParametersMerger.RemoveByName(asset, OscParameterSpec.LegacyNames);

            Assert.AreEqual(2, removed);
            Assert.AreEqual(1, asset.parameters.Length);
            Assert.AreEqual("SomeOtherToggle", asset.parameters[0].name);

            Object.DestroyImmediate(asset);
        }
    }
}

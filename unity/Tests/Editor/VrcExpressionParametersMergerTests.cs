using System.Linq;
using NUnit.Framework;
using UnityEngine;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    public class VrcExpressionParametersMergerTests
    {
        [Test]
        public void Merge_AddsAllTenParameters_WhenAssetEmpty()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            var added = VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);

            Assert.AreEqual(10, added);
            Assert.AreEqual(10, asset.parameters.Length);
            CollectionAssert.AreEquivalent(
                OscParameterSpec.All.Select(s => s.Name).ToArray(),
                asset.parameters.Select(p => p.name).ToArray());
            Assert.IsTrue(asset.parameters.All(p => p.valueType == VRCExpressionParameters.ValueType.Float));

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
                    name = "MouthOpen",
                    valueType = VRCExpressionParameters.ValueType.Float,
                },
            };

            var firstPass = VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);
            Assert.AreEqual(9, firstPass, "should skip the pre-existing MouthOpen");
            Assert.AreEqual(10, asset.parameters.Length);

            var secondPass = VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.All);
            Assert.AreEqual(0, secondPass, "re-running should add nothing new");
            Assert.AreEqual(10, asset.parameters.Length, "must not duplicate");

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void IsFullyWired_ReflectsWhetherAllTenArePresent()
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
            Assert.AreEqual(11, asset.parameters.Length);

            var removed = VrcExpressionParametersMerger.Remove(asset, OscParameterSpec.All);
            Assert.AreEqual(10, removed);
            Assert.AreEqual(1, asset.parameters.Length);
            Assert.AreEqual("SomeOtherToggle", asset.parameters[0].name);
            Assert.IsFalse(VrcExpressionParametersMerger.IsFullyWired(asset, OscParameterSpec.All));

            // Removing again is a harmless no-op.
            Assert.AreEqual(0, VrcExpressionParametersMerger.Remove(asset, OscParameterSpec.All));

            Object.DestroyImmediate(asset);
        }
    }
}

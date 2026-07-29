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
            // Everything is Float except the hand-gesture ints (issue #8)
            // and the arm-tracked Bool gates (issue #28 phase 2).
            Assert.IsTrue(asset.parameters.All(p => p.valueType ==
                (p.name.StartsWith("VCO_Gesture")
                    ? VRCExpressionParameters.ValueType.Int
                    : p.name.EndsWith("ArmTracked")
                        ? VRCExpressionParameters.ValueType.Bool
                        : VRCExpressionParameters.ValueType.Float)));

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
        public void SpecsToDeclare_IncludesCoreAlways_OptionalsOnlyWhenWired()
        {
            // Issue #24: unwired optional extras must not be declared (they
            // would cost expression-parameter bits for nothing). With hands
            // off and arms off, exactly the Core set remains.
            var none = AvatarSetupWindow.SpecsToDeclare(new string[0], HandMode.Off, includeArms: false);
            CollectionAssert.AreEquivalent(
                OscParameterSpec.Core.Select(s => s.Name).ToArray(),
                none.Select(s => s.Name).ToArray());

            var withPuff = AvatarSetupWindow.SpecsToDeclare(
                new[] { "v2/CheekPuffLeft", "v2/JawOpen" }, HandMode.Off, includeArms: false);
            Assert.IsTrue(withPuff.Any(s => s.Name == "v2/CheekPuffLeft"));
            Assert.IsFalse(withPuff.Any(s => s.Name == "v2/CheekPuffRight"));
            Assert.AreEqual(OscParameterSpec.Core.Count() + 1, withPuff.Count);
        }

        [Test]
        public void SpecsToDeclare_HandAndArmDeclarationsFollowTheSelectedMode()
        {
            // Issue #8 phase 3 / issue #28: gesture ints only in Gestures
            // mode, curl floats only in FingerCurls mode (10 floats = 80
            // bits — never ride along unused), arm floats only with the arm
            // toggle on.
            var gestures = AvatarSetupWindow.SpecsToDeclare(new string[0], HandMode.Gestures, includeArms: false)
                .Select(s => s.Name).ToArray();
            Assert.Contains("VCO_GestureLeft", gestures);
            Assert.Contains("VCO_GestureRight", gestures);
            Assert.IsFalse(gestures.Any(n => n.EndsWith("Curl")), "no curl floats in gestures mode");
            Assert.IsFalse(gestures.Any(n => n.Contains("ArmUpDown")), "no arm floats with arms off");

            var curls = AvatarSetupWindow.SpecsToDeclare(new string[0], HandMode.FingerCurls, includeArms: false)
                .Select(s => s.Name).ToArray();
            Assert.AreEqual(10, curls.Count(n => n.EndsWith("Curl")), "all 10 curl floats in curls mode");
            foreach (var side in new[] { "L", "R" })
            {
                foreach (var finger in new[] { "Thumb", "Index", "Middle", "Ring", "Little" })
                {
                    Assert.Contains($"VCO_{side}_{finger}Curl", curls);
                }
            }
            Assert.IsFalse(curls.Any(n => n.StartsWith("VCO_Gesture")), "no gesture ints in curls mode");

            var arms = AvatarSetupWindow.SpecsToDeclare(new string[0], HandMode.Off, includeArms: true)
                .Select(s => s.Name).ToArray();
            foreach (var side in new[] { "L", "R" })
            {
                Assert.Contains($"VCO_{side}_ArmTracked", arms);
                Assert.Contains($"VCO_{side}_ArmUpDown", arms);
                Assert.Contains($"VCO_{side}_ArmAcross", arms);
                Assert.Contains($"VCO_{side}_Elbow", arms);
            }
            Assert.AreEqual(OscParameterSpec.Core.Count() + 8, arms.Length,
                "arm toggle declares 2 Bool gates + 6 floats (issue #28 phase 2)");
        }

        [Test]
        public void Merge_DeclaresArmAndCurlFloats_WithZeroDefaults()
        {
            // The arm floats' 0 default matters doubly: it is also the value
            // the tracker decays them to while the arm is untracked (issue
            // #28 phase 2 contract).
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            VrcExpressionParametersMerger.Merge(
                asset,
                OscParameterSpec.All.Where(s =>
                    s.Kind == OscParamKind.FingerCurl || s.Kind == OscParamKind.ArmFloat));

            Assert.AreEqual(16, asset.parameters.Length, "10 curls + 6 arm direction/elbow floats");
            foreach (var p in asset.parameters)
            {
                Assert.AreEqual(VRCExpressionParameters.ValueType.Float, p.valueType, p.name);
                Assert.AreEqual(0f, p.defaultValue, 1e-6f, p.name);
                Assert.IsTrue(p.networkSynced, p.name);
            }

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void Merge_DeclaresArmTrackedAsBool_DefaultFalse_NetworkSynced()
        {
            // Issue #28 phase 2: the ArmTracked gates must declare as Bool
            // (the Animator's If/IfNot conditions key off a Bool parameter)
            // resting false, so an avatar with no tracker keeps its arms on
            // idle animation.
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            VrcExpressionParametersMerger.Merge(
                asset, OscParameterSpec.All.Where(s => s.Kind == OscParamKind.ArmTracked));

            Assert.AreEqual(2, asset.parameters.Length);
            foreach (var p in asset.parameters)
            {
                Assert.IsTrue(p.name.EndsWith("ArmTracked"), p.name);
                Assert.AreEqual(VRCExpressionParameters.ValueType.Bool, p.valueType, p.name);
                Assert.AreEqual(0f, p.defaultValue, 1e-6f, $"{p.name} rests false");
                Assert.IsTrue(p.networkSynced, p.name);
            }

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

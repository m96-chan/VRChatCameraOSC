using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    /// <summary>
    /// Issue #29: VRCFT-style binary encoding of the face float parameters —
    /// declaration names/types/defaults, budget math, the Direct-Blend-Tree
    /// decode layer, and float ⇄ binary migration. The encoding convention
    /// (bit suffix = power of two, LSB first; quantize floor(v·2^N) with
    /// ≥0.99999 → all ones; Negative = sign Bool) is the exact
    /// BinaryBaseParameter contract the Rust tracker ports
    /// (src/mapping/avatar.rs) — these tests pin the wizard side of it.
    /// </summary>
    public class OscBinaryParameterTests
    {
        const string TestDir = "Assets/_VRChatCameraOscBinaryTests";

        AnimatorController _controller;

        static OscParamSpec SpecOf(string name) => OscParameterSpec.All.First(s => s.Name == name);

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscBinaryTests");
            }
            _controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Test.controller");
        }

        [TearDown]
        public void TearDown()
        {
            AssetDatabase.DeleteAsset(TestDir);
        }

        // ------------------------------------------------------------------
        // Naming + budget
        // ------------------------------------------------------------------

        [Test]
        public void BinaryNames_PowerOfTwoSuffixes_NegativeOnlyForSignedParams()
        {
            CollectionAssert.AreEqual(
                new[] { "v2/JawOpen1", "v2/JawOpen2", "v2/JawOpen4", "v2/JawOpen8" },
                OscParameterSpec.BinaryNames(SpecOf("v2/JawOpen"), 4).ToArray());
            CollectionAssert.AreEqual(
                new[]
                {
                    "v2/Head/Yaw1", "v2/Head/Yaw2", "v2/Head/Yaw4", "v2/Head/Yaw8",
                    "v2/Head/YawNegative",
                },
                OscParameterSpec.BinaryNames(SpecOf("v2/Head/Yaw"), 4).ToArray());
            // Resolution follows the setting: 8-bit runs the suffixes to 128.
            var eight = OscParameterSpec.BinaryNames(SpecOf("v2/JawOpen"), 8).ToArray();
            Assert.AreEqual(8, eight.Length);
            Assert.AreEqual("v2/JawOpen128", eight.Last());
        }

        [Test]
        public void IsFace_CoversBlendShapeEyeLidHeadPose_NothingElse()
        {
            // The binary toggle applies to the face floats only — gesture
            // ints, curls, and arm params keep their declarations (issue #29
            // scope).
            Assert.IsTrue(OscParameterSpec.IsFace(OscParamKind.BlendShape));
            Assert.IsTrue(OscParameterSpec.IsFace(OscParamKind.EyeLid));
            Assert.IsTrue(OscParameterSpec.IsFace(OscParamKind.HeadPose));
            Assert.IsFalse(OscParameterSpec.IsFace(OscParamKind.GestureInt));
            Assert.IsFalse(OscParameterSpec.IsFace(OscParamKind.FingerCurl));
            Assert.IsFalse(OscParameterSpec.IsFace(OscParamKind.ArmFloat));
            Assert.IsFalse(OscParameterSpec.IsFace(OscParamKind.ArmTracked));
        }

        [Test]
        public void BitCost_Binary_FaceFloatsCostBitsPlusNegative_OthersUnchanged()
        {
            // 4-bit: unsigned face float 4 bits, signed (head) 4+1; the
            // non-face kinds keep their normal costs.
            Assert.AreEqual(4, OscParameterSpec.BitCost(SpecOf("v2/JawOpen"), binaryFace: true, binaryBits: 4));
            Assert.AreEqual(4, OscParameterSpec.BitCost(SpecOf("v2/EyeLidLeft"), binaryFace: true, binaryBits: 4));
            Assert.AreEqual(5, OscParameterSpec.BitCost(SpecOf("v2/Head/Yaw"), binaryFace: true, binaryBits: 4));
            Assert.AreEqual(8, OscParameterSpec.BitCost(SpecOf("VCO_GestureLeft"), binaryFace: true, binaryBits: 4));
            Assert.AreEqual(1, OscParameterSpec.BitCost(SpecOf("VCO_L_ArmTracked"), binaryFace: true, binaryBits: 4));
            // Off = the plain costs.
            Assert.AreEqual(8, OscParameterSpec.BitCost(SpecOf("v2/JawOpen"), binaryFace: false, binaryBits: 4));

            // The core face set (9 unsigned + 3 signed head axes) at 4-bit:
            // 9×4 + 3×5 = 51 bits, down from 12×8 = 96.
            Assert.AreEqual(51, OscParameterSpec.BitCost(OscParameterSpec.Core, binaryFace: true, binaryBits: 4));
            Assert.AreEqual(96, OscParameterSpec.BitCost(OscParameterSpec.Core, binaryFace: false, binaryBits: 4));
        }

        [Test]
        public void BitCost_Binary_FullyWiredAvatarScenario_FitsCurlsAndArms()
        {
            // The issue #29 point, at the maximum wiring (core 12 + all 12
            // optionals = 24 face params): 4-bit binary = 21×4 + 3×5 = 99
            // bits (was 24×8 = 192), so face + gestures + arms = 165, and
            // even the curls variant (99 + 80 + 50 = 229) fits under 256
            // with room for avatar-own parameters — impossible with floats
            // (192 + 80 + 50 = 322).
            var allOptionalNames = OscParameterSpec.All
                .Where(s => s.Optional)
                .Select(s => s.Name)
                .ToArray();
            var gestures = AvatarSetupWindow.SpecsToDeclare(
                allOptionalNames, HandMode.Gestures, includeArms: true);
            Assert.AreEqual(99 + 16 + 50, OscParameterSpec.BitCost(gestures, binaryFace: true, binaryBits: 4));
            var curls = AvatarSetupWindow.SpecsToDeclare(
                allOptionalNames, HandMode.FingerCurls, includeArms: true);
            Assert.AreEqual(99 + 80 + 50, OscParameterSpec.BitCost(curls, binaryFace: true, binaryBits: 4));
        }

        // ------------------------------------------------------------------
        // Declarations
        // ------------------------------------------------------------------

        [Test]
        public void Merge_Binary4Bit_DeclaresBoolGroups_CountNamesTypesSynced()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            var added = VrcExpressionParametersMerger.Merge(
                asset, OscParameterSpec.Core, binaryFace: true, binaryBits: 4);

            // 9 unsigned × 4 bits + 3 head axes × (4 bits + Negative) = 51.
            Assert.AreEqual(51, added);
            Assert.AreEqual(51, asset.parameters.Length);
            CollectionAssert.AreEquivalent(
                OscParameterSpec.DeclarationNames(OscParameterSpec.Core, true, 4).ToArray(),
                asset.parameters.Select(p => p.name).ToArray());
            foreach (var p in asset.parameters)
            {
                Assert.AreEqual(VRCExpressionParameters.ValueType.Bool, p.valueType, p.name);
                Assert.IsTrue(p.networkSynced, p.name);
                Assert.IsFalse(p.saved, p.name);
            }
            // No float declaration of the face names must remain/appear.
            Assert.IsFalse(asset.parameters.Any(p => p.name == "v2/JawOpen"));
            Assert.IsFalse(asset.parameters.Any(p => p.name == "v2/Head/Yaw"));

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void Merge_Binary_EyeLidBitDefaultsEncodeNeutralOpen_EverythingElseZero()
        {
            // Binary edition of the issue #21 eyes-open-at-rest guard: the
            // 0.75 neutral encodes to round(0.75 × 15) = 11 = 1011b, i.e.
            // bits 1, 2 and 8 rest true (decodes to 11/15 ≈ 0.733 — the
            // closest 4-bit step to 0.75). All other bits (and every
            // Negative) rest false.
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];

            VrcExpressionParametersMerger.Merge(
                asset, OscParameterSpec.Core, binaryFace: true, binaryBits: 4);

            float DefaultOf(string name) => asset.parameters.First(p => p.name == name).defaultValue;
            foreach (var lid in new[] { "v2/EyeLidLeft", "v2/EyeLidRight" })
            {
                Assert.AreEqual(1f, DefaultOf(lid + "1"), 1e-6f, lid);
                Assert.AreEqual(1f, DefaultOf(lid + "2"), 1e-6f, lid);
                Assert.AreEqual(0f, DefaultOf(lid + "4"), 1e-6f, lid);
                Assert.AreEqual(1f, DefaultOf(lid + "8"), 1e-6f, lid);
            }
            foreach (var p in asset.parameters.Where(p => !p.name.StartsWith("v2/EyeLid")))
            {
                Assert.AreEqual(0f, p.defaultValue, 1e-6f, p.name);
            }

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void SyncDeclarations_FloatToBinaryAndBack_MigratesInPlace()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new[]
            {
                // Avatar-own parameter — must survive every migration.
                new VRCExpressionParameters.Parameter
                {
                    name = "Hair",
                    valueType = VRCExpressionParameters.ValueType.Bool,
                    networkSynced = true,
                },
            };
            var specs = AvatarSetupWindow.SpecsToDeclare(new string[0], HandMode.Gestures, includeArms: true);

            // Apply float mode, then re-Apply with binary on: the float
            // face declarations must be replaced by the bit Bools.
            VrcExpressionParametersMerger.SyncDeclarations(asset, specs, binaryFace: false, binaryBits: 4);
            Assert.IsTrue(asset.parameters.Any(p => p.name == "v2/JawOpen"));

            VrcExpressionParametersMerger.SyncDeclarations(asset, specs, binaryFace: true, binaryBits: 4);
            Assert.IsFalse(asset.parameters.Any(p => p.name == "v2/JawOpen"), "float face declaration must be dropped");
            Assert.IsTrue(asset.parameters.Any(p => p.name == "v2/JawOpen8"));
            Assert.IsTrue(asset.parameters.Any(p => p.name == "v2/Head/YawNegative"));
            // Non-face declarations ride along unchanged in both modes.
            Assert.IsTrue(asset.parameters.Any(p => p.name == "VCO_GestureLeft"));
            Assert.IsTrue(asset.parameters.Any(p => p.name == "VCO_L_ArmTracked"));

            // Resolution change sweeps the stale wider bits.
            VrcExpressionParametersMerger.SyncDeclarations(asset, specs, binaryFace: true, binaryBits: 6);
            Assert.IsTrue(asset.parameters.Any(p => p.name == "v2/JawOpen32"));
            VrcExpressionParametersMerger.SyncDeclarations(asset, specs, binaryFace: true, binaryBits: 4);
            Assert.IsFalse(asset.parameters.Any(p => p.name == "v2/JawOpen16"), "stale 6-bit params must be swept");
            Assert.IsFalse(asset.parameters.Any(p => p.name == "v2/JawOpen32"), "stale 6-bit params must be swept");

            // And back to float mode: bit Bools out, floats in.
            VrcExpressionParametersMerger.SyncDeclarations(asset, specs, binaryFace: false, binaryBits: 4);
            Assert.IsTrue(asset.parameters.Any(p => p.name == "v2/JawOpen"));
            Assert.IsFalse(asset.parameters.Any(p => p.name == "v2/JawOpen1"), "bit Bools must be dropped");
            Assert.IsFalse(asset.parameters.Any(p => p.name.EndsWith("Negative")));

            Assert.IsTrue(asset.parameters.Any(p => p.name == "Hair"), "avatar-own parameter must survive");
            Object.DestroyImmediate(asset);
        }

        [Test]
        public void ForeignSyncedBits_DoesNotCountOurBinaryBitBools()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new[]
            {
                new VRCExpressionParameters.Parameter
                {
                    name = "Hair",
                    valueType = VRCExpressionParameters.ValueType.Bool,
                    networkSynced = true,
                },
                new VRCExpressionParameters.Parameter
                {
                    name = "v2/JawOpen8",
                    valueType = VRCExpressionParameters.ValueType.Bool,
                    networkSynced = true,
                },
                new VRCExpressionParameters.Parameter
                {
                    name = "v2/Head/YawNegative",
                    valueType = VRCExpressionParameters.ValueType.Bool,
                    networkSynced = true,
                },
            };

            Assert.AreEqual(1, VrcExpressionParametersMerger.ForeignSyncedBits(asset),
                "our bit Bools must not count as the avatar's own");

            Object.DestroyImmediate(asset);
        }

        [Test]
        public void IsFullyWiredNames_TracksTheBinaryDeclarations()
        {
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];
            var binaryNames = OscParameterSpec.DeclarationNames(OscParameterSpec.Core, true, 4).ToArray();

            Assert.IsFalse(VrcExpressionParametersMerger.IsFullyWiredNames(asset, binaryNames));

            VrcExpressionParametersMerger.Merge(asset, OscParameterSpec.Core, binaryFace: true, binaryBits: 4);
            Assert.IsTrue(VrcExpressionParametersMerger.IsFullyWiredNames(asset, binaryNames));
            // ... and the float-mode names are NOT wired in binary mode.
            Assert.IsFalse(VrcExpressionParametersMerger.IsFullyWired(asset, OscParameterSpec.Core));

            Object.DestroyImmediate(asset);
        }

        // ------------------------------------------------------------------
        // Decode layer (Direct Blend Tree)
        // ------------------------------------------------------------------

        [Test]
        public void AddBinaryDecodeLayer_UnsignedParam_DirectTreeOfBitWeightedAapClips()
        {
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller, new[] { SpecOf("v2/JawOpen") }, binaryBits: 4);

            // VRChat casts the synced Bool onto a same-named Float animator
            // parameter (creators.vrchat.com/avatars/animator-parameters,
            // bool → float: true is 1.0) — the bit params must be Floats.
            foreach (var bit in new[] { "v2/JawOpen1", "v2/JawOpen2", "v2/JawOpen4", "v2/JawOpen8" })
            {
                Assert.IsTrue(
                    _controller.parameters.Any(p =>
                        p.name == bit && p.type == AnimatorControllerParameterType.Float),
                    bit);
            }
            // The decode target (base name) and the constant-1 weight.
            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "v2/JawOpen" && p.type == AnimatorControllerParameterType.Float));
            var one = _controller.parameters.Single(p => p.name == OscAnimatorLayerBuilder.ConstantOneParam);
            Assert.AreEqual(AnimatorControllerParameterType.Float, one.type);
            Assert.AreEqual(1f, one.defaultFloat, 1e-6f, "constant-1 weight must default to 1");

            var layer = _controller.layers.Single(l => l.name == "OSC_BinaryDecode");
            var state = layer.stateMachine.states.Single().state;
            Assert.IsTrue(state.writeDefaultValues, "DBT/AAP layers require write defaults ON");

            var root = (BlendTree)state.motion;
            Assert.AreEqual(BlendTreeType.Direct, root.blendType);
            Assert.AreEqual(4, root.children.Length);
            // One child per bit, weighted by that bit's Float, whose clip
            // writes the AAP to 2^k/(2^N−1) — the VRCFT decode weights
            // (2^4−1 = 15 denominator so all-bits-set decodes to exactly 1,
            // matching the encoder's ≥0.99999 → all-ones saturation).
            for (var k = 0; k < 4; k++)
            {
                var child = root.children[k];
                Assert.AreEqual("v2/JawOpen" + (1 << k), child.directBlendParameter);
                var clip = (AnimationClip)child.motion;
                var binding = AnimationUtility.GetCurveBindings(clip).Single();
                Assert.AreEqual(typeof(Animator), binding.type);
                Assert.AreEqual("v2/JawOpen", binding.propertyName);
                var value = AnimationUtility.GetEditorCurve(clip, binding).Evaluate(0f);
                Assert.AreEqual((1 << k) / 15f, value, 1e-6f, $"bit {1 << k} contribution");
            }
        }

        [Test]
        public void AddBinaryDecodeLayer_SignedHeadAxis_SignSelectTreeWithMirroredContributions()
        {
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller, new[] { SpecOf("v2/Head/Yaw") }, binaryBits: 4);

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "v2/Head/YawNegative" && p.type == AnimatorControllerParameterType.Float));

            var layer = _controller.layers.Single(l => l.name == "OSC_BinaryDecode");
            var root = (BlendTree)layer.stateMachine.states.Single().state.motion;
            Assert.AreEqual(BlendTreeType.Direct, root.blendType);
            // A signed param contributes ONE root child: the sign-select
            // Simple1D on the Negative float (0 = positive tree, 1 =
            // negative tree), always-on via the constant-1 direct weight.
            var signChild = root.children.Single();
            Assert.AreEqual(OscAnimatorLayerBuilder.ConstantOneParam, signChild.directBlendParameter);
            var signTree = (BlendTree)signChild.motion;
            Assert.AreEqual(BlendTreeType.Simple1D, signTree.blendType);
            Assert.AreEqual("v2/Head/YawNegative", signTree.blendParameter);
            Assert.IsFalse(signTree.useAutomaticThresholds);
            CollectionAssert.AreEqual(
                new[] { 0f, 1f }, signTree.children.Select(c => c.threshold).ToArray());

            // Positive branch sums +2^k/15, negative branch −2^k/15 —
            // magnitude bits + sign Bool, the BinaryBaseParameter contract.
            for (var branch = 0; branch < 2; branch++)
            {
                var sign = branch == 0 ? 1f : -1f;
                var tree = (BlendTree)signTree.children[branch].motion;
                Assert.AreEqual(BlendTreeType.Direct, tree.blendType);
                Assert.AreEqual(4, tree.children.Length);
                for (var k = 0; k < 4; k++)
                {
                    Assert.AreEqual("v2/Head/Yaw" + (1 << k), tree.children[k].directBlendParameter);
                    var clip = (AnimationClip)tree.children[k].motion;
                    var binding = AnimationUtility.GetCurveBindings(clip).Single();
                    Assert.AreEqual("v2/Head/Yaw", binding.propertyName);
                    Assert.AreEqual(
                        sign * (1 << k) / 15f,
                        AnimationUtility.GetEditorCurve(clip, binding).Evaluate(0f),
                        1e-6f);
                }
            }
        }

        [Test]
        public void AddBinaryDecodeLayer_EyeLidDrivingLayerIsUntouched_NeutralPreserved()
        {
            // The decode feeds the SAME float the existing eyelid layer
            // blends on — the inverted 0.75-neutral tree must be exactly the
            // float-mode construction (decoded 0 = closed at weight 100,
            // decoded ≈0.75 = open at weight 0).
            var avatarRoot = new GameObject("Avatar");
            try
            {
                var meshGo = new GameObject("Face");
                meshGo.transform.SetParent(avatarRoot.transform);
                var renderer = meshGo.AddComponent<SkinnedMeshRenderer>();
                var mesh = new Mesh
                {
                    vertices = new[] { Vector3.zero, Vector3.up, Vector3.right },
                    triangles = new[] { 0, 1, 2 },
                };
                mesh.AddBlendShapeFrame("Blink", 100f, new Vector3[3], null, null);
                renderer.sharedMesh = mesh;

                OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                    _controller, new[] { SpecOf("v2/EyeLidLeft") }, binaryBits: 4);
                OscAnimatorLayerBuilder.AddEyeLidLayer(
                    _controller, avatarRoot.transform, "v2/EyeLidLeft", renderer, "Blink");

                var lidLayer = _controller.layers.Single(l => l.name == "OSC_v2_EyeLidLeft");
                var tree = (BlendTree)lidLayer.stateMachine.states.Single().state.motion;
                Assert.AreEqual("v2/EyeLidLeft", tree.blendParameter, "driving layer reads the decoded AAP");
                Assert.AreEqual(0f, tree.children[0].threshold, 1e-6f);
                Assert.AreEqual(OscParameterSpec.EyeLidNeutral, tree.children[1].threshold, 1e-6f);
                StringAssert.EndsWith("_100", tree.children[0].motion.name);
                StringAssert.EndsWith("_0", tree.children[1].motion.name);
                // Exactly one v2/EyeLidLeft Float parameter — decode and
                // driving layer share it.
                Assert.AreEqual(1, _controller.parameters.Count(p => p.name == "v2/EyeLidLeft"));
            }
            finally
            {
                Object.DestroyImmediate(avatarRoot);
            }
        }

        [Test]
        public void AddBinaryDecodeLayer_ReRunning_ReplacesRatherThanDuplicates()
        {
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller, new[] { SpecOf("v2/JawOpen") }, binaryBits: 4);
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller, new[] { SpecOf("v2/JawOpen") }, binaryBits: 4);

            Assert.AreEqual(1, _controller.layers.Count(l => l.name == "OSC_BinaryDecode"));
            Assert.AreEqual(1, _controller.parameters.Count(p => p.name == "v2/JawOpen1"));

            // Re-running at a lower resolution drops the stale wider bits.
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller, new[] { SpecOf("v2/JawOpen") }, binaryBits: 6);
            Assert.IsTrue(_controller.parameters.Any(p => p.name == "v2/JawOpen32"));
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller, new[] { SpecOf("v2/JawOpen") }, binaryBits: 4);
            Assert.IsFalse(_controller.parameters.Any(p => p.name == "v2/JawOpen32"));
        }

        [Test]
        public void RemoveLayer_BinaryDecode_CleansBitParamsTreesAndClips_KeepsBaseFloats()
        {
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller,
                new[] { SpecOf("v2/JawOpen"), SpecOf("v2/Head/Yaw") },
                binaryBits: 4);

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(
                _controller, OscAnimatorLayerBuilder.BinaryDecodeKey));

            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(
                _controller, OscAnimatorLayerBuilder.BinaryDecodeKey));
            // Bit params, Negative, and the constant-1 weight are gone ...
            Assert.IsFalse(_controller.parameters.Any(p =>
                p.name == OscAnimatorLayerBuilder.ConstantOneParam
                || p.name.StartsWith("v2/JawOpen1")
                || p.name == "v2/Head/YawNegative"));
            // ... but the base floats stay — the driving layers own them
            // (their own RemoveLayer drops them, tested elsewhere).
            Assert.IsTrue(_controller.parameters.Any(p => p.name == "v2/JawOpen"));
            Assert.IsTrue(_controller.parameters.Any(p => p.name == "v2/Head/Yaw"));
            // No orphaned sub-assets.
            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller));
            Assert.IsFalse(assets.OfType<BlendTree>().Any(t => t.name.StartsWith("OSC_BinaryDecode")), "orphaned trees");
            Assert.IsFalse(assets.OfType<AnimationClip>().Any(c => c.name.StartsWith("OSC_BinaryDecode")), "orphaned clips");
        }

        [Test]
        public void AddBinaryDecodeLayer_WithNoSpecs_AddsNothing()
        {
            OscAnimatorLayerBuilder.AddBinaryDecodeLayer(
                _controller, System.Linq.Enumerable.Empty<OscParamSpec>(), binaryBits: 4);

            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(
                _controller, OscAnimatorLayerBuilder.BinaryDecodeKey));
            Assert.IsFalse(_controller.parameters.Any(p =>
                p.name == OscAnimatorLayerBuilder.ConstantOneParam));
        }
    }
}

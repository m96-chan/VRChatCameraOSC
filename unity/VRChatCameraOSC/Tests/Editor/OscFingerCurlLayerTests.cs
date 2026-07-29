using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    /// <summary>
    /// Issue #8 phase 3: per-finger curl floats (<c>VCO_*Curl</c>, 0 =
    /// straight, 1 = fully curled) driving ONE nested-blend-tree layer per
    /// hand (<see cref="OscAnimatorLayerBuilder.AddFingerCurlLayer"/>), and
    /// the mutual exclusion with the gesture-int pose layers
    /// (<see cref="OscAnimatorLayerBuilder.ApplyHandMode"/>) — both modes
    /// write the same Fingers muscle group, which Override layers fight
    /// over whole-group (issue #27).
    /// </summary>
    public class OscFingerCurlLayerTests
    {
        const string TestDir = "Assets/_VRChatCameraOscCurlTests";

        static readonly string[] Fingers = { "Thumb", "Index", "Middle", "Ring", "Little" };

        AnimatorController _controller;

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscCurlTests");
            }
            _controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Curls.controller");
        }

        [TearDown]
        public void TearDown()
        {
            AssetDatabase.DeleteAsset(TestDir);
        }

        [Test]
        public void Spec_DeclaresTenCurlFloats_ModeGatedNotCore()
        {
            foreach (var side in new[] { "L", "R" })
            {
                foreach (var finger in Fingers)
                {
                    var name = $"VCO_{side}_{finger}Curl";
                    var spec = OscParameterSpec.All.Single(s => s.Name == name);
                    Assert.AreEqual(OscParamKind.FingerCurl, spec.Kind, name);
                    Assert.AreEqual(0f, spec.Min, name);
                    Assert.AreEqual(1f, spec.Max, name);
                    Assert.AreEqual(0f, spec.DefaultValue, name);
                    Assert.IsTrue(OscParameterSpec.IsModeGated(spec.Kind), name);
                    Assert.IsFalse(OscParameterSpec.Core.Any(s => s.Name == name),
                        $"{name} must not be always-declared (80 bits for all 10)");
                }
            }
        }

        [Test]
        public void AddFingerCurlLayer_Left_OneOverrideLayer_FiveFloatParams_FingersMask()
        {
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: true);

            foreach (var p in OscAnimatorLayerBuilder.CurlParams(leftHand: true))
            {
                Assert.IsTrue(_controller.parameters.Any(x =>
                    x.name == p && x.type == AnimatorControllerParameterType.Float), p);
            }

            var layer = _controller.layers.Single(l => l.name == "OSC_VCO_L_FingerCurls");
            Assert.AreEqual(AnimatorLayerBlendingMode.Override, layer.blendingMode);
            Assert.AreEqual(1f, layer.defaultWeight, 1e-6f);

            Assert.IsNotNull(layer.avatarMask);
            for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
            {
                Assert.AreEqual(part == AvatarMaskBodyPart.LeftFingers,
                    layer.avatarMask.GetHumanoidBodyPartActive(part), $"mask, {part}");
            }

            // No tracking control: fingers aren't IK-held on Desktop.
            foreach (var child in layer.stateMachine.states)
            {
                Assert.IsEmpty(child.state.behaviours, child.state.name);
            }
        }

        [Test]
        public void AddFingerCurlLayer_NestedTree_Depth5_ThumbToLittle_Thresholds0And1()
        {
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: true);

            var layer = _controller.layers.Single(l => l.name == "OSC_VCO_L_FingerCurls");
            var motion = layer.stateMachine.defaultState.motion;

            // Walk the straight-most branch: each level must blend on that
            // finger's curl param with exactly two children at 0 and 1.
            var expectedParams = OscAnimatorLayerBuilder.CurlParams(leftHand: true);
            for (var depth = 0; depth < Fingers.Length; depth++)
            {
                var tree = (BlendTree)motion;
                Assert.AreEqual(expectedParams[depth], tree.blendParameter, $"depth {depth}");
                Assert.IsFalse(tree.useAutomaticThresholds, $"depth {depth}");
                Assert.AreEqual(2, tree.children.Length, $"depth {depth}");
                Assert.AreEqual(0f, tree.children[0].threshold, 1e-6f, $"depth {depth} straight slot");
                Assert.AreEqual(1f, tree.children[1].threshold, 1e-6f, $"depth {depth} curled slot");
                motion = tree.children[0].motion;
            }
            Assert.IsInstanceOf<AnimationClip>(motion, "the tree must bottom out in leaf clips at depth 5");
        }

        static IEnumerable<AnimationClip> Leaves(Motion motion)
        {
            if (motion is AnimationClip clip)
            {
                yield return clip;
            }
            else if (motion is BlendTree tree)
            {
                foreach (var child in tree.children)
                {
                    foreach (var leaf in Leaves(child.motion))
                    {
                        yield return leaf;
                    }
                }
            }
        }

        [Test]
        public void AddFingerCurlLayer_Has32Leaves_EachWritingAll20MusclesWithRealDuration()
        {
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: true);

            var root = _controller.layers.Single(l => l.name == "OSC_VCO_L_FingerCurls")
                .stateMachine.defaultState.motion;
            var leaves = Leaves(root).ToArray();
            Assert.AreEqual(32, leaves.Length, "2^5 straight/curled combinations");
            Assert.AreEqual(32, leaves.Distinct().Count(), "each combination gets its own clip");

            var expectedProps = OscAnimatorLayerBuilder.FingerMuscleProperties(true).ToArray();
            foreach (var leaf in leaves)
            {
                // Real length by convention (zero-length clips broke
                // exit-time machinery elsewhere, issue #25).
                Assert.Greater(leaf.length, 0.5f, leaf.name);
                var bound = AnimationUtility.GetCurveBindings(leaf).Select(b => b.propertyName).ToArray();
                CollectionAssert.AreEquivalent(expectedProps, bound,
                    $"{leaf.name}: every leaf must write ALL 20 finger muscles (issue #27 group lesson)");
            }
        }

        static float ValueOf(AnimationClip clip, string prop) => AnimationUtility.GetEditorCurve(
            clip,
            AnimationUtility.GetCurveBindings(clip).Single(b => b.propertyName == prop)).Evaluate(0f);

        [Test]
        public void AddFingerCurlLayer_LeafValues_CurledMinus1_StraightPlus1_SpreadsNeutral()
        {
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: true);

            var root = (BlendTree)_controller.layers.Single(l => l.name == "OSC_VCO_L_FingerCurls")
                .stateMachine.defaultState.motion;

            // Branch selector: child [c0][c1]... per finger, 0 = straight.
            AnimationClip Leaf(params int[] curled)
            {
                Motion m = root;
                foreach (var c in curled)
                {
                    m = ((BlendTree)m).children[c].motion;
                }
                return (AnimationClip)m;
            }

            // All straight: every stretch joint +1.
            var straight = Leaf(0, 0, 0, 0, 0);
            // All curled: every stretch joint -1.
            var fist = Leaf(1, 1, 1, 1, 1);
            // Only the index curled: index -1, everything else +1.
            var indexOnly = Leaf(0, 1, 0, 0, 0);

            foreach (var finger in Fingers)
            {
                foreach (var joint in new[] { "1", "2", "3" })
                {
                    var prop = $"LeftHand.{finger}.{joint} Stretched";
                    Assert.AreEqual(1f, ValueOf(straight, prop), 1e-6f, $"straight {prop}");
                    Assert.AreEqual(-1f, ValueOf(fist, prop), 1e-6f, $"fist {prop}");
                    Assert.AreEqual(finger == "Index" ? -1f : 1f, ValueOf(indexOnly, prop), 1e-6f,
                        $"index-only {prop}");
                }
                // Spreads ride at neutral 0 in every leaf — written (so the
                // whole-group override always carries them) but not driven.
                foreach (var leaf in new[] { straight, fist, indexOnly })
                {
                    Assert.AreEqual(0f, ValueOf(leaf, $"LeftHand.{finger}.Spread"), 1e-6f, leaf.name);
                }
            }
        }

        [Test]
        public void AddFingerCurlLayer_Right_UsesRightParamsAndMuscles()
        {
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: false);

            var layer = _controller.layers.Single(l => l.name == "OSC_VCO_R_FingerCurls");
            var root = (BlendTree)layer.stateMachine.defaultState.motion;
            Assert.AreEqual("VCO_R_ThumbCurl", root.blendParameter);

            var leaf = Leaves(root).First();
            var bound = AnimationUtility.GetCurveBindings(leaf).Select(b => b.propertyName).ToArray();
            Assert.IsTrue(bound.All(p => p.StartsWith("RightHand.")), "right layer must animate RightHand.* only");
            CollectionAssert.AreEquivalent(OscAnimatorLayerBuilder.FingerMuscleProperties(false).ToArray(), bound);

            for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
            {
                Assert.AreEqual(part == AvatarMaskBodyPart.RightFingers,
                    layer.avatarMask.GetHumanoidBodyPartActive(part), $"mask, {part}");
            }
        }

        [Test]
        public void AddFingerCurlLayer_ReApply_ReplacesWithoutDuplicatesOrLeakedSubAssets()
        {
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: true);
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: true);

            Assert.AreEqual(1, _controller.layers.Count(l => l.name == "OSC_VCO_L_FingerCurls"));
            Assert.AreEqual(5, _controller.parameters.Count(p => p.name.EndsWith("Curl")));

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.AreEqual(32, assets.OfType<AnimationClip>().Count(),
                "re-Apply must not leak orphaned leaf clips");
            Assert.AreEqual(31, assets.OfType<BlendTree>().Count(),
                "1+2+4+8+16 nested trees, no orphans");
            Assert.AreEqual(1, assets.OfType<AvatarMask>().Count(m => m.name == "OSC_LeftFingersMask"));
        }

        [Test]
        public void RemoveLayer_FingerCurlLayers_RoundTripToZero()
        {
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: true);
            OscAnimatorLayerBuilder.AddFingerCurlLayer(_controller, leftHand: false);

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(
                _controller, OscAnimatorLayerBuilder.FingerCurlLeftKey));
            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(
                _controller, OscAnimatorLayerBuilder.FingerCurlRightKey));

            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.FingerCurlLeftKey));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.FingerCurlRightKey));
            Assert.IsFalse(_controller.parameters.Any(p => p.name.EndsWith("Curl")),
                "all 10 curl float parameters must be removed too");

            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.IsFalse(assets.OfType<AnimationClip>().Any(), "orphaned leaf clips");
            Assert.IsFalse(assets.OfType<BlendTree>().Any(), "orphaned nested trees");
            Assert.IsFalse(assets.OfType<AvatarMask>().Any(m => m.name.StartsWith("OSC_")), "orphaned masks");
        }

        // ---- Mode exclusivity (the issue #27 lesson applied to hands):
        // gesture poses and per-finger curls both write the per-hand
        // Fingers muscle groups, so applying one mode must remove the other.

        [Test]
        public void ApplyHandMode_Curls_RemovesGestureLayers_AndViceVersa()
        {
            OscAnimatorLayerBuilder.ApplyHandMode(_controller, HandMode.Gestures);
            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "VCO_GestureLeft"));
            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "VCO_GestureRight"));

            OscAnimatorLayerBuilder.ApplyHandMode(_controller, HandMode.FingerCurls);
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "VCO_GestureLeft"),
                "curls mode must remove the gesture layers (same Fingers muscle group)");
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "VCO_GestureRight"));
            Assert.IsFalse(_controller.parameters.Any(p => p.name.StartsWith("VCO_Gesture")));
            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.FingerCurlLeftKey));
            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.FingerCurlRightKey));

            OscAnimatorLayerBuilder.ApplyHandMode(_controller, HandMode.Gestures);
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, OscAnimatorLayerBuilder.FingerCurlLeftKey),
                "gestures mode must remove the curl layers");
            Assert.IsFalse(_controller.parameters.Any(p => p.name.EndsWith("Curl")));
            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "VCO_GestureLeft"));
        }

        [Test]
        public void ApplyHandMode_Off_RemovesBothModesCompletely()
        {
            OscAnimatorLayerBuilder.ApplyHandMode(_controller, HandMode.Gestures);
            OscAnimatorLayerBuilder.ApplyHandMode(_controller, HandMode.FingerCurls);

            OscAnimatorLayerBuilder.ApplyHandMode(_controller, HandMode.Off);

            Assert.IsFalse(_controller.layers.Any(l => l.name.StartsWith("OSC_VCO_")));
            Assert.IsFalse(_controller.parameters.Any(p => p.name.StartsWith("VCO_")));
            var assets = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .Where(a => a != null).ToArray();
            Assert.IsFalse(assets.OfType<AnimationClip>().Any(), "orphaned clips");
            Assert.IsFalse(assets.OfType<AvatarMask>().Any(m => m.name.StartsWith("OSC_")), "orphaned masks");
        }
    }
}

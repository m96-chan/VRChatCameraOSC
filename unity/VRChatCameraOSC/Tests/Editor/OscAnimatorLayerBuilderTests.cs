using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using VRC.SDKBase;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    public class OscAnimatorLayerBuilderTests
    {
        const string TestDir = "Assets/_VRChatCameraOscTests";

        AnimatorController _controller;
        GameObject _avatarRoot;
        SkinnedMeshRenderer _renderer;

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscTests");
            }
            _controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Test.controller");

            _avatarRoot = new GameObject("Avatar");
            var meshGo = new GameObject("Face");
            meshGo.transform.SetParent(_avatarRoot.transform);
            _renderer = meshGo.AddComponent<SkinnedMeshRenderer>();

            var mesh = new Mesh();
            mesh.vertices = new[] { Vector3.zero, Vector3.up, Vector3.right };
            mesh.triangles = new[] { 0, 1, 2 };
            mesh.AddBlendShapeFrame("Smile", 100f, new Vector3[3], null, null);
            mesh.AddBlendShapeFrame("Pucker", 100f, new Vector3[3], null, null);
            _renderer.sharedMesh = mesh;
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_avatarRoot);
            AssetDatabase.DeleteAsset(TestDir);
        }

        [Test]
        public void AddBlendShapeLayer_AddsParameterAndLayer()
        {
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "MouthOpen", _renderer, "Smile");

            Assert.IsTrue(_controller.parameters.Any(p =>
                p.name == "MouthOpen" && p.type == AnimatorControllerParameterType.Float));
            Assert.IsTrue(_controller.layers.Any(l => l.name == "OSC_MouthOpen"));
        }

        [Test]
        public void AddBlendShapeLayer_ReRunning_ReplacesRatherThanDuplicates()
        {
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "MouthOpen", _renderer, "Smile");
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "MouthOpen", _renderer, "Smile");

            Assert.AreEqual(1, _controller.layers.Count(l => l.name == "OSC_MouthOpen"));
            Assert.AreEqual(1, _controller.parameters.Count(p => p.name == "MouthOpen"));
        }

        [Test]
        public void AddSignedBlendShapeLayer_WithBothShapes_AddsOneLayer()
        {
            OscAnimatorLayerBuilder.AddSignedBlendShapeLayer(
                _controller, _avatarRoot.transform, "MouthWide", _renderer, "Smile", _renderer, "Pucker");

            Assert.IsTrue(_controller.parameters.Any(p => p.name == "MouthWide"));
            Assert.IsTrue(_controller.layers.Any(l => l.name == "OSC_MouthWide"));
        }

        [Test]
        public void AddSignedBlendShapeLayer_NegativeShapeOptional_DoesNotThrow()
        {
            Assert.DoesNotThrow(() =>
                OscAnimatorLayerBuilder.AddSignedBlendShapeLayer(
                    _controller, _avatarRoot.transform, "MouthWide", _renderer, "Smile", null, null));

            Assert.IsTrue(_controller.layers.Any(l => l.name == "OSC_MouthWide"));
        }

        [Test]
        public void AddHeadPoseLayer_UsesAdditiveBlending()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");

            var layer = _controller.layers.First(l => l.name == "OSC_HeadRoll");
            Assert.AreEqual(AnimatorLayerBlendingMode.Additive, layer.blendingMode);
            Assert.IsTrue(_controller.parameters.Any(p => p.name == "HeadRoll"));
        }

        [Test]
        public void AddHeadPoseLayer_RestrictsToHeadOnlyMask_ToStopChestBonePerturbationLeaking()
        {
            // Real bug (issue #16): Humanoid retargeting can leak tiny
            // perturbations into Chest/Spine from a Head-only muscle clip,
            // which a chest-mounted VRCPhysBone (e.g. wings) can amplify into
            // a visible spin under real (continuous, jittery) OSC head data
            // even though a single static parameter scrub looks harmless.
            // The mask must contain the layer's effect to Head only.
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");

            var layer = _controller.layers.First(l => l.name == "OSC_HeadRoll");
            Assert.IsNotNull(layer.avatarMask, "head-pose layer must have a mask");
            Assert.IsTrue(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Head));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.Body));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftArm));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.RightArm));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.LeftLeg));
            Assert.IsFalse(layer.avatarMask.GetHumanoidBodyPartActive(AvatarMaskBodyPart.RightLeg));
        }

        [Test]
        public void AddHeadPoseLayer_SharesOneMaskAcrossAllThreeAxes()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadYaw", "Head Turn Left-Right");
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadPitch", "Head Nod Down-Up");

            var masks = _controller.layers
                .Where(l => l.name.StartsWith("OSC_Head"))
                .Select(l => l.avatarMask)
                .ToArray();
            Assert.AreEqual(3, masks.Length);
            Assert.AreSame(masks[0], masks[1]);
            Assert.AreSame(masks[1], masks[2]);
        }

        [Test]
        public void AddHeadPoseLayer_AttachesExactlyOneTrackingControl_HeadAnimationOthersNoChange()
        {
            // VRChat fact (creators.vrchat.com/avatars/state-behaviors/): the
            // Head bone is IK-driven on Desktop; only a
            // VRCAnimatorTrackingControl with trackingHead = Animation makes
            // this layer's muscle curves win. Every other tracked part must
            // stay NoChange so this layer never stomps on hands/eyes/etc.
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");

            var layer = _controller.layers.First(l => l.name == "OSC_HeadRoll");
            var state = layer.stateMachine.states.Single().state;
            var behaviours = state.behaviours.OfType<VRCAnimatorTrackingControl>().ToArray();

            Assert.AreEqual(1, behaviours.Length, "exactly one VRCAnimatorTrackingControl must be attached");
            var behaviour = behaviours[0];
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.Animation, behaviour.trackingHead);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingLeftHand);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingRightHand);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingHip);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingLeftFoot);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingRightFoot);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingLeftFingers);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingRightFingers);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingEyes);
            Assert.AreEqual(VRC_AnimatorTrackingControl.TrackingType.NoChange, behaviour.trackingMouth);
        }

        [Test]
        public void AddBlendShapeLayer_DoesNotAttachTrackingControl()
        {
            // Only head-pose layers need VRCAnimatorTrackingControl; blend
            // shape layers must be left alone.
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "MouthOpen", _renderer, "Smile");

            var layer = _controller.layers.First(l => l.name == "OSC_MouthOpen");
            var state = layer.stateMachine.states.Single().state;
            Assert.IsEmpty(state.behaviours);
        }

        [Test]
        public void RemoveLayer_HeadPoseLayer_LeavesNoOrphanedTrackingControlAsset()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");

            OscAnimatorLayerBuilder.RemoveLayer(_controller, "HeadRoll");

            var orphaned = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .OfType<VRCAnimatorTrackingControl>()
                .Any();
            Assert.IsFalse(orphaned, "removing a head-pose layer must not leave a StateMachineBehaviour sub-asset behind");
        }

        [Test]
        public void RemoveLayer_KeepsSharedMask_WhileOtherHeadLayersStillUseIt()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadYaw", "Head Turn Left-Right");

            OscAnimatorLayerBuilder.RemoveLayer(_controller, "HeadRoll");

            var remaining = _controller.layers.First(l => l.name == "OSC_HeadYaw");
            Assert.IsNotNull(remaining.avatarMask, "surviving layer's mask must not have been destroyed");
        }

        [Test]
        public void RemoveLayer_CleansUpSharedMask_OnceNoLayerReferencesIt()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");
            OscAnimatorLayerBuilder.RemoveLayer(_controller, "HeadRoll");

            var maskStillInAsset = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(_controller))
                .OfType<AvatarMask>()
                .Any(m => m.name == "OSC_HeadOnlyMask");
            Assert.IsFalse(maskStillInAsset, "orphaned mask sub-asset must be cleaned up");
        }

        [Test]
        public void RemoveLayer_IsTheInverseOfAdd_LeavesOtherLayersAlone()
        {
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "MouthOpen", _renderer, "Smile");
            OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, "MouthSmile", _renderer, "Smile");

            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "MouthOpen"));
            var removed = OscAnimatorLayerBuilder.RemoveLayer(_controller, "MouthOpen");

            Assert.IsTrue(removed);
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "MouthOpen"));
            Assert.IsFalse(_controller.parameters.Any(p => p.name == "MouthOpen"));
            // The other parameter's layer must survive untouched.
            Assert.IsTrue(OscAnimatorLayerBuilder.HasLayer(_controller, "MouthSmile"));
            Assert.IsTrue(_controller.parameters.Any(p => p.name == "MouthSmile"));
        }

        [Test]
        public void RemoveLayer_WhenAbsent_ReturnsFalseAndDoesNotThrow()
        {
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "MouthOpen"));
            Assert.IsFalse(OscAnimatorLayerBuilder.RemoveLayer(_controller, "MouthOpen"));
        }

        [Test]
        public void RemoveLayer_WorksForSignedBlendShapeWithSharedZeroClip()
        {
            // The zero clip is reused across the -1/0 (and possibly 1) children;
            // removal must not double-destroy it.
            OscAnimatorLayerBuilder.AddSignedBlendShapeLayer(
                _controller, _avatarRoot.transform, "MouthWide", _renderer, "Smile", null, null);

            Assert.DoesNotThrow(() => OscAnimatorLayerBuilder.RemoveLayer(_controller, "MouthWide"));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "MouthWide"));
        }

        [Test]
        public void RemoveLayer_WorksForHeadPoseLayer()
        {
            OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, "HeadRoll", "Head Tilt Left-Right");

            Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, "HeadRoll"));
            Assert.IsFalse(OscAnimatorLayerBuilder.HasLayer(_controller, "HeadRoll"));
        }

        [Test]
        public void AllTenParameters_CanBeWiredWithoutError()
        {
            foreach (var spec in OscParameterSpec.All)
            {
                switch (spec.Kind)
                {
                    case OscParamKind.BlendShape:
                        OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, spec.Name, _renderer, "Smile");
                        break;
                    case OscParamKind.SignedBlendShape:
                        OscAnimatorLayerBuilder.AddSignedBlendShapeLayer(
                            _controller, _avatarRoot.transform, spec.Name, _renderer, "Smile", _renderer, "Pucker");
                        break;
                    case OscParamKind.HeadPose:
                        var muscle = spec.Name == "HeadRoll" ? "Head Tilt Left-Right"
                            : spec.Name == "HeadYaw" ? "Head Turn Left-Right"
                            : "Head Nod Down-Up";
                        OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, spec.Name, muscle);
                        break;
                }
            }

            Assert.AreEqual(10, _controller.parameters.Length);
            // CreateAnimatorControllerAtPath seeds one default "Base Layer";
            // AddLayer only ever replaces layers it owns (named "OSC_*").
            Assert.AreEqual(10, _controller.layers.Count(l => l.name.StartsWith("OSC_")));
            foreach (var spec in OscParameterSpec.All)
            {
                Assert.IsTrue(
                    _controller.layers.Any(l => l.name == "OSC_" + spec.Name),
                    $"missing layer for {spec.Name}");
            }
        }

        [Test]
        public void WireThenRemoveAllTen_RoundTripsBackToZero()
        {
            foreach (var spec in OscParameterSpec.All)
            {
                switch (spec.Kind)
                {
                    case OscParamKind.BlendShape:
                        OscAnimatorLayerBuilder.AddBlendShapeLayer(_controller, _avatarRoot.transform, spec.Name, _renderer, "Smile");
                        break;
                    case OscParamKind.SignedBlendShape:
                        OscAnimatorLayerBuilder.AddSignedBlendShapeLayer(
                            _controller, _avatarRoot.transform, spec.Name, _renderer, "Smile", _renderer, "Pucker");
                        break;
                    case OscParamKind.HeadPose:
                        var muscle = spec.Name == "HeadRoll" ? "Head Tilt Left-Right"
                            : spec.Name == "HeadYaw" ? "Head Turn Left-Right"
                            : "Head Nod Down-Up";
                        OscAnimatorLayerBuilder.AddHeadPoseLayer(_controller, spec.Name, muscle);
                        break;
                }
            }

            foreach (var spec in OscParameterSpec.All)
            {
                Assert.IsTrue(OscAnimatorLayerBuilder.RemoveLayer(_controller, spec.Name));
            }

            Assert.AreEqual(0, _controller.parameters.Length);
            Assert.AreEqual(0, _controller.layers.Count(l => l.name.StartsWith("OSC_")));
        }
    }
}

using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    /// <summary>
    /// Covers the playable-layer plumbing around head pose (issue #25): head
    /// layers live on the <b>Additive</b> playable layer
    /// (<see cref="AvatarSetupWindow.EnsureAdditiveController"/>), and the
    /// Gesture-era experiment (layers on Gesture + first-layer-mask swap) is
    /// cleaned up on migration
    /// (<see cref="AvatarSetupWindow.RestoreGestureMask"/>).
    /// </summary>
    public class AvatarSetupWindowGestureControllerTests
    {
        const string TestDir = "Assets/_VRChatCameraOscGestureTests";

        GameObject _avatarRoot;
        VRCAvatarDescriptor _descriptor;

        [SetUp]
        public void SetUp()
        {
            if (!AssetDatabase.IsValidFolder(TestDir))
            {
                AssetDatabase.CreateFolder("Assets", "_VRChatCameraOscGestureTests");
            }

            _avatarRoot = new GameObject("Avatar");
            _descriptor = _avatarRoot.AddComponent<VRCAvatarDescriptor>();
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_avatarRoot);
            AssetDatabase.DeleteAsset(TestDir);
        }

        static VRCAvatarDescriptor.CustomAnimLayer[] SingleLayer(
            VRCAvatarDescriptor.AnimLayerType type, bool isDefault, AnimatorController controller = null)
        {
            return new[]
            {
                new VRCAvatarDescriptor.CustomAnimLayer
                {
                    type = type,
                    isDefault = isDefault,
                    animatorController = controller,
                    isEnabled = true,
                },
            };
        }

        [Test]
        public void TryGetGestureController_WhenLayerIsDefault_ReturnsNull()
        {
            _descriptor.baseAnimationLayers = SingleLayer(VRCAvatarDescriptor.AnimLayerType.Gesture, isDefault: true);

            Assert.IsNull(AvatarSetupWindow.TryGetGestureController(_descriptor));
        }

        [Test]
        public void TryGetGestureController_WhenCustomControllerAssigned_ReturnsIt()
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Existing.controller");
            _descriptor.baseAnimationLayers = SingleLayer(VRCAvatarDescriptor.AnimLayerType.Gesture, isDefault: false, controller);

            Assert.AreSame(controller, AvatarSetupWindow.TryGetGestureController(_descriptor));
        }

        [Test]
        public void EnsureAdditiveController_WithExistingCustomController_ReturnsItUnchanged()
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Additive.controller");
            _descriptor.baseAnimationLayers = SingleLayer(VRCAvatarDescriptor.AnimLayerType.Additive, isDefault: false, controller);

            var result = AvatarSetupWindow.EnsureAdditiveController(_descriptor);

            Assert.AreSame(controller, result);
        }

        [Test]
        public void EnsureAdditiveController_WhenDefault_CreatesAndAssignsBlankController()
        {
            // VRChat's default Additive layer is empty, so replacing it with a
            // blank custom controller loses nothing (unlike Gesture, where a
            // blank controller would lose the default hand gestures).
            _descriptor.baseAnimationLayers = SingleLayer(VRCAvatarDescriptor.AnimLayerType.Additive, isDefault: true);

            var result = AvatarSetupWindow.EnsureAdditiveController(_descriptor);

            Assert.IsNotNull(result);
            var layer = _descriptor.baseAnimationLayers
                .Single(l => l.type == VRCAvatarDescriptor.AnimLayerType.Additive);
            Assert.IsFalse(layer.isDefault);
            Assert.IsTrue(layer.isEnabled);
            Assert.AreSame(result, layer.animatorController);
        }

        /// <summary>Recreates the state a Gesture-era (issue #25 first
        /// attempt) wizard left behind: an <c>OSC_GestureMask</c> sub-asset
        /// swapped into the controller's first layer and the descriptor's
        /// Gesture slot.</summary>
        AnimatorController GestureEraMaskedController()
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Gesture.controller");
            var ours = new AvatarMask { name = AvatarSetupWindow.GestureMaskName };
            for (var part = AvatarMaskBodyPart.Root; part < AvatarMaskBodyPart.LastBodyPart; part++)
            {
                ours.SetHumanoidBodyPartActive(
                    part,
                    part == AvatarMaskBodyPart.Head ||
                    part == AvatarMaskBodyPart.LeftFingers ||
                    part == AvatarMaskBodyPart.RightFingers);
            }
            AssetDatabase.AddObjectToAsset(ours, controller);
            var layers = controller.layers;
            layers[0].avatarMask = ours;
            controller.layers = layers;

            var descLayers = SingleLayer(VRCAvatarDescriptor.AnimLayerType.Gesture, isDefault: false, controller);
            descLayers[0].mask = ours;
            _descriptor.baseAnimationLayers = descLayers;
            return controller;
        }

        [Test]
        public void RestoreGestureMask_SwapsBackAndDeletesCombinedSubAsset()
        {
            var controller = GestureEraMaskedController();

            AvatarSetupWindow.RestoreGestureMask(_descriptor, controller);

            var first = controller.layers[0].avatarMask;
            // Restored by equivalence: the stock vrc_HandsOnly if findable in
            // this project (it ships inside the SDK package), else null.
            Assert.IsTrue(first == null || first.name == "vrc_HandsOnly", $"got: {first?.name}");
            var gestureLayer = _descriptor.baseAnimationLayers
                .Single(l => l.type == VRCAvatarDescriptor.AnimLayerType.Gesture);
            Assert.AreSame(first, gestureLayer.mask);
            var lingering = AssetDatabase.LoadAllAssetsAtPath(AssetDatabase.GetAssetPath(controller))
                .OfType<AvatarMask>()
                .Any(m => m.name == AvatarSetupWindow.GestureMaskName);
            Assert.IsFalse(lingering, "combined mask sub-asset must be cleaned up");
        }

        [Test]
        public void RestoreGestureMask_NoOp_WhenFirstLayerMaskIsNotOurs()
        {
            var controller = AnimatorController.CreateAnimatorControllerAtPath(TestDir + "/Gesture.controller");
            var foreign = new AvatarMask { name = "SomeoneElsesMask" };
            AssetDatabase.AddObjectToAsset(foreign, controller);
            var layers = controller.layers;
            layers[0].avatarMask = foreign;
            controller.layers = layers;
            _descriptor.baseAnimationLayers = SingleLayer(VRCAvatarDescriptor.AnimLayerType.Gesture, isDefault: false, controller);

            AvatarSetupWindow.RestoreGestureMask(_descriptor, controller);

            Assert.AreSame(foreign, controller.layers[0].avatarMask, "foreign masks must be left alone");
        }
    }
}

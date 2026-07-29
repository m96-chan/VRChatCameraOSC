using System.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace VRChatCameraOsc.AvatarSetup.Tests
{
    public class MuscleNameValidityTests
    {
        // A misspelled muscle-curve property silently animates nothing —
        // verify our three head muscle names against Unity's own list, and
        // print the real head/neck names for reference.
        [Test]
        public void HeadPoseMuscleNames_ExistInHumanTrait()
        {
            var names = HumanTrait.MuscleName;
            foreach (var n in names.Where(n => n.Contains("Head") || n.Contains("Neck")))
            {
                Debug.Log($"HumanTrait muscle: '{n}'");
            }
            foreach (var expected in new[] { "Head Nod Down-Up", "Head Tilt Left-Right", "Head Turn Left-Right" })
            {
                Assert.Contains(expected, names, $"'{expected}' is not a real muscle name");
            }
        }

        // Finger quirk (issue #8): unlike the head muscles, the finger names
        // in HumanTrait.MuscleName ("Left Thumb 1 Stretched") are NOT the
        // animatable curve property names — a muscle-curve binding must use
        // the Animation-window spelling ("LeftHand.Thumb.1 Stretched").
        // Verify both directions empirically: the HumanTrait entry exists for
        // every binding property we use, AND a clip bound with our spelling
        // registers as humanoid motion (clip.humanMotion) while the
        // HumanTrait spelling does not.
        [Test]
        public void FingerMuscleBindingProperties_AreRealHumanoidMuscles()
        {
            var traitNames = HumanTrait.MuscleName;
            foreach (var n in traitNames.Where(n => n.Contains("Thumb") || n.Contains("Index") || n.Contains("Little")))
            {
                Debug.Log($"HumanTrait finger muscle: '{n}'");
            }

            foreach (var left in new[] { true, false })
            {
                var props = OscAnimatorLayerBuilder.FingerMuscleProperties(left).ToArray();
                Assert.AreEqual(20, props.Length, "4 muscles per finger, 5 fingers");
                foreach (var prop in props)
                {
                    // "LeftHand.Thumb.1 Stretched" -> "Left Thumb 1 Stretched"
                    var traitName = prop.Replace("Hand.", " ").Replace(".", " ");
                    Assert.Contains(traitName, traitNames,
                        $"binding '{prop}' has no HumanTrait muscle '{traitName}'");
                    Assert.IsTrue(BindsHumanoidMuscle(prop),
                        $"'{prop}' does not bind a humanoid muscle curve");
                }
            }

            // Control group: the HumanTrait spelling and a made-up finger
            // name must NOT bind — proving humanMotion is a real check, and
            // that the binding-name/trait-name distinction actually exists.
            Assert.IsFalse(BindsHumanoidMuscle("Left Thumb 1 Stretched"),
                "HumanTrait spelling unexpectedly binds — the finger-name quirk is gone?");
            Assert.IsFalse(BindsHumanoidMuscle("LeftHand.Pinky.1 Stretched"),
                "a bogus finger name should not bind");
        }

        // Arm muscles (issue #28): expect the head-style behavior — the
        // HumanTrait.MuscleName spellings ("Left Arm Down-Up") ARE the
        // animatable curve property names, unlike the finger quirk above.
        // Verify all 9 per-arm names empirically: present in HumanTrait AND
        // a clip bound with that exact spelling registers as humanoid
        // motion (clip.humanMotion).
        [Test]
        public void ArmMuscleBindingProperties_AreRealHumanoidMuscles()
        {
            var traitNames = HumanTrait.MuscleName;
            foreach (var n in traitNames.Where(n =>
                n.Contains("Arm") || n.Contains("Forearm") || n.Contains("Shoulder") ||
                (n.Contains("Hand") && !n.Contains("."))))
            {
                Debug.Log($"HumanTrait arm muscle: '{n}'");
            }

            foreach (var left in new[] { true, false })
            {
                var props = OscAnimatorLayerBuilder.ArmMuscleProperties(left).ToArray();
                Assert.AreEqual(9, props.Length, "9 muscles per arm group");
                foreach (var prop in props)
                {
                    Assert.Contains(prop, traitNames, $"'{prop}' is not a real HumanTrait muscle name");
                    Assert.IsTrue(BindsHumanoidMuscle(prop),
                        $"'{prop}' does not bind a humanoid muscle curve");
                }
            }

            // Control group: a made-up arm muscle must not bind.
            Assert.IsFalse(BindsHumanoidMuscle("Left Arm Up-Down"),
                "a bogus arm muscle name should not bind");
        }

        static bool BindsHumanoidMuscle(string propertyName)
        {
            var clip = new AnimationClip();
            var binding = EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), propertyName);
            AnimationUtility.SetEditorCurve(clip, binding, AnimationCurve.Constant(0f, 1f, 1f));
            var isHuman = clip.humanMotion;
            Object.DestroyImmediate(clip);
            return isHuman;
        }
    }
}

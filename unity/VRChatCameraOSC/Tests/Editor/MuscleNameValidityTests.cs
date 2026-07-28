using System.Linq;
using NUnit.Framework;
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
    }
}

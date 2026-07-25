using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// Builds FX Animator Controller layers that drive an existing blend shape
    /// or the Humanoid head bone from one of the 10 OSC float parameters
    /// (issue #16). Each parameter gets its own layer, named
    /// <c>OSC_&lt;ParamName&gt;</c>, so re-running the wizard replaces rather
    /// than duplicates.
    ///
    /// Deliberately Animator-only: VRChat strips arbitrary MonoBehaviours from
    /// uploaded avatars, so nothing here can be a runtime script — see the
    /// package README.
    /// </summary>
    public static class OscAnimatorLayerBuilder
    {
        const float BlendShapeFullWeight = 100f;

        /// <summary>0..1 parameter driving a single blend shape from 0 to 100.</summary>
        public static void AddBlendShapeLayer(
            AnimatorController controller,
            Transform avatarRoot,
            string paramName,
            SkinnedMeshRenderer renderer,
            string blendShapeName)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, 0f), 0f);
            tree.AddChild(BlendShapeClip(controller, avatarRoot, renderer, blendShapeName, BlendShapeFullWeight), 1f);
            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Override, null);
        }

        /// <summary>
        /// -1..1 parameter. <paramref name="negativeRenderer"/>/<paramref name="negativeBlendShape"/>
        /// may be null to leave the negative half undriven (common: many
        /// avatars only have a "wide/smile" shape, not a "pucker" one).
        /// </summary>
        public static void AddSignedBlendShapeLayer(
            AnimatorController controller,
            Transform avatarRoot,
            string paramName,
            SkinnedMeshRenderer positiveRenderer,
            string positiveBlendShape,
            SkinnedMeshRenderer negativeRenderer,
            string negativeBlendShape)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            var zero = ZeroClip(controller, paramName);

            if (negativeRenderer != null && !string.IsNullOrEmpty(negativeBlendShape))
            {
                tree.AddChild(BlendShapeClip(controller, avatarRoot, negativeRenderer, negativeBlendShape, BlendShapeFullWeight), -1f);
            }
            else
            {
                tree.AddChild(zero, -1f);
            }

            tree.AddChild(zero, 0f);

            if (positiveRenderer != null && !string.IsNullOrEmpty(positiveBlendShape))
            {
                tree.AddChild(BlendShapeClip(controller, avatarRoot, positiveRenderer, positiveBlendShape, BlendShapeFullWeight), 1f);
            }
            else
            {
                tree.AddChild(zero, 1f);
            }

            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Override, null);
        }

        /// <summary>
        /// -1..1 parameter driving one Humanoid muscle via an additive layer,
        /// so it composes with whatever else animates the avatar instead of
        /// overriding it. <paramref name="muscleName"/> must be one of
        /// <see cref="HumanTrait.MuscleName"/> (e.g. "Head Nod Down-Up").
        /// </summary>
        public static void AddHeadPoseLayer(AnimatorController controller, string paramName, string muscleName)
        {
            var tree = NewTree(controller, paramName, BlendTreeType.Simple1D);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, -1f), -1f);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, 0f), 0f);
            tree.AddChild(MuscleClip(controller, paramName, muscleName, 1f), 1f);
            AddLayer(controller, paramName, tree, AnimatorLayerBlendingMode.Additive, null);
        }

        /// <summary>Whether an <c>OSC_&lt;paramName&gt;</c> layer currently exists — the
        /// "ON" state the wizard's toggle button reads to decide Apply vs. Remove.</summary>
        public static bool HasLayer(AnimatorController controller, string paramName)
        {
            return controller != null && controller.layers.Any(l => l.name == $"OSC_{paramName}");
        }

        /// <summary>
        /// Removes the <c>OSC_&lt;paramName&gt;</c> layer, its parameter, and
        /// every sub-asset (BlendTree + AnimationClips) it owns — the "OFF"
        /// side of the wizard's apply/remove toggle. No-op (returns false) if
        /// the layer isn't present.
        /// </summary>
        public static bool RemoveLayer(AnimatorController controller, string paramName)
        {
            var layerName = $"OSC_{paramName}";
            var layers = controller.layers.ToList();
            var index = layers.FindIndex(l => l.name == layerName);
            if (index < 0)
            {
                return false;
            }

            var layer = layers[index];
            layers.RemoveAt(index);
            controller.layers = layers.ToArray();

            if (layer.stateMachine != null)
            {
                var visited = new HashSet<int>();
                foreach (var childState in layer.stateMachine.states)
                {
                    DestroyMotion(childState.state.motion, visited);
                }
                AssetDatabase.RemoveObjectFromAsset(layer.stateMachine);
                Object.DestroyImmediate(layer.stateMachine, true);
            }

            controller.parameters = controller.parameters.Where(p => p.name != paramName).ToArray();
            return true;
        }

        static void DestroyMotion(Motion motion, HashSet<int> visited)
        {
            if (motion == null || !visited.Add(motion.GetInstanceID()))
            {
                return;
            }

            if (motion is BlendTree tree)
            {
                foreach (var child in tree.children)
                {
                    DestroyMotion(child.motion, visited);
                }
                AssetDatabase.RemoveObjectFromAsset(tree);
                Object.DestroyImmediate(tree, true);
            }
            else if (motion is AnimationClip clip)
            {
                AssetDatabase.RemoveObjectFromAsset(clip);
                Object.DestroyImmediate(clip, true);
            }
        }

        static BlendTree NewTree(AnimatorController controller, string paramName, BlendTreeType type)
        {
            EnsureFloatParameter(controller, paramName);
            var tree = new BlendTree
            {
                name = $"OSC_{paramName}",
                blendType = type,
                blendParameter = paramName,
            };
            AssetDatabase.AddObjectToAsset(tree, controller);
            return tree;
        }

        static void EnsureFloatParameter(AnimatorController controller, string paramName)
        {
            if (controller.parameters.Any(p => p.name == paramName))
            {
                return;
            }
            controller.AddParameter(paramName, AnimatorControllerParameterType.Float);
        }

        /// <summary>
        /// Replaces an existing <c>OSC_&lt;paramName&gt;</c> layer if present
        /// (idempotent re-runs), otherwise appends a new one.
        /// </summary>
        static void AddLayer(
            AnimatorController controller,
            string paramName,
            BlendTree tree,
            AnimatorLayerBlendingMode blendingMode,
            AvatarMask mask)
        {
            var layerName = $"OSC_{paramName}";
            var layers = controller.layers.ToList();
            var existingIndex = layers.FindIndex(l => l.name == layerName);
            if (existingIndex >= 0)
            {
                layers.RemoveAt(existingIndex);
            }

            var stateMachine = new AnimatorStateMachine { name = layerName, hideFlags = HideFlags.HideInHierarchy };
            AssetDatabase.AddObjectToAsset(stateMachine, controller);
            var state = stateMachine.AddState(layerName);
            state.motion = tree;
            state.writeDefaultValues = false;
            stateMachine.defaultState = state;

            layers.Add(new AnimatorControllerLayer
            {
                name = layerName,
                stateMachine = stateMachine,
                defaultWeight = 1f,
                blendingMode = blendingMode,
                avatarMask = mask,
            });
            controller.layers = layers.ToArray();
        }

        static AnimationClip BlendShapeClip(
            AnimatorController controller,
            Transform avatarRoot,
            SkinnedMeshRenderer renderer,
            string blendShapeName,
            float weight)
        {
            var clip = new AnimationClip { name = $"{renderer.name}_{blendShapeName}_{weight:0}" };
            var path = AnimationUtility.CalculateTransformPath(renderer.transform, avatarRoot);
            var binding = EditorCurveBinding.FloatCurve(path, typeof(SkinnedMeshRenderer), "blendShape." + blendShapeName);
            AnimationUtility.SetEditorCurve(clip, binding, AnimationCurve.Constant(0f, 0f, weight));
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        static AnimationClip MuscleClip(AnimatorController controller, string paramName, string muscleName, float value)
        {
            var clip = new AnimationClip { name = $"OSC_{paramName}_{value:0.##}" };
            var binding = EditorCurveBinding.FloatCurve(string.Empty, typeof(Animator), muscleName);
            AnimationUtility.SetEditorCurve(clip, binding, AnimationCurve.Constant(0f, 0f, value));
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }

        static AnimationClip ZeroClip(AnimatorController controller, string paramName)
        {
            var clip = new AnimationClip { name = $"OSC_{paramName}_zero" };
            AssetDatabase.AddObjectToAsset(clip, controller);
            return clip;
        }
    }
}

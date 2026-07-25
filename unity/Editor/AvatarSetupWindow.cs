using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;
using VRC.SDK3.Avatars.Components;
using VRC.SDK3.Avatars.ScriptableObjects;

namespace VRChatCameraOsc.AvatarSetup
{
    /// <summary>
    /// Editor wizard (issue #16): given a Humanoid avatar with existing blend
    /// shapes, generates/merges the VRC Expression Parameters and FX Animator
    /// Controller layers so VRChat's own OSC-to-parameter routing has
    /// something to drive. Does not create blend shapes, does not run at
    /// avatar runtime — see the package README.
    /// </summary>
    public class AvatarSetupWindow : EditorWindow
    {
        /// <summary>Humanoid muscle name per head-pose OSC parameter (confirmed
        /// against <see cref="HumanTrait.MuscleName"/> — see the package README).</summary>
        static readonly Dictionary<string, string> HeadPoseMuscles = new Dictionary<string, string>
        {
            { "HeadRoll", "Head Tilt Left-Right" },
            { "HeadYaw", "Head Turn Left-Right" },
            { "HeadPitch", "Head Nod Down-Up" },
        };

        VRCAvatarDescriptor _avatar;
        readonly Dictionary<string, SkinnedMeshRenderer> _positiveRenderer = new Dictionary<string, SkinnedMeshRenderer>();
        readonly Dictionary<string, string> _positiveBlendShape = new Dictionary<string, string>();
        readonly Dictionary<string, SkinnedMeshRenderer> _negativeRenderer = new Dictionary<string, SkinnedMeshRenderer>();
        readonly Dictionary<string, string> _negativeBlendShape = new Dictionary<string, string>();
        bool _includeHeadPose = true;
        Vector2 _scroll;

        [MenuItem("VRChatCameraOSC/Avatar Setup Wizard")]
        public static void Open() => GetWindow<AvatarSetupWindow>("VRChatCameraOSC Setup");

        void OnGUI()
        {
            EditorGUILayout.HelpBox(
                "Wires this app's 10 OSC parameters to blend shapes/head bone you " +
                "already have on this Humanoid avatar. Doesn't create blend shapes, " +
                "doesn't add any runtime script.",
                MessageType.Info);

            var newAvatar = (VRCAvatarDescriptor)EditorGUILayout.ObjectField(
                "Avatar", _avatar, typeof(VRCAvatarDescriptor), true);
            if (newAvatar != _avatar)
            {
                _avatar = newAvatar;
                _positiveRenderer.Clear();
                _positiveBlendShape.Clear();
                _negativeRenderer.Clear();
                _negativeBlendShape.Clear();
            }

            if (_avatar == null)
            {
                return;
            }

            var animator = _avatar.GetComponent<Animator>();
            if (animator == null || !animator.isHuman)
            {
                EditorGUILayout.HelpBox("This avatar isn't Humanoid — out of scope (issue #16).", MessageType.Error);
                return;
            }

            var renderers = _avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true)
                .Where(r => r.sharedMesh != null && r.sharedMesh.blendShapeCount > 0)
                .ToArray();
            if (renderers.Length == 0)
            {
                EditorGUILayout.HelpBox("No SkinnedMeshRenderer with blend shapes found under this avatar.", MessageType.Warning);
            }

            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            foreach (var spec in OscParameterSpec.All)
            {
                if (spec.Kind == OscParamKind.HeadPose)
                {
                    continue;
                }
                DrawBlendShapePicker(spec, renderers);
            }
            EditorGUILayout.EndScrollView();

            EditorGUILayout.Space();
            _includeHeadPose = EditorGUILayout.ToggleLeft(
                "Wire HeadRoll / HeadYaw / HeadPitch to the Humanoid head bone (additive layer)",
                _includeHeadPose);

            EditorGUILayout.Space();
            var wired = VrcExpressionParametersMerger.IsFullyWired(_avatar.expressionParameters, OscParameterSpec.All);
            EditorGUILayout.LabelField("Status", wired ? "ON — wired" : "OFF — not wired");
            var newWired = GUILayout.Toggle(wired, wired ? "ON (click to remove)" : "OFF (click to apply)", "Button");
            if (newWired != wired)
            {
                if (newWired)
                {
                    RunSetup();
                }
                else
                {
                    RunRemove();
                }
            }
        }

        void DrawBlendShapePicker(OscParamSpec spec, SkinnedMeshRenderer[] renderers)
        {
            EditorGUILayout.LabelField(spec.Name, EditorStyles.boldLabel);
            EditorGUI.indentLevel++;
            DrawOneShapePicker(spec.Kind == OscParamKind.SignedBlendShape ? "Positive (+1)" : "Blend shape",
                spec.Name, renderers, _positiveRenderer, _positiveBlendShape);
            if (spec.Kind == OscParamKind.SignedBlendShape)
            {
                DrawOneShapePicker("Negative (-1, optional)", spec.Name, renderers, _negativeRenderer, _negativeBlendShape);
            }
            EditorGUI.indentLevel--;
        }

        void DrawOneShapePicker(
            string label,
            string paramName,
            SkinnedMeshRenderer[] renderers,
            Dictionary<string, SkinnedMeshRenderer> rendererMap,
            Dictionary<string, string> shapeMap)
        {
            var rendererOptions = new[] { "(skip)" }.Concat(renderers.Select(r => r.name)).ToArray();
            rendererMap.TryGetValue(paramName, out var currentRenderer);
            var rendererIndex = currentRenderer == null ? 0 : System.Array.IndexOf(renderers, currentRenderer) + 1;

            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label, GUILayout.Width(140));
            var newRendererIndex = EditorGUILayout.Popup(rendererIndex, rendererOptions, GUILayout.Width(160));

            if (newRendererIndex == 0)
            {
                rendererMap.Remove(paramName);
                shapeMap.Remove(paramName);
                EditorGUILayout.EndHorizontal();
                return;
            }

            var renderer = renderers[newRendererIndex - 1];
            rendererMap[paramName] = renderer;

            var shapeNames = System.Linq.Enumerable.Range(0, renderer.sharedMesh.blendShapeCount)
                .Select(renderer.sharedMesh.GetBlendShapeName)
                .ToArray();
            shapeMap.TryGetValue(paramName, out var currentShape);
            var shapeIndex = System.Array.IndexOf(shapeNames, currentShape);
            if (shapeIndex < 0)
            {
                shapeIndex = 0;
            }
            var newShapeIndex = EditorGUILayout.Popup(shapeIndex, shapeNames);
            if (shapeNames.Length > 0)
            {
                shapeMap[paramName] = shapeNames[newShapeIndex];
            }
            EditorGUILayout.EndHorizontal();
        }

        void RunSetup()
        {
            var expressionParameters = EnsureExpressionParameters(_avatar);
            var added = VrcExpressionParametersMerger.Merge(expressionParameters, OscParameterSpec.All);

            var controller = EnsureFxController(_avatar);
            foreach (var spec in OscParameterSpec.All)
            {
                switch (spec.Kind)
                {
                    case OscParamKind.BlendShape:
                        if (_positiveRenderer.TryGetValue(spec.Name, out var r) &&
                            _positiveBlendShape.TryGetValue(spec.Name, out var shape))
                        {
                            OscAnimatorLayerBuilder.AddBlendShapeLayer(controller, _avatar.transform, spec.Name, r, shape);
                        }
                        break;
                    case OscParamKind.SignedBlendShape:
                        _positiveRenderer.TryGetValue(spec.Name, out var posR);
                        _positiveBlendShape.TryGetValue(spec.Name, out var posShape);
                        _negativeRenderer.TryGetValue(spec.Name, out var negR);
                        _negativeBlendShape.TryGetValue(spec.Name, out var negShape);
                        if (posR != null || negR != null)
                        {
                            OscAnimatorLayerBuilder.AddSignedBlendShapeLayer(
                                controller, _avatar.transform, spec.Name, posR, posShape, negR, negShape);
                        }
                        break;
                    case OscParamKind.HeadPose:
                        if (_includeHeadPose)
                        {
                            OscAnimatorLayerBuilder.AddHeadPoseLayer(controller, spec.Name, HeadPoseMuscles[spec.Name]);
                        }
                        break;
                }
            }

            EditorUtility.SetDirty(controller);
            AssetDatabase.SaveAssets();
            EditorUtility.DisplayDialog(
                "VRChatCameraOSC Setup",
                $"Done. Added {added} new expression parameter(s). FX layers wired for the selected blend shapes" +
                (_includeHeadPose ? " and head pose." : "."),
                "OK");
        }

        /// <summary>
        /// The "OFF" side of the toggle: removes every VRChatCameraOSC
        /// parameter and <c>OSC_*</c> layer this wizard could have added.
        /// Leaves the Expression Parameters asset and FX Animator Controller
        /// themselves in place (only their VRChatCameraOSC-owned contents are
        /// removed) since they may hold unrelated setup.
        /// </summary>
        void RunRemove()
        {
            var removedParams = 0;
            if (_avatar.expressionParameters != null)
            {
                removedParams = VrcExpressionParametersMerger.Remove(_avatar.expressionParameters, OscParameterSpec.All);
            }

            var removedLayers = 0;
            var controller = TryGetFxController(_avatar);
            if (controller != null)
            {
                foreach (var spec in OscParameterSpec.All)
                {
                    if (OscAnimatorLayerBuilder.RemoveLayer(controller, spec.Name))
                    {
                        removedLayers++;
                    }
                }
                EditorUtility.SetDirty(controller);
            }

            AssetDatabase.SaveAssets();
            EditorUtility.DisplayDialog(
                "VRChatCameraOSC Setup",
                $"Removed {removedParams} expression parameter(s) and {removedLayers} FX layer(s).",
                "OK");
        }

        static VRCExpressionParameters EnsureExpressionParameters(VRCAvatarDescriptor avatar)
        {
            if (avatar.expressionParameters != null)
            {
                return avatar.expressionParameters;
            }
            var asset = ScriptableObject.CreateInstance<VRCExpressionParameters>();
            asset.parameters = new VRCExpressionParameters.Parameter[0];
            var path = AssetPathNextToAvatar(avatar, "ExpressionParameters.asset");
            AssetDatabase.CreateAsset(asset, path);
            Undo.RecordObject(avatar, "Assign VRC Expression Parameters");
            avatar.expressionParameters = asset;
            EditorUtility.SetDirty(avatar);
            return asset;
        }

        /// <summary>Read-only: the avatar's FX controller if it already has one, else null.</summary>
        static AnimatorController TryGetFxController(VRCAvatarDescriptor avatar)
        {
            var layers = avatar.baseAnimationLayers;
            var fxIndex = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.FX);
            return fxIndex >= 0 ? layers[fxIndex].animatorController as AnimatorController : null;
        }

        static AnimatorController EnsureFxController(VRCAvatarDescriptor avatar)
        {
            var existing = TryGetFxController(avatar);
            if (existing != null)
            {
                return existing;
            }

            var layers = avatar.baseAnimationLayers;
            var fxIndex = System.Array.FindIndex(layers, l => l.type == VRCAvatarDescriptor.AnimLayerType.FX);
            var path = AssetPathNextToAvatar(avatar, "FX.controller");
            var controller = AnimatorController.CreateAnimatorControllerAtPath(path);

            Undo.RecordObject(avatar, "Assign VRChatCameraOSC FX Controller");
            if (fxIndex >= 0)
            {
                layers[fxIndex].animatorController = controller;
                layers[fxIndex].isDefault = false;
            }
            avatar.baseAnimationLayers = layers;
            EditorUtility.SetDirty(avatar);
            return controller;
        }

        static string AssetPathNextToAvatar(VRCAvatarDescriptor avatar, string fileName)
        {
            var scenePath = avatar.gameObject.scene.IsValid() ? "Assets" : null;
            var prefabPath = PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(avatar.gameObject);
            var dir = !string.IsNullOrEmpty(prefabPath)
                ? Path.GetDirectoryName(prefabPath)
                : (scenePath ?? "Assets");
            var path = $"{dir}/VRChatCameraOSC_{avatar.gameObject.name}_{fileName}".Replace("\\", "/");
            return AssetDatabase.GenerateUniqueAssetPath(path);
        }
    }
}

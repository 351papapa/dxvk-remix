# Emissive flare materials

The Flare Materials section of Material Options identifies legacy-draw flare cards
from an uncompressed 32-bit RGBA/BGRA DDS. It derives the four channel-order and
vertical-orientation hashes used by the old local runtime. DDS dimensions, masks,
pitch and payload size are checked. Unsupported files leave the submitted material
alone. Optional albedo and emission masks use the existing texture asset manager.
An empty or missing emission mask uses the albedo texture as its falloff.

The `rtx.fork.lensFlare.*` names are retained. The master defaults off, paths are
empty and the neutral emission level is one. Game-specific paths and appearance
belong in the game's configuration. Existing configurations with explicit level
and texture paths retain those values. The old unused corona override is not
activated by this feature.

Matched materials use explicit additive emission and retain vertex lamp tint and
RGB fade. The additive `kEmissive` blend retains the previous runtime semantics:
vertex alpha alone does not attenuate emission. They bypass the legacy emissive-blend intensity override and clear the v2
GPU legacy-defaults flag so legacy-only specular controls cannot affect this
authored material. This adapts the active legacy-draw path of the previous local runtime; API materials
continue to use their authored public API emission settings.

The device owns identity and replacement-texture caches. Reference-file modification times
are checked once per frame. Saves that retain the exact timestamp require changing
the reference option or reloading the runtime. Failed reads are retried only when inputs change,
avoiding per-draw disk access and repeated warnings. A generation stored on the
scene's replacement instance invalidates the preserved material when its reference
or options change. The v2 frontend resolves these materials through SceneInterface;
its scene event stream requests retranslation when the reference changes. Both
scene-owned D3D9 objects and transient draws use that entry point. The internal
legacy draw path also tracks the generation.
Material preparation and file identity are host metadata; lighting, blending and
instance rendering continue through the GPU pipeline.

The implementation derives from the user's local `rtx_fork_texture_emissive.cpp`
at `c17b36af` plus saved working-tree changes. Device ownership, bounded DDS loading,
cache generations and the generic UI are local adaptations. Repository licensing
and retained NVIDIA notices apply; the bundled dependencies are unmodified.

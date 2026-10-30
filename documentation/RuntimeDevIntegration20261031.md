# Local runtime integration with upstream main and dev

The integrated runtime is based on the preserved local source snapshot `19ed7709`,
including the uncommitted changes present in the migration checkout. Upstream
`main` at `cfe02cce57977f872c05448a9ca1050c12de30de` was merged first in
`f8cdc80d3d11f5d070d6404a01c16ea35226cd68`. The subsequent dev integration targets
`90583a4c1c810d238ccdc2524887b4afa991b4af` from RemixProjGroup/dxvk-remix.

The original checkout and game installation are independent of this integration
checkout. No remote branch is updated by this workflow.

## Integration decisions

- Preserve local GPU instancing, GPU skinning, retained-scene handling, wetness,
  precipitation, window lighting, water foam, grass, particle sky lighting and
  local sphere-light sampling.
- Keep upstream material-description deduplication for file-backed materials.
  Materials referencing live `game:0x` textures still validate their current game
  resource and synchronize the result. A failed lookup must remain retryable.
- Apply the local window and street-light emission gates inside upstream's
  nonzero-emission branch, before exposure is calculated.
- Include all wetness, particle-sky and combined shaders for the nine new
  upstream slim/validation G-buffer stems (27 additional header references).
- Keep upstream SAB average-search controls and the local particle volume gain
  in distinct rows of `CompositeArgs`.
- Use upstream's pinned RTXDI `639c99fd0d8df6bf81ea9730d71a071d6eff5dba` and NUMOS
  `6ba250d472659dc3207b433c927d3b7fad5737d7` without modifying either dependency.

## GPU parameter layout

NUMOS `AtmosphereArgs` expands from 1424 to 1440 bytes in the dev pin. The later
fields of both host/shader structures move by 16 bytes. The new SAB controls add
a further 16 bytes to `CompositeArgs`.

| Field or structure | Bytes |
| --- | ---: |
| `sizeof(RaytraceArgs)` | 8800 |
| `RaytraceArgs::softOmmFlags` | 8788 |
| `RaytraceArgs::particleFroxelConsumerScale` | 8792 |
| `RaytraceArgs::particleFroxelRawScale` | 8796 |
| `sizeof(CompositeArgs)` | 5792 |
| `CompositeArgs::cloudMotionVectorMissDepth` | 5768 |
| `CompositeArgs::stochasticAlphaBlendRadianceVolumeFroxelGain` | 5772 |
| `CompositeArgs::stochasticAlphaBlendAverageSearchMatches` | 5776 |

An independent probe compiled the production headers as C++ and SPIR-V and
checked all 366 reflected member offsets. Runtime binaries and shaders must be
rebuilt together. Client and server bridge binaries must also come from this
same source revision.

## Validation scope

Build the release runtime twice so generated shader headers are relinked, and
build both bridge architectures. Run the existing standalone runtime tests in
`_Comp64UnitTest` and bridge logic/payload tests in their architecture-specific
directories. Verify cross-architecture API serialization and streaming before
testing the game. A source merge or successful compilation alone does not
establish gameplay compatibility or a performance improvement.

The proposed zero-puddle ripple optimization is separate from this upstream
integration. It is not included in the integration baseline.

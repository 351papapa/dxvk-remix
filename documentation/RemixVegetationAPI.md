# Remix Vegetation API

Procedural grass reads host-provided contact spheres through `SetGameValue` and
uses resolved NUMOS wind directly on the render thread.

The contact text convention uses the existing API without additional exports or
structs. Fields use decimal numbers with a dot as the decimal separator and no
thousands separator. The key holds the latest complete snapshot, not an event queue.

## Contacts: `__vegetation.pushers_v1`

The host publishes the entire active contact list with `SetGameValue`:

```text
count;x,y,z,radius,weight;x,y,z,radius,weight
```

For example, one contact in game world coordinates:

```text
1;10.0000,20.0000,3.0000,0.2500,1.0000
```

- `count`: number of records; procedural grass consumes at most 16.
- `x,y,z`: world position in the same game coordinate system as submitted geometry.
- `radius`: nonnegative radius in game units. The grass `contactRadius` option
  adds an extra margin around it.
- `weight`: contact strength, clamped to `[0,1]` by the runtime.

Use finite values. The runtime stops at the first incomplete or non-finite
record and accepts the valid prefix. The portable spelling uses semicolons
between records and commas between fields, as shown above.

Publish `0;` when there are no contacts, the player is unavailable, the feature
is disabled, or the host is unloading. GameStateStore values persist across
device reset and do not expire automatically. The runtime does not decay stale
contacts: the producer must republish decreasing weights for a recovery trail
and then clear it. A trail record needs no persistent identity.

Grass evaluates contacts in the scene's horizontal plane and currently rejects
contacts with a vertical separation of 3 or more game units. The host is
responsible for choosing suitable contact centres and radii for its world scale.


## Grass tuning

The menu path is **Alt+X > Remix Plus Developer Menu > Game Setup > Step 2: Parameter Tuning > PROCEDURAL GRASS**.
These options also work through `SetConfigVariable` and configuration files:

| Option | Meaning |
| :-- | :-- |
| `rtx.grass.windReference` | Cloud speed in km/s used to normalize vegetation wind; default `0.05`. |
| `rtx.grass.windStrengthMultiplier` | Strength before the ambient angle clamp, range `0–10`, default `1`. Zero removes ambient bending. |
| `rtx.grass.windMaxRadians` | Ambient lean cap including gusts and multiplier; default `0.45` radians (about 26 degrees). |
| `rtx.grass.gustStrength`, `gustFrequency`, `swayFrequency` | Travelling gust strength and animation frequencies in Hz. |
| `rtx.grass.contactRadius`, `contactMaxRadians` | Extra contact margin and maximum contact lean; separate from ambient wind. |

Contact bending remains available when the wind multiplier is zero. Setting
the contact limit to zero leaves ambient wind working. These artistic controls
are independent of graphics quality presets. See [RTX Options](../RtxOptions.md)
for defaults and the other grass settings.

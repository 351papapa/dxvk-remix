# Terrain wetness

The surface wetness layer uses a water level over two value-noise fields and,
when enabled, measured ground relief. It adapts the generation method in
xoxor4d/dxvk-remix game/gta4 at 7f5210e1a7cd1356bf0271230f82e6132b08ad8c.

`rtx.wetness.rainMapEnabled` traces static, fully opaque scene geometry into a
camera-centered height/cover map, then measures relief and slope and blurs rain
cover. Cars, characters, particles, decals and animated water are excluded.
The map window snaps to its texel grid. Distances are in meters and support both
Y-up and Z-up scenes. The optional map and surface retention default off.

The two signed noise samples use frequencies 1 and 2.7 relative to the cell
size. Their weights are 2/3 and 1/3. The water level rises with wetness over
the resulting ground height minus the measured depression.
Slope drains the surface, a damp shore surrounds the edge, and standing water
replaces normal-map detail and roughness. Six directional wind waves fade with
pixel footprint. Full puddle detail is evaluated for primary/PSR hits; indirect
world hits use the existing base wet layer; vehicle detail keeps its existing
behavior. Contact particles evaluate the same puddle
function, with the matching previous-frame map, constants and TLAS.

Rain-map resources belong to the device. Preparation snapshots the parameters;
the two GPU passes run after the current scene upload and constants and before
surface shading. A defined miss texture is bound while the map is being written.
The map fades over its outer fifth, and points away from mapped ground do not
inherit that ground's depression. Bridge decks use an upward rain ray and the
procedural water level. Existing no-rain zones and force-disable rules still apply.

Actual animated-water geometry excludes submerged ground from puddles and rain
contact particles. The map caches water height above each ground texel; ambiguous
shore edges and stacked surfaces use an upward query against the matching ordered
and unordered scene. A nearer solid surface blocks water above another room.
This uses water geometry in the scene, not a global sea level or camera medium.

This is a height-field pooling approximation, not fluid simulation. A single
height/cover layer cannot fully represent stacked floors. Only geometry resident
in the scene acceleration structure can contribute. At default resolution there
are 512 x 512 downward ray queries each frame and water-boundary queries for valid
ground texels, plus 16 relief samples and the cover tent filter (normally 13 x 13
taps, capped at 25 x 25). Measure this GPU cost
in representative scenes; this feature does not promise an FPS gain.

## Configuration

Enable the existing wetness system and `rainMapEnabled` for measured depressions.
The GTA4 Standard preset sets the cell size to 2.8 m, coverage to 0.7 and
detail distance to 400 m. Its shipped rtx.conf sets puddle roughness to 0.035.
The reference runtime supplies the 0.03 m microrelief amplitude.
The VC configuration example sets `worldGroundMinRoughness = 0.2` for the
rain-wet base of static world ground. The runtime default is 0. The floor is
capped at the authored dry roughness before the standing-water layer is applied.
Nearby sloping ground can show rain impacts even when its slope prevents pooling.
Rain impacts and standing water share the same water-boundary and rain-cover check.

The **Terrain: Puddles** debug view (891) shows standing-water coverage in red,
damp shores in green and base ground wetness at 30% brightness in blue.
Submerged ground can retain blue while its red and green channels are zero.
Coverage controls water level, not a guaranteed percentage of screen pixels.

`surfaceRetentionEnabled` holds authored world-ground wetness after rain stops for
`surfaceHoldSeconds` (90 by default), then dries it over `surfaceDrySeconds` (120).
The live rainfall remains separate and drives rain impacts. Without a fully
authored weather amount, the existing per-surface wetness tags remain in use.
The weather preset's original transition time remains unchanged.

All controls are in the Wetness / Rain panel. Numeric settings are bounded before
shader upload. Ultra/High/Medium/Low use the same rain-map extent/resolution and
detail distance to preserve puddle placement; Custom leaves these values alone.

Old threshold/noise-frequency controls are removed. Do not copy their values to
the new cell-size/water-level controls. NUMOS weather presets can override
surface parameters: migrate their puddle keys as well as rtx.conf, keeping the
same terrain parameters in rainy and dry presets so retained water dries in place.
See docs/examples/vc-terrain-puddles.conf for the game-side reference settings.

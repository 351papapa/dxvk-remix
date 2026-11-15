# HAZE light distance controls

The Sky > Haze page offers separate controls for local-light participation and
spotlight reach. These affect NUMOS aerial perspective only. Surface lighting,
coronas, emissive materials, sunlight and the separate volumetric system retain
their existing settings.

| Setting | Meaning |
| --- | --- |
| `rtx.numos.hazeLightDistanceMeters` | Maximum camera-to-emitter distance for host lights in HAZE. Zero disables this limit. |
| `rtx.numos.hazeLightFadeMeters` | Distance over which lights fade before the camera limit. Zero uses the final 20% of the limit; values above the limit use the whole range. |
| `rtx.numos.hazeSpotlightReachMeters` | Maximum distance from a shaped light to its HAZE contribution. Zero keeps the existing power-derived reach. |

All three controls default to zero. They are authored scene/look controls and
remain independent of graphics presets. Camera distance uses the rendered view,
including a free camera. Distances use the same real game-unit calibration as
NUMOS, without the artistic World Scale or Haze World Scale multipliers.

Camera fade multiplies only the HAZE copy of a light's radiance before NUMOS
selects and ranks lights. As a light fades, its power-derived reach also shrinks.
Lights at or beyond the limit never enter this list. The global light budget
still applies to the remaining candidates; the complete local-list fix is still
required when more than sixteen lights overlap a column.

Spotlight reach limits the influence sphere, not the cone angle or the emitter
size. It does not extend lights whose existing reach is already shorter. It
applies only to lights with shaping, leaving NUMOS lightning unchanged. The
existing **Local Light Range Window** setting controls the smooth distance
falloff `(1 - (distance/reach)^2)^2`. Enable that setting to avoid an abrupt end;
the reach control does not silently override it. The falloff changes brightness
throughout the sphere, so a 30-metre reach is an upper bound rather than a promise
of a uniformly bright 30-metre column.

The controls update the host-side light copies and preserve the packed light
layout, NUMOS shader interface and complete-list representation. The GPU probe
observes the transformed bytes actually uploaded. No frame-time improvement is
claimed until a fixed-view, settled-weather GPU comparison has been recorded.

## Car headlight focus

The current Remix and NUMOS shaping functions use a cone-softness weight S and,
for a positive focus exponent n, return `S + (1-S) * pow(cos(theta), n) * S`.
Inside a zero-softness cone S is one, so changing focus has no effect there.
Focus is not a length control. Change cone angle for width, cone softness for
the edge transition, and the separate HAZE spotlight reach for length. This
feature does not alter the global light-shaping formula.

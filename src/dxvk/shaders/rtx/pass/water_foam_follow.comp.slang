#include "rtx/pass/water_foam_follow_args.h"
#include "rtx/pass/water_ocean_wave.slangh"

layout(push_constant) ConstantBuffer<WaterFoamFollowArgs> args;
layout(binding = WATER_FOAM_FOLLOW_INPUT) StructuredBuffer<float> input;
layout(binding = WATER_FOAM_FOLLOW_POSITIONS) StructuredBuffer<float3> positions;
struct FoamVertex { float3 position; float3 normal; float2 uv; uint color; };
layout(binding = WATER_FOAM_FOLLOW_SOURCE) StructuredBuffer<FoamVertex> source;
layout(binding = WATER_FOAM_FOLLOW_TARGET) RWStructuredBuffer<FoamVertex> target;

bool triangle(float3 a, float3 b, float3 c, float2 p, out float3 heightGradient) {
  float den = (b.y-c.y)*(a.x-c.x)+(c.x-b.x)*(a.y-c.y);
  heightGradient = 0;
  if (!isfinite(den) || abs(den)<1e-8f) { return false; }
  float u = ((b.y-c.y)*(p.x-c.x)+(c.x-b.x)*(p.y-c.y))/den;
  float v = ((c.y-a.y)*(p.x-c.x)+(a.x-c.x)*(p.y-c.y))/den;
  float w = 1-u-v;
  if (!isfinite(u+v+w) || min(min(u,v),w)<-.0001f) { return false; }
  float3 n = cross(b-a,c-a);
  heightGradient = float3(u*a.z+v*b.z+w*c.z,-n.x/n.z,-n.y/n.z);
  return all(isfinite(heightGradient));
}
bool cell(int2 c, float2 p, out float3 heightGradient) {
  heightGradient = 0;
  if (any(c<0) || any(c>=60)) { return false; }
  uint id = c.y*61+c.x;
  float3 a=positions[id],b=positions[id+1],d=positions[id+62],e=positions[id+61];
  if (triangle(a,b,d,p,heightGradient)) { return true; }
  return triangle(a,d,e,p,heightGradient);
}
bool fineCell(int2 c, float2 p, out float3 heightGradient) {
  heightGradient = 0;
  if (any(c < 0) || any(c >= 90)) { return false; }
  if (input[WATER_FINE_MASK + c.y * 90 + c.x] <= 0.0f) { return false; }
  uint id = WATER_COARSE_COUNT + c.y * 91 + c.x;
  float3 a = positions[id], b = positions[id + 1];
  float3 d = positions[id + 92], e = positions[id + 91];
  // Match the surface index buffer's diagonal exactly, including its actual
  // horizontal Gerstner displacement, wake height and local hull pressure.
  if (triangle(a, b, d, p, heightGradient)) { return true; }
  return triangle(a, d, e, p, heightGradient);
}

bool fineSurface(float2 p, out float3 heightGradient) {
  heightGradient = 0;
  if (input[9] <= 0.0f) { return false; }
  // Weather amplification limits q * sum(k*a) to 0.75. This conservative XY
  // reach is valid for any wave gain and catches triangles displaced outside
  // the nominal fine rectangle without scanning every fine-grid cell.
  float2 sums = oceanWaveSums(float4(input[16], input[17], input[18], input[19]),
    float4(input[24], input[25], input[26], input[27]));
  float reach = (sums.y > 0.0f ? 0.75f * sums.x / sums.y : 0.0f) + input[9];
  if (any(p < input[8] - reach) || any(p > input[29] + reach)) { return false; }
  float2 grid = clamp((p - input[8]) / input[9], float2(0.0f), float2(89.9999f));
  // Invert the deformed XY mesh instead of assuming the foam stayed in its
  // undeformed cell. A large swell can move it several fine cells sideways.
  [loop] for (uint iteration = 0; iteration < 6; ++iteration) {
    int2 c = int2(grid);
    float2 f = grid - float2(c);
    uint id = WATER_COARSE_COUNT + c.y * 91 + c.x;
    float2 a = positions[id].xy, b = positions[id + 1].xy;
    float2 d = positions[id + 92].xy, e = positions[id + 91].xy;
    float2 current = lerp(lerp(a, b, f.x), lerp(e, d, f.x), f.y);
    float2 error = current - p;
    if (dot(error, error) < 1e-8f) { break; }
    float2 dx = lerp(b - a, d - e, f.y);
    float2 dy = lerp(e - a, d - b, f.x);
    float determinant = dx.x * dy.y - dx.y * dy.x;
    if (!isfinite(determinant) || abs(determinant) < 1e-8f) { break; }
    float2 step = float2(error.x * dy.y - error.y * dy.x,
      dx.x * error.y - dx.y * error.x) / determinant;
    if (!all(isfinite(step))) { break; }
    grid = clamp(grid - clamp(step, float2(-16.0f), float2(16.0f)), float2(0.0f), float2(89.9999f));
  }
  int2 c = int2(grid);
  if (fineCell(c, p, heightGradient)) { return true; }
  // Bilinear inversion only locates a cell; return a true triangle height and
  // normal. The neighbour search also handles rounding at cell/patch edges.
  [loop] for (int y = -1; y <= 1; ++y) {
    [loop] for (int x = -1; x <= 1; ++x) {
      if ((x != 0 || y != 0) && fineCell(c + int2(x,y), p, heightGradient)) { return true; }
    }
  }
  return false;
}

[shader("compute")]
[numthreads(WATER_SURFACE_GROUP_SIZE,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint id=tid.x;
  if (id>=args.vertexCount) { return; }
  FoamVertex vertex=source[id];
  float2 p=vertex.position.xy-float2(input[0],input[1]);
  float2 g=p/input[3];
  if (all(g>=0) && all(g<60)) {
    int2 c=int2(g);
    float3 hg=0;
    bool found=fineSurface(p,hg);
    float fineHi=input[29];
    if (!found && (p.x<input[8] || p.y<input[8] || p.x>fineHi || p.y>fineHi)) {
      found=cell(c,p,hg);
      for (int y=-1;y<=1 && !found;++y) {
        for (int x=-1;x<=1 && !found;++x) {
          if (x!=0 || y!=0) { found=cell(c+int2(x,y),p,hg); }
        }
      }
    }
    if (!found) {
      uint k=c.y*61+c.x;
      float a=positions[k].z,b=positions[k+1].z,e=positions[k+61].z,d=positions[k+62].z;
      float2 f=g-float2(c);
      hg.x=lerp(lerp(a,b,f.x),lerp(e,d,f.x),f.y);
      hg.y=lerp(b-a,d-e,f.y)/input[3];
      hg.z=lerp(e-a,d-b,f.x)/input[3];
    }
    vertex.position.z=input[2]+hg.x+.035f;
    vertex.normal=normalize(float3(-hg.y,-hg.z,1));
  }
  target[id]=vertex;
}

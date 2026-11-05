#include "rtx/pass/water_surface_args.h"
#include "rtx/pass/water_ocean_wave.slangh"
#include "rtx/pass/water_shore_wave.slangh"
#include "rtx/pass/water_shore_depth.slangh"
#include "rtx/pass/water_beach_mask.slangh"
#include "rtx/pass/water_surface_clip.slangh"

layout(push_constant) ConstantBuffer<WaterSurfaceArgs> args;
layout(binding = WATER_SURFACE_INPUT) RWStructuredBuffer<float> input;
layout(binding = WATER_SURFACE_POSITIONS) RWStructuredBuffer<float3> positions;
layout(binding = WATER_SURFACE_BEACH_MASK) StructuredBuffer<uint> beachMask;
layout(binding = WATER_SURFACE_SHORE_DEPTH) StructuredBuffer<float4> shoreDepth;
struct WaterVertex { float3 position; float3 normal; float2 uv; uint color; };
layout(binding = WATER_SURFACE_VERTICES) RWStructuredBuffer<WaterVertex> vertices;
#ifdef WATER_GPU_INTERACTION
layout(binding = 225) StructuredBuffer<float2> interaction;
#endif
#ifdef WATER_GPU_HULL
layout(binding = 226) StructuredBuffer<float2> hull;
layout(binding = 227) StructuredBuffer<float4> hullDomain;
float hullHeight(float2 world) {
  float2 g=(world-hullDomain[0].xy)/hullDomain[0].z;
  if(any(g<0)||any(g>=128)) { return 0; }
  int2 c=int2(g);float2 f=frac(g);uint i=c.y*129+c.x;
  return lerp(lerp(hull[i].x,hull[i+1].x,f.x),lerp(hull[i+129].x,hull[i+130].x,f.x),f.y);
}
#endif

float3 ocean(float2 p,bool fine) {
  float4 k=float4(input[16],input[17],input[18],input[19]);
  OceanWave wave = sampleOceanWave(p,
    float4(input[16], input[17], input[18], input[19]),
    float4(input[20], input[21], input[22], input[23]),
    float4(input[24], input[25], input[26], input[27]) * args.oceanWaveStrength
      * oceanGridFilter(k,fine?input[9]:input[3]), input[28]);
  return float3(wave.horizontal, wave.height);
}

// Keep the attenuation field identical for coarse geometry, fine geometry
// and the analytic foam crest sampler. Interpolate wet * fade, not wet/fade
// separately: a dry corner must not contribute any ocean displacement.
float coarseWaveFade(int2 c) {
  float2 p = float2(c) * input[3];
  float edge = min(min(p.x, p.y), min(input[12] - p.x, input[12] - p.y));
  float a = saturate(edge / 9.0f);
  return input[11] > 0.0f ? input[WATER_FADE + c.y * 61 + c.x] : a * a * (3.0f - 2.0f * a);
}

float coarseSurfaceWeight(int2 c) {
  return input[WATER_WET + c.y * 61 + c.x] > 0.0f ? coarseWaveFade(c) : 0.0f;
}

float coarseOceanWeight(int2 c) {
  float shoreFade = args.beachWaveHeight > 0.0f ? shoreOceanAttenuation(smoothShoreBed(shoreDepth,c,false),
    beachSurfaceMask(beachMask, float2(input[0], input[1]) + float2(c) * input[3])) : 1.0f;
  return coarseSurfaceWeight(c) * shoreFade;
}

float3 coarseOceanOffset(int2 c) {
  return ocean(float2(c) * input[3] + float2(input[0], input[1]),false) * coarseOceanWeight(c);
}

float beachWave(int2 c, bool fine) {
  if (args.beachWaveHeight <= 0.0f) { return 0.0f; }
  float4 bed = smoothShoreBed(shoreDepth, c, fine);
  if (bed.y <= 0.0f) { return 0.0f; }
  const float spacing = fine ? input[9] : input[3];
  float depth = bed.x;
  float2 world = float2(input[0], input[1]) + (fine ? input[8] : 0.0f) + float2(c) * spacing;
  return sampleShoreWave(depth, bed.z, bed.w, world, args.timeSeconds, args.beachWaveHeight,
    args.beachWaveLength, args.beachWavePeriod, args.beachWaveDepth, beachSurfaceMask(beachMask, world), spacing).height;
}
float wake(float2 p) {
  float trail=0,splash=0;
  float duration=clamp(input[14],2,8);
  for(uint i=0;i<84;i++) {
    uint b=WATER_SECTIONS+i*8;
    float age=input[b+4],gain=input[b+6];
    if(age>=duration || gain<=0) continue;
    float2 delta=p-float2(input[b],input[b+1]),dir=float2(input[b+2],input[b+3]);
    float along=dot(delta,dir),side=abs(dot(delta,float2(-dir.y,dir.x)));
    float width=input[b+5],len=input[b+7];
    if(abs(along)>len*3 || side>width+.5f*age+.88f) continue;
    float q=(side-width-.5f*age)/.22f,t=age/duration;
    float life=1-t*t*(3-2*t),birth=min(age/.08f,1.f);
    float ridge=(1-2*q*q)*exp(-q*q);
    float middle=.22f*cos(age*15)*exp(-side*side/.08f);
    trail+=.07f*gain*birth*life*exp(-along*along/(len*len))*(ridge+middle);

    // Two diverging height ridges behind the hull. Taper both the downstream
    // end and the existing side bound so the extra term reaches zero before
    // either early-out boundary (these ridges do not model vorticity).
    float downstream = saturate(-along / max(len, 1.0f));
    float armWidth = max(width * (0.85f + 1.65f * downstream), 0.35f);
    float armCoordinate = abs(side) / armWidth;
    float arm = exp(-pow((armCoordinate - 0.72f) * 3.2f, 2.0f));
    float tail = saturate(-along / max(3.0f * len, 0.05f));
    float tailFade = 1.0f - smoothstep(0.60f, 1.0f, tail);
    float sideFade = 1.0f - smoothstep(max(0.0f, width + .5f * age + .44f),
      max(.01f, width + .5f * age + .88f), side);
    trail += .045f * gain * birth * life * downstream * arm * tailFade * sideFade;
  }
  for(uint i=0;i<4;i++) {
    uint b=WATER_SPLASHES+i*4;
    float age=input[b+2],gain=input[b+3];
    if(age>=2 || gain<=0) continue;
    float r=length(p-float2(input[b],input[b+1])),q=(r-.2f-1.2f*age)/.3f,life=1-age/2;
    splash+=.1f*gain*min(age/.06f,1.f)*life*life*(1-2*q*q)*exp(-q*q);
  }
  return .08f*tanh(trail/.08f)+.12f*tanh(splash/.12f);
}
float3 point(int2 c,bool fine) {
  c=clamp(c,int2(0),int2(fine?90:60));
  return positions[(fine?WATER_COARSE_COUNT:0)+c.y*(fine?91:61)+c.x];
}
// Harmonic slopes keep interpolation inside the sampled height interval.
float waterHeightSlope(float a,float b) {
  return a*b>0.0f ? 2.0f*a*b/(a+b) : 0.0f;
}
float waterHeightCurve(float a,float b,float c,float d,float t) {
  float m=waterHeightSlope(b-a,c-b),n=waterHeightSlope(c-b,d-c);
  float t2=t*t,t3=t2*t;
  return (2.0f*t3-3.0f*t2+1.0f)*b+(t3-2.0f*t2+t)*m
    +(-2.0f*t3+3.0f*t2)*c+(t3-t2)*n;
}
float smoothWaterHeight(int2 cell,float2 f) {
  float row[4];
  [unroll] for(int y=0;y<4;++y) {
    int2 p=cell+int2(0,y-1);
    row[y]=waterHeightCurve(point(p+int2(-1,0),false).z,point(p,false).z,
      point(p+int2(1,0),false).z,point(p+int2(2,0),false).z,f.x);
  }
  return waterHeightCurve(row[0],row[1],row[2],row[3],f.y);
}
#ifdef WATER_GPU_HULL
float coarseHull(int2 c) {
  uint i=c.y*61+c.x;
  if(input[WATER_WET+i]<=0) { return 0; }
  float2 p=float2(c)*input[3];
  float a=saturate(min(min(p.x,p.y),min(input[12]-p.x,input[12]-p.y))/9.f);
  float fade=input[11]>0?input[WATER_FADE+i]:a*a*(3-2*a);
  return hullHeight(point(c,false).xy+float2(input[0],input[1]))*fade;
}
#endif
float3 normal(int2 c,bool fine) {
  float3 a=point(c+int2(1,0),fine)-point(c-int2(1,0),fine);
  float3 b=point(c+int2(0,1),fine)-point(c-int2(0,1),fine);
  float3 n=cross(a,b);
  return n*rsqrt(max(dot(n,n),1e-8f));
}
float3 renderNormal(int2 c,bool fine) {
  float2 p=(fine?input[8]:0.0f)+float2(c)*(fine?input[9]:input[3]);
  float3 n=normal(c,fine);
  if(fine) {
    float edge=min(min(c.x,c.y),min(90-c.x,90-c.y))*input[9];
    float blend=saturate(edge/.7f);blend=blend*blend*(3-2*blend);
    if(blend<1) {
    float2 g=p/input[3];int2 cell=int2(g);float2 f=frac(g);
    float3 nc=lerp(lerp(normal(cell,false),normal(cell+int2(1,0),false),f.x),
             lerp(normal(cell+int2(0,1),false),normal(cell+int2(1,1),false),f.x),f.y);
    n=normalize(lerp(nc,n,blend));
    }
  }
  return n;
}
WaterClipVertex renderCorner(int2 c,bool fine) {
  uint id=(fine?WATER_COARSE_COUNT:0)+c.y*(fine?91:61)+c.x;
  WaterClipVertex v;
  v.position=positions[id]+float3(input[0],input[1],input[2]);
  v.normal=renderNormal(c,fine);
  v.uv=(fine?input[8]:0.0f)+float2(c)*(fine?input[9]:input[3]);
  v.gap=positions[id].z+shoreBedAt(shoreDepth,c,fine).z;
  return v;
}
void writeRenderVertex(uint id,WaterClipVertex source) {
  WaterVertex v;v.position=source.position;
  v.normal=normalize(source.normal);v.uv=source.uv;v.color=0xffffffffu;
  vertices[id]=v;
}
bool emitRenderTriangle(uint base,WaterClipVertex a,WaterClipVertex b,WaterClipVertex c,bool draw) {
  // Preserve the native water plane. The real opaque beach determines the visible intersection.
  [unroll] for(uint i=0;i<6;++i) { writeRenderVertex(base+i,a); }
  if(draw) {
    writeRenderVertex(base+1,b);writeRenderVertex(base+2,c);
  }
  return draw;
}
[shader("compute")]
[numthreads(WATER_SURFACE_GROUP_SIZE,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
  uint id=tid.x;
  if(args.phase==0) {
    if(id>=WATER_COARSE_COUNT) return;
    float2 p=float2(id%61,id/61)*input[3];
    float3 w=input[WATER_WET+id]>0 ? ocean(p+float2(input[0],input[1]),false) : float3(0);
    if (args.beachWaveHeight > 0.0f) {
      w *= shoreOceanAttenuation(smoothShoreBed(shoreDepth,int2(id%61,id/61),false),
        beachSurfaceMask(beachMask, p + float2(input[0], input[1])));
    }
    if (input[WATER_WET+id] > 0.0f) { w.z += beachWave(int2(id % 61, id / 61), false); }
    float fade=coarseWaveFade(int2(id % 61, id / 61));
#ifdef WATER_GPU_INTERACTION
    positions[id]=float3(p, interaction[id].x)+w*fade;
#else
    positions[id]=float3(p, input[WATER_HEIGHT+id])+w*fade;
#endif
#ifdef WATER_GPU_HULL
    if(input[WATER_WET+id]>0) { positions[id].z+=hullHeight(positions[id].xy+float2(input[0],input[1]))*fade; }
#endif
  } else if(args.phase==1) {
    if(id>=WATER_FINE_COUNT) return;
    int2 c=int2(id%91,id/91);
    float2 p=input[8]+float2(c)*input[9],g=p/input[3];
    int2 cell=min(int2(g),int2(59));
    float2 f=g-float2(cell);
    float3 v=lerp(lerp(point(cell,false),point(cell+int2(1,0),false),f.x),
                  lerp(point(cell+int2(0,1),false),point(cell+int2(1,1),false),f.x),f.y);
    float edge=min(min(c.x,c.y),min(90-c.x,90-c.y))*input[9];
    float fade=saturate(edge/.7f);fade=fade*fade*(3-2*fade);
    // Keep interpolated beach/interaction height, but replace only the coarse
    // ocean component with the continuous spectrum at this fine-grid parameter.
    // A zero edge fade preserves the exact coarse boundary and its normals.
    float3 coarseOcean = lerp(lerp(coarseOceanOffset(cell), coarseOceanOffset(cell + int2(1,0)), f.x),
      lerp(coarseOceanOffset(cell + int2(0,1)), coarseOceanOffset(cell + int2(1,1)), f.x), f.y);
    float oceanWeight = lerp(lerp(coarseOceanWeight(cell), coarseOceanWeight(cell + int2(1,0)), f.x),
      lerp(coarseOceanWeight(cell + int2(0,1)), coarseOceanWeight(cell + int2(1,1)), f.x), f.y);
    float3 fineOcean = ocean(p + float2(input[0],input[1]),true) * oceanWeight;
    v += (fineOcean - coarseOcean) * fade;
    // Smooth the coarse height field without creating extrema or following the beach upward.
    float4 bed=smoothShoreBed(shoreDepth,c,true);
    if(args.beachWaveHeight>0.0f && bed.y>0.99f && bed.x<args.beachWaveDepth*2.2f) {
      float coarseHeight=lerp(lerp(point(cell,false).z,point(cell+int2(1,0),false).z,f.x),
        lerp(point(cell+int2(0,1),false).z,point(cell+int2(1,1),false).z,f.x),f.y);
      float blend=smoothstep(0.0f,2.0f*input[3],edge)
        *beachSurfaceMask(beachMask,p+float2(input[0],input[1]));
      v.z+=(smoothWaterHeight(cell,f)-coarseHeight)*blend;
    }
    if(input[WATER_FINE_WET+id]>0) v.z+=wake(p+float2(input[0],input[1]))*fade;
#ifdef WATER_GPU_HULL
    // Replace interpolated coarse hull displacement with high-resolution sampling.
    float2 origin=float2(input[0],input[1]);
    float coarse=lerp(lerp(coarseHull(cell),coarseHull(cell+int2(1,0)),f.x),
      lerp(coarseHull(cell+int2(0,1)),coarseHull(cell+int2(1,1)),f.x),f.y);
    if(input[WATER_FINE_WET+id]>0) { v.z+=(hullHeight(v.xy+origin)-coarse)*fade; }
#endif
    positions[WATER_COARSE_COUNT+id]=v;
  } else {
    if(id>=WATER_RENDER_CELL_COUNT) { return; }
    bool fine=id>=3600;uint local=fine?id-3600:id;
    int side=fine?90:60;int2 c=int2(local%side,local/side);
    uint mask=(fine?WATER_FINE_MASK:WATER_COARSE_MASK)+local;
    bool native=input[mask]>0.0f;
    int half=int(input[10]);
    bool hole=!fine && all(c>=30-half) && all(c<30+half);
    bool draw=!hole && native;
    WaterClipVertex a=renderCorner(c,fine),b=renderCorner(c+int2(1,0),fine);
    WaterClipVertex d=renderCorner(c+int2(1,1),fine),e=renderCorner(c+int2(0,1),fine);
    bool first=emitRenderTriangle(id*12,a,b,d,draw);
    bool second=emitRenderTriangle(id*12+6,a,d,e,draw);
    // Native masks are independent of the sampled shoreline field.
    input[mask]=(first||second)?1.0f:0.0f;
  }
}

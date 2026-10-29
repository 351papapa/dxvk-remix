#include "rtx/pass/water_flow_args.h"
layout(push_constant) ConstantBuffer<WaterFlowArgs> args;
layout(binding = FLOW_STATE_SOURCE) StructuredBuffer<float4> state;
layout(binding = FLOW_STATE_TARGET) RWStructuredBuffer<float4> nextState;
layout(binding = FLOW_PRESSURE_SOURCE) StructuredBuffer<float> pressure;
layout(binding = FLOW_PRESSURE_TARGET) RWStructuredBuffer<float> nextPressure;
layout(binding = FLOW_DIVERGENCE) RWStructuredBuffer<float> divergence;
layout(binding = FLOW_SOURCES) StructuredBuffer<WaterFlowSource> sources;
layout(binding = FLOW_FORCES) StructuredBuffer<WaterFlowForce> forces;
layout(binding = FLOW_WATER_MAP) StructuredBuffer<float> waterMap;
struct FoamVertex { float3 position; float3 normal; float2 uv; uint color; };
layout(binding = FLOW_VERTICES) RWStructuredBuffer<FoamVertex> vertices;
layout(binding = FLOW_INDICES) RWStructuredBuffer<uint> indices;

float level(float2 world) {
  int2 c=int2(floor((world-float2(args.waterMapOriginX,args.waterMapOriginY))/args.waterMapCellSize));
  if(any(c<0)||any(c>=128)) { return -1e20f; }
  return waterMap[c.y*128+c.x];
}
bool wet(int2 c) {
  if(any(c<=0)||any(c>=FLOW_SIDE-1)) { return false; }
  return level(float2(args.originX,args.originY)+float2(c)*args.spacing)>-10000;
}
bool sameWater(int2 a,int2 b) {
  if(!wet(a)||!wet(b)) { return false; }
  float2 origin=float2(args.originX,args.originY);
  return abs(level(origin+float2(a)*args.spacing)-level(origin+float2(b)*args.spacing))<.1f;
}
float4 readState(int2 c) {
  if(!wet(c)) { return 0; }
  return state[c.y*FLOW_SIDE+c.x];
}
float4 sampleState(float2 g) {
  int2 c=int2(floor(g));float2 f=frac(g);
  return lerp(lerp(readState(c),readState(c+int2(1,0)),f.x),
    lerp(readState(c+int2(0,1)),readState(c+int2(1,1)),f.x),f.y);
}
float2 velocity(float2 g) {
  return float2(sampleState(g-float2(.5f,0)).x,sampleState(g-float2(0,.5f)).y);
}
float hash(uint n) {
  n^=n>>16;n*=0x7feb352du;n^=n>>15;n*=0x846ca68bu;n^=n>>16;
  return float(n&0xffffffu)/16777216.f;
}
float noise(float2 p) {
  int2 c=int2(floor(p));float2 f=frac(p);f=f*f*(3-2*f);
  float a=hash(uint(c.x)*73856093u^uint(c.y)*19349663u);
  float b=hash(uint(c.x+1)*73856093u^uint(c.y)*19349663u);
  float d=hash(uint(c.x+1)*73856093u^uint(c.y+1)*19349663u);
  float e=hash(uint(c.x)*73856093u^uint(c.y+1)*19349663u);
  return lerp(lerp(a,b,f.x),lerp(e,d,f.x),f.y);
}
float2 boatForce(float2 p) {
  float2 result=0;
  for(uint i=0;i<args.forceCount;++i) {
    WaterFlowForce force=forces[i];
    float2 motion=float2(force.vx,force.vy);
    float speed=length(motion);
    if(speed<.2f) { continue; }
    float2 dir=motion/speed,side=float2(-dir.y,dir.x);
    float2 delta=p-float2(force.x,force.y);
    float radius=max(force.radius,.6f),extent=max(force.length,radius);
    float along=dot(delta,dir),across=dot(delta,side);
    if(abs(along)>extent*3 || abs(across)>radius*5) { continue; }
    float core=exp(-along*along/(extent*extent)-across*across/(radius*radius));
    result+=dir*min(speed*.35f,4.f)*core*args.strength;
    // A counter-rotating pair is an authored boat force; transport and pressure solve its subsequent motion.
    for(int sign=-1;sign<=1;sign+=2) {
      float2 r=delta-side*(sign*radius);
      float shape=exp(-dot(r,r)/(radius*radius*2));
      result+=float2(-r.y,r.x)/radius*(sign*min(speed*.3f,5.f)*shape*args.swirl);
    }
  }
  return result;
}
[shader("compute")]
[numthreads(FLOW_GROUP_SIZE,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
  uint id=tid.x;if(id>=FLOW_COUNT) { return; }
  int2 c=int2(id%FLOW_SIDE,id/FLOW_SIDE);
  float2 origin=float2(args.originX,args.originY),world=origin+float2(c)*args.spacing;
  bool water=wet(c);
  if(args.operation==0) { nextState[id]=0;nextPressure[id]=0;divergence[id]=0;return; }
  if(args.operation==1) {
    int2 old=c+int2(args.shiftX,args.shiftY);
    float4 value=0;
    if(water&&all(old>=0)&&all(old<FLOW_SIDE)) { value=state[old.y*FLOW_SIDE+old.x]; }
    nextState[id]=value;return;
  }
  float4 value=water?state[id]:float4(0);
  if(args.operation==2) {
    if(water) {
      for(uint i=0;i<args.sourceCount;++i) {
        WaterFlowSource source=sources[i];
        float2 r=(world-float2(source.x,source.y))/max(source.radius,.01f);
        if(any(abs(r)>2.f)) { continue; }
        float add=source.amount*exp(-dot(r,r)*2);
        value.z=min(value.z+add,1.6f);
        value.xy=lerp(value.xy,float2(source.vx,source.vy),saturate(add));
      }
      float2 acceleration=float2(boatForce(world+float2(.5f,0)*args.spacing).x,
        boatForce(world+float2(0,.5f)*args.spacing).y);
      value.xy=clamp(value.xy+acceleration*args.dt,-6.f,6.f);
      if(!sameWater(c,c+int2(1,0))) { value.x=0; }
      if(!sameWater(c,c+int2(0,1))) { value.y=0; }
    }
    nextState[id]=value;return;
  }
  if(args.operation==3) {
    float2 faceX=float2(c)+float2(.5f,0),faceY=float2(c)+float2(0,.5f);
    float2 backX=faceX-velocity(faceX)*args.dt/args.spacing;
    float2 backY=faceY-velocity(faceY)*args.dt/args.spacing;
    value.x=water&&sameWater(c,c+int2(1,0))?sampleState(backX-float2(.5f,0)).x:0;
    value.y=water&&sameWater(c,c+int2(0,1))?sampleState(backY-float2(0,.5f)).y:0;
    value.xy*=exp(-args.drag*args.dt);
    nextState[id]=value;return;
  }
  if(args.operation==4) {
    float left=sameWater(c,c-int2(1,0))?readState(c-int2(1,0)).x:0;
    float down=sameWater(c,c-int2(0,1))?readState(c-int2(0,1)).y:0;
    divergence[id]=water?(value.x-left+value.y-down)/args.spacing:0;
    nextPressure[id]=0;return;
  }
  if(args.operation==5) {
    float sum=0;uint count=0;
    const int2 dirs[4]={int2(-1,0),int2(1,0),int2(0,-1),int2(0,1)};
    for(uint i=0;i<4;++i) {
      int2 n=c+dirs[i];
      if(sameWater(c,n)) { sum+=pressure[n.y*FLOW_SIDE+n.x];++count; }
    }
    nextPressure[id]=water&&count>0?(sum-divergence[id]*args.spacing*args.spacing)/count:0;
    return;
  }
  if(args.operation==6) {
    if(water&&sameWater(c,c+int2(1,0))) { value.x-=(pressure[id+1]-pressure[id])/args.spacing; } else { value.x=0; }
    if(water&&sameWater(c,c+int2(0,1))) { value.y-=(pressure[id+FLOW_SIDE]-pressure[id])/args.spacing; } else { value.y=0; }
    nextState[id]=value;return;
  }
  if(args.operation==7) {
    float2 back=float2(c)-velocity(float2(c))*args.dt/args.spacing;
    value.z=water?max(0.f,sampleState(back).z*exp(-args.dt*3/max(args.lifetime,.5f))):0;
    if(value.z<.00001f) { value.z=0; }
    nextState[id]=value;return;
  }
  if(args.operation==8) {
    float a=noise(world*.42f+float2(.13f,-.09f)*args.time);
    float b=noise(world*1.1f+float2(-.21f,.16f)*args.time);
    float density=1-exp(-value.z*2.8f);
    float alpha=density*saturate((density-(a*.38f+b*.18f))/.5f);
    FoamVertex vertex;
    vertex.position=float3(world,water?level(world)+.035f:0);
    vertex.normal=float3(0,0,1);
    vertex.uv=world/5+float2(-.045f,.035f)*args.time-value.xy*.14f+float2(a-.5f,b-.5f)*.22f;
    vertex.color=(uint(saturate(alpha)*255)<<24)|0x00ffffffu;
    vertices[id]=vertex;
    if(all(c<FLOW_SIDE-1)) {
      uint base=(c.y*(FLOW_SIDE-1)+c.x)*6;
      bool valid=water&&sameWater(c,c+int2(1,0))&&sameWater(c,c+int2(0,1))&&sameWater(c,c+int2(1,1));
      if(valid) { valid=max(max(value.z,state[id+1].z),max(state[id+FLOW_SIDE].z,state[id+FLOW_SIDE+1].z))>=.015f; }
      indices[base]=id; indices[base+1]=valid?id+1:id; indices[base+2]=valid?id+FLOW_SIDE+1:id;
      indices[base+3]=id; indices[base+4]=valid?id+FLOW_SIDE+1:id; indices[base+5]=valid?id+FLOW_SIDE:id;
    }
  }
}

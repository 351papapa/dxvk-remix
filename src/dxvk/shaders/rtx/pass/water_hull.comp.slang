#include "rtx/pass/water_hull_args.h"
layout(push_constant) ConstantBuffer<WaterHullArgs> args;
layout(binding=222) StructuredBuffer<float2> source;
layout(binding=223) RWStructuredBuffer<float2> target;
layout(binding=224) StructuredBuffer<float> waterMap;
layout(binding=225) StructuredBuffer<WaterHullForce> forces;
layout(binding=226) RWStructuredBuffer<float> pressure;
float level(int2 c) {
  float2 world=float2(args.originX,args.originY)+float2(c)*args.spacing;
  int2 m=int2(floor((world-float2(args.waterMapOriginX,args.waterMapOriginY))/args.waterMapCellSize));
  if(any(m<0)||any(m>=128)) { return -1e20f; }
  return waterMap[m.y*128+m.x];
}
bool wet(int2 c) { return all(c>0)&&all(c<HULL_SIDE-1)&&level(c)>-10000; }
float totalHeight(int2 c,int2 n,float own) {
  if(!wet(n)||abs(level(c)-level(n))>.1f) { return own; }
  uint i=n.y*HULL_SIDE+n.x;
  return source[i].x+pressure[i];
}
[shader("compute")]
[numthreads(64,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
  uint id=tid.x;if(id>=HULL_COUNT) { return; }
  int2 c=int2(id%HULL_SIDE,id/HULL_SIDE);
  if(args.operation==0) { target[id]=0;return; }
  if(args.operation==1) {
    int2 old=c+int2(args.shiftX,args.shiftY);
    float2 v=0;
    if(wet(c)&&all(old>0)&&all(old<HULL_SIDE-1)) { v=source[old.y*HULL_SIDE+old.x]; }
    target[id]=v;return;
  }
  if(args.operation==2) {
    float p=0;
    if(wet(c)) {
      float2 world=float2(args.originX,args.originY)+float2(c)*args.spacing;
      for(uint i=0;i<args.forceCount;++i) {
        WaterHullForce boat=forces[i];
        int2 sourceCell=int2(round((float2(boat.x,boat.y)-float2(args.originX,args.originY))/args.spacing));
        if(abs(level(c)-level(sourceCell))>.1f) { continue; }
        float2 motion=float2(boat.vx,boat.vy);
        float speed=length(motion);
        if(speed<.2f) { continue; }
        float2 dir=motion/speed,side=float2(-dir.y,dir.x);
        // Events are emitted at the stern; move the pressure footprint toward the hull.
        float2 delta=world-float2(boat.x,boat.y)-dir*boat.length;
        float2 q=float2(dot(delta,dir)/max(boat.length,1.f),dot(delta,side)/max(boat.radius,.6f));
        if(any(abs(q)>3)) { continue; }
        p+=min(.45f*(speed/12)*(speed/12)*args.strength,.9f)*exp(-dot(q,q)*.5f);
      }
    }
    pressure[id]=min(p,1.2f);return;
  }
  if(!wet(c)) { target[id]=0;return; }
  float2 value=source[id];
  float own=value.x+pressure[id];
  float lap=totalHeight(c,c+int2(1,0),own)+totalHeight(c,c-int2(1,0),own)
    +totalHeight(c,c+int2(0,1),own)+totalHeight(c,c-int2(0,1),own)-4*own;
  float edge=min(min(c.x,c.y),min(128-c.x,128-c.y));
  float loss=1-saturate(edge/10.f);
  value.y=(value.y+args.speed*args.speed*lap/(args.spacing*args.spacing)*args.dt)*exp(-(args.damping+8*loss*loss)*args.dt);
  float next=value.x+value.y*args.dt;
  value.x=clamp(next,-args.maxHeight,args.maxHeight);
  if((next>=args.maxHeight&&value.y>0)||(next<=-args.maxHeight&&value.y<0)) { value.y=0; }
  target[id]=value;
}

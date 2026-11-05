#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>
#include "remix/remix_c.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include "remixapi/water_gpu_simulation.h"

static void trace(const char* operation) { FILE* file = nullptr; fopen_s(&file, "water-stage.log", "a"); if (file) { fprintf(file, "%lu %s\n", GetTickCount(), operation); fclose(file); } }
static void check(HRESULT hr, const char* operation) {
  trace(operation);
  if (FAILED(hr)) {
    std::cerr << operation << " failed: 0x" << std::hex << hr << '\n';
    ExitProcess(2);
  }
}

struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  DWORD color;
  float u, v;
};

int wmain(int argc, wchar_t** argv) {
  if (argc != 3) {
    std::cerr << "Pass runtime output directory\n";
    return 1;
  }
  const std::wstring directory = argv[1];
  SetDllDirectoryW(directory.c_str());
  trace("Load library");
  const HMODULE dll = LoadLibraryExW((directory + L"\\d3d9.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!dll) {
    std::cerr << "Runtime load failed: " << GetLastError() << '\n';
    return 2;
  }
  auto create = reinterpret_cast<IDirect3D9* (WINAPI*)(UINT)>(GetProcAddress(dll, "Direct3DCreate9"));
  if (!create) {
    return 3;
  }
  const HWND window = CreateWindowExW(0, L"STATIC", L"Remix GPU water validation", WS_OVERLAPPEDWINDOW,
                                     0, 0, 320, 240, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
  if (!window) {
    return 4;
  }
  trace("Create API");
  IDirect3D9* api = create(D3D_SDK_VERSION);
  if (!api) {
    return 5;
  }
  D3DPRESENT_PARAMETERS presentation {};
  presentation.BackBufferWidth = 320;
  presentation.BackBufferHeight = 240;
  presentation.BackBufferFormat = D3DFMT_X8R8G8B8;
  presentation.BackBufferCount = 1;
  presentation.SwapEffect = D3DSWAPEFFECT_DISCARD;
  presentation.hDeviceWindow = window;
  presentation.Windowed = TRUE;
  presentation.EnableAutoDepthStencil = TRUE;
  presentation.AutoDepthStencilFormat = D3DFMT_D24S8;
  presentation.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
  IDirect3DDevice9* device = nullptr;
  trace("Create device");
  check(api->CreateDevice(0, D3DDEVTYPE_HAL, window, D3DCREATE_HARDWARE_VERTEXPROCESSING,
                         &presentation, &device), "CreateDevice");
  const int phase = _wtoi(argv[2]);
  D3DMATRIX world {};
  world._11 = world._22 = world._33 = world._44 = 1.0f;
  D3DMATRIX view = world;
  view._22 = -0.70710678f; view._23 = -0.70710678f; view._32 = 0.70710678f; view._33 = -0.70710678f; view._43 = 8.48528137f;
  D3DMATRIX projection {};
  projection._11 = 1.3f;
  projection._22 = 1.7f;
  projection._33 = 100.0f / 99.9f;
  projection._34 = 1.0f;
  projection._43 = -0.1f * projection._33;
  check(device->SetTransform(D3DTS_WORLD, &world), "Set world");
  check(device->SetTransform(D3DTS_VIEW, &view), "Set view");
  check(device->SetTransform(D3DTS_PROJECTION, &projection), "Set projection");
  check(device->SetRenderState(D3DRS_LIGHTING, FALSE), "Disable fixed lighting");
  check(device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE), "Disable culling");
  check(device->SetFVF(D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_DIFFUSE | D3DFVF_TEX1), "Set FVF");
  IDirect3DTexture9* texture = nullptr;
  check(device->CreateTexture(4, 4, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &texture, nullptr), "Create texture");
  D3DLOCKED_RECT locked {};
  check(texture->LockRect(0, &locked, nullptr, 0), "Lock texture");
  for (int y = 0; y < 4; ++y) {
    auto row = reinterpret_cast<DWORD*>(static_cast<char*>(locked.pBits) + y * locked.Pitch);
    for (int x = 0; x < 4; ++x) {
      row[x] = 0xff40c080u;
    }
  }
  check(texture->UnlockRect(0), "Unlock texture");
  check(device->SetTexture(0, texture), "Set texture");

  auto apiCheck = [](remixapi_ErrorCode result, const char* operation) {
    trace(operation);
    if (result != REMIXAPI_ERROR_CODE_SUCCESS) { std::cerr << operation << " failed: " << int(result) << '\n'; ExitProcess(20); }
  };
  remixapi_Interface remix {};
  auto initialize = reinterpret_cast<PFN_remixapi_InitializeLibrary>(GetProcAddress(dll,"remixapi_InitializeLibrary"));
  remixapi_InitializeLibraryInfo init {};
  init.sType=REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO;
  init.version=REMIXAPI_VERSION_MAKE(REMIXAPI_VERSION_MAJOR,REMIXAPI_VERSION_MINOR,REMIXAPI_VERSION_PATCH);
  if(!initialize)return 6;
  apiCheck(initialize(&init,&remix),"Initialize Remix API");
#if defined(_WIN64)
  apiCheck(remix.dxvk_RegisterD3D9Device(static_cast<IDirect3DDevice9Ex*>(device)),"Register native D3D9 device");
#endif
  remixapi_MaterialInfoTranslucentEXT water {};
  water.sType=REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_TRANSLUCENT_EXT;
  water.refractiveIndex=1.333f;water.transmittanceColor={.5f,.85f,.95f};water.transmittanceMeasurementDistance=7.5f;
  remixapi_MaterialInfo material {};
  material.sType=REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;material.pNext=&water;material.hash=0x57415445524C4142ull;
  material.spriteSheetRow=material.spriteSheetCol=1;
  remixapi_MaterialHandle waterHandle {},foamHandle {},mediumHandle {};
  apiCheck(remix.CreateMaterial(&material,&waterHandle),"Create GPU water material");
  material.hash=0x574154455243414Dull;water.transmittanceColor={1,1,1};
  apiCheck(remix.CreateMaterial(&material,&mediumHandle),"Create camera medium");
  remixapi_MaterialInfoOpaqueEXT foam {};
  foam.sType=REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
  foam.albedoConstant={1,1,1};foam.opacityConstant=1;foam.roughnessConstant=.8f;
  material.pNext=&foam;material.hash=0x574C4649454C4432ull;
  apiCheck(remix.CreateMaterial(&material,&foamHandle),"Create GPU wake material");
  if(!remix.SetCameraMediumMaterial)return 21;
  remixapi_CameraMediumInfo medium {};medium.sType=REMIXAPI_STRUCT_TYPE_CAMERA_MEDIUM_INFO;
  if(remix.SetCameraMediumMaterial(nullptr)!=REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS)return 22;
  apiCheck(remix.SetCameraMediumMaterial(&medium),"Clear camera medium");
  if(phase==5){
    medium.medium=mediumHandle;apiCheck(remix.SetCameraMediumMaterial(&medium),"Enter camera medium");
    view._22=-.16439899f;view._23=.98639392f;view._32=.98639392f;view._33=.16439899f;view._42=0;view._43=6.08276253f;
    check(device->SetTransform(D3DTS_VIEW,&view),"Underwater view");
  }
  remixapi_LightInfoDistantEXT distant {};distant.sType=REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISTANT_EXT;
  distant.direction={0,0,-1};distant.angularDiameterDegrees=.53f;distant.volumetricRadianceScale=1;
  remixapi_LightInfo light {};light.sType=REMIXAPI_STRUCT_TYPE_LIGHT_INFO;light.pNext=&distant;light.hash=0x5741544552544553ull;light.radiance={3,3,3};
  remixapi_LightHandle lightHandle {};apiCheck(remix.CreateLight(&light,&lightHandle),"Create test sunlight");
  namespace gpu=remixapi::water::gpu;namespace sim=remixapi::water::simulation;
  HANDLE waterMutex=CreateMutexW(nullptr,FALSE,gpu::kMutexName),simMutex=CreateMutexW(nullptr,FALSE,sim::kMutexName);
  HANDLE waterMapping=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,sizeof(gpu::Frame),gpu::kMappingName);
  HANDLE simMapping=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,sizeof(sim::Shared),sim::kMappingName);
  if(!waterMutex||!simMutex||!waterMapping||!simMapping)return 23;
  auto shared=static_cast<gpu::Frame*>(MapViewOfFile(waterMapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(gpu::Frame)));
  auto simulation=static_cast<sim::Shared*>(MapViewOfFile(simMapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(sim::Shared)));
  if(!shared||!simulation)return 24;
  WaitForSingleObject(waterMutex,INFINITE);*shared=gpu::Frame{};ReleaseMutex(waterMutex);
  WaitForSingleObject(simMutex,INFINITE);memset(simulation,0,sizeof(*simulation));simulation->version=sim::kVersion;simulation->bytes=sizeof(*simulation);
  simulation->waterMapReady=1;std::fill_n(simulation->waterMap,128*128,0.0f);ReleaseMutex(simMutex);
  auto input=std::make_unique<gpu::Frame>();auto packet=std::make_unique<sim::Packet>();
  float* f=input->data;
  f[0]=-30;f[1]=-30;f[2]=0;f[3]=1;f[5]=phase==1?0:.35f;f[6]=12;f[7]=.4f;
  f[8]=15;f[9]=1.f/3;f[10]=15;f[11]=1;f[12]=60;f[14]=5;
  std::fill_n(f+gpu::kWet,gpu::kCoarseCount,1.f);std::fill_n(f+gpu::kFade,gpu::kCoarseCount,1.f);
  std::fill_n(f+gpu::kCoarseMask,3600,1.f);std::fill_n(f+gpu::kFineMask,8100,1.f);std::fill_n(f+gpu::kFineWet,gpu::kFineCount,1.f);
  std::vector<Vertex> ground={{-20,-20,-7,0,-.2873479f,.9578263f,0xffffffff,0,0},{20,-20,-7,0,-.2873479f,.9578263f,0xffffffff,1,0},
    {-20,20,5,0,-.2873479f,.9578263f,0xffffffff,0,1},{-20,20,5,0,-.2873479f,.9578263f,0xffffffff,0,1},
    {20,-20,-7,0,-.2873479f,.9578263f,0xffffffff,1,0},{20,20,5,0,-.2873479f,.9578263f,0xffffffff,1,1}};
  IDirect3DVertexBuffer9* vertices=nullptr;
  check(device->CreateVertexBuffer(UINT(ground.size()*sizeof(Vertex)),0,0,D3DPOOL_MANAGED,&vertices,nullptr),"Create shoreline ground");
  void* mapped=nullptr;check(vertices->Lock(0,0,&mapped,0),"Lock ground");memcpy(mapped,ground.data(),ground.size()*sizeof(Vertex));check(vertices->Unlock(),"Unlock ground");
  uint32_t waterAcks=0,solverAcks=0,consumed=0,lastReadSequence=0,lastEpoch=0;
  const unsigned frameCount=phase==10?660u:180u;
  for(uint32_t frame=0;frame<frameCount;++frame){
    if(frame==60&&(phase==2||phase==3))apiCheck(remix.SetConfigVariable("rtx.water.enabled","False"),"Disable water");
    if(frame==85&&phase==3)apiCheck(remix.SetConfigVariable("rtx.water.enabled","True"),"Re-enable water");
    if(frame==70&&phase==4){f[0]-=2;f[1]+=3;}
    if(frame==70&&phase==7)f[3]=std::numeric_limits<float>::quiet_NaN();
    f[4]=phase==8?1.0f:float(frame)/60;
    gpu::prepareWaveConstants(f);input->tick=GetTickCount();input->sequence=frame+1;input->active=1;
    WaitForSingleObject(waterMutex,INFINITE);
    waterAcks+=shared->ready!=0;const auto ready=shared->ready,readyTick=shared->readyTick;
    *shared=*input;shared->ready=ready;shared->readyTick=readyTick;ReleaseMutex(waterMutex);
    *packet=sim::Packet{};packet->surface=*input;packet->sequence=frame+1;packet->resetSerial=1;
    packet->originX=f[0];packet->originY=f[1];packet->spacing=1;packet->steps=phase==8?0:2;
    for(unsigned i=0;i<gpu::kCoarseCount;++i){packet->mask[i][0]=packet->mask[i][1]=1;}
    if(phase==6&&frame==45){packet->impulseCount=1;packet->impulses[0]={30,30,2,5,25,25,35,35};}
    if((phase==14||phase==15)&&frame==45){
      packet->steps=0;packet->impulseCount=phase==14?2:1;
      packet->impulses[0]={24,22,3,3,20,18,28,26};
      packet->impulses[1]={36,22,3,-2,32,18,40,26};
    }
    packet->flow.enabled=sim::kFoamFlow|sim::kHullWater;packet->flow.resetSerial=1;
    packet->flow.originX=-32;packet->flow.originY=-32;packet->flow.dt=phase==8?0:1.f/60;packet->flow.time=f[4];
    if(phase==6){packet->flow.sourceCount=1;packet->flow.sources[0]={0,0,1,.5f,1,0};packet->flow.forceCount=1;packet->flow.forces[0]={0,0,2,0,1,3};}
    WaitForSingleObject(simMutex,INFINITE);solverAcks+=simulation->ready!=0;
    if(simulation->epoch!=lastEpoch){lastEpoch=simulation->epoch;lastReadSequence=0;}
    consumed+=simulation->readSequence-lastReadSequence;lastReadSequence=simulation->readSequence;
    if(simulation->writeSequence-simulation->readSequence>=sim::kQueueSize){++simulation->epoch;simulation->writeSequence=simulation->readSequence=0;}
    simulation->tick=GetTickCount();simulation->active=1;simulation->packets[simulation->writeSequence%sim::kQueueSize]=*packet;++simulation->writeSequence;
    if(phase==15&&frame==45){
      packet->impulses[0]=packet->impulses[1];
      simulation->packets[simulation->writeSequence%sim::kQueueSize]=*packet;++simulation->writeSequence;
    }
    ReleaseMutex(simMutex);
    check(device->Clear(0,nullptr,D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER,0xff101010,1,0),"Clear");
    check(device->BeginScene(),"BeginScene");
    apiCheck(remix.DrawLightInstance(lightHandle),"Draw sunlight");
    check(device->SetStreamSource(0,vertices,0,sizeof(Vertex)),"Bind ground");
    check(device->DrawPrimitive(D3DPT_TRIANGLELIST,0,2),"Draw ground");
    check(device->EndScene(),"EndScene");check(device->Present(nullptr,nullptr,nullptr,nullptr),"Present");Sleep(20);
  }
  medium.medium=nullptr;apiCheck(remix.SetCameraMediumMaterial(&medium),"Exit camera medium");
  WaitForSingleObject(waterMutex,INFINITE);shared->active=0;ReleaseMutex(waterMutex);
  WaitForSingleObject(simMutex,INFINITE);simulation->active=0;ReleaseMutex(simMutex);
  std::cout<<"GPU handshake surface="<<waterAcks<<" solver="<<solverAcks<<" consumed="<<consumed<<" phase="<<phase<<'\n';
  UnmapViewOfFile(shared);UnmapViewOfFile(simulation);CloseHandle(waterMapping);CloseHandle(simMapping);CloseHandle(waterMutex);CloseHandle(simMutex);
  remix.DestroyLight(lightHandle);remix.DestroyMaterial(mediumHandle);remix.DestroyMaterial(waterHandle);remix.DestroyMaterial(foamHandle);
  vertices->Release();texture->Release();device->Release();api->Release();DestroyWindow(window);
  return waterAcks>30 && solverAcks>30 && consumed>0 ? 0:25;
}

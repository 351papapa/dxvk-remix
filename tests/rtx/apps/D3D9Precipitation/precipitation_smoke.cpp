#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>
#include "remix/remix_c.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static void trace(const char* operation) { FILE* file = nullptr; fopen_s(&file, "precipitation-stage.log", "a"); if (file) { fprintf(file, "%lu %s\n", GetTickCount(), operation); fclose(file); } }
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
  const HWND window = CreateWindowExW(0, L"STATIC", L"GPU precipitation validation", WS_POPUP,
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

  std::vector<Vertex> ground = {
    {-25,-25,0,0,0,1,0xffaaaaaau,0,1}, {-25,25,0,0,0,1,0xffaaaaaau,0,0}, {25,-25,0,0,0,1,0xffaaaaaau,1,1},
    {25,-25,0,0,0,1,0xffaaaaaau,1,1}, {-25,25,0,0,0,1,0xffaaaaaau,0,0}, {25,25,0,0,0,1,0xffaaaaaau,1,0}
  };
  const bool yUp = phase == 2 || phase == 19;
  if (yUp) {
    for (auto& v : ground) { v.z=-v.y;v.y=0;v.ny=1;v.nz=0; }
    D3DMATRIX rotate=world,adjusted{};rotate._22=0;rotate._23=1;rotate._32=-1;rotate._33=0;
    for(int a=0;a<4;++a)for(int b=0;b<4;++b)for(int c=0;c<4;++c)adjusted.m[a][b]+=rotate.m[a][c]*view.m[c][b];
    view=adjusted;check(device->SetTransform(D3DTS_VIEW,&view),"Y-up view");
  }
  auto apiCheck=[](remixapi_ErrorCode result,const char* operation){
    if(result!=REMIXAPI_ERROR_CODE_SUCCESS){std::cerr<<operation<<" failed "<<int(result)<<'\n';ExitProcess(20);}
  };
  remixapi_Interface remix{};
  auto initialize=reinterpret_cast<PFN_remixapi_InitializeLibrary>(GetProcAddress(dll,"remixapi_InitializeLibrary"));
  remixapi_InitializeLibraryInfo init{};init.sType=REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO;
  init.version=REMIXAPI_VERSION_MAKE(REMIXAPI_VERSION_MAJOR,REMIXAPI_VERSION_MINOR,REMIXAPI_VERSION_PATCH);
  if(!initialize)return 6;
  apiCheck(initialize(&init,&remix),"Initialize Remix");
#if defined(_WIN64)
  apiCheck(remix.dxvk_RegisterD3D9Device(static_cast<IDirect3DDevice9Ex*>(device)),"Register native device");
#endif
  auto config=[&](const char* key,const char* value){apiCheck(remix.SetConfigVariable(key,value),key);};
  auto game=[&](const char* key,const char* value){apiCheck(remix.SetGameValue(key,value),key);};
  if(yUp)config("rtx.zUp","False");
  if(phase==1)config("rtx.weather.precipitation.enable","False");
  if(phase==8){config("rtx.weather.precipitation.fallSpeed","0.5");config("rtx.weather.precipitation.streak","0");config("rtx.weather.precipitation.turbulence","0.5");}
  const bool contact = phase>=9 && phase<=15 || phase==19;
  if(contact)config("rtx.weather.precipitation.enable","False");
  if(phase==16||phase==17||phase==18){config("rtx.weather.preset","RainTest");config("rtx.weather.transitionSeconds","1");}
  if(phase==18)config("rtx.weather.preset","InvalidTest");
  if(phase==14)config("rtx.weather.preset","RainTest");
  if(phase==15)config("rtx.wetness.respectNoRainZones","True");
  if(phase==15)game("__wetness.norain_zones_v1","1 1 -25 25 -25 25 -1 2");
  remixapi_MaterialInfoOpaqueEXT opaque{};opaque.sType=REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
  opaque.albedoConstant={1,.4f,.1f};opaque.opacityConstant=1;opaque.roughnessConstant=.8f;opaque.alphaTestType=7;opaque.blendType_hasvalue=1;opaque.blendType_value=0;
  remixapi_MaterialInfo mi{};mi.sType=REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;mi.pNext=&opaque;mi.hash=0x5052454349504D54ull;
  mi.emissiveIntensity=5;mi.emissiveColorConstant={1,.2f,.05f};mi.spriteSheetRow=mi.spriteSheetCol=1;
  remixapi_MaterialHandle material{};apiCheck(remix.CreateMaterial(&mi,&material),"Create contact material");
  remixapi_HardcodedVertex emitterVertices[4]{};
  const float corners[4][2]={{-2,-2},{2,-2},{-2,2},{2,2}};
  for(int i=0;i<4;++i){emitterVertices[i].position[0]=corners[i][0];emitterVertices[i].position[yUp?2:1]=yUp?-corners[i][1]:corners[i][1];
    emitterVertices[i].position[yUp?1:2]=.2f;emitterVertices[i].normal[yUp?1:2]=1;emitterVertices[i].color=0xffffffffu;}
  uint32_t emitterIndices[6]={0,1,2,3,2,1};
  remixapi_MeshInfoSurfaceTriangles surface{};surface.vertices_values=emitterVertices;surface.vertices_count=4;
  surface.indices_values=emitterIndices;surface.indices_count=6;surface.material=material;
  remixapi_MeshInfo meshInfo{};meshInfo.sType=REMIXAPI_STRUCT_TYPE_MESH_INFO;meshInfo.hash=0x5052454349504D53ull;
  meshInfo.surfaces_values=&surface;meshInfo.surfaces_count=1;
  remixapi_MeshHandle mesh{};apiCheck(remix.CreateMesh(&meshInfo,&mesh),"Create contact emitter");
  const char* contactOption=phase==11 ? "rtx.particles.puddleEmitterMeshes" : (phase==12||phase==13) ? "rtx.particles.waterContactEmitterMeshes" : "rtx.particles.wetRoadEmitterMeshes";
  if(contact)config(contactOption,"0x5052454349504D53");
  remixapi_InstanceInfoParticleSystemEXT particles{};particles.sType=REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_PARTICLE_SYSTEM_EXT;
  particles.maxNumParticles=1000;particles.hideEmitter=1;particles.minTimeToLive=particles.maxTimeToLive=2;
  particles.spawnRatePerSecond=400;particles.initialVelocityFromNormal=100;particles.billboardType=0;particles.spriteSheetMode=0;
  remixapi_Float4D colors[2]={{1,.5f,.1f,1},{1,.5f,.1f,1}};particles.minColor={colors,2};particles.maxColor={colors,2};
  remixapi_Float2D sizes[2]={{20,20},{20,20}};particles.minSize={sizes,2};particles.maxSize={sizes,2};
  remixapi_InstanceInfo instance{};instance.sType=REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;instance.pNext=&particles;instance.mesh=mesh;instance.doubleSided=1;
  instance.transform.matrix[0][0]=instance.transform.matrix[1][1]=instance.transform.matrix[2][2]=1;
  remixapi_MeshHandle waterMesh{};
  if(phase==12||phase==13){
    auto waterOpaque=opaque;waterOpaque.blendType_hasvalue=0;waterOpaque.albedoConstant={.1f,.25f,.8f};
    auto waterMaterialInfo=mi;waterMaterialInfo.pNext=&waterOpaque;waterMaterialInfo.hash=0x505245435741544Dull;
    waterMaterialInfo.emissiveIntensity=0;
    remixapi_MaterialHandle waterMaterial{};apiCheck(remix.CreateMaterial(&waterMaterialInfo,&waterMaterial),"Create water material");
    surface.material=waterMaterial;
    // The contact ray searches 35 cm below the emitter. Keep the plane within that authored range.
    for(auto& v:emitterVertices){v.position[0]*=10;v.position[yUp?2:1]*=10;v.position[yUp?1:2]=-.1f;}
    meshInfo.hash=0x5052454357415452ull;apiCheck(remix.CreateMesh(&meshInfo,&waterMesh),"Create water contact plane");
  }
  remixapi_LightInfoDistantEXT sun{};sun.sType=REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISTANT_EXT;sun.direction=yUp?remixapi_Float3D{0,-1,0}:remixapi_Float3D{0,0,-1};sun.angularDiameterDegrees=.53f;sun.volumetricRadianceScale=1;
  remixapi_LightInfo light{};light.sType=REMIXAPI_STRUCT_TYPE_LIGHT_INFO;light.pNext=&sun;light.hash=0x5052454353554Eull;light.radiance={3,3,3};
  remixapi_LightHandle lightHandle{};apiCheck(remix.CreateLight(&light,&lightHandle),"Create sunlight");
  IDirect3DVertexBuffer9* groundBuffer=nullptr;
  check(device->CreateVertexBuffer(UINT(ground.size()*sizeof(Vertex)),0,0,D3DPOOL_MANAGED,&groundBuffer,nullptr),"Create ground");
  void* mapped=nullptr;check(groundBuffer->Lock(0,0,&mapped,0),"Lock ground");memcpy(mapped,ground.data(),ground.size()*sizeof(Vertex));check(groundBuffer->Unlock(),"Unlock ground");
  auto pumpMessages=[](){MSG message{};while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageW(&message);}};
  for(uint32_t frame=0;frame<240;++frame){
    pumpMessages();
    if(frame==60&&(phase==3||phase==4))config("rtx.weather.precipitation.enable","False");
    if(frame==120&&phase==4)config("rtx.weather.precipitation.enable","True");
    if(frame==60&&phase==5)game("__weather.precipitation_enable","0");
    if(frame==60&&phase==14)config("rtx.weather.preset","DryTest");
    if(frame==60&&phase==17)config("rtx.weather.preset","DryTest");
    check(device->Clear(0,nullptr,D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER,0xff101010u,1,0),"Clear");
    check(device->BeginScene(),"BeginScene");
    world._41=phase==12?100.f:0.f;world._42=world._43=0;
    check(device->SetTransform(D3DTS_WORLD,&world),"Ground transform");
    check(device->SetStreamSource(0,groundBuffer,0,sizeof(Vertex)),"Bind ground");
    check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(210),(2u<<24)|0x100u|(phase==10?0u:255u)),"Road wetness tag");
    check(device->DrawPrimitive(D3DPT_TRIANGLELIST,0,2),"Draw ground");
    if(phase==6||phase==7){
      world._11=phase==7?.45f:1;world._41=phase==7?-12.f:0;world._43=15;
      check(device->SetTransform(D3DTS_WORLD,&world),"Roof transform");
      check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(210),0),"Dry roof");
      check(device->DrawPrimitive(D3DPT_TRIANGLELIST,0,2),"Draw roof");world._11=1;
    }
    if(waterMesh){auto wi=instance;wi.mesh=waterMesh;wi.pNext=nullptr;wi.categoryFlags=REMIXAPI_INSTANCE_CATEGORY_BIT_ANIMATED_WATER;apiCheck(remix.DrawInstance(&wi),"Draw water plane");}
    if(contact)apiCheck(remix.DrawInstance(&instance),"Draw contact emitter");
    apiCheck(remix.DrawLightInstance(lightHandle),"Draw sunlight");
    check(device->EndScene(),"EndScene");check(device->Present(nullptr,nullptr,nullptr,nullptr),"Present");Sleep(5);
  }
  groundBuffer->Release();texture->Release();device->Release();api->Release();
  // Bridge releases are asynchronous. Keep the surface alive while the server drains its tail frame.
  for(int wait=0;wait<200;++wait){pumpMessages();Sleep(5);}
  DestroyWindow(window);
  std::cout<<"Precipitation GPU fixture completed phase="<<phase<<'\n';return 0;
}

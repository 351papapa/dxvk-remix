#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>
#include "remix/remix_c.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static void trace(const char* operation) { FILE* file = nullptr; fopen_s(&file, "lens-rain-stage.log", "a"); if (file) { fprintf(file, "%lu %s\n", GetTickCount(), operation); fclose(file); } }
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
  const int phase = _wtoi(argv[2]);
  const UINT width=(phase==10||phase==16)?240:320,height=(phase==10||phase==16)?320:240;
  const HWND window = CreateWindowExW(0, L"STATIC", L"GPU lens rain validation", WS_POPUP,
                                     0, 0, width, height, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
  if (!window) {
    return 4;
  }
  trace("Create API");
  IDirect3D9* api = create(D3D_SDK_VERSION);
  if (!api) {
    return 5;
  }
  D3DPRESENT_PARAMETERS presentation {};
  presentation.BackBufferWidth = width;
  presentation.BackBufferHeight = height;
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
  check(device->CreateTexture(64, 64, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &texture, nullptr), "Create texture");
  D3DLOCKED_RECT locked {};
  check(texture->LockRect(0, &locked, nullptr, 0), "Lock texture");
  for (int y = 0; y < 64; ++y) {
    auto row = reinterpret_cast<DWORD*>(static_cast<char*>(locked.pBits) + y * locked.Pitch);
    for (int x = 0; x < 64; ++x) {
      row[x] = ((x/4+y/4)%2) ? 0xffe06020u : 0xff1040e0u;
    }
  }
  check(texture->UnlockRect(0), "Unlock texture");
  check(device->SetTexture(0, texture), "Set texture");

  std::vector<Vertex> ground = {
    {-25,-25,0,0,0,1,0xffaaaaaau,0,1}, {-25,25,0,0,0,1,0xffaaaaaau,0,0}, {25,-25,0,0,0,1,0xffaaaaaau,1,1},
    {25,-25,0,0,0,1,0xffaaaaaau,1,1}, {-25,25,0,0,0,1,0xffaaaaaau,0,0}, {25,25,0,0,0,1,0xffaaaaaau,1,0}
  };
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
  config("rtx.lensRain.forceAmount",phase==0||phase==16?"0":"1");
  if(phase==8){config("rtx.lensRain.dropCount","1");config("rtx.lensRain.slideSpeed","0");}
  if(phase==9)config("rtx.lensRain.dropCount","1024");
  if(phase==4||phase==15) {
    config("rtx.lensRain.respectShelter","True");
    game("__wetness.norain_zones_v1","1 1 -30 30 -30 30 -30 30");
  }
  if(phase==12) {
    config("rtx.lensRain.forceAmount","-1");
    config("rtx.weather.preset","RainTest");
  }
  if(phase==14)config("rtx.lensRain.slideSpeed","0");
  IDirect3DVertexBuffer9* groundBuffer=nullptr;
  check(device->CreateVertexBuffer(UINT(ground.size()*sizeof(Vertex)),0,0,D3DPOOL_MANAGED,&groundBuffer,nullptr),"Create ground");
  void* mapped=nullptr;check(groundBuffer->Lock(0,0,&mapped,0),"Lock ground");memcpy(mapped,ground.data(),ground.size()*sizeof(Vertex));check(groundBuffer->Unlock(),"Unlock ground");
  auto pumpMessages=[](){MSG message{};while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageW(&message);}};
  for(uint32_t frame=0;frame<240;++frame){
    pumpMessages();
    if(phase==2 || phase==3&&frame<60) {
      const std::string motion="1 "+std::to_string(frame)+" 25 0 0";
      game("__lensRain.motion_v1",motion.c_str());
    }
    if(phase==13)game("__lensRain.motion_v1","1 5 nan 0 0");
    if(frame==60&&phase==5)config("rtx.lensRain.forceAmount","0");
    if(frame==60&&(phase==6||phase==7))config("rtx.lensRain.lensRainEnable","False");
    if(frame==120&&phase==7)config("rtx.lensRain.lensRainEnable","True");
    if(phase==11&&frame==60)config("rtx.lensRain.dropCount","1024");
    if(phase==11&&frame==120)config("rtx.lensRain.dropCount","32");
    if(phase==15&&frame==120)game("__wetness.norain_zones_v1","1 0");
    check(device->Clear(0,nullptr,D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER,0xff101010u,1,0),"Clear");
    check(device->BeginScene(),"BeginScene");
    world._41=world._42=world._43=0;
    check(device->SetTransform(D3DTS_WORLD,&world),"Ground transform");
    check(device->SetStreamSource(0,groundBuffer,0,sizeof(Vertex)),"Bind ground");
    check(device->DrawPrimitive(D3DPT_TRIANGLELIST,0,2),"Draw ground");
    check(device->EndScene(),"EndScene");check(device->Present(nullptr,nullptr,nullptr,nullptr),"Present");Sleep(5);
  }
  groundBuffer->Release();texture->Release();device->Release();api->Release();
  // Bridge releases are asynchronous. Keep the surface alive while the server drains its tail frame.
  for(int wait=0;wait<200;++wait){pumpMessages();Sleep(5);}
  DestroyWindow(window);
  std::cout<<"Lens rain GPU fixture completed phase="<<phase<<'\n';return 0;
}

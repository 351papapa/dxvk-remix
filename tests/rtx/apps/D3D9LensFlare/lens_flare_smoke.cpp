#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>
#include "remix/remix_c.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static void trace(const char* operation) { FILE* file = nullptr; fopen_s(&file, "lens-flare-stage.log", "a"); if (file) { fprintf(file, "%lu %s\n", GetTickCount(), operation); fclose(file); } }
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
  const UINT width=320,height=240;
  const HWND window = CreateWindowExW(0, L"STATIC", L"GPU lens flare validation", WS_POPUP,
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
  auto replaceReference=[](const char* source){
    if(!CopyFileA(source,"identity-hot.dds",FALSE))return false;
    HANDLE file=CreateFileA("identity-hot.dds",FILE_WRITE_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,0,nullptr);
    if(file==INVALID_HANDLE_VALUE)return false;
    FILETIME now;GetSystemTimeAsFileTime(&now);
    const bool result=SetFileTime(file,nullptr,nullptr,&now)!=FALSE;CloseHandle(file);return result;
  };
  const std::string assets="G:/Codex/2026-10-29/ban/work/rpv2-source-migration/flare-assets/";
  config("rtx.fork.lensFlare.enabled",phase==0?"False":"True");
  if(phase==2)config("rtx.fork.lensFlare.identityDds",(assets+"missing.dds").c_str());
  if(phase==3){config("rtx.fork.lensFlare.albedoPath",(assets+"missing.dds").c_str());config("rtx.fork.lensFlare.emissivePath",(assets+"missing.dds").c_str());}
  if(phase==4)for(auto& v:ground)v.color=0xff40ff80u;
  if(phase==9)config("rtx.fork.lensFlare.identityDds",(assets+"wrong.dds").c_str());
  if(phase==10)config("rtx.fork.lensFlare.identityDds",(assets+"malformed.dds").c_str());
  if(phase==11)config("rtx.fork.lensFlare.albedoPath",(assets+"green.dds").c_str());
  if(phase==12)config("rtx.fork.lensFlare.emissivePath",(assets+"green.dds").c_str());
  if(phase==13||phase==14){
    if(!replaceReference((assets+"identity.dds").c_str()))return 21;
    config("rtx.fork.lensFlare.identityDds","identity-hot.dds");
  }
  if(phase==15){
    for(auto& v:ground)v.color=0x80aaaaaau;
    check(device->SetTextureStageState(0,D3DTSS_ALPHAOP,D3DTOP_MODULATE),"Alpha modulation");
    check(device->SetTextureStageState(0,D3DTSS_ALPHAARG2,D3DTA_DIFFUSE),"Alpha vertex input");
  }
  if(phase==19)for(auto& v:ground)v.color=0xff555555u;
  if(phase==20)for(auto& v:ground)v.color=0xff000000u;
  if(phase==16)config("rtx.vertexColorIsBakedLighting","False");
  if(phase==17)config("rtx.enablePreservePath","False");
  if(phase==18)config("rtx.fork.lensFlare.emissiveLevel","0");
  IDirect3DVertexBuffer9* groundBuffer=nullptr;
  check(device->CreateVertexBuffer(UINT(ground.size()*sizeof(Vertex)),0,0,D3DPOOL_MANAGED,&groundBuffer,nullptr),"Create ground");
  void* mapped=nullptr;check(groundBuffer->Lock(0,0,&mapped,0),"Lock ground");memcpy(mapped,ground.data(),ground.size()*sizeof(Vertex));check(groundBuffer->Unlock(),"Unlock ground");
  auto pumpMessages=[](){MSG message{};while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageW(&message);}};
  for(uint32_t frame=0;frame<240;++frame){
    pumpMessages();
    if(frame==60&&(phase==5||phase==6))config("rtx.fork.lensFlare.enabled","False");
    if(frame==120&&phase==6)config("rtx.fork.lensFlare.enabled","True");
    if(frame==60&&phase==7)config("rtx.fork.lensFlare.emissiveLevel","2");
    if(frame==60&&phase==8)config("rtx.fork.lensFlare.identityDds",(assets+"wrong.dds").c_str());
    if(frame==120&&phase==8)config("rtx.fork.lensFlare.identityDds",(assets+"identity.dds").c_str());
    if(frame==60&&(phase==13||phase==14)&&!replaceReference((assets+"wrong.dds").c_str()))return 22;
    if(frame==120&&phase==13&&!replaceReference((assets+"identity.dds").c_str()))return 23;
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
  std::cout<<"Lens flare GPU fixture completed phase="<<phase<<'\n';return 0;
}

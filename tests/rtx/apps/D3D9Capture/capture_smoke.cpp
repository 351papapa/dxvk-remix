#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>
#include <cstdint>
#include <iostream>
#include <string>

static void check(HRESULT hr, const char* operation) {
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
  if (argc != 2) {
    std::cerr << "Pass runtime output directory\n";
    return 1;
  }
  const std::wstring directory = argv[1];
  SetDllDirectoryW(directory.c_str());
  const HMODULE dll = LoadLibraryExW((directory + L"\\d3d9.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!dll) {
    std::cerr << "Runtime load failed: " << GetLastError() << '\n';
    return 2;
  }
  auto create = reinterpret_cast<IDirect3D9* (WINAPI*)(UINT)>(GetProcAddress(dll, "Direct3DCreate9"));
  if (!create) {
    return 3;
  }
  const HWND window = CreateWindowExW(0, L"STATIC", L"Remix draw capture validation", WS_OVERLAPPEDWINDOW,
                                     0, 0, 320, 240, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
  if (!window) {
    return 4;
  }
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
  check(api->CreateDevice(0, D3DDEVTYPE_HAL, window, D3DCREATE_HARDWARE_VERTEXPROCESSING,
                         &presentation, &device), "CreateDevice");
  D3DMATRIX world {};
  world._11 = world._22 = world._33 = world._44 = 1.0f;
  D3DMATRIX view = world;
  view._43 = 5.0f;
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
  const Vertex triangle[] = {
    {-1, -1, 0, 0, 0, -1, 0xffffffffu, 0, 1},
    { 0,  1, 0, 0, 0, -1, 0xffffffffu, .5f, 0},
    { 1, -1, 0, 0, 0, -1, 0xffffffffu, 1, 1},
  };
  const uint32_t session = GetTickCount() | 1u;
  std::cout << "CAPTURE_SESSION=" << session << std::endl;
  for (uint32_t frame = 0; frame < 28; ++frame) {
    check(device->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xff101010u, 1.0f, 0), "Clear");
    check(device->BeginScene(), "BeginScene");
    const uint32_t captureSession = frame < 10 ? 0 : session + (frame >= 20 ? 1 : 0);
    check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(213), captureSession), "Set session");
    for (uint32_t token = 1; token <= 2; ++token) {
      check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(214), captureSession ? token : 0), "Set token");
      check(device->DrawPrimitiveUP(D3DPT_TRIANGLELIST, 1, triangle, sizeof(Vertex)), "Draw");
    }
    check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(214), 0), "Clear token");
    check(device->EndScene(), "EndScene");
    check(device->Present(nullptr, nullptr, nullptr, nullptr), "Present");
    Sleep(20);
  }
  check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(213), 0), "Clear session");
  texture->Release();
  device->Release();
  api->Release();
  DestroyWindow(window);
  // Runtime workers are shut down by process exit; do not unload the DLL while they unwind.
  std::cout << "Capture draw submission complete\n";
  return 0;
}

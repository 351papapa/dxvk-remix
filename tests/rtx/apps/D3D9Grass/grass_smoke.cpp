#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>
#include "remix/remix_c.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static void trace(const char* operation) { FILE* file = nullptr; fopen_s(&file, "grass-stage.log", "a"); if (file) { fprintf(file, "%lu %s\n", GetTickCount(), operation); fclose(file); } }
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
  const HWND window = CreateWindowExW(0, L"STATIC", L"Remix draw capture validation", WS_OVERLAPPEDWINDOW,
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
  std::vector<Vertex> triangle = {
    {-4,-4,0, 0,0,1, 0xffffffffu, 0,1},
    {-4,4,0, 0,0,1, 0xffffffffu, 0,0},
    {4,-4,0, 0,0,1, 0xffffffffu, 1,1},
    {4,-4,0, 0,0,1, 0xffffffffu, 1,1},
    {-4,4,0, 0,0,1, 0xffffffffu, 0,0},
    {4,4,0, 0,0,1, 0xffffffffu, 1,0}
  };
  if (phase == 2) {
    for (auto& vertex : triangle) {
      vertex.z = -vertex.y; vertex.y = 0;
      vertex.ny = 1; vertex.nz = 0;
    }
    D3DMATRIX rotate = world;
    rotate._22 = 0; rotate._23 = 1; rotate._32 = -1; rotate._33 = 0;
    D3DMATRIX adjusted {};
    for (int row = 0; row < 4; ++row) {
      for (int col = 0; col < 4; ++col) {
        for (int k = 0; k < 4; ++k) adjusted.m[row][col] += rotate.m[row][k] * view.m[k][col];
      }
    }
    check(device->SetTransform(D3DTS_VIEW, &adjusted), "Y-up view");
  }
  remixapi_Interface remix {};
  auto initialize = reinterpret_cast<PFN_remixapi_InitializeLibrary>(GetProcAddress(dll, "remixapi_InitializeLibrary"));
  remixapi_InitializeLibraryInfo info {};
  info.sType = REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO;
  info.version = REMIXAPI_VERSION_MAKE(REMIXAPI_VERSION_MAJOR, REMIXAPI_VERSION_MINOR, REMIXAPI_VERSION_PATCH);
  if (!initialize || initialize(&info, &remix) != REMIXAPI_ERROR_CODE_SUCCESS || !remix.SetGameValue) return 7;
  auto set = [&remix](const char* key, const char* value) {
    if (remix.SetGameValue(key, value) != REMIXAPI_ERROR_CODE_SUCCESS) ExitProcess(8);
  };
  IDirect3DVertexBuffer9* vertices = nullptr;
  check(device->CreateVertexBuffer(UINT(triangle.size() * sizeof(Vertex)), 0, 0, D3DPOOL_MANAGED, &vertices, nullptr), "Create vertices");
  void* mapped = nullptr;
  check(vertices->Lock(0, 0, &mapped, 0), "Lock vertices");
  memcpy(mapped, triangle.data(), UINT(triangle.size() * sizeof(Vertex)));
  check(vertices->Unlock(), "Unlock vertices");
  check(device->SetStreamSource(0, vertices, 0, sizeof(Vertex)), "Bind vertices");
  IDirect3DIndexBuffer9* indices = nullptr;
  if (phase == 7 || phase == 8) {
    const bool wide = phase == 8;
    check(device->CreateIndexBuffer(wide ? 24 : 12, 0, wide ? D3DFMT_INDEX32 : D3DFMT_INDEX16, D3DPOOL_MANAGED, &indices, nullptr), "Create indices");
    check(indices->Lock(0, 0, &mapped, 0), "Lock indices");
    for (unsigned i = 0; i < 6; ++i) {
      if (wide) static_cast<uint32_t*>(mapped)[i] = i;
      else static_cast<uint16_t*>(mapped)[i] = uint16_t(i);
    }
    check(indices->Unlock(), "Unlock indices");
    check(device->SetIndices(indices), "Bind indices");
  }
  std::vector<IDirect3DVertexBuffer9*> extraGeometry;
  if (phase == 9) {
    for (unsigned i = 0; i < 256; ++i) {
      Vertex extra[] = {{1000.f+i*2,0,0,0,0,1,0xffffffffu,0,0},
                        {1001.f+i*2,0,0,0,0,1,0xffffffffu,1,0},
                        {1000.f+i*2,1,0,0,0,1,0xffffffffu,0,1}};
      IDirect3DVertexBuffer9* buffer = nullptr;
      check(device->CreateVertexBuffer(sizeof(extra), 0, 0, D3DPOOL_MANAGED, &buffer, nullptr), "Create extra geometry");
      check(buffer->Lock(0, 0, &mapped, 0), "Lock extra geometry");
      memcpy(mapped, extra, sizeof(extra));
      check(buffer->Unlock(), "Unlock extra geometry");
      extraGeometry.push_back(buffer);
    }
  }
  for (uint32_t frame = 0; frame < 140; ++frame) {
    if (frame == 60 && phase == 3) {
      if (remix.SetConfigVariable("rtx.grass.enabled", "False") != REMIXAPI_ERROR_CODE_SUCCESS) return 9;
    }
    if (frame == 60 && phase == 4) {
      if (remix.SetConfigVariable("rtx.grass.reloadVersion", "1") != REMIXAPI_ERROR_CODE_SUCCESS) return 9;
    }
    if (frame == 60 && phase == 5) {
      if (remix.SetConfigVariable("rtx.grass.maxInstances", "32") != REMIXAPI_ERROR_CODE_SUCCESS) return 9;
    }
    check(device->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0xff101010u, 1.0f, 0), "Clear");
    check(device->BeginScene(), "BeginScene");
    // Capture only during the first two frames; subsequent draws exercise retained reuse.
    check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(213), frame < 2 ? 31337 : 0), "Capture session");
    check(device->SetRenderState(static_cast<D3DRENDERSTATETYPE>(214), frame < 2 ? 1 : 0), "Capture token");
    check(device->SetStreamSource(0, vertices, 0, sizeof(Vertex)), "Bind ground");
    if (indices) {
      check(device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, 6, 0, 2), "Draw indexed ground");
    } else {
      check(device->DrawPrimitive(D3DPT_TRIANGLELIST, 0, UINT(triangle.size() / 3)), "Draw ground");
    }
    for (auto* buffer : extraGeometry) {
      check(device->SetStreamSource(0, buffer, 0, sizeof(Vertex)), "Bind extra geometry");
      check(device->DrawPrimitive(D3DPT_TRIANGLELIST, 0, 1), "Draw extra geometry");
    }
    check(device->EndScene(), "EndScene");
    check(device->Present(nullptr, nullptr, nullptr, nullptr), "Present");
    Sleep(20);
  }
  for (auto* buffer : extraGeometry) buffer->Release();
  if (indices) indices->Release();
  trace("Releasing vertices");
  vertices->Release();
  trace("Released vertices");
  trace("Releasing texture");
  texture->Release();
  trace("Released texture");
  trace("Releasing device");
  device->Release();
  trace("Released device");
  trace("Releasing api");
  api->Release();
  trace("Released api");
  DestroyWindow(window);
  // Runtime workers are shut down by process exit; do not unload the DLL while they unwind.
  std::cout << "Grass draw submission complete\n";
  return 0;
}

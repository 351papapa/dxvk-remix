# Font atlas upload

The embedded Inter and Roboto fonts produce a coverage-only atlas. The frontend uploads the existing Alpha8 data to an R8_UNORM image. Its image view maps components to (ONE, ONE, ONE, R), supplying the same white RGB and filtered alpha as the RGBA8 atlas. The sampler and text geometry are unchanged. No color-font input is registered by this frontend.

This path does not call ImFontAtlas::GetTexDataAsRGBA32, so initialization needs neither its four-byte-per-pixel CPU expansion allocation nor its conversion loop. For N atlas pixels, the GPU texture and upload carry N bytes rather than 4N. Rendering and component expansion remain on the GPU.

The observed initialization crash was a null write inside GetTexDataAsRGBA32 at imgui_draw.cpp:2065. The existing conversion loop did not check the expansion allocation. A monitored rerun succeeded; the original allocation-failure trigger is not established, so this fix does not claim to solve arbitrary system memory exhaustion.

Before changing the upload, the current NVIDIA upstream createFontsTexture implementation was reviewed on 2026-10-30. It still expands and uploads RGBA8. This local change affects only the frontend upload and leaves the bundled ImGui implementation unchanged. NVIDIA MIT notices remain in the source and LICENSE-MIT.

Upstream comparison: https://github.com/NVIDIAGameWorks/dxvk-remix/blob/main/src/dxvk/imgui/dxvk_imgui.cpp

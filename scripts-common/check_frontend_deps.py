"""Frontend include rules.

1. The backend never includes frontend code or an API module: a source file under one of the backend directories may
   not include a path containing "frontends/", nor a header of src/d3d9/, src/dxso/, src/d3d11/, src/d3d10/,
   src/dxgi/ or src/dxbc/.
2. The frontend door: a file under src/frontends/ includes backend headers only from DOOR (the scene interface, the
   value types a draw is made of, the util / option / UI layers and the DXVK device types) or from SCOPED_DOOR, which
   names the frontends each of its headers is for.
3. No D3D9 in the backend: no backend file names the D3D9 frontend's draw record or its services (DrawCallState,
   DrawCallService, D3D9Rtx), a D3D9 SDK type (D3D*, IDirect3D*, LPDIRECT3D*), or a type the D3D9 module declares
   (a class, struct, enum or alias of src/d3d9/, src/dxso/ or src/frontends/d3d9/ named D3D9* or Dxso*).
4. No D3D11 or DXGI in the backend: no backend file names a D3D11, D3D10 or DXGI SDK type (ID3D11*, D3D11_*,
   PFN_D3D11*, the same for D3D10, IDXGI*, DXGI_*), or a type the D3D11, D3D10, DXGI or DXBC module declares (a class,
   struct, enum or alias of src/d3d11/, src/d3d10/, src/dxgi/ or src/dxbc/ named D3D11*, D3D10*, Dxgi*, DXGI* or
   Dxbc*).

There is no allowlist. Rules 3 and 4 cover the interface headers too, and ignore comments and string literals.
D3DKMT_ names are Windows display-kernel types, and DXGI_ERROR_ and DXGI_STATUS_ names are HRESULT codes of
winerror.h, so neither counts as an SDK name. Module types count only where the module declares them, because a plain
prefix pattern would also hit backend names that only mention an API, such as the perf counters perf::D3D9Ctr and
Api::RenderD3D11.

Run by src/frontends/meson.build at configure time. Usage: check_frontend_deps.py <source root>
"""
import os
import re
import sys

BACKEND_DIRS = ["src/dxvk", "src/util", "src/vulkan", "src/spirv", "src/lssusd"]
FRONTEND_DIR = "src/frontends"
SOURCE_EXTS = (".h", ".hpp", ".cpp", ".c", ".inl", ".slang", ".slangh")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')

# Backend headers every frontend may include, relative to the source root. Everything under src/util/ is allowed too.
DOOR = {
    "src/dxvk/rtx_render/rtx_frontend_interface.h",
    "src/dxvk/rtx_render/rtx_frontend_ui.h",
    "src/dxvk/rtx_render/rtx_fork_material_derivation.h",
    "src/dxvk/rtx_render/rtx_imgui.h",
    "src/dxvk/rtx_render/rtx_materials.h",
    "src/dxvk/rtx_render/rtx_matrix_helpers.h",
    "src/dxvk/rtx_render/rtx_option.h",
    "src/dxvk/rtx_render/rtx_options.h",
    "src/dxvk/rtx_render/rtx_replacement_material_texture_type.h",
    "src/dxvk/rtx_render/rtx_scene_types.h",
    "src/dxvk/rtx_render/translator/rtx_translator_lifetime.h",
    "src/dxvk/dxvk_buffer.h",
    "src/dxvk/dxvk_context.h",
    "src/dxvk/dxvk_device.h",
    "src/dxvk/dxvk_image.h",
    "src/dxvk/dxvk_sampler.h",
}

# Backend headers only some frontends may include: the files or directories allowed to, and why.
SCOPED_DOOR = {
    "src/dxvk/rtx_render/rtx_remix_api_device.h": (
        ["src/frontends/api/"],
        "the registered API device and the backend calls of the API's D3D9 device and D3D11 host"),
    "src/dxvk/rtx_render/rtx_fork_api_recorder.h": (
        ["src/frontends/api/"], "the recording of the API's dxvk_ calls and of RenderD3D11"),
    "src/dxvk/dxvk_perf_counters.h": (
        ["src/frontends/api/"], "the timing scopes of the API entry points"),
    "src/dxvk/rtx_render/rtx_color_grade.h": (
        ["src/frontends/api/"], "the color grading exports"),
}


class ApiModule:
    """An API the backend may not include or name: the directories of its module, the directories whose declarations
    count as its types, the prefixes of those types, and its SDK names."""

    def __init__(self, include_dirs, type_dirs, type_prefixes, sdk_re):
        self.include_dirs = include_dirs
        self.type_dirs = type_dirs
        self.prefixed_re = re.compile(r"\b(?:%s)[A-Za-z0-9_]+\b" % type_prefixes)
        self.declared_re = re.compile(
            r"\b(?:class|struct|union|enum(?:\s+class|\s+struct)?|using|typedef\b[^;{}]*?)\s+((?:%s)[A-Za-z0-9_]+)\b"
            % type_prefixes)
        self.sdk_re = sdk_re


API_MODULES = [
    ApiModule(
        ["src/d3d9", "src/dxso"],
        ["src/d3d9", "src/dxso", "src/frontends/d3d9"],
        "D3D9|Dxso",
        re.compile(r"\b(?!D3DKMT_)(?:D3D[A-Z_][A-Z0-9_]*|D3D9_[A-Za-z0-9_]*|IDirect3D\w*|LPDIRECT3D\w*)\b")),
    ApiModule(
        ["src/d3d11", "src/d3d10", "src/dxgi", "src/dxbc"],
        ["src/d3d11", "src/d3d10", "src/dxgi", "src/dxbc"],
        "D3D11|D3D10|Dxgi|DXGI|Dxbc",
        re.compile(r"\b(?:ID3D1[01]\w*|D3D1[01]_\w*|PFN_D3D1[01]\w*|IDXGI\w*|DXGI_(?!ERROR_|STATUS_)\w*)\b")),
]

DRAW_CALL_RE = re.compile(r"\b(?:DrawCallState|DrawCallService|D3D9Rtx)\b")
COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"')


def norm(path):
    return path.replace("\\", "/")


def source_files(root, rel):
    for dirpath, _, filenames in os.walk(os.path.join(root, rel)):
        for name in filenames:
            if name.endswith(SOURCE_EXTS):
                yield os.path.join(dirpath, name)


def includes(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                m = INCLUDE_RE.match(line)
                if m:
                    yield lineno, norm(m.group(1))
    except OSError:
        return


def resolve(root, path, include):
    """The included file relative to the source root, or None when it is not a file of this tree."""
    for base in (os.path.dirname(path), os.path.join(root, "src"), os.path.join(root, "src/dxvk")):
        candidate = os.path.normpath(os.path.join(base, include))
        if os.path.isfile(candidate):
            return norm(os.path.relpath(candidate, root))
    return None


def is_backend(rel):
    return any(rel.startswith(d + "/") for d in BACKEND_DIRS)


def is_api_module(rel):
    return any(rel.startswith(d + "/") for module in API_MODULES for d in module.include_dirs)


def code(path):
    """A file's code without comments and string literals."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return STRING_RE.sub('""', COMMENT_RE.sub(" ", f.read()))
    except OSError:
        return ""


def module_types(root, module):
    """The types with the module's prefixes that its type directories declare."""
    names = set()
    for rel in module.type_dirs:
        for path in source_files(root, rel):
            names.update(module.declared_re.findall(code(path)))
    return names


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    failed = False

    violations = []
    for rel in BACKEND_DIRS:
        for path in source_files(root, rel):
            for lineno, include in includes(path):
                target = resolve(root, path, include)
                if "frontends/" in include or (target is not None and is_api_module(target)):
                    violations.append(f"{norm(os.path.relpath(path, root))}:{lineno}: {include}")
    if violations:
        print("Backend files include frontend or API module code (rule 1):")
        for v in violations:
            print("  " + v)
        failed = True

    outside = []
    for path in source_files(root, FRONTEND_DIR):
        source = norm(os.path.relpath(path, root))
        for lineno, include in includes(path):
            target = resolve(root, path, include)
            if target is None or not is_backend(target):
                continue
            if target.startswith("src/util/") or target in DOOR:
                continue
            scoped = SCOPED_DOOR.get(target)
            if scoped is not None and any(source.startswith(allowed) for allowed in scoped[0]):
                continue
            outside.append(f"{source}:{lineno}: {include}")
    if outside:
        print("Frontend files include backend headers outside the frontend door "
              "(rule 2):")
        for v in outside:
            print("  " + v)
        failed = True

    types = [module_types(root, module) for module in API_MODULES]
    named = []
    for rel in BACKEND_DIRS:
        for path in source_files(root, rel):
            text = code(path)
            names = set(DRAW_CALL_RE.findall(text))
            for module, declared in zip(API_MODULES, types):
                names |= set(module.sdk_re.findall(text))
                names |= {name for name in module.prefixed_re.findall(text) if name in declared}
            if names:
                named.append(f"{norm(os.path.relpath(path, root))}: {', '.join(sorted(names))}")
    if named:
        print("Backend files name the D3D9 frontend's record, or SDK or module types of D3D9, D3D11 or DXGI "
              "(rules 3 and 4):")
        for v in named:
            print("  " + v)
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

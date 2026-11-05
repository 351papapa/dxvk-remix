#!/usr/bin/env python3
"""Externalize only gbuffer's generated SPIR-V arrays without changing /O2.

audit reads source/cache and can write a report.
generate writes declaration overrides, an AMD64 COFF object, and its manifest.
patch-ninja changes only the generated gbuffer compile/archive edges and adds a
custom generation edge. apply performs generate followed by patch-ninja.
apply is idempotent and replaces a previous generated rule after Meson regeneration.
scripts/build-validated.ps1 owns the ordered configure/shader/COFF/link workflow.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
import zlib

DEFAULT_RUNTIME = Path(__file__).resolve().parents[1]
INCLUDE = re.compile(rb'^\s*#\s*include\s*[<"]rtx_shaders/([^>"\r\n]+)[>"]', re.M)
ANY_INCLUDE = re.compile(rb'^\s*#\s*include\s*[<"]([^>"\r\n]+)[>"]', re.M)
ARRAY = re.compile(rb'\A\s*const\s+uint32_t\s+([A-Za-z_][A-Za-z_0-9]*)\s*\[\s*\]\s*=\s*\{(.*?)\};\s*(?:const\s+size_t\s+([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([0-9]+)\s*;\s*)?\Z', re.S)
BEGIN = "# BEGIN SUNFLARE GBUFFER EXTERNAL SPIRV"
END = "# END SUNFLARE GBUFFER EXTERNAL SPIRV"
SECTION_FLAGS = 0x40301040  # INITIALIZED_DATA | COMDAT | ALIGN_4BYTES | MEM_READ


def configured_paths(runtime: Path, build: Path) -> tuple[Path, Path, dict]:
    info = json.loads((build / "meson-info/meson-info.json").read_text(encoding="utf-8"))
    source = Path(info["directories"]["source"])
    configured_build = Path(info["directories"]["build"])
    if not runtime.samefile(source) or not build.samefile(configured_build):
        raise ValueError("Requested source/build directory is not the configured physical directory")
    # Keep the configured source spelling and a source-local build alias. Resolving
    # the junction can introduce spaces into the existing Ninja compile paths.
    relative_build = build.relative_to(runtime)
    alias = source / relative_build
    if not alias.samefile(configured_build):
        raise ValueError("Configured source has no matching build-directory alias")
    return source, alias, info


def absolute(path: str | Path) -> Path:
    # Do not resolve the build-directory junction into its spaced J: target.
    return Path(os.path.abspath(path))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_changed(path: Path, data: bytes) -> None:
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def declaration(entry: dict) -> bytes:
    text = f'#pragma once\n#include <cstdint>\n#include <cstddef>\nextern "C" const uint32_t {entry["symbol"]}[{entry["words"]}];\n'
    if entry.get("size_symbol"):
        text += f'const size_t {entry["size_symbol"]} = {entry["spv_bytes"]};\n'
    return text.encode("ascii")


def inspect(runtime: Path, build: Path) -> dict:
    source = runtime / "src/dxvk/rtx_render/rtx_pathtracer_gbuffer.cpp"
    shader_dir = build / "src/dxvk/rtx_shaders"
    source_data = source.read_bytes()
    direct = sorted(set(x.decode("ascii") for x in INCLUDE.findall(source_data)))
    pending = list(direct)
    seen: set[str] = set()
    entries: list[dict] = []
    symbols: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        if Path(name).name != name or not name.endswith(".h"):
            raise ValueError(f"Unexpected generated header path: {name}")
        seen.add(name)
        header = shader_dir / name
        data = header.read_bytes()
        includes = sorted(set(x.decode("ascii") for x in INCLUDE.findall(data)))
        pending.extend(x for x in includes if x not in seen)
        entry = {"header": name, "header_bytes": len(data), "header_sha256": digest(data)}
        match = ARRAY.fullmatch(data)
        if match:
            symbol = match.group(1).decode("ascii")
            if symbol in symbols:
                raise ValueError(f"Duplicate shader symbol: {symbol}")
            symbols.add(symbol)
            spv_name = symbol + ".spv"
            spv = (shader_dir / spv_name).read_bytes()
            if len(spv) % 4 or len(spv) < 20 or spv[:4] != b"\x03\x02\x23\x07":
                raise ValueError(f"Invalid little-endian SPIR-V: {spv_name}")
            body = match.group(2)
            if body.count(b"0x") != len(spv) // 4:
                raise ValueError(f"Array word count disagrees with SPIR-V: {name}")
            # bytes.fromhex ignores whitespace, then swap each 32-bit hex word
            # into its in-memory little-endian representation. No Python-int list.
            words = array.array("I")
            if words.itemsize != 4:
                raise RuntimeError("This generator requires 32-bit array('I') elements")
            words.frombytes(bytes.fromhex(body.replace(b"0x", b"").replace(b",", b"").decode("ascii")))
            words.byteswap()
            if words.tobytes() != spv:
                raise ValueError(f"Generated header is NOT byte-identical to SPIR-V: {name}")
            size_symbol = match.group(3).decode("ascii") if match.group(3) else None
            if size_symbol and (size_symbol != symbol + "_sizeInBytes" or int(match.group(4)) != len(spv)):
                raise ValueError(f"Generated size constant disagrees with SPIR-V: {name}")
            entry.update(kind="array", symbol=symbol, spv=spv_name, spv_bytes=len(spv),
                         words=len(spv) // 4, spv_sha256=digest(spv),
                         size_symbol=size_symbol, spv_crc32=zlib.crc32(spv))
            override = declaration(entry)
            entry.update(override_bytes=len(override), override_sha256=digest(override))
        elif includes and len(data) < 1024 * 1024:
            # Preserve macro/variant tables verbatim, recursively overriding the
            # shader arrays they include rather than flattening their behavior.
            if any(not x.startswith(b"rtx_shaders/") for x in ANY_INCLUDE.findall(data)):
                raise ValueError(f"Wrapper has an unsupported relative/system include: {name}")
            entry.update(kind="wrapper", includes=includes, override_bytes=len(data),
                         override_sha256=digest(data))
        else:
            raise ValueError(f"Header is neither a plain shader array nor a known include wrapper: {name}")
        entries.append(entry)
        if len(entries) % 100 == 0:
            print(f"Verified {len(entries)} generated headers", file=sys.stderr, flush=True)
    entries.sort(key=lambda x: x["header"])
    arrays = [x for x in entries if x["kind"] == "array"]
    if not arrays or len(arrays) > 32767:
        raise ValueError("Shader count exceeds standard COFF signed section-index capacity")
    return {"schema_version": 1, "runtime": runtime.as_posix(), "build_directory": build.as_posix(),
            "source": source.as_posix(), "source_sha256": digest(source_data),
            "shader_directory": shader_dir.as_posix(), "generator_sha256": file_digest(Path(__file__)),
            "summary": {"direct_header_count": len(direct), "header_count": len(entries),
                        "shader_count": len(arrays), "wrapper_count": len(entries) - len(arrays),
                        "original_header_bytes": sum(x["header_bytes"] for x in entries),
                        "spirv_bytes": sum(x["spv_bytes"] for x in arrays),
                        "override_header_bytes": sum(x["override_bytes"] for x in entries),
                        "all_headers_byte_identical_to_spirv": True,
                        "coff_machine": "AMD64", "coff_section_alignment": 4,
                        "coff_comdat_selection": "EXACT_MATCH", "optimization_flags_changed": False},
            "entries": entries}


def write_coff(path: Path, inventory: dict) -> None:
    entries = [x for x in inventory["entries"] if x["kind"] == "array"]
    shader_dir = Path(inventory["shader_directory"])
    string_data = bytearray(b"\x00\x00\x00\x00")
    symbols = bytearray()
    sections = bytearray()
    cursor = 20 + 40 * len(entries)
    offsets = []
    for number, entry in enumerate(entries, 1):
        cursor = (cursor + 3) & ~3
        offsets.append(cursor)
        size = entry["spv_bytes"]
        sections += struct.pack("<8sIIIIIIHHI", b".rdata\0\0", 0, 0, size, cursor, 0, 0, 0, 0, SECTION_FLAGS)
        # Section symbol + section-definition auxiliary record, then its public
        # C-linkage data symbol. EXACT_MATCH rejects mismatched duplicate data.
        symbols += struct.pack("<8sIhHBB", b".rdata\0\0", 0, number, 0, 3, 1)
        symbols += struct.pack("<IHHIhBBH", size, 0, 0, entry["spv_crc32"], 0, 4, 0, 0)
        symbol = entry["symbol"].encode("ascii")
        if len(symbol) <= 8:
            symbol_field = symbol.ljust(8, b"\0")
        else:
            symbol_field = struct.pack("<II", 0, len(string_data))
            string_data += symbol + b"\0"
        symbols += struct.pack("<8sIhHBB", symbol_field, 0, number, 0, 2, 0)
        cursor += size
    struct.pack_into("<I", string_data, 0, len(string_data))
    if cursor + len(symbols) + len(string_data) >= 2**32:
        raise ValueError("Object would exceed standard COFF 32-bit file offsets")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<HHIIIHH", 0x8664, len(entries), 0, cursor, len(entries) * 3, 0, 0))
        stream.write(sections)
        for entry, offset in zip(entries, offsets):
            stream.write(b"\0" * (offset - stream.tell()))
            blob = (shader_dir / entry["spv"]).read_bytes()
            if digest(blob) != entry["spv_sha256"]:
                raise ValueError(f"SPIR-V changed during generation: {entry['spv']}")
            stream.write(blob)
        if stream.tell() != cursor:
            raise AssertionError("COFF symbol-table offset mismatch")
        stream.write(symbols)
        stream.write(string_data)
    verify_coff(temporary, inventory)
    os.replace(temporary, path)


def verify_coff(path: Path, inventory: dict) -> None:
    entries = [x for x in inventory["entries"] if x["kind"] == "array"]
    with path.open("rb") as stream:
        machine, count, stamp, symbol_pos, symbol_count, optional, flags = struct.unpack("<HHIIIHH", stream.read(20))
        if (machine, count, stamp, symbol_count, optional, flags) != (0x8664, len(entries), 0, len(entries) * 3, 0, 0):
            raise ValueError("COFF file header validation failed")
        sections = [struct.unpack("<8sIIIIIIHHI", stream.read(40)) for _ in entries]
        for entry, section in zip(entries, sections):
            if section[3] != entry["spv_bytes"] or section[9] != SECTION_FLAGS or section[4] % 4:
                raise ValueError(f"COFF section validation failed: {entry['symbol']}")
            stream.seek(section[4])
            if digest(stream.read(section[3])) != entry["spv_sha256"]:
                raise ValueError(f"COFF payload hash mismatch: {entry['symbol']}")
        stream.seek(symbol_pos)
        symbol_bytes = stream.read(symbol_count * 18)
        string_length = struct.unpack("<I", stream.read(4))[0]
        string_bytes = struct.pack("<I", string_length) + stream.read(string_length - 4)
        for index, entry in enumerate(entries):
            section_symbol = struct.unpack_from("<8sIhHBB", symbol_bytes, index * 54)
            auxiliary = struct.unpack_from("<IHHIhBBH", symbol_bytes, index * 54 + 18)
            if section_symbol != (b".rdata\0\0", 0, index + 1, 0, 3, 1):
                raise ValueError(f"COFF section symbol validation failed: {entry['symbol']}")
            if auxiliary != (entry["spv_bytes"], 0, 0, entry["spv_crc32"], 0, 4, 0, 0):
                raise ValueError(f"COFF COMDAT auxiliary validation failed: {entry['symbol']}")
            raw, value, section, kind, storage, aux = struct.unpack_from("<8sIhHBB", symbol_bytes, index * 54 + 36)
            if raw[:4] == b"\0" * 4:
                offset = struct.unpack_from("<I", raw, 4)[0]
                name = string_bytes[offset:string_bytes.index(0, offset)].decode("ascii")
            else:
                name = raw.rstrip(b"\0").decode("ascii")
            if (name, value, section, kind, storage, aux) != (entry["symbol"], 0, index + 1, 0, 2, 0):
                raise ValueError(f"COFF external symbol validation failed: {entry['symbol']}")


def generate(root: Path, inventory: dict) -> dict:
    shader_dir = Path(inventory["shader_directory"])
    for entry in inventory["entries"]:
        if entry["kind"] == "array":
            data = declaration(entry)
        else:
            data = (shader_dir / entry["header"]).read_bytes()
        if digest(data) != entry["override_sha256"]:
            raise ValueError(f"Header changed during generation: {entry['header']}")
        write_changed(root / "include/rtx_shaders" / entry["header"], data)
    obj = root / "gbuffer_shader_blobs.obj"
    write_coff(obj, inventory)
    inventory["generated_root"] = root.as_posix()
    inventory["coff_object"] = {"path": obj.as_posix(), "bytes": obj.stat().st_size, "sha256": file_digest(obj)}
    write_changed(root / "manifest.json", json_bytes(inventory))
    # Ninja compares every declared output with the newest input. Even unchanged
    # declarations must advance together or this multi-output edge stays dirty.
    completed = time.time_ns()
    for entry in inventory["entries"]:
        os.utime(root / "include/rtx_shaders" / entry["header"], ns=(completed, completed))
    for output in (obj, root / "manifest.json"):
        os.utime(output, ns=(completed, completed))
    return inventory


def ninja_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


def ninja_command(arguments: list[str]) -> str:
    return subprocess.list2cmdline(arguments).replace("$", "$$")


def normalize_ninja_paths(build: Path, info: dict) -> dict:
    physical = Path(info["directories"]["build"])
    if not physical.samefile(build):
        raise ValueError("Cannot normalize a different physical build directory")
    ninja = build / "build.ninja"
    original = ninja.read_text(encoding="utf-8")
    modified = original
    replacements = 0
    spellings = set()
    for forward in (False, True):
        old = physical.as_posix() if forward else str(physical)
        new = build.as_posix() if forward else str(build)
        for escaped_space in (False, True):
            for escaped_colon in (False, True):
                before, after = old, new
                if escaped_space:
                    before, after = before.replace(" ", "$ "), after.replace(" ", "$ ")
                if escaped_colon:
                    before, after = before.replace(":", "$:"), after.replace(":", "$:")
                if before != after:
                    spellings.add((before, after))
    for before, after in sorted(spellings, key=lambda pair: -len(pair[0])):
        # Match this root only, never a sibling whose name extends the prefix.
        pattern = re.compile(re.escape(before) + r"(?=$|[/\\\s\"'])", re.IGNORECASE)
        modified, count = pattern.subn(lambda match: after, modified)
        replacements += count
    if modified != original:
        backup = ninja.with_name("build.ninja.before-path-alias." + digest(original.encode("utf-8"))[:16])
        write_changed(backup, original.encode("utf-8"))
        write_changed(ninja, modified.encode("utf-8"))
    return {"ninja": str(ninja), "configured_build": str(physical), "build_alias": str(build), "replacements": replacements}


def patch_ninja(build: Path, root: Path, inventory: dict, write: bool = True) -> dict:
    ninja = build / "build.ninja"
    original = ninja.read_text(encoding="utf-8")
    relative_root = root.relative_to(build).as_posix()
    obj = relative_root + "/gbuffer_shader_blobs.obj"
    base = original
    installed = BEGIN in base or END in base
    if installed:
        if base.count(BEGIN) != 1 or base.count(END) != 1:
            raise ValueError("Ambiguous G-buffer generated rule markers")
        begin, end = base.index(BEGIN), base.index(END)
        if end < begin:
            raise ValueError("Reversed G-buffer generated rule markers")
        base = base[:begin].rstrip() + "\n" + base[end + len(END):].lstrip("\r\n")
    lines = base.rstrip().splitlines()
    compile_indices = [i for i, line in enumerate(lines)
                       if line.startswith("build ") and "rtx_pathtracer_gbuffer.cpp.obj: cpp_COMPILER_RSP " in line]
    archive_indices = [i for i, line in enumerate(lines) if line.startswith("build src/dxvk/libdxvk.a: STATIC_LINKER_RSP ")]
    if len(compile_indices) != 1 or len(archive_indices) != 1:
        raise ValueError("Expected exactly one gbuffer compile and one libdxvk archive edge")
    compile_index, archive_index = compile_indices[0], archive_indices[0]
    include_arg = ninja_command(["-I" + (root / "include").as_posix()])
    if installed:
        for index in (compile_index, archive_index):
            token = " " + ninja_path(obj)
            if lines[index].count(token) != 1:
                raise ValueError("Existing COFF dependency is missing or ambiguous; reconfigure Meson")
            lines[index] = lines[index].replace(token, "", 1)
    line = lines[compile_index]
    before, separator, after = line.partition(" || ")
    if before.endswith(" |"):
        before = before[:-2]
    if " | " not in before:
        before += " |"
    lines[compile_index] = before + " " + ninja_path(obj) + (separator + after if separator else "")
    args_index = compile_index + 1
    while args_index < len(lines) and not lines[args_index].startswith(" ARGS = "):
        if lines[args_index].startswith("build "):
            raise ValueError("gbuffer compile edge has no ARGS assignment")
        args_index += 1
    if args_index >= len(lines) or '"/O2"' not in lines[args_index]:
        raise ValueError("Expected existing /O2 compilation; refusing to change a different configuration")
    if installed:
        prefix = " ARGS = " + include_arg + " "
        if not lines[args_index].startswith(prefix):
            raise ValueError("Existing generated include path differs; reconfigure Meson")
        lines[args_index] = " ARGS = " + lines[args_index][len(prefix):]
    lines[args_index] = " ARGS = " + include_arg + " " + lines[args_index][len(" ARGS = "):]
    archive_regular, order_sep, archive_order = lines[archive_index].partition(" || ")
    archive_before, archive_sep, archive_after = archive_regular.partition(" | ")
    lines[archive_index] = (archive_before + " " + ninja_path(obj)
                            + (archive_sep + archive_after if archive_sep else "")
                            + (order_sep + archive_order if order_sep else ""))
    outputs = [obj, relative_root + "/manifest.json"]
    outputs += [relative_root + "/include/rtx_shaders/" + e["header"] for e in inventory["entries"]]
    # Shader files are side-effects of the stamp-producing shader command.
    # The stamp must be a normal dependency, so same-build regeneration dirties
    # the COFF edge even if Ninja examined the old side-effect mtimes earlier.
    inputs = [Path(__file__).absolute(), Path(inventory["source"]), "src/dxvk/_built_shaders.txt"]
    shader_dir = Path(inventory["shader_directory"])
    for entry in inventory["entries"]:
        inputs.append(shader_dir / entry["header"])
        if entry["kind"] == "array":
            inputs.append(shader_dir / entry["spv"])
    command = ninja_command([sys.executable, str(Path(__file__).absolute()), "generate",
                             "--runtime", inventory["runtime"], "--build-dir", str(build), "--generated-root", str(root)])
    addition = ["", BEGIN,
                "build " + " ".join(ninja_path(x) for x in outputs) + ": CUSTOM_COMMAND "
                + " ".join(ninja_path(x) for x in inputs),
                " COMMAND = " + command,
                " DESC = Verifying and externalizing unchanged gbuffer SPIR-V data",
                END, ""]
    modified = "\n".join(lines + addition)
    backup = ninja.with_name("build.ninja.before-gbuffer-external." + digest(original.encode("utf-8"))[:16])
    changed = modified != original
    if changed and write:
        write_changed(backup, original.encode("utf-8"))
        write_changed(ninja, modified.encode("utf-8"))
    return {"ninja": ninja.as_posix(), "backup": backup.as_posix(),
            "changed": changed,
            "original_sha256": digest(original.encode("utf-8")), "modified_sha256": digest(modified.encode("utf-8")),
            "compile_target": lines[compile_index].split(": ", 1)[0][len("build "):],
            "custom_input_count": len(inputs), "custom_output_count": len(outputs),
            "other_compile_edges_changed": 0, "optimization_flags_changed": False}


def verify_generated(root: Path, inventory: dict) -> dict:
    saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if any(saved.get(key) != value for key, value in inventory.items()):
        raise ValueError("COFF manifest does not match current source, generator and shader inputs")
    for entry in inventory["entries"]:
        if file_digest(root / "include/rtx_shaders" / entry["header"]) != entry["override_sha256"]:
            raise ValueError("Generated declaration differs: " + entry["header"])
    verify_coff(root / "gbuffer_shader_blobs.obj", inventory)
    return saved


def ensure_generated(root: Path, inventory: dict) -> dict:
    try:
        saved = verify_generated(root, inventory)
        inputs = [Path(__file__), Path(inventory["source"]),
                  Path(inventory["build_directory"]) / "src/dxvk/_built_shaders.txt"]
        outputs = [root / "manifest.json", root / "gbuffer_shader_blobs.obj"]
        for entry in inventory["entries"]:
            inputs.append(Path(inventory["shader_directory"]) / entry["header"])
            if entry["kind"] == "array":
                inputs.append(Path(inventory["shader_directory"]) / entry["spv"])
            outputs.append(root / "include/rtx_shaders" / entry["header"])
        if max(path.stat().st_mtime_ns for path in inputs) <= min(path.stat().st_mtime_ns for path in outputs):
            return saved
    except (OSError, ValueError, KeyError, struct.error):
        pass
    return generate(root, inventory)


def pe_imports(path: Path) -> dict:
    """Read normal and delay-load import names without loading executable code."""
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        pe = struct.unpack_from("<I", data, 0x3c)[0]
        if data[:2] != b"MZ" or data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("Invalid PE file: " + str(path))
        machine, count = struct.unpack_from("<HH", data, pe + 4)
        optional_size = struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        magic = struct.unpack_from("<H", data, optional)[0]
        if magic != 0x20b or machine != 0x8664:
            raise ValueError("Expected AMD64 PE32+ dependency: " + str(path))
        image_base = struct.unpack_from("<Q", data, optional + 24)[0]
        headers_size = struct.unpack_from("<I", data, optional + 60)[0]
        directory_count = struct.unpack_from("<I", data, optional + 108)[0]
        sections = []
        for i in range(count):
            section = optional + optional_size + i * 40
            virtual_size, rva, raw_size, raw = struct.unpack_from("<IIII", data, section + 8)
            sections.append((rva, max(virtual_size, raw_size), raw, raw_size))

        def offset(rva: int) -> int:
            if rva < headers_size:
                return rva
            for start, size, raw, raw_size in sections:
                if start <= rva < start + size and rva - start < raw_size:
                    return raw + rva - start
            raise ValueError(f"Invalid PE RVA {rva:x}: {path}")

        def name(rva: int) -> str:
            start = offset(rva)
            end = data.find(b"\0", start, start + 4096)
            if end == -1:
                raise ValueError("Unterminated PE import name")
            value = data[start:end].decode("ascii")
            if Path(value).name != value or "/" in value or "\\" in value:
                raise ValueError("Unexpected PE import path: " + value)
            return value

        result = {"machine": "AMD64", "normal_imports": [], "delay_imports": []}
        for index, field, stride in ((1, "normal_imports", 20), (13, "delay_imports", 32)):
            if index >= directory_count:
                continue
            rva, size = struct.unpack_from("<II", data, optional + 112 + index * 8)
            if not rva:
                continue
            start = offset(rva)
            for i in range(min(65536, size // stride + 1)):
                values = struct.unpack_from("<" + "I" * (stride // 4), data, start + i * stride)
                if not any(values):
                    break
                name_rva = values[3] if index == 1 else values[1]
                if index == 13 and not (values[0] & 1):
                    name_rva -= image_base
                result[field].append(name(name_rva))
            else:
                raise ValueError("Unterminated PE import table: " + str(path))
        return result


def output_files(output: Path) -> list[Path]:
    if not (output / "d3d9.dll").is_file():
        raise ValueError("An existing complete _output distribution is required")
    return sorted(path for path in output.rglob("*") if path.is_file() and path.suffix.lower() in (".dll", ".exe"))


def output_snapshot(output: Path) -> dict:
    return {path.relative_to(output).as_posix(): file_digest(path) for path in output_files(output)
            if path != output / "d3d9.dll"}


def audit_output(output: Path, candidate: Path, baseline: Path) -> dict:
    expected = json.loads(baseline.read_text(encoding="utf-8"))
    actual = output_snapshot(output)
    if actual != expected:
        changed = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
        raise ValueError("Supporting distribution files changed during build: " + ", ".join(changed))
    system = Path(os.environ["SystemRoot"]) / "System32"
    files, missing = [], []
    for original in output_files(output):
        path = candidate if original == output / "d3d9.dll" else original
        info = pe_imports(path)
        imports = sorted(set(info["normal_imports"] + info["delay_imports"]), key=str.lower)
        resolved = {}
        for dependency in imports:
            if dependency.lower().startswith(("api-ms-win-", "ext-ms-win-")):
                resolved[dependency] = "Windows API set"
                continue
            match = next((folder / dependency for folder in (original.parent, output, system)
                          if (folder / dependency).is_file()), None)
            if match is None:
                missing.append(f"{original.name} -> {dependency}")
            else:
                resolved[dependency] = str(match)
        files.append({"path": original.relative_to(output).as_posix(), **info, "resolved": resolved})
    if missing:
        raise ValueError("Unresolved normal/delay imports: " + "; ".join(missing))
    return {"scope": "All existing package PE files, including normal and delay imports. Dynamic LoadLibrary behavior is not executed.",
            "candidate": str(candidate), "dll_sha256": file_digest(candidate), "supporting_sha256": actual,
            "missing_imports": missing, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["config", "normalize-ninja", "snapshot-output", "audit-output", "publish",
                                        "audit", "generate", "patch-ninja", "apply", "verify"])
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    parser.add_argument("--build-dir")
    parser.add_argument("--generated-root")
    parser.add_argument("--report", help="Optional JSON report path (audit does not write build cache)")
    parser.add_argument("--baseline", type=Path, help="Required supporting-file snapshot for publication/audit")
    args = parser.parse_args()
    runtime = absolute(args.runtime)
    build = absolute(args.build_dir) if args.build_dir else runtime / "_Comp64Release"
    runtime, build, info = configured_paths(runtime, build)
    root = absolute(args.generated_root) if args.generated_root else build / "src/dxvk/gbuffer_external"
    root.relative_to(build)  # Bound generated writes to the explicitly named build directory.
    root.resolve().relative_to(build.resolve())
    if args.mode == "config":
        result = {"runtime": str(runtime), "build": str(build), "meson_version": info["meson_version"]["full"]}
    elif args.mode == "normalize-ninja":
        result = normalize_ninja_paths(build, info)
    elif args.mode == "snapshot-output":
        result = output_snapshot(runtime / "_output")
    elif args.mode in ("audit-output", "publish"):
        if args.baseline is None:
            parser.error("--baseline is required")
        inventory = inspect(runtime, build)
        verify_generated(root, inventory)
        if patch_ninja(build, root, inventory, write=False)["changed"]:
            raise ValueError("Ninja COFF rule changed or vanished after linking; rerun the controlled build")
        candidate = build / "src/d3d9/d3d9.dll"
        if candidate.stat().st_mtime_ns < (root / "gbuffer_shader_blobs.obj").stat().st_mtime_ns:
            raise ValueError("Runtime DLL predates the verified COFF object")
        result = audit_output(runtime / "_output", candidate, args.baseline)
        if args.mode == "publish":
            destination = runtime / "_output/d3d9.dll"
            temporary = destination.with_name("d3d9.dll.validated.tmp")
            shutil.copy2(candidate, temporary)
            if file_digest(temporary) != result["dll_sha256"]:
                raise ValueError("Runtime DLL copy hash mismatch")
            os.replace(temporary, destination)
            result["published"] = str(destination)
            result["published_sha256"] = file_digest(destination)
    elif args.mode == "verify":
        inventory = inspect(runtime, build)
        verify_generated(root, inventory)
        if patch_ninja(build, root, inventory, write=False)["changed"]:
            raise ValueError("Ninja COFF rule does not match the configured source/generator")
        result = {"verified": True, "shader_count": inventory["summary"]["shader_count"]}
    else:
        inventory = inspect(runtime, build)
        if args.mode == "generate":
            inventory = generate(root, inventory)
        elif args.mode == "apply":
            inventory = ensure_generated(root, inventory)
        if args.mode in ("patch-ninja", "apply"):
            inventory["ninja_patch"] = patch_ninja(build, root, inventory)
        result = {"mode": args.mode, "summary": inventory["summary"],
                  "coff_object": inventory.get("coff_object"), "ninja_patch": inventory.get("ninja_patch"),
                  "report": args.report}
    if args.report:
        write_changed(absolute(args.report), json_bytes(inventory if args.mode in ("audit", "generate", "patch-ninja", "apply") else result))
    if args.mode in ("audit-output", "publish"):
        result = {key: value for key, value in result.items() if key not in ("files", "supporting_sha256")}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

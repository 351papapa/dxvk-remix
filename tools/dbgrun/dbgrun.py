"""Run a command under a minimal Win32 debugger. On a second-chance exception: write a minidump,
print the exception and a symbolized stack for every thread (faulting thread first), kill the process.
Usage: python dbgrun.py <dump_dir> <symbol_path> -- <exe> [args...]"""
import ctypes as C, ctypes.wintypes as W, subprocess, sys, os, time

k32 = C.WinDLL("kernel32", use_last_error=True)
dbh = C.WinDLL("dbghelp", use_last_error=True)
k32.CreateFileW.restype = W.HANDLE

DEBUG_ONLY_THIS_PROCESS = 0x2
DBG_CONTINUE, DBG_EXCEPTION_NOT_HANDLED = 0x00010002, 0x80010001
EXCEPTION_DEBUG_EVENT, CREATE_THREAD, CREATE_PROCESS, EXIT_THREAD, EXIT_PROCESS, LOAD_DLL, UNLOAD_DLL, OUTPUT_STRING, RIP = range(1, 10)

class EXCEPTION_RECORD(C.Structure):
    _fields_ = [("Code", W.DWORD), ("Flags", W.DWORD), ("Rec", C.c_void_p), ("Address", C.c_void_p),
                ("NumParams", W.DWORD), ("Info", C.c_uint64 * 15)]

class DEBUG_EVENT(C.Structure):
    _fields_ = [("Code", W.DWORD), ("Pid", W.DWORD), ("Tid", W.DWORD), ("u", C.c_byte * 168)]

class EXC_INFO(C.Structure):
    _fields_ = [("Rec", EXCEPTION_RECORD), ("FirstChance", W.DWORD)]

class CREATE_PROCESS_INFO(C.Structure):
    _fields_ = [("hFile", W.HANDLE), ("hProcess", W.HANDLE), ("hThread", W.HANDLE)]

class CREATE_THREAD_INFO(C.Structure):
    _fields_ = [("hThread", W.HANDLE)]

class LOAD_DLL_INFO(C.Structure):
    _fields_ = [("hFile", W.HANDLE)]

class STARTUPINFO(C.Structure):
    _fields_ = [("cb", W.DWORD), ("r", C.c_byte * 100)]

class PROCESS_INFORMATION(C.Structure):
    _fields_ = [("hProcess", W.HANDLE), ("hThread", W.HANDLE), ("pid", W.DWORD), ("tid", W.DWORD)]

class ADDRESS64(C.Structure):
    _fields_ = [("Offset", C.c_uint64), ("Segment", W.WORD), ("Mode", C.c_int)]

class STACKFRAME64(C.Structure):
    _fields_ = [("AddrPC", ADDRESS64), ("AddrReturn", ADDRESS64), ("AddrFrame", ADDRESS64), ("AddrStack", ADDRESS64),
                ("AddrBStore", ADDRESS64), ("FuncTableEntry", C.c_void_p), ("Params", C.c_uint64 * 4),
                ("Far", W.BOOL), ("Virtual", W.BOOL), ("Reserved", C.c_uint64 * 3), ("KdHelp", C.c_byte * 256)]

class SYMBOL_INFO(C.Structure):
    _fields_ = [("SizeOfStruct", W.ULONG), ("TypeIndex", W.ULONG), ("Reserved", C.c_uint64 * 2), ("Index", W.ULONG),
                ("Size", W.ULONG), ("ModBase", C.c_uint64), ("Flags", W.ULONG), ("Value", C.c_uint64),
                ("Address", C.c_uint64), ("Register", W.ULONG), ("Scope", W.ULONG), ("Tag", W.ULONG),
                ("NameLen", W.ULONG), ("MaxNameLen", W.ULONG), ("Name", C.c_char * 1024)]

class LINE64(C.Structure):
    _fields_ = [("SizeOfStruct", W.DWORD), ("Key", C.c_void_p), ("LineNumber", W.DWORD), ("FileName", C.c_char_p),
                ("Address", C.c_uint64)]

class MINIDUMP_EXCEPTION_INFORMATION(C.Structure):
    _pack_ = 4
    _fields_ = [("ThreadId", W.DWORD), ("ExceptionPointers", C.c_void_p), ("ClientPointers", W.BOOL)]

class EXCEPTION_POINTERS(C.Structure):
    _fields_ = [("Rec", C.c_void_p), ("Ctx", C.c_void_p)]

CONTEXT_SIZE, CONTEXT_FULL = 0x4D0, 0x10001F  # CONTROL|INTEGER|SEGMENTS|FLOATING_POINT|DEBUG_REGISTERS

def get_context(hthread):
    raw = C.create_string_buffer(CONTEXT_SIZE + 16)
    addr = (C.addressof(raw) + 15) & ~15
    C.c_uint32.from_address(addr + 0x30).value = CONTEXT_FULL
    if not k32.GetThreadContext(W.HANDLE(hthread), C.c_void_p(addr)):
        return None, None
    return raw, addr

def symbolize(hproc, pc):
    dbh.SymGetModuleBase64.restype = C.c_uint64
    dbh.SymGetModuleBase64.argtypes = [W.HANDLE, C.c_uint64]
    dbh.SymFromAddr.argtypes = [W.HANDLE, C.c_uint64, C.c_void_p, C.c_void_p]
    dbh.SymGetLineFromAddr64.argtypes = [W.HANDLE, C.c_uint64, C.c_void_p, C.c_void_p]
    s = SYMBOL_INFO(); s.SizeOfStruct = 88; s.MaxNameLen = 1000
    disp = C.c_uint64(0)
    name = "?"
    mod = dbh.SymGetModuleBase64(hproc, C.c_uint64(pc))
    modname = ""
    if mod:
        buf = C.create_unicode_buffer(260)
        k32.K32GetModuleBaseNameW(hproc, C.c_void_p(mod), buf, 260)
        modname = buf.value
    if dbh.SymFromAddr(hproc, C.c_uint64(pc), C.byref(disp), C.byref(s)):
        name = f"{s.Name.decode(errors='replace')}+0x{disp.value:x}"
    line = LINE64(); line.SizeOfStruct = C.sizeof(LINE64); d32 = W.DWORD(0)
    where = ""
    if dbh.SymGetLineFromAddr64(hproc, C.c_uint64(pc), C.byref(d32), C.byref(line)):
        where = f"  [{os.path.basename(line.FileName.decode(errors='replace'))}:{line.LineNumber}]"
    return f"{modname}!{name}{where}"

def walk(hproc, hthread, maxf=40):
    raw, addr = get_context(hthread)
    if not addr:
        return ["  (GetThreadContext failed)"]
    rd = lambda off: C.c_uint64.from_address(addr + off).value
    f = STACKFRAME64()
    f.AddrPC.Offset, f.AddrPC.Mode = rd(0xF8), 3
    f.AddrFrame.Offset, f.AddrFrame.Mode = rd(0xA0), 3
    f.AddrStack.Offset, f.AddrStack.Mode = rd(0x98), 3
    out = []
    fta = C.cast(dbh.SymFunctionTableAccess64, C.c_void_p)
    gmb = C.cast(dbh.SymGetModuleBase64, C.c_void_p)
    dbh.StackWalk64.argtypes = [W.DWORD, W.HANDLE, W.HANDLE, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
    for _ in range(maxf):
        if not dbh.StackWalk64(0x8664, hproc, W.HANDLE(hthread), C.byref(f), C.c_void_p(addr), None, fta, gmb, None):
            break
        if not f.AddrPC.Offset:
            break
        out.append(f"  {f.AddrPC.Offset:016x} {symbolize(hproc, f.AddrPC.Offset)}")
    return out

def main():
    dump_dir, sympath = sys.argv[1], sys.argv[2]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    os.makedirs(dump_dir, exist_ok=True)
    si = STARTUPINFO(); si.cb = 104; pi = PROCESS_INFORMATION()
    cmdline = subprocess.list2cmdline(cmd)
    if not k32.CreateProcessW(None, C.create_unicode_buffer(cmdline), None, None, True, DEBUG_ONLY_THIS_PROCESS,
                              None, None, C.byref(si), C.byref(pi)):
        print("CreateProcess failed", C.get_last_error()); return 2
    k32.DebugSetProcessKillOnExit(True)
    threads = {}
    hproc = None
    seen_initial_bp = False
    first_chance_av = 0
    ev = DEBUG_EVENT()
    dbh.SymSetOptions(0x2 | 0x10 | 0x200)  # DEFERRED_LOADS | LOAD_LINES | FAIL_CRITICAL_ERRORS
    while True:
        if not k32.WaitForDebugEvent(C.byref(ev), 0xFFFFFFFF):
            print("WaitForDebugEvent failed", C.get_last_error()); return 2
        status = DBG_CONTINUE
        u = C.addressof(ev.u) + 4  # union is 8-aligned after the 3 DWORDs
        if ev.Code == CREATE_PROCESS:
            info = CREATE_PROCESS_INFO.from_address(u)
            hproc = info.hProcess; threads[ev.Tid] = info.hThread
            if info.hFile: k32.CloseHandle(W.HANDLE(info.hFile))
        elif ev.Code == CREATE_THREAD:
            threads[ev.Tid] = CREATE_THREAD_INFO.from_address(u).hThread
        elif ev.Code == EXIT_THREAD:
            threads.pop(ev.Tid, None)
        elif ev.Code == LOAD_DLL:
            h = LOAD_DLL_INFO.from_address(u).hFile
            if h: k32.CloseHandle(W.HANDLE(h))
        elif ev.Code == EXIT_PROCESS:
            code = C.c_uint32.from_address(u).value
            print(f"DBGRUN: process exited 0x{code:08x}, first-chance AVs seen {first_chance_av}")
            return 0 if code == 0 else 1
        elif ev.Code == EXCEPTION_DEBUG_EVENT:
            ex = EXC_INFO.from_address(u)
            code = ex.Rec.Code
            if code == 0x80000003 and not seen_initial_bp:
                seen_initial_bp = True
            elif ex.FirstChance:
                if code == 0xC0000005: first_chance_av += 1
                status = DBG_EXCEPTION_NOT_HANDLED
            else:
                ts = time.strftime("%H%M%S")
                print(f"DBGRUN: SECOND-CHANCE exception 0x{code:08x} at {ex.Rec.Address or 0:#x} tid {ev.Tid} ({time.strftime('%T')})")
                if code == 0xC0000005 and ex.Rec.NumParams >= 2:
                    kind = {0: "read", 1: "write", 8: "execute"}.get(ex.Rec.Info[0], str(ex.Rec.Info[0]))
                    print(f"  access violation: {kind} of {ex.Rec.Info[1]:#x}")
                # Minidump with the exception record.
                path = os.path.join(dump_dir, f"crash-{ts}.dmp")
                f = k32.CreateFileW(path, 0x40000000, 0, None, 2, 0x80, None)
                raw, caddr = get_context(threads.get(ev.Tid))
                rec = EXCEPTION_RECORD.from_buffer_copy(ex.Rec)
                # ClientPointers=FALSE needs the pointers in our address space.
                ep = EXCEPTION_POINTERS(C.addressof(rec), caddr)
                mei = MINIDUMP_EXCEPTION_INFORMATION(ev.Tid, C.addressof(ep), False)
                flags = 0x1 | 0x4 | 0x20 | 0x40 | 0x800 | 0x1000  # DataSegs|HandleData|UnloadedModules|IndirectMem|FullMemInfo|ThreadInfo
                dbh.MiniDumpWriteDump.argtypes = [W.HANDLE, W.DWORD, W.HANDLE, W.DWORD, C.c_void_p, C.c_void_p, C.c_void_p]
                ok = dbh.MiniDumpWriteDump(hproc, ev.Pid, W.HANDLE(f), flags, C.byref(mei), None, None)
                k32.CloseHandle(W.HANDLE(f))
                print(f"  minidump {'written' if ok else 'FAILED'}: {path}")
                dbh.SymInitializeW(W.HANDLE(hproc), sympath, True)
                print(f"--- faulting thread {ev.Tid}")
                print("\n".join(walk(W.HANDLE(hproc), threads.get(ev.Tid))))
                for tid, h in threads.items():
                    if tid == ev.Tid: continue
                    st = walk(W.HANDLE(hproc), h, 12)
                    if any("d3d9" in l or "remix" in l.lower() for l in st):
                        print(f"--- thread {tid}"); print("\n".join(st))
                sys.stdout.flush()
                k32.TerminateProcess(W.HANDLE(hproc), 0xDEAD)
                status = DBG_EXCEPTION_NOT_HANDLED
        k32.ContinueDebugEvent(ev.Pid, ev.Tid, status)

if __name__ == "__main__":
    sys.exit(main())

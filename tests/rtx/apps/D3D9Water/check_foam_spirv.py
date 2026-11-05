"""Check compiled foam kernels for the negative-coordinate addressing regression."""
from pathlib import Path
import struct
import sys

folder = Path(sys.argv[1])
for suffix in ('', '_culled', '_live', '_live_culled', '_live_pair_reuse', '_live_culled_pair_reuse'):
    path = folder / ('water_foam_surface' + suffix + '.spv')
    data = path.read_bytes()
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203, f'Invalid SPIR-V: {path}'
    offset = 5
    local_size = None
    while offset < len(words):
        size, opcode = words[offset] >> 16, words[offset] & 0xffff
        assert size and offset + size <= len(words), f'Invalid instruction: {path}'
        operands = words[offset + 1:offset + size]
        assert opcode not in (138, 139), f'Signed remainder reintroduced into foam world addressing: {path}'
        if opcode == 16 and operands[1] == 17:
            local_size = operands[2:5]
        offset += size
    assert local_size == (8, 8, 1), f'Unexpected foam workgroup {local_size}: {path}'
    print(path.name + ': unsigned world addressing; workgroup 8x8x1 PASS')

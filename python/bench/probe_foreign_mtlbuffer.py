"""Probe: can torch (MPS) write into an MTLBuffer it did not allocate?

Builds an MTLBuffer with newBufferWithBytesNoCopy over a page-aligned mmap
(shared memory candidate), wraps it as a kDLMetal DLPack capsule, imports it
with torch.from_dlpack and runs copy_ / aminmax(out=) into it on torch's stream.
Then checks torch.mps.Event record/query semantics (does record() commit?).
"""
import ctypes, ctypes.util, mmap, time, sys
import torch

objc = ctypes.CDLL(ctypes.util.find_library("objc"))
metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
objc.objc_msgSend.restype = ctypes.c_void_p
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]
metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p

def sel(s): return objc.sel_registerName(s.encode())

def msg(obj, s, *args, argtypes=(), restype=ctypes.c_void_p):
    f = objc.objc_msgSend
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + list(argtypes)
    f.restype = restype
    return f(obj, sel(s), *args)

dev = metal.MTLCreateSystemDefaultDevice()
assert dev

PAGE = mmap.PAGESIZE
H, W = 1024, 1024
nbytes = H * W * 4 + 4096                 # payload + a stats page
length = (nbytes + PAGE - 1) // PAGE * PAGE
mm = mmap.mmap(-1, length, flags=mmap.MAP_SHARED | mmap.MAP_ANON)   # stand-in for shm
addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
assert addr % PAGE == 0
MTLResourceStorageModeShared = 0 << 4
buf = msg(dev, "newBufferWithBytesNoCopy:length:options:deallocator:",
          ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_size_t(MTLResourceStorageModeShared),
          ctypes.c_void_p(0),
          argtypes=[ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p])
assert buf, "newBufferWithBytesNoCopy failed"
contents = msg(buf, "contents")
print(f"MTLBuffer {buf:#x} contents {contents:#x} mmap {addr:#x} same={contents == addr}")

# --- DLPack capsule over the MTLBuffer (legacy DLManagedTensor) ---
class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]
class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]
class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int),
                ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]
DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", DELETER)]

kDLMetal = 8
keep = []
def make_capsule(shape, dtype_code, bits, byte_offset):
    shp = (ctypes.c_int64 * len(shape))(*shape)
    strides = []
    s = 1
    for d in reversed(shape):
        strides.append(s); s *= d
    strd = (ctypes.c_int64 * len(shape))(*reversed(strides))
    m = DLManagedTensor()
    m.dl_tensor.data = buf
    m.dl_tensor.device = DLDevice(kDLMetal, 0)
    m.dl_tensor.ndim = len(shape)
    m.dl_tensor.dtype = DLDataType(dtype_code, bits, 1)
    m.dl_tensor.shape = shp
    m.dl_tensor.strides = strd
    m.dl_tensor.byte_offset = byte_offset
    deleter = DELETER(lambda p: None)
    m.deleter = deleter
    keep.extend([m, shp, strd, deleter])
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return ctypes.pythonapi.PyCapsule_New(ctypes.addressof(m), b"dltensor", None)

try:
    snap = torch.from_dlpack(make_capsule((H, W), 2, 32, 0))      # f32 payload at offset 0
    stats = torch.from_dlpack(make_capsule((2,), 2, 32, H * W * 4))   # 2 f32 at the stats page
except Exception as e:
    print("from_dlpack(kDLMetal foreign buffer) FAILED:", type(e).__name__, e)
    sys.exit(1)
print("from_dlpack ok:", snap.device, snap.dtype, tuple(snap.shape), "storage ptr",
      hex(snap.untyped_storage().data_ptr()), "offset", snap.storage_offset(),
      "| stats", stats.device, tuple(stats.shape), "offset", stats.storage_offset())

src = torch.rand(H, W, device="mps")
snap.copy_(src)                       # GPU blit on torch's stream into OUR buffer
lo, hi = torch.aminmax(src)
try:
    torch.aminmax(src, out=(stats[0:1].squeeze(), stats[1:2].squeeze()))
    aminmax_out = "aminmax(out=) ok"
except Exception as e:
    stats[0] = lo; stats[1] = hi
    aminmax_out = f"aminmax(out=) failed ({type(e).__name__}: {str(e)[:80]}); used elementwise copies"
ev = torch.mps.Event()
t0 = time.perf_counter(); ev.record(); t_rec = (time.perf_counter() - t0) * 1e3
q0 = ev.query()
# Does record() commit? If yes, query() turns true WITHOUT any synchronize().
t1 = time.perf_counter()
while not ev.query() and time.perf_counter() - t1 < 2.0:
    time.sleep(0.0005)
print(f"event: record {t_rec:.3f} ms, query right after = {q0}, completed without sync after "
      f"{(time.perf_counter() - t1) * 1e3:.2f} ms -> {ev.query()}")
print(aminmax_out)

# Read the bytes back from the CPU side of the mmap (what a viewer process would see)
import numpy as np
cpu_view = np.frombuffer(mm, dtype=np.float32, count=H * W).reshape(H, W)
ref = src.cpu().numpy()
print("bytes in mmap == src:", np.array_equal(cpu_view, ref), "| stats page:",
      np.frombuffer(mm, dtype=np.float32, count=2, offset=H * W * 4), "ref", lo.item(), hi.item())

# Cost of the snapshot path per call (steady state), no sync anywhere
def one():
    snap.copy_(src)
    torch.aminmax(src, out=(stats[0:1].squeeze(), stats[1:2].squeeze()))
    ev.record()
    ev.query()
for _ in range(20): one()
torch.mps.synchronize()
N = 200
t0 = time.perf_counter()
for _ in range(N): one()
t_cpu = (time.perf_counter() - t0) / N * 1e3
torch.mps.synchronize()
t_all = (time.perf_counter() - t0) / N * 1e3
print(f"snapshot path CPU cost per call: {t_cpu:.3f} ms (incl. GPU drain amortised: {t_all:.3f} ms) at {H}x{W}")

from tinygrad import Tensor, dtypes, Context, getenv, UOp, fetch
from tinygrad.uop.ops import Ops, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen import Renderer
from tinygrad.codegen.opt import Opt, OptOps
from dataclasses import dataclass

# ************************* implementation of the problem ************************

def myhash(a: Tensor) -> Tensor:
  a = (a + 0x7ED55D16) + (a << 12)
  a = (a ^ 0xC761C23C) ^ (a >> 19)
  a = (a + 0x165667B1) + (a << 5)
  a = (a + 0xD3A2646C) ^ (a << 9)
  a = (a + 0xFD7046C5) + (a << 3)
  a = (a ^ 0xB55A4F09) ^ (a >> 16)
  return a

def select_with_where_tree(values: Tensor, relative_idx: Tensor) -> Tensor:
  n = values.shape[0]
  if n == 1: return values[0].expand(relative_idx.shape)

  mid = n // 2
  left = select_with_where_tree(values[:mid], relative_idx)
  right = select_with_where_tree(values[mid:], relative_idx - mid)

  go_left = relative_idx < mid
  return go_left.where(left, right)

def tree_traversal(forest: Tensor, val: Tensor, height: int, rounds: int, where_tree_threshold=3) -> Tensor:
  # All walkers start at idx=0
  idx = Tensor.zeros(val.shape, device=val.device, dtype=dtypes.uint32)

  for r in range(rounds):
    level = r % (height + 1)
    level_start = (1 << level) - 1
    level_size = 1 << level

    if level == 0:
      # At root (level 0), all walkers are at idx=0
      # No gather needed, just broadcast the root value
      node_val = forest[0].expand(val.shape)
      idx = idx * 0  # Reset to 0
    elif level <= where_tree_threshold:
      # Small level: use where-tree
      level_values = forest[level_start : level_start + level_size]
      relative_idx = (idx - level_start)
      node_val = select_with_where_tree(level_values, relative_idx)
    else:
      # Large level: use gather
      node_val = forest.gather(0, idx)

    val = myhash(val ^ node_val)
    idx = (idx << 1) + (1 + (val & 1))

    # No wrap check needed! At round 10 (level becomes 0), we reset idx above.

  return val.contiguous(arg=(Opt(OptOps.UPCAST, 0, 8),))

# ************************* renderer for VLIW machine *************************

def loop_unrolling(sink:UOp):
  rng = [x for x in sink.toposort() if x.op is Ops.RANGE]
  if len(rng) == 0: return None
  print(f"unrolling loop with size {rng[0].vmax+1}")
  unrolled_sinks = [sink.substitute({rng[0]:rng[0].const_like(i)}).src[0] for i in range(rng[0].vmax+1)]
  return UOp.sink(*unrolled_sinks, arg=sink.arg)

global_addrs = []

vliw_prepare = PatternMatcher([
  # loop unrolling (should be a part of tinygrad)
  (UPat(Ops.SINK, name="sink"), loop_unrolling),
  # cast is fake
  (UPat(Ops.CAST, name="c"), lambda c: c.src[0]),
  # rewrites to hardcode the addresses in memory
  (UPat(Ops.PARAM, name="dg"), lambda dg: UOp.const(dtypes.uint, global_addrs[dg.arg])),
  # INDEX is just plus
  (UPat(Ops.INDEX, name="i"), lambda i: i.src[0]+i.src[1]),
  # scalar MULACC not in renderer
  (UPat(Ops.ADD, src=[UPat(Ops.MUL, src=(UPat(name="a"), UPat(name="b"))), UPat(name="c")]), lambda a, b, c: UOp(Ops.MULACC, a.dtype, (a, b, c))),
])+symbolic

@dataclass
class ScheduledUOp:
  idx: int
  elements: tuple[int,int] | None = None

class RegisterAllocator:
  def __init__(self, uops: list[UOp]):
    self.uops = uops
    self.r: dict[UOp, int] = {}
    self.uses: dict[UOp, int] = {}
    self.free_segments: list[tuple[int, int]] = []
    self.next_reg = 0
    self.zero_reg = 0
    self.pinned: set[UOp] = set()
    self.alloc_ops = {Ops.STORE, Ops.SINK, Ops.GEP}
    
    # Pin CONST(0) to zero_reg if it exists
    for u in uops:
      if u.op is Ops.CONST and u.arg == 0:
        self.r[u] = self.zero_reg
        self.pinned.add(u)
        break
    self.next_reg = max(self.next_reg, self.zero_reg + 1)

  def get_reg(self, u: UOp) -> int:
    if u.op is Ops.GEP:
      return self.get_reg(u.src[0]) + u.arg[0]
    return self.r[u]
  
  def alloc(self, u: UOp):
    if u not in self.alloc_ops and u not in self.r:
      k = u.dtype.count
      for idx, (s, l) in enumerate(self.free_segments):
        if l >= k:
          self.r[u] = s
          self.free_segments[idx:idx+1] = ([(s + k, l - k)] if l > k else [])
          return
      self.r[u] = self.next_reg
      self.next_reg += k

  def free(self, u: UOp):
    if u in self.r and u not in self.pinned:
      self.free_segments.append((self.r[u], u.dtype.count))
      del self.r[u]

  def use(self, u: UOp, inc: bool):
    # increment or decrement use count
    cu = u.src[0] if u.op is Ops.GEP else u
    if cu not in self.alloc_ops and cu not in self.pinned:
      self.uses[cu] = self.uses.get(cu, 0) + (1 if inc else -1)


class VLIWPacker:
  DEMOTABLE = {Ops.ADD, Ops.MUL, Ops.XOR, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.CMPLT}
  def __init__(self, uops: list[UOp], code_for_op, SLOT_LIMITS):
    self.uops = uops
    self.code_for_op = code_for_op
    self.SLOT_LIMITS = SLOT_LIMITS
    self.users: list[list[int]] = []
    self.deps_left: list[int] = []
    self.uop_to_idx: dict[UOp, int] = {}
    self.build_deps()


  def _get_engine(self, u: UOp) -> str:
    """Determine engine for a UOp."""
    if u.op is Ops.CONST:
      return "load"
    if u.op is Ops.VECTORIZE:
      return "valu" if all(s == u.src[0] for s in u.src) else "alu"
    if u.op is Ops.LOAD:
      return "load"
    if u.op is Ops.STORE:
      return "store"
    if u.op is Ops.MULACC:
      return "valu"
    if u.op is Ops.WHERE:
      return "flow"
    if u.op in self.code_for_op:
      return "valu" if u.dtype.count > 1 else "alu"
    raise NotImplementedError(f"unhandled op {u.op}")

  def build_deps(self):
    self.uop_to_idx = {u: i for i, u in enumerate(self.uops)}
    n = len(self.uops)
    self.users: list[list[int]] = [[] for _ in range(n)]
    self.deps_left: list[int] = [0] * n
    
    for i, u in enumerate(self.uops):
      # SINK and GEP don't produce values
      if u.op is Ops.SINK or u.op is Ops.GEP:
        if u.op is Ops.GEP: 
          self.uop_to_idx[u] = self.uop_to_idx.get(u.src[0])
        continue
      for src in u.src:
        if src in self.uop_to_idx:
          pred = self.uop_to_idx[src]
          self.users[pred].append(i)  
          self.deps_left[i] += 1  
    
    self.depth = self._compute_depth()
    self.batch_ids, self.offsets = self._assign_batches()

  def _compute_depth(self) -> list[int]:
    n = len(self.uops)
    depth = [0] * n
    for u, idx in self.uop_to_idx.items():
      if u.src:
        depth[idx] = 1 + max(depth[self.uop_to_idx.get(d, 0)] for d in u.src)
    return depth

  def _assign_batches(self) -> tuple[list[int | None], list[int]]:
    def walk(u: UOp, bi: int):
        idx = self.uop_to_idx.get(u)
        if idx is None or visited_by[idx] == bi: return  
        visited_by[idx] = bi
        if batch_ids[idx] is None: batch_ids[idx] = bi
        # shared UOp
        elif batch_ids[idx] != bi: batch_ids[idx] = None
        for s in u.src: walk(s, bi)

    n = len(self.uops)
    batch_ids= [None] * n
    sink = self.uops[-1]
    batch_count = len(sink.src)
    visited_by = [None] * n  
    for bi, root in enumerate(sink.src): walk(root, bi)

    # stagger batches evenly across the schedule
    schedule_length = max(self.depth) + 1 if self.depth else 1
    spacing = schedule_length // batch_count
    offsets = [(bi * spacing) % schedule_length for bi in range(batch_count)]
    return batch_ids, offsets

  def _priority(self, i: int) -> int:
    bid = self.batch_ids[i]
    # start scheduling with low depth and initial batches
    return self.depth[i] + (self.offsets[bid] if bid is not None else 0)

  def _slot_count(self, i: int) -> int:
    uop = self.uops[i]
    # vectorize is len(src) ALU moves
    if uop.op is Ops.VECTORIZE:
      if not all(s == uop.src[0] for s in uop.src):  
        return len(set(uop.src))
    if uop.op is Ops.MULACC and uop.dtype.count == 1: print("scalar MULACC"); return 2 
    return 1

  def pack(self):
    n = len(self.uops)
    ready = [i for i in range(n) if self.deps_left[i] == 0 and self.uops[i].op not in {Ops.SINK, Ops.GEP}]
    bundles = [] # engine : list of scheduled ops (cycle i)
    pending_split = None # instruction pending (4, 8) scheduling

    while ready or pending_split is not None:
      current, slots, next_ready = {}, {e: 0 for e in self.SLOT_LIMITS}, []
      # finish pending split first (4, 8)
      if pending_split is not None:
        current.setdefault("alu", []).append(ScheduledUOp(pending_split, (4, 8)))
        slots["alu"] += 4
        for u in self.users[pending_split]:
          self.deps_left[u] -= 1
          if self.deps_left[u] == 0: next_ready.append(u)
        pending_split = None  

      # sort by batch and dependency depth
      ready.sort(key=self._priority)
      for i in ready:
        u = self.uops[i]
        eng = self._get_engine(u)
        cost = self._slot_count(i)
        if slots[eng] + cost <= self.SLOT_LIMITS[eng]:
          current.setdefault(eng, []).append(ScheduledUOp(i, None))
          slots[eng] += cost
          for u in self.users[i]:
            self.deps_left[u] -= 1
            if self.deps_left[u] == 0: next_ready.append(u)
          continue
        # valu full, try demotion
        if eng == "valu" and u.op in self.DEMOTABLE and u.dtype.count == 8:
          alu_spare = self.SLOT_LIMITS["alu"] - slots["alu"]
          if alu_spare >= 8: # 8 wide demotion
            current.setdefault("alu", []).append(ScheduledUOp(i, (0, 8)))
            slots["alu"] += 8
            for u in self.users[i]:
              self.deps_left[u] -= 1
              if self.deps_left[u] == 0: next_ready.append(u)
            continue
          if alu_spare >= 4 and pending_split is None: # 4+4 split demotion
            current.setdefault("alu", []).append(ScheduledUOp(i, (0, 4)))
            slots["alu"] += 4
            pending_split = i
            continue  
        # slot limited (leave for next bundles)
        next_ready.append(i) 
      assert current is not None, "Deadlock while packing"
      bundles.append(current)
      ready = next_ready

    return bundles

  def emit(self, bundles):
    ra = RegisterAllocator(self.uops)

    # Emit helper for base (non-sliced)
    def emit_base(i: int, elements: tuple[int, int] | None = None):
      u = self.uops[i]
      op = u.op
      v = "broadcast" if u.op is Ops.VECTORIZE and all(s == u.src[0] for s in u.src) else "moves"

      # demoted emission
      if elements is not None:
        lo, hi = elements
        op2 = self.code_for_op[u.op]
        return [
          (op2, ra.get_reg(u) + k, ra.get_reg(u.src[0]) + k, ra.get_reg(u.src[1]) + k)
          for k in range(lo, hi)
        ]

      # skip emitting CONST(0) if it's mapped to pinned zero_reg
      if op is Ops.CONST:
        if u.arg == 0 and ra.get_reg(u) == ra.zero_reg:
          return None
        return ("const", ra.get_reg(u), u.arg)

      if op is Ops.VECTORIZE:
        if v == "broadcast":
          return ("vbroadcast", ra.get_reg(u), ra.get_reg(u.src[0]))
        # v == "moves": emit ALU moves (+ 0), skip if src already in place
        out = []
        dst_base = ra.get_reg(u)
        for lane, s in enumerate(u.src):
          dst = dst_base + lane
          src = ra.get_reg(s)
          if src != dst:
            out.append(("+", dst, src, ra.zero_reg))
        return out

      if op is Ops.LOAD:
        return (("vload" if u.dtype.count > 1 else "load"), ra.get_reg(u), ra.get_reg(u.src[0]))

      if op is Ops.STORE:
        return (("vstore" if u.src[1].dtype.count > 1 else "store"), ra.get_reg(u.src[0]), ra.get_reg(u.src[1]))

      if op is Ops.MULACC:
        return ("multiply_add", ra.get_reg(u), ra.get_reg(u.src[0]), ra.get_reg(u.src[1]), ra.get_reg(u.src[2]))

      if op is Ops.WHERE:
        return ("vselect", ra.get_reg(u), ra.get_reg(u.src[0]), ra.get_reg(u.src[1]), ra.get_reg(u.src[2]))

      return (self.code_for_op[op], ra.get_reg(u), ra.get_reg(u.src[0]), ra.get_reg(u.src[1]))

    # Build use counts from scheduled occurrences
    for bundle in bundles:
      for eng, items in bundle.items():
        for it in items:
          u = self.uops[it.idx]
          if u.op is Ops.VECTORIZE and all(s == u.src[0] for s in u.src):
            ra.use(u.src[0], inc=True)
          else:
            for s in u.src:
              ra.use(s, inc=True)

    result = []
    result.append({"load": [("const", ra.zero_reg, 0)]})

    # Main emission loop (bundle-safe freeing)
    for bundle in bundles:
      out = {}
      for eng, items in bundle.items():
        tup = []
        for it in items:
          u = self.uops[it.idx]
          ra.alloc(u)

          if u.op is Ops.VECTORIZE and all(s == u.src[0] for s in u.src):
            ra.use(u.src[0], inc=False)
          else:
            for s in u.src:
              ra.use(s, inc=False)

          # emit instruction
          tup.extend(t if isinstance(t:=emit_base(it.idx, it.elements), list) else [t])
        out[eng] = tup
      result.append(out)

      # free no longer used registers
      for cu, cnt in list(ra.uses.items()):
        if cnt == 0:
          ra.free(cu)
          del ra.uses[cu]

    result.append({"flow": [("halt",)]})
    return repr(result)

class VLIWRenderer(Renderer):
  has_local = False  # TODO: this should be the default / cleaned up
  # this says this backend supports MULACC + more. decompositions uses this
  code_for_op: dict = {Ops.MULACC: None, Ops.ADD: "+", Ops.MUL: "*",
                       Ops.XOR: "^", Ops.AND: "&", Ops.OR: "|",
                       Ops.SHL: "<<", Ops.SHR: ">>", Ops.CMPLT: "<"}
  # this matcher runs while still in graph form
  pre_matcher = vliw_prepare

  def render(self, uops:list[UOp]):
    print(f"rendering with {len(uops)} uops")
    print(f"len(uops[-1].src): {len(uops[-1].src)}")
    SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1, "debug": 64}
    packer = VLIWPacker(uops, self.code_for_op, SLOT_LIMITS)
    bundles = packer.pack()
    return packer.emit(bundles)


# ************************* test and render *************************

import sys, types
PROBLEM_URL = "https://raw.githubusercontent.com/anthropics/original_performance_takehome/refs/heads/main/tests/frozen_problem.py"
sys.modules["problem"] = problem = types.ModuleType("problem")
exec(fetch(PROBLEM_URL).read_text(), problem.__dict__)

if __name__ == "__main__":
  batch_size = getenv("BS", 256)
  height = 10
  rounds = getenv("ROUNDS", 16)

  # build problem
  tree = problem.Tree.generate(height)
  inp = problem.Input.generate(tree, batch_size, rounds)
  mem = problem.build_mem_image(tree, inp)
  global_addrs.extend([mem[6], mem[6], mem[4]])  # output, input, forest

  # *** verify the kernel in tinygrad compared to reference ***

  forest_t = Tensor(tree.values, dtype=dtypes.uint32)
  val_t = Tensor(inp.values, dtype=dtypes.uint32)

  if getenv("VERIFY", 1):
    # verify on normal tinygrad device
    with Context(PCONTIG=2):
      out = tree_traversal(forest_t, val_t, height, rounds)
      val_out = out.tolist()
    problem.reference_kernel(tree, inp)
    assert val_out == inp.values
    print("verification passed")

  # *** render to device ***

  from tinygrad.codegen import get_program
  with Context(PCONTIG=2, DEVECTORIZE=2, SPEC=0):
    out = tree_traversal(forest_t, val_t, height, rounds)
    sink = out.schedule()[-1].ast
    prg = get_program(sink, VLIWRenderer())

  # *** run on Machine and compare ***

  # NOTE: the scratch size needs to be reduced to 1536 when you have a register allocator
  src = eval(prg.src)
  max_regs = max(t[1] for instr in src for v in instr.values() for t in v if len(t) > 1) + 8
  print(f"{max_regs:5d} regs used" + ("" if max_regs <= 1536 else "       <-- WARNING: TOO MANY REGISTERS, MUST BE <= 1536"))
  enable_trace = getenv("TRACE", 0)
  machine = problem.Machine(mem, src, problem.DebugInfo(scratch_map={}), n_cores=1, trace=bool(enable_trace), scratch_size=max_regs)
  machine.run()
  print(f"ran for {machine.cycle:5d} cycles" + ("" if machine.cycle <= 1363 else "  <-- EVEN CLAUDE GOT 1363"))
  if enable_trace:
    print(f"Trace saved to trace.json")
    print(f"  View in Chrome: chrome://tracing (load trace.json)")
    print(f"  View online: https://ui.perfetto.dev (drag trace.json)")

  # compare to reference
  ref_mem = mem.copy()
  for _ in problem.reference_kernel2(ref_mem, {}): pass
  assert machine.mem[mem[6]:mem[6]+mem[2]] == ref_mem[mem[6]:mem[6]+mem[2]]
  print("compare passed!")

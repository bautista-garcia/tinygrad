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
])+symbolic



@dataclass(frozen=True)
class ScheduledOp:
  idx: int
  engine: str
  elements: tuple[int,int] | None = None


class VLIWPacker:
  DEMOTABLE = {Ops.ADD, Ops.MUL, Ops.XOR, Ops.AND, Ops.OR, Ops.SHL, Ops.SHR, Ops.CMPLT}
  def __init__(self, uops: list[UOp], code_for_op, SLOT_LIMITS):
    self.uops = uops
    self.code_for_op = code_for_op
    self.SLOT_LIMITS = SLOT_LIMITS
    self.instructions: list[tuple] = []  # (engine, uop, variant)
    self.users: list[list[int]] = []
    self.deps_left: list[int] = []
    self.build_deps()
    # self.batch_id = self.assign_batch_ids()
    # self.depth = self.compute_depth()
    # self.offsets = self.compute_offsets()


  def demotable(self, i: int) -> bool:
    eng, uop, variant = self.instructions[i]
    return eng == "valu" and variant is None and uop.op in self.DEMOTABLE and uop.dtype.count == 8

  def build_deps(self):
    def_instr: dict[UOp, list[int]] = {}  # maps uop -> list of instruction indices it produces

    def add(engine: str, uop: UOp, variant, src_uops: list[UOp]):
      i = len(self.instructions)
      self.instructions.append((engine, uop, variant))
      self.users.append([])
      self.deps_left.append(0)
      for s in src_uops:
        for pred in def_instr.get(s, []):
          self.users[pred].append(i)
          self.deps_left[i] += 1
      return i

    for u in self.uops:
      if u.op is Ops.SINK: continue
      if u.op is Ops.GEP:
        def_instr[u] = def_instr.get(u.src[0], [])
        continue

      match u.op:
        case Ops.CONST:
          def_instr[u] = [add("load", u, None, [])]
        case Ops.VECTORIZE:
          if all(s == u.src[0] for s in u.src):
            def_instr[u] = [add("valu", u, "broadcast", [u.src[0]])]
          else:
            def_instr[u] = [add("flow", u, lane, [s]) for lane, s in enumerate(u.src)]
        case Ops.LOAD:
          def_instr[u] = [add("load", u, None, [u.src[0]])]
        case Ops.STORE:
          add("store", u, None, list(u.src[:2]))
        case Ops.MULACC:
          def_instr[u] = [add("valu", u, None, list(u.src))]
        case Ops.WHERE:
          def_instr[u] = [add("flow", u, None, list(u.src))]
        case _ if u.op in self.code_for_op:
          engine = "valu" if u.dtype.count > 1 else "alu"
          def_instr[u] = [add(engine, u, None, list(u.src))]
        case _:
          raise NotImplementedError(f"unhandled op {u.op}")
        
  # def assign_batch_ids(self) -> list[int|None]:
  #   sink = self.uops[-1]
  #   assert sink.op is Ops.SINK and len(sink.src) == 32

  #   uop_batch: dict[UOp, int|None] = {}
  #   seen: set[tuple[UOp,int]] = set()

  #   def walk(u: UOp, bi: int):
  #     k = (u, bi)
  #     if k in seen: return
  #     seen.add(k)

  #     if u not in uop_batch: uop_batch[u] = bi
  #     elif uop_batch[u] != bi: uop_batch[u] = None  # shared across batches

  #     for s in u.src: walk(s, bi)

  #   for bi, root in enumerate(sink.src):
  #     walk(root, bi)

  #   return [uop_batch.get(uop, None) for _, uop, _ in self.instructions]
  
  # def compute_depth(self) -> list[int]:
  #   n = len(self.instructions)
  #   depth = [0]*n
  #   for i in range(n-1, -1, -1):
  #     if self.users[i]:
  #       depth[i] = 1 + max(depth[u] for u in self.users[i])
  #   return depth
  
  # def compute_offsets(self) -> list[int]:
  #   sink = self.uops[-1]
  #   B = len(sink.src)   # 32
  #   sched_len = max(self.depth) + 1 if self.depth else 1
  #   spacing = max(1, sched_len // B)
  #   return [(bi * spacing) % sched_len for bi in range(B)]

  # def prio(self, i):
  #   bid = self.batch_id[i]
  #   off = self.offsets[bid] if bid is not None else 0
  #   return -(self.depth[i] + off)   # more depth => smaller key => earlier


  def pack(self):
    # heuristics for scheduling
    n = len(self.instructions)
    is_load = [self.instructions[i][0] == "load" for i in range(n)]
    unlocks_load = [any(is_load[u] for u in self.users[i]) for i in range(n)]
    outdeg = [len(self.users[i]) for i in range(n)]

    ready = [i for i in range(n) if self.deps_left[i] == 0]
    bundles: list[dict[str, list[ScheduledOp]]] = []
    pending_split: int | None = None

    while ready or pending_split is not None:
      current, slots, next_ready = {}, {e: 0 for e in self.SLOT_LIMITS}, []
      # finish pending split first (hi half)
      if pending_split is not None:
        current.setdefault("alu", []).append(ScheduledOp(pending_split, "alu", (4, 8)))
        slots["alu"] += 4
        for u in self.users[pending_split]:
          self.deps_left[u] -= 1
          if self.deps_left[u] == 0: next_ready.append(u)
        pending_split = None  # reset

      if not ready:
        if current: bundles.append(current)
        break

      ready.sort(key=lambda i: (-outdeg[i], -unlocks_load[i]))

      for i in ready:
        eng = self.instructions[i][0]

        # base scheduling
        if slots[eng] < self.SLOT_LIMITS[eng]:
          current.setdefault(eng, []).append(ScheduledOp(i, eng, None))
          slots[eng] += 1
          for u in self.users[i]:
            self.deps_left[u] -= 1
            if self.deps_left[u] == 0: next_ready.append(u)
          continue

        # valu -> alu demotion
        if eng == "valu" and self.demotable(i):
          alu_spare = self.SLOT_LIMITS["alu"] - slots["alu"]
          if alu_spare >= 8: # 8 wide demotion
            current.setdefault("alu", []).append(ScheduledOp(i, "alu", (0, 8)))
            slots["alu"] += 8
            for u in self.users[i]:
              self.deps_left[u] -= 1
              if self.deps_left[u] == 0: next_ready.append(u)
            continue
          if alu_spare >= 4 and pending_split is None: # 4+4 split demotion
            current.setdefault("alu", []).append(ScheduledOp(i, "alu", (0, 4)))
            slots["alu"] += 4
            pending_split = i
            continue  

        next_ready.append(i)

      if current: bundles.append(current)
      ready = next_ready

    return bundles

  def emit(self, bundles):
    # reg alloc
    reg, r = 0, {}
    for u in self.uops:
      if u.op not in {Ops.STORE, Ops.SINK, Ops.GEP}:
        r[u] = reg; reg += u.dtype.count
      elif u.op is Ops.GEP:
        r[u] = r[u.src[0]] + u.arg[0]

    def lane_reg(u: UOp, lane: int):
      return r[u] + lane if u.dtype.count > 1 else r[u]

    def emit_base(i: int):
      _, u, v = self.instructions[i]
      op = u.op
      if op is Ops.CONST: return ("const", r[u], u.arg)
      if op is Ops.VECTORIZE:
        if v == "broadcast": return ("vbroadcast", r[u], r[u.src[0]])
        s = u.src[v]
        return None if r[s] == r[u] + v else ("add_imm", r[u] + v, r[s], 0)
      if op is Ops.LOAD:  return (("vload" if u.dtype.count > 1 else "load"),  r[u], r[u.src[0]])
      if op is Ops.STORE: return (("vstore" if u.src[1].dtype.count > 1 else "store"), r[u.src[0]], r[u.src[1]])
      if op is Ops.MULACC: return ("multiply_add", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]])
      if op is Ops.WHERE:  return ("vselect", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]])
      return (self.code_for_op[op], r[u], r[u.src[0]], r[u.src[1]])

    result = []
    for bundle in bundles:
      out = {}
      for eng, items in bundle.items():
        tup = []
        for it in items:
          it = it if isinstance(it, ScheduledOp) else ScheduledOp(it, eng, None)

          if it.elements is None:
            t = emit_base(it.idx)
            if t is not None: tup.append(t)
          else:
            _, u, _ = self.instructions[it.idx]
            op = self.code_for_op[u.op]  
            lo, hi = it.elements
            tup += [(op, lane_reg(u, k), lane_reg(u.src[0], k), lane_reg(u.src[1], k)) for k in range(lo, hi)]

        if tup: out[eng] = tup
      if out: result.append(out)

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

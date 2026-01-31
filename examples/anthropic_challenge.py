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
            # Non-broadcast VECTORIZE: ALU moves (not flow)
            def_instr[u] = [add("alu", u, "moves", list(u.src))]
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

    # Build reverse mapping: deps[i] = list of instruction indices that i depends on
    n = len(self.instructions)
    self.deps: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
      for user_idx in self.users[i]:
        self.deps[user_idx].append(i)

    # Build uop_to_idx mapping for batch assignment
    self.uop_to_idx: dict[UOp, int] = {}
    for idx, (_, uop, _) in enumerate(self.instructions):
      if uop not in self.uop_to_idx:
        self.uop_to_idx[uop] = idx

    # Compute depth, batch_ids, and offsets
    self.depth = self._compute_depth()
    self.batch_ids, self.offsets = self._assign_batches()

  def _compute_depth(self) -> list[int]:
    """Compute critical path depth for each instruction (used for scheduling priority)."""
    n = len(self.instructions)
    depth = [0] * n
    for i in range(n):
      if self.deps[i]:
        depth[i] = 1 + max(depth[d] for d in self.deps[i])
    return depth

  def _assign_batches(self) -> tuple[list[int | None], list[int]]:
    """Assign batch IDs and compute stagger offsets."""
    n = len(self.instructions)
    batch_ids: list[int | None] = [None] * n
    sink = self.uops[-1]
    assert sink.op is Ops.SINK
    batch_count = len(sink.src)

    if batch_count > 1:
      # visited_by[i] = which batch last visited instruction i
      visited_by: list[int | None] = [None] * n
      def walk(u: UOp, bi: int):
        idx = self.uop_to_idx.get(u)
        if idx is None: return
        if visited_by[idx] == bi: return  # already visited by this batch
        visited_by[idx] = bi
        if batch_ids[idx] is None: batch_ids[idx] = bi
        elif batch_ids[idx] != bi: batch_ids[idx] = None  # shared across batches
        for s in u.src: walk(s, bi)
      for bi, root in enumerate(sink.src): walk(root, bi)

    # stagger batches evenly across the schedule
    schedule_length = max(self.depth) + 1 if self.depth else 1
    spacing = schedule_length // batch_count if batch_count > 0 else 1
    offsets = [(bi * spacing) % schedule_length for bi in range(batch_count)]
    return batch_ids, offsets

  def _priority(self, i: int) -> int:
    """Scheduling priority: lower = scheduled earlier."""
    bid = self.batch_ids[i]
    # shared instructions (bid=None) get offset 0, so they're scheduled first
    return self.depth[i] + (self.offsets[bid] if bid is not None else 0)

  def _engine(self, uop: UOp) -> str:
    """Get the engine for a UOp."""
    # Find the instruction index for this UOp
    idx = self.uop_to_idx.get(uop)
    if idx is not None:
      eng, _, variant = self.instructions[idx]
      # Non-broadcast VECTORIZE is ALU (not flow)
      if eng == "alu" and variant == "moves":
        return "alu"
      return eng
    return "none"

  def _slot_count(self, i: int) -> int:
    """Return slot cost for instruction i."""
    _, uop, variant = self.instructions[i]
    # Non-broadcast VECTORIZE costs = number of unique scalar sources (ALU moves)
    if uop.op is Ops.VECTORIZE and variant == "moves":
      return len(set(uop.src))
    return 1

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

      # group ready ops by engine and sort by priority
      ready_by_engine: dict[str, list[int]] = {k: [] for k in [*self.SLOT_LIMITS.keys(), "none"]}
      for i in ready:
        engine, _, _ = self.instructions[i]
        ready_by_engine[engine].append(i)
      for eng in ready_by_engine:
        ready_by_engine[eng].sort(key=self._priority)

      for eng in ready_by_engine:
        for i in ready_by_engine[eng]:
          # base scheduling
          cost = self._slot_count(i)
          if slots[eng] + cost <= self.SLOT_LIMITS[eng]:
            current.setdefault(eng, []).append(ScheduledOp(i, eng, None))
            slots[eng] += cost
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
    # reg alloc - find zero constant for ALU moves (VECTORIZE), fallback to reg 0
    reg, r, zero_reg = 0, {}, 0
    for u in self.uops:
      if u.op not in {Ops.STORE, Ops.SINK, Ops.GEP}:
        r[u] = reg; reg += u.dtype.count
        if u.op is Ops.CONST and u.arg == 0: zero_reg = r[u]
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
        # v == "moves": emit ALU moves (+ 0), skip if src already in place
        return [("+", r[u] + lane, r[s], zero_reg) for lane, s in enumerate(u.src) if r[s] != r[u] + lane]
      if op is Ops.LOAD:  return (("vload" if u.dtype.count > 1 else "load"),  r[u], r[u.src[0]])
      if op is Ops.STORE: return (("vstore" if u.src[1].dtype.count > 1 else "store"), r[u.src[0]], r[u.src[1]])
      if op is Ops.MULACC: return ("multiply_add", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]])
      if op is Ops.WHERE:  return ("vselect", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]])
      return (self.code_for_op[op], r[u], r[u.src[0]], r[u.src[1]])

    result = []
    # Ensure zero register exists (needed for ALU moves in VECTORIZE)
    if zero_reg == 0:
      # Allocate zero register if no CONST(0) exists
      zero_reg = reg
      result.append({"load": [("const", zero_reg, 0)]})
    
    for bundle in bundles:
      out = {}
      for eng, items in bundle.items():
        tup = []
        for it in items:
          it = it if isinstance(it, ScheduledOp) else ScheduledOp(it, eng, None)

          if it.elements is None:
            t = emit_base(it.idx)
            if t is not None:
              if isinstance(t, list): tup.extend(t)
              else: tup.append(t)
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

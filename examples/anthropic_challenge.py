from tinygrad import Tensor, dtypes, Context, getenv, UOp, fetch
from tinygrad.uop.ops import Ops, PatternMatcher, UPat
from tinygrad.uop.symbolic import symbolic
from tinygrad.codegen import Renderer
from tinygrad.codegen.opt import Opt, OptOps

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

class VLIWRenderer(Renderer):
  has_local = False  # TODO: this should be the default / cleaned up
  # this says this backend supports MULACC + more. decompositions uses this
  code_for_op: dict = {Ops.MULACC: None, Ops.ADD: "+", Ops.MUL: "*",
                       Ops.XOR: "^", Ops.AND: "&", Ops.OR: "|",
                       Ops.SHL: "<<", Ops.SHR: ">>", Ops.CMPLT: "<"}
  # this matcher runs while still in graph form
  pre_matcher = vliw_prepare

  def render(self, uops:list[UOp]):

    # TODO: this is a minimal renderer. for low cycle count, make it good
    # to get speed, you need to add VLIW packing
    # to get under 1536 regs, you need to add a register allocator
    # we left the fun parts to you

    print(f"rendering with {len(uops)} uops")
    
    # Slot limits per engine
    SLOT_LIMITS = {
      "alu": 12,
      "valu": 6,
      "load": 2,
      "store": 2,
      "flow": 1,
      "debug": 64
    }
    
    # Phase 1: Generate instructions and build dependency graph in single forward pass
    instructions: list[tuple[str, tuple]] = []  # (engine, instr_tuple) pairs
    reads: list[set[int]] = []   # reads[i] = registers read by instruction i
    writes: list[set[int]] = []  # writes[i] = registers written by instruction i
    engines: list[str] = []      # engines[i] = engine for instruction i

    # per-instruction memory access metadata (keyed by address register)
    mem_reads_addr: list[set[int]] = []   # mem_reads_addr[i] = address regs read (LOAD)
    mem_writes_addr: list[set[int]] = []  # mem_writes_addr[i] = address regs written (STORE)
    
    # register dependency tracking
    last_write: dict[int, int] = {}  # last_write[reg] = instruction index that last wrote reg
    users: list[list[int]] = []      # users[i] = instructions that depend on i
    deps_left: list[int] = []        # deps_left[i] = remaining dependencies

    # memory dependency tracking (by address register)
    last_mem_write: dict[int, int] = {}  # last_mem_write[addr_reg] = last store to that address
    last_mem_read: dict[int, int] = {}   # last_mem_read[addr_reg] = last load from that address
    
    def build_deps(engine: str, instr_tuple: tuple, r_set: set[int], w_set: set[int]):
      """Add instruction and build register + memory dependencies."""
      i = len(instructions)
      instructions.append((engine, instr_tuple))
      reads.append(r_set)
      writes.append(w_set)
      engines.append(engine)
      users.append([])
      deps_left.append(0)
      mem_reads_addr.append(set())
      mem_writes_addr.append(set())
      
      # Build register dependencies: RAW and WAW
      for reg in r_set:
        if reg in last_write:
          users[last_write[reg]].append(i)
          deps_left[i] += 1
      for reg in w_set:
        if reg in last_write:
          users[last_write[reg]].append(i)
          deps_left[i] += 1
        last_write[reg] = i

      # Build memory dependencies keyed by address register for LOAD/STORE
      if engine in ("load", "store"):
        # Convention: LOAD: (op, dest, addr_reg), STORE: (op, addr_reg, src)
        addr_reg = instr_tuple[2] if engine == "load" else instr_tuple[1]
        if engine == "load":
          # LOAD depends on last STORE to the same address (memory RAW)
          if addr_reg in last_mem_write:
            users[last_mem_write[addr_reg]].append(i)
            deps_left[i] += 1
          last_mem_read[addr_reg] = i
          mem_reads_addr[i].add(addr_reg)
        else:
          # STORE after last STORE (WAW) and after last LOAD (WAR) to the same address
          if addr_reg in last_mem_write:
            users[last_mem_write[addr_reg]].append(i)
            deps_left[i] += 1
          if addr_reg in last_mem_read:
            users[last_mem_read[addr_reg]].append(i)
            deps_left[i] += 1
          last_mem_write[addr_reg] = i
          mem_writes_addr[i].add(addr_reg)
    
    reg = 0
    r: dict[UOp, int] = {}
    for u in uops:
      assert u.dtype.count in (1,8), "dtype count must be 1 or 8"

      # dumb register allocator
      if u.op not in {Ops.STORE, Ops.SINK, Ops.GEP}:
        r[u] = reg
        reg += u.dtype.count

      # Generate instruction and build read/writes 
      match u.op:
        case Ops.SINK:
          build_deps("flow", ("halt",), set(), set())
        case Ops.CONST:
          build_deps("load", ("const", r[u], u.arg), set(), {r[u]})
        case Ops.GEP:
          # a GEP is just an alias to a special register in the vector
          r[u] = r[u.src[0]] + u.arg[0]
        case Ops.VECTORIZE:
          if all(s == u.src[0] for s in u.src):
            build_deps("valu", ("vbroadcast", r[u], r[u.src[0]]), {r[u.src[0]]}, set(range(r[u], r[u] + 8)))
          else:
            for i, s in enumerate(u.src):
              if r[s] != r[u] + i:
                build_deps("flow", ("add_imm", r[u]+i, r[s], 0), {r[s]}, {r[u]+i})
        case Ops.LOAD:
          op = "vload" if u.dtype.count > 1 else "load"
          count = 8 if op == "vload" else 1
          build_deps("load", (op, r[u], r[u.src[0]]), {r[u.src[0]]}, set(range(r[u], r[u] + count)))
        case Ops.STORE:
          op = "vstore" if u.src[1].dtype.count > 1 else "store"
          count = 8 if op == "vstore" else 1
          build_deps("store", (op, r[u.src[0]], r[u.src[1]]),
                     {r[u.src[0]]} | set(range(r[u.src[1]], r[u.src[1]] + count)), set())
        case Ops.MULACC:
          assert u.dtype.count == 8
          build_deps("valu", ("multiply_add", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]]),
                    set(range(r[u.src[0]], r[u.src[0]] + 8)) | set(range(r[u.src[1]], r[u.src[1]] + 8)) | set(range(r[u.src[2]], r[u.src[2]] + 8)),
                    set(range(r[u], r[u] + 8)))
        case Ops.WHERE:
          assert u.dtype.count == 8
          build_deps("flow", ("vselect", r[u], r[u.src[0]], r[u.src[1]], r[u.src[2]]),
                    set(range(r[u.src[0]], r[u.src[0]] + 8)) | set(range(r[u.src[1]], r[u.src[1]] + 8)) | set(range(r[u.src[2]], r[u.src[2]] + 8)),
                    set(range(r[u], r[u] + 8)))
        case _ if u.op in self.code_for_op:
          cat = "valu" if u.dtype.count > 1 else "alu"
          count = 8 if cat == "valu" else 1
          build_deps(cat, (self.code_for_op[u.op], r[u], r[u.src[0]], r[u.src[1]]),
                    set(range(r[u.src[0]], r[u.src[0]] + count)) | set(range(r[u.src[1]], r[u.src[1]] + count)),
                    set(range(r[u], r[u] + count)))
        case _:
          raise NotImplementedError(f"unhandled op {u.op}")
    
    # Scheduler
    ready = [i for i in range(len(instructions)) if deps_left[i] == 0]
    scheduled = [False] * len(instructions)
    bundles = []
    
    while ready or any(not s for s in scheduled):
      # Start new bundle
      current_bundle = {}
      bundle_slots = {e: 0 for e in SLOT_LIMITS}
      bundle_writes = set()
      bundle_mem_reads = set()   # address registers read in this bundle
      bundle_mem_writes = set()  # address registers written in this bundle
      bundle_added = False
      
      # Try to fill bundle from ready list
      remaining_ready = []
      for i in ready:
        if scheduled[i]:
          continue
        
        engine = engines[i]
        
        # Check if instruction fits in current bundle
        mem_conflict = (
          (mem_reads_addr[i] & bundle_mem_writes) or
          (mem_writes_addr[i] & (bundle_mem_reads | bundle_mem_writes))
        )
        fits = (
          bundle_slots[engine] < SLOT_LIMITS[engine] and
          not (reads[i] & bundle_writes) and  # No RAW hazard
          not (writes[i] & bundle_writes) and # No WAW hazard
          not mem_conflict                    # No memory hazards on same addr_reg
        )
        
        if fits:
          # Add to bundle
          current_bundle.setdefault(engine, []).append(instructions[i][1])
          bundle_slots[engine] += 1
          bundle_writes.update(writes[i])
          bundle_mem_reads.update(mem_reads_addr[i])
          bundle_mem_writes.update(mem_writes_addr[i])
          scheduled[i] = True
          bundle_added = True
          
          # Update dependencies: mark dependent instructions as potentially ready
          for u in users[i]:
            deps_left[u] -= 1
            if deps_left[u] == 0:
              remaining_ready.append(u)
        else:
          remaining_ready.append(i)
      
      # Forward progress guarantee: if bundle empty and ready list non-empty,
      # schedule at least one instruction (relax slot/register hazards if needed,
      # but still respect memory hazards to the same addr_reg)
      if not bundle_added and remaining_ready:
        # Pick first ready instruction that satisfies at least slot limit
        for i in remaining_ready:
          if scheduled[i]:
            continue
          engine = engines[i]
          mem_conflict = (
            (mem_reads_addr[i] & bundle_mem_writes) or
            (mem_writes_addr[i] & (bundle_mem_reads | bundle_mem_writes))
          )
          if bundle_slots[engine] < SLOT_LIMITS[engine] and not mem_conflict:
            current_bundle.setdefault(engine, []).append(instructions[i][1])
            bundle_slots[engine] += 1
            bundle_writes.update(writes[i])
            bundle_mem_reads.update(mem_reads_addr[i])
            bundle_mem_writes.update(mem_writes_addr[i])
            scheduled[i] = True
            bundle_added = True
            
            for u in users[i]:
              deps_left[u] -= 1
              if deps_left[u] == 0:
                remaining_ready.append(u)
            break
      
      # Emit bundle if non-empty
      if current_bundle:
        bundles.append(current_bundle)
      
      # Update ready list for next cycle
      ready = remaining_ready
    
    return repr(bundles)

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
  machine = problem.Machine(mem, src, problem.DebugInfo(scratch_map={}), n_cores=1, trace=False, scratch_size=max_regs)
  machine.run()
  print(f"ran for {machine.cycle:5d} cycles" + ("" if machine.cycle <= 1363 else "  <-- EVEN CLAUDE GOT 1363"))

  # compare to reference
  ref_mem = mem.copy()
  for _ in problem.reference_kernel2(ref_mem, {}): pass
  assert machine.mem[mem[6]:mem[6]+mem[2]] == ref_mem[mem[6]:mem[6]+mem[2]]
  print("compare passed!")

# Add a QBE backend to dcc (Phase 1: scalars + control flow)

## Context

dcc currently has exactly one backend (`src/backend_x64/`): it lowers TACKY IR
straight to x86-64 assembly text, doing its own register allocation
(`register_allocate.rs`), spilling (`replace_pseudoregisters.rs`), and
instruction legalization (`fixup.rs`). Adding a second backend that targets
[QBE](https://c9x.me/compile/) (emit QBE IL text, shell out to the `qbe`
tool for real assembly) gives a much simpler, ABI-agnostic backend to compare
against, since QBE does register allocation, calling-convention handling, and
instruction legalization itself. There is currently no backend
abstraction/trait anywhere in the compiler — codegen is called as a fixed,
unconditional sequence directly from `compiler.rs`.

This plan covers **Phase 1 only**: scalar types (int/long/short + unsigned
variants, char variants, double, pointers), arithmetic/comparison/bitwise
ops, all control flow, and simple (no struct-passing/returning) function
calls/recursion. Structs, array indexing, and struct-by-value ABI handling
are an explicit follow-up phase (see Non-goals).

Danielle plans to implement this herself; this document is a reference design,
not a task list for Claude to execute.

## Pre-existing work: branch `qbe` (origin/qbe)

Two commits so far: `d12a3e6 "Make the backend pluggable at the code level"`
(rewritten from an earlier `b22a2dd` after `src/common/backend.rs` was added
in — it was initially committed missing that file, which broke the build;
fixed now) plus a routine dependency-version bump merged in from a branch
named `artemis-qbe` (`derive_more`/`clap`/`logos`/etc. minor bumps only — no
backend code, unrelated to QBE, safe to ignore). **The branch builds cleanly
as of 2026-09-08.** It uses a real `Backend` **trait** rather than the plain
enum-dispatch this plan originally sketched — the rest of this document
(Driver/CLI plumbing, Module layout) is written to match that choice rather
than re-argue for the enum. What's there:

- `src/common/backend.rs`:
  ```rust
  pub(crate) trait Backend {
      fn emit(
          self,
          code: ir::Program,
          symbols: SymbolTable,
          types: &TypeTable,
      ) -> Result<String, CompilerError>;
  }
  ```
  Note it's `pub(crate)`, not `pub` — fine, since `backend_qbe` is a sibling
  module in the same crate.
- `X64Backend` has already been refactored into `src/backend_x64/mod.rs` to
  implement this trait: `struct X64Backend { debug, stage, source_name }`,
  `X64Backend::new(debug, stage, source_name)`, `impl Backend for X64Backend
  { fn emit(self, code, symbols, &types) -> Result<String, CompilerError>
  {...} }` — same steps as before (translate → backend_table → allocate →
  replace_pseudoregisters/fixup → emit_assembly), just reshaped into that
  method.
- `write_debug_file`/`write_debug_text_file` moved from `compiler.rs` into
  `common/mod.rs` as `pub fn`s; a new `pub fn swap_suffix(filename,
  old_suffix, new_suffix)` was added there too and is already used by both
  `X64Backend::emit` and `main.rs`'s `assemble_source`/`compile_source`.
- **Not yet done, still needed**: `compiler.rs::compile()`'s signature is
  unchanged — it unconditionally builds an `X64Backend` and calls
  `.emit(...)`. There is no backend-selection parameter, no CLI flag, and no
  `backend_qbe/` module yet. That's the rest of this plan.

## Staged rollout, by book chapter

Each stage should reach a green run of the conformance suite before moving
on:

```
../writing-a-c-compiler-tests/test_compiler /path/to/dcc --chapter <N> -- --backend qbe
```

QBE will happily accept subtly-wrong IL and still link a working-looking
binary (unlike `backend_x64`, where mistakes tend to just crash) — the
per-chapter test suite, not "it compiled," is the real correctness signal.
Don't skip ahead of a red stage.

| Stage | Chapter(s) | New TACKY surface | Backend work added |
|---|---|---|---|
| 1 | 1–4 (constants, unary, binary arithmetic, logical/relational + short-circuit) | `Return`, `Unary`, `Binary` (arith/bitwise/compare), `Copy`, `Jump`/`JumpIfZero`/`JumpIfNotZero`/`Label` (short-circuit `&&`/`\|\|` already lower to these before "if" even exists) | finish the `qbe` branch's WIP driver plumbing (`Backend` trait is already committed and builds; still need a `BackendKind` enum + `--backend` flag + `qbe` shell-out), then `backend_qbe/mod.rs` skeleton + `QbeBackend`, `types.rs` for `Int` only, the block splitter in `translate_ir.rs`, straight-line instruction lowering, `emit_qbe.rs` |
| 2 | 5 (local variables) | more `Copy`/variable traffic, no new instruction kind | — |
| 3 | 6 (if / conditional expressions) | same instructions, first real branch coverage | — |
| 4 | 7 (compound statements/blocks) | none (scoping is a semantic-analysis concern) | — |
| 5 | 8 (loops, break/continue) + goto/switch extra credit if you want it now | still the same instruction set — `switch` desugars to compare/jump chains and `goto` is just more `Label`/`Jump` in TACKY already | — |
| 6 | 9 (functions) | `FunCall`, non-`Void` `Return`, params | `@start` alloc-scan starts to matter once functions take address-taken params |
| 7 | 10 (file-scope vars, static/extern) | `StaticVariable`/`StaticConstant` | `data` emission, `export`/global handling |
| 8 | 11 (long) | `Long` type, `SignExtend`/`Truncate` to/from `Long` | extend `types.rs` |
| 9 | 12 (unsigned) | `Unsigned`/`UnsignedLong`, `ZeroExtend` | `udiv`/`urem`/`shr`/unsigned comparisons |
| 10 | 13 (floating point) | `Double`, double arithmetic/comparisons, `*ToDouble`/`DoubleTo*` | int↔double conversions (mnemonics already verified against the installed `qbe`) |
| 11 | 14 (pointers) | `GetAddress`/`Load`/`Store` | `alloc4`/`alloc8` for address-taken locals, `Pointer` → `l` |

**Phase 1 ends here.** Chapters 15 (arrays), 16 (chars & strings), 17
(dynamic memory), 18 (structs) all hinge on `AddPtr`/`CopyToOffset`/
`CopyFromOffset` and general array/struct value semantics, which are out of
scope for this plan (see Non-goals) — treat them as one bundled Phase 2, not
further staged sub-chapters, since none of them is independently reachable
without the others. Optional bonus once Stage 11 is green: the
string-literal `data`-emission carve-out (see Function/program structure)
can be pulled forward to unlock `puts("literal")`-style smoke tests without
full array support — but it won't make chapter 16's suite pass on its own
(indexing/`AddPtr` is still missing), so don't treat a chapter-16 pass as a
real target yet.

Chapters 19 (TACKY optimizations) and 20 (register allocation) need **no**
QBE-specific work at all: chapter 19's passes run on TACKY before either
backend sees it, and chapter 20 is moot for QBE, which does its own register
allocation.

## Key finding that shapes the whole design

QBE does **not** require strict SSA input — the frontend can reassign a
`%temp` multiple times across blocks and QBE reconstructs SSA internally.
Combined with QBE owning register allocation and ABI lowering, this means
TACKY's flat 3-address instruction stream maps **almost 1:1** onto QBE IL: one
QBE `%name` per TACKY `Identifier`, no allocator, no spill pass, no
legalization pass needed on dcc's side at all.

## Module layout

New `src/backend_qbe/`, parallel to `backend_x64/` (as already reshaped on the
`qbe` branch — see above) but dropping everything QBE subsumes:

```
src/backend_qbe/
  mod.rs            // pub struct QbeBackend { debug, stage, source_name } + impl Backend for QbeBackend,
                     // mirroring X64Backend exactly; everything else here is a private `mod`
                     // (matching the qbe branch's convention: backend_x64/mod.rs now keeps
                     // translate_ir/emit_asm/etc. private and exposes only the X64Backend struct)
  qbe_ast.rs         // minimal typed IL AST (kept for phase-2 extensibility/consistency
                     // with backend_x64's style, even though phase 1's 1:1 mapping doesn't strictly need it)
  translate_ir.rs    // TACKY -> qbe_ast lowering (instruction selection)
  types.rs           // CType -> QBE type mapping (base/ABI/ext type, alloc size+align)
  emit_qbe.rs        // qbe_ast -> QBE IL text
```

`QbeBackend::emit` (the `Backend` trait method — see Driver/CLI plumbing)
does: `translate_ir` (TACKY → `qbe_ast`) → `emit_qbe` (→ text) → write to
`swap_suffix(&self.source_name, ".c", ".ssa")` → shell out to `qbe` →
return the resulting `.s` path, using `common::swap_suffix` and
`common::write_debug_text_file` exactly as `X64Backend::emit` already does.

Not needed, and why: `register_allocate.rs`/`replace_pseudoregisters.rs` (QBE
allocates/spills), `fixup.rs` (QBE legalizes its own IL), `backend_table.rs`'s
eightbyte classification (no by-value structs in phase 1; QBE does ABI
classification internally when it is relevant), `platform.rs` (emit
unmangled `$name`; QBE's `-t <target>`, defaulting to host, handles
platform-specific symbol mangling itself — verify this on the first smoke
test on macOS/arm64).

`common/` (TACKY IR, `CType`, `SymbolTable`, `TypeTable`) needs no changes.

## Type lowering (`CType` → QBE)

| `CType` | SSA temp type | ABI type (param/ret/call) | data type | alloc |
|---|---|---|---|---|
| Int / Unsigned | `w` | `w` | `w` | `alloc4` |
| Long / UnsignedLong / Pointer | `l` | `l` | `l` | `alloc8` |
| Short | `w` (sign-ext canonical) | `sh` | `h` | `alloc4` |
| UnsignedShort | `w` (zero-ext canonical) | `uh` | `h` | `alloc4` |
| Char / SignedChar | `w` (sign-ext canonical) | `sb` | `b` | `alloc4` |
| UnsignedChar | `w` (zero-ext canonical) | `ub` | `b` | `alloc4` |
| Double | `d` | `d` | `d` | `alloc8` |
| Void (return only) | — | omitted | n/a | n/a |

**Invariant:** every `w` temp representing a narrower-than-32-bit C type is
kept fully sign/zero-extended at every point it's live (QBE has no sub-word
arithmetic — `addb`/`addh` don't exist). `SignExtend`/`ZeroExtend`/`Truncate`
re-derive the narrow representation only where TACKY already asks for it:

- `SignExtend` to `Long` → `extsw`; to `Int` from a narrower signed type →
  plain `copy` (already canonical).
- `ZeroExtend` to `UnsignedLong` → `extuw`; to `Unsigned` from narrower →
  plain `copy`.
- `Truncate` from `Long`/pointer to `Int`/`Unsigned` → `copy` (QBE truncates
  `l`→`w` context automatically — verify on first test). Truncate to
  `Short`/`Char` family → `extsb`/`extub`/`extsh`/`extuh` to re-canonicalize.

`DoubleToInt`/`DoubleToUInt`/`IntToDouble`/`UIntToDouble` map onto QBE's
`dtosi`/`dtoui`/`swtof`/`uwtof`/`sltof`/`ultof` family (verify exact mnemonic
spelling against the installed `qbe` binary — possible version drift). This
is notably simpler than `backend_x64`, which hand-codes out-of-range branches
for unsigned⟷double (`translate_ir.rs:680-832`) — QBE should handle that
internally.

## Instruction lowering (TACKY → QBE IL)

- Arithmetic/bitwise: `add/sub/mul/and/or/xor` directly; `Complement` → `xor
  %s, -1` (no native not); `Negate` → `neg` (verify it accepts `d`, else
  fallback `sub d_0, %s`).
- `Divide`/`Remainder`: `div`/`udiv`/`rem`/`urem` by signedness, or `div` for
  double.
- Shifts: `shl`, `sar` (signed), `shr` (unsigned).
- Comparisons: `ceq{w,l}/cne/cslt/csle/csgt/csge` (signed),
  `cult/cule/cugt/cuge` (unsigned int), `ceqd/cned/cltd/...` (double) →
  produce a `w` boolean.
- `Copy` → `copy`.
- `GetAddress(Var(x))` → `copy %x.addr` (local, address-taken — see below) or
  `copy $x` (global/static).
- `Load`/`Store` → width/sign-selected `loadw/loadl/loadd/loadsb/loadub/
  loadsh/loaduh` and `storew/storel/stored/storeb/storeh` (store width comes
  from the *source* value's type).
- `Jump(L)` → `jmp @L`. `JumpIfZero`/`JumpIfNotZero` on int/pointer → `jnz
  %cond, @a, @b` directly (no comparison needed — `jnz` tests "≠0" natively);
  on double, materialize `ceqd %cond, d_0` first, then `jnz`.
- `Label` → closes current block, opens `@name:`. Blocks that merely fall
  into the next block need no synthesized `jmp` (QBE auto-inserts fallthrough
  when a block's terminator is omitted and it's followed immediately by the
  next block in file order) — but `jnz` still always names both targets
  explicitly.
- `Return(None)` → `ret`; `Return(Some(v))` → `ret %v`.
- `FunCall(name, args, dest)` → `[%dest =T] call $name(<abitype> %arg, ...)`.
- `AddPtr` / `CopyToOffset` / `CopyFromOffset` → **out of scope, `todo!()`**.

## Function/program structure

- Each function gets a synthesized empty `@start` entry block (entry blocks
  can never be jump targets in QBE; a `goto` back to the function's first
  real label would otherwise violate that). `@start` holds `alloc4`/`alloc8`
  for every local whose address is taken (`GetAddress(Var(x), _)` where `x`
  is not static) — write a small dedicated scan for this; don't reuse
  `optimizer`'s existing address-taken analysis, which conflates a different
  concept.
- Split `Function.body` into blocks the same way TACKY's optimizer CFG
  builder already does (`src/optimizer/control_flow.rs`: new block at each
  `Label`, and after each terminator) — model the QBE splitter on that same
  logic without pulling in its `Entry`/`Exit`/annotation machinery, which is
  more apparatus than a single linear pass needs. Two-pass: assign every
  block a label first, then emit (so `jnz`/`jmp` fallthrough targets are
  known).
- Every function body already ends in a defensive `Return`
  (`src/tacky/emit.rs:987-995`), so the last block always has a real
  terminator — no synthesized trailing `ret` needed.
- `StaticVariable`/`StaticConstant` → `[export] data $name = align <n> {
  ... }`; all-zero init → `z <n>`. **Recommended phase-1 inclusion**: emit
  string-literal `StaticConstant`s (`CType::Array(Char,_)` +
  `StaticInit::StringInit`) as `data` with each byte as a numeric `b`
  constant (side-steps unclear QBE string-escaping rules) — this is
  distinct from general array/struct value semantics and is needed for any
  `puts`/`printf`-based smoke test. Adjust scope if you'd rather exclude it.

## Naming

`$name` for real C symbols (functions, globals, string constants) — used
verbatim, must match what `gcc`'s link step expects. `%name` for locals
(TACKY's own `Identifier`, already unique per scope); address-taken slots get
a `%name.addr`-style suffix. `@name` for `CodeLabel`s. Sanitize every
backend-internal name (`[A-Za-z0-9_]` only, replacing e.g. TACKY's `.` in
synthesized names with `_`) since QBE's identifier lexical rules aren't
documented — real linker-visible C symbols are never sanitized.

## Calling convention

No analog of `backend_table.rs` needed. The backend states each param/return's
QBE ABI type (per the table above); QBE handles physical register
assignment, stack spilling/alignment, and sub-word extension per its own
`-t <target>` ABI knowledge. No eightbyte classification since phase 1 has no
by-value structs.

## Driver / CLI plumbing

This matches what's already committed on the `qbe` branch (see "Pre-existing
work" above; `src/common/backend.rs` now exists and the branch builds — no
fix needed there anymore), completing the parts left unfinished on it.

- **`src/common/backend.rs`** (already present):
  ```rust
  pub(crate) trait Backend {
      fn emit(
          self,
          code: crate::tacky::ir::Program,
          symbols: crate::common::symbol_table::SymbolTable,
          types: &crate::common::type_table::TypeTable,
      ) -> Result<String, crate::errors::CompilerError>;
  }
  ```
  Note `emit` takes `self` by value, not `&self` or `self: Box<Self>` — this
  makes the trait **not** object-safe as `dyn Backend`, which is fine and
  intentional: it's a shared *shape* both backends implement (constructor +
  one-shot consuming `emit`), not a vtable to dispatch through. Selecting
  between them at the call site is still a plain, monomorphic `match` (no
  `dyn`/`Box` ceremony) — the trait just makes the parallel structure between
  `X64Backend` and the new `QbeBackend` explicit and enforced by the
  compiler, which the plain-`match`-over-free-functions design this plan
  originally sketched didn't give you.
- **`QbeBackend`**, in `src/backend_qbe/mod.rs`, mirroring `X64Backend`
  exactly: `pub struct QbeBackend { debug: bool, stage: Stage, source_name:
  String }`, `QbeBackend::new(debug, stage, source_name)`, and `impl Backend
  for QbeBackend` whose `emit` does: `translate_ir::translate` (TACKY →
  `qbe_ast`) → `emit_qbe::emit` (→ QBE IL text) → write to
  `swap_suffix(&self.source_name, ".c", ".ssa")` (dump via
  `write_debug_text_file` first if `self.debug`) → shell out to `qbe` (below)
  → if `self.stage == Stage::Codegen`, `process::exit(0)` before shelling out
  (matching `X64Backend`'s `Stage::Codegen` early-exit, which happens before
  its final emission step) → return the `.s` path.
- **New `BackendKind` enum + CLI flag.** `compiler.rs::compile()`'s signature
  gains a `backend: BackendKind` parameter (`pub enum BackendKind { X64, Qbe
  }`, default `X64`). Its current unconditional tail —
  ```rust
  let backend = X64Backend::new(debug, stage, source_name.to_owned());
  backend.emit(tacky_program, symbol_table, &type_table)
  ```
  — becomes:
  ```rust
  match backend {
      BackendKind::X64 => X64Backend::new(debug, stage, source_name.to_owned())
          .emit(tacky_program, symbol_table, &type_table),
      BackendKind::Qbe => QbeBackend::new(debug, stage, source_name.to_owned())
          .emit(tacky_program, symbol_table, &type_table),
  }
  ```
  New `--backend {x64,qbe}` CLI flag in `main.rs` (clap `ValueEnum`, default
  `x64`), threaded into the `compile(...)` call. Both arms return a `.s` path
  so `main.rs`'s `assemble_source`/`compile_source` need **no changes**.
- **`qbe` shell-out**, in `backend_qbe/mod.rs`, mirroring `preprocess_source`'s
  style (`process::Command::new("qbe").arg("-o").arg(asm_path).arg(ssa_path)`
  — **verified: `-o <file>` must come before the positional input file, or
  `qbe` errors with "cannot open '-o'"** — no `-t` needed, host
  auto-detection defaults to `arm64_apple` on this machine and produces
  correctly Mach-O-mangled (`_`-prefixed) symbols with no extra work; check
  `output.status.success()`, print stderr and exit on failure). Use
  `common::swap_suffix(&ssa_name, ".ssa", ".s")` for the output path, same
  helper `X64Backend`/`main.rs` already use.

## Testing

Book's official conformance suite is checked out at
`../writing-a-c-compiler-tests` (sibling of this repo). Its `test_compiler`
driver forwards trailing args straight through to the compiler under test, so
once `--backend qbe` exists:

```
../writing-a-c-compiler-tests/test_compiler /path/to/dcc --chapter <N> -- --backend qbe
```

(the `--` is required so its own argparse doesn't choke on `--backend`).
Use this as the primary correctness check per chapter as phase-1 features
come online (chapters roughly map to phase-1 scope through structs; stop
before the structs chapter).

In-repo, add a `tests/qbe_backend.rs` integration test (new `tests/` dir —
none exists yet) with small fixture `.c` files under `tests/fixtures/qbe/`,
each compiled via `dcc::compile(..., BackendKind::Qbe)`, then `gcc`-assembled
and run, asserting on exit code/stdout. `qbe` is now installed
(`/opt/homebrew/bin/qbe`, default target `arm64_apple` on this machine — see
"Verified against the installed `qbe`" below), but the skip-gracefully-if-
`qbe`-is-missing behavior is still worth keeping for other machines/CI.
Consider making `main.rs`'s `assemble_source`/`compile_source` `pub` and
reusing them from the test harness instead of duplicating the shell-out.

## Non-goals for Phase 1

- `AddPtr`, `CopyToOffset`, `CopyFromOffset` — left `todo!()`.
- `CType::Structure`, general `CType::Array` value semantics (indexing,
  pointer decay, local/stack arrays) — only the string-literal data-emission
  carve-out above is in scope.
- Struct-by-value params/returns, `returns_in_memory` handling.
- Variadic functions (moot — dcc's parser has no `...` support at all today).
- QBE features never exercised: `blit`, `vastart`/`vaarg`, user `type :name =
  {...}` aggregates, `env` params, thread-linkage, `phi` (unnecessary given
  non-SSA input).

## Verified against the installed `qbe` (2026-09-04, arm64_apple host target)

All five items below were confirmed end-to-end (`qbe` → `gcc` → run the
binary with edge-case values), not just checked for acceptance by `qbe`:

1. ✅ Int↔double conversion mnemonics are exactly `dtosi`/`dtoui`/`swtof`/
   `uwtof`/`sltof`/`ultof` as assumed.
2. ✅ A sub-word (`sb`/`ub`/`sh`) parameter is already canonical `w` inside
   the function body — no extra extension instruction needed. Verified with
   `test_sb(-5)`, `test_ub(200)`, `test_sh(-1000)` all round-tripping
   correctly through a real C caller.
3. ✅ `neg` accepts `d` directly, lowering to `fneg`.
4. ✅ Unmangled `$name` round-trips correctly: `qbe -o out.s` on this host
   (default target `arm64_apple`) emits `_`-prefixed Mach-O symbols itself —
   confirmed no `platform.rs`-equivalent is needed.
5. ✅ `l`→`w` truncation via plain `copy` correctly discards high bits at
   runtime (`0x100000005` truncated to `w` → `5`).

Two more things worth carrying into implementation, found while verifying:
- **`-o` must precede the positional input file** in the `qbe` invocation
  (`qbe -o out.s in.ssa`) — `qbe in.ssa -o out.s` fails with `cannot open
  '-o'`. See the corrected snippet above.
- **Always pass an explicit `align <n>`** on every `data` item — omitting it
  did not error but silently picked 8-byte alignment even for a single-byte
  item in testing. Use `CType::alignment(type_table)` (or `1` for raw string
  bytes) rather than relying on any default.
- `z <n>` zero-fill data is confirmed to land in `.bss`, and `jnz`/block-
  fallthrough elision behave exactly as documented (no manual comparison or
  extra `jmp` needed) — both previously flagged medium-confidence, now
  confirmed.

## Critical files

- `src/compiler.rs` — pipeline driver, new `Backend` enum + dispatch
- `src/tacky/ir.rs` — the IR this backend consumes (unchanged)
- `src/backend_x64/translate_ir.rs` — reference for existing instruction
  selection and double-conversion handling to compare against
- `src/common/ctype.rs`, `src/common/symbol_table.rs` — types/`StaticInit`
  this backend must lower
- `src/main.rs` — CLI flag, existing `assemble_source`/`compile_source` reused as-is

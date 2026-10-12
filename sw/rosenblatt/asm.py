"""Two-pass assembler for the Rosenblatt ISA.

Syntax summary (docs/isa.md has the full reference):

    ; comment          # comment          // comment
    label:  addi r1, r1, 4
            ldw  r2, [r1 + 8]           ; byte address = r1 + 8
            stw  [r1 - 4], r2
            s2r  r3, %gtid
            li   r4, 0x12345678         ; pseudo-op: becomes lui+ori when needed
    .equ    N, 64                       ; constants (may be redefined; evaluated in order)
    .alias  acc, r5                     ; register alias
    .rept   4, k                        ; repeat a block, binding k = 0..3
            ldbu r6, [r2 + k]
            mad  r(8 + k), r6, r7       ; computed register number
    .endr
    .if     N > 32                      ; conditional assembly
            ...
    .else
            ...
    .endif

Pseudo-instructions: nop, mov, li, bra, neg, not, relu.

Structured, divergence-safe control flow (expands into ssy / bz / bnz / sync):

    if.nz r1  /  if.z r1   ...   [else ...]   endif
    loop                   ...   endloop.nz r1  /  endloop.z r1      (do-while, per thread)

Several kernels can share the instruction memory: assemble each at its own
`base` address and combine them with `link()`.
"""

from __future__ import annotations

import ast
import operator
import re
import sys
from dataclasses import dataclass, field

from . import isa
from .isa import BY_MNEMONIC, Fn, Op, SR


class AsmError(Exception):
    def __init__(self, msg: str, lineno: int | None = None, text: str = ""):
        where = f"line {lineno}: " if lineno else ""
        super().__init__(f"{where}{msg}" + (f"\n    {text.strip()}" if text else ""))
        self.lineno = lineno


@dataclass
class Program:
    words: list[int]
    labels: dict[str, int]
    source_map: list[tuple[int, int, str]] = field(default_factory=list)  # (addr, lineno, text)
    base: int = 0

    def __len__(self) -> int:
        return len(self.words)

    @property
    def end(self) -> int:
        return self.base + len(self.words)

    def hex(self) -> str:
        return "".join(f"{w:08x}\n" for w in self.words)

    def listing(self) -> str:
        src: dict[int, str] = {}
        for addr, _lineno, text in self.source_map:
            src.setdefault(addr, text)
        inv: dict[int, str] = {}
        for name, addr in sorted(self.labels.items(), key=lambda kv: len(kv[0])):
            inv.setdefault(addr, name)
        out = []
        for i, w in enumerate(self.words):
            addr = self.base + i
            lab = f"{inv[addr]}:" if addr in inv else ""
            out.append(f"{addr:5d}  {w:08x}  {lab:<14} {isa.disassemble(w, inv):<30} ; "
                       f"{src.get(addr, '').strip()}")
        return "\n".join(out)


def link(programs: dict[str, Program]) -> Program:
    """Combine programs assembled at different bases into one image starting at 0.

    Each program's labels are exported as 'name.label', and 'name' itself is the
    program's base address (its entry point).
    """
    end = max((p.end for p in programs.values()), default=0)
    words = [0] * end
    owner = [""] * end
    labels: dict[str, int] = {}
    smap: list[tuple[int, int, str]] = []
    for name, p in programs.items():
        for i, w in enumerate(p.words):
            a = p.base + i
            if owner[a]:
                raise AsmError(f"'{name}' overlaps '{owner[a]}' at address {a}")
            owner[a] = name
            words[a] = w
        labels[name] = p.base
        labels.update({f"{name}.{k}": v for k, v in p.labels.items()})
        smap += p.source_map
    return Program(words, labels, smap, 0)


# ------------------------------------------------------------ expressions

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Div: operator.floordiv, ast.Mod: operator.mod,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
}
_UNOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert,
          ast.Not: lambda v: int(not v)}
_CMPOPS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
           ast.Eq: operator.eq, ast.NotEq: operator.ne}


class _Unresolved(Exception):
    pass


def evaluate(expr: str, symbols: dict[str, int]) -> int:
    """Evaluate an integer expression (+ - * / % << >> & | ^ ~, comparisons, and/or)."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"bad expression '{expr}'") from e

    def ev(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        if isinstance(node, ast.Name):
            if node.id not in symbols:
                raise _Unresolved(node.id)
            return symbols[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMPOPS:
            return int(_CMPOPS[type(node.ops[0])](ev(node.left), ev(node.comparators[0])))
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            return int(all(vals) if isinstance(node.op, ast.And) else any(vals))
        raise ValueError(f"unsupported expression '{expr}'")

    return ev(tree)


# ---------------------------------------------------------------- parsing


@dataclass
class Stmt:
    lineno: int
    text: str
    mnemonic: str            # lower-case; directives keep their leading dot
    operands: list[str]
    labels: list[str] = field(default_factory=list)


@dataclass
class Rept:
    stmt: Stmt
    body: list


@dataclass
class Cond:
    stmt: Stmt
    then: list
    other: list


_LABEL_RE = re.compile(r"^\s*([A-Za-z_.$][\w.$]*)\s*:(?!:)")
_REG_RE = re.compile(r"^r(\d+)$")
_REGEXPR_RE = re.compile(r"^r\((.+)\)$")
_MEM_RE = re.compile(r"^\[\s*([^\]]*?)\s*\]$")


def _strip_comment(line: str) -> str:
    for marker in (";", "#", "//"):
        pos = line.find(marker)
        if pos >= 0:
            line = line[:pos]
    return line.rstrip()


def _split_operands(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _parse(source: str) -> list[Stmt]:
    stmts: list[Stmt] = []
    pending: list[str] = []
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = _strip_comment(raw)
        while (m := _LABEL_RE.match(line)) is not None:
            pending.append(m.group(1))
            line = line[m.end():]
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        ops = _split_operands(parts[1]) if len(parts) > 1 else []
        stmts.append(Stmt(lineno, raw, parts[0].lower(), ops, pending))
        pending = []
    if pending:
        stmts.append(Stmt(0, "", ".label_only", [], pending))
    return stmts


def _group(stmts: list[Stmt]) -> list:
    """Nest .rept/.endr and .if/.else/.endif into Rept and Cond nodes."""
    root: list = []
    stack: list[tuple[object, list]] = [(None, root)]
    for s in stmts:
        m = s.mnemonic
        if m == ".rept":
            if not s.operands:
                raise AsmError(".rept needs a count", s.lineno, s.text)
            node = Rept(s, [])
            stack[-1][1].append(node)
            stack.append((node, node.body))
        elif m == ".endr":
            if not isinstance(stack[-1][0], Rept):
                raise AsmError(".endr without .rept", s.lineno, s.text)
            stack.pop()
        elif m == ".if":
            if len(s.operands) != 1:
                raise AsmError(".if takes one expression", s.lineno, s.text)
            node = Cond(s, [], [])
            stack[-1][1].append(node)
            stack.append((node, node.then))
        elif m == ".else":
            node = stack[-1][0]
            if not isinstance(node, Cond) or stack[-1][1] is node.other:
                raise AsmError(".else without .if", s.lineno, s.text)
            stack[-1] = (node, node.other)
        elif m == ".endif":
            if not isinstance(stack[-1][0], Cond):
                raise AsmError(".endif without .if", s.lineno, s.text)
            stack.pop()
        else:
            stack[-1][1].append(s)
    if len(stack) > 1:
        node = stack[-1][0]
        raise AsmError(f"unterminated {node.stmt.mnemonic}", node.stmt.lineno, node.stmt.text)
    return root


class _Macros:
    """Expands if/else/endif and loop/endloop into ssy/branch/sync with fresh labels."""

    def __init__(self):
        self.n = 0

    def fresh(self, kind: str) -> str:
        self.n += 1
        return f"__{kind}{self.n}"

    def expand(self, seq: list, in_rept: bool = False) -> list:
        out: list = []
        stack: list[dict] = []

        def emit(s: Stmt, mnem: str, ops: list[str], labels: list[str] | None = None) -> None:
            out.append(Stmt(s.lineno, s.text, mnem, ops, labels or []))

        for s in seq:
            if isinstance(s, Rept):
                out.append(Rept(s.stmt, self.expand(s.body, in_rept=True)))
                continue
            if isinstance(s, Cond):
                out.append(Cond(s.stmt, self.expand(s.then, in_rept), self.expand(s.other, in_rept)))
                continue
            m = s.mnemonic
            if in_rept and (s.labels or m in ("if.nz", "if.z", "loop")):
                raise AsmError("labels, if and loop are not allowed inside .rept "
                               "(they would be defined once per repetition)", s.lineno, s.text)
            if m in ("if.nz", "if.z"):
                if len(s.operands) != 1:
                    raise AsmError(f"{m} takes one register", s.lineno, s.text)
                ctx = {"kind": "if", "join": self.fresh("join"), "else": self.fresh("else"),
                       "has_else": False, "stmt": s}
                stack.append(ctx)
                emit(s, "ssy", [ctx["join"]], s.labels)
                # if.nz r: threads with r == 0 skip the then-block
                emit(s, "bz" if m == "if.nz" else "bnz", [s.operands[0], ctx["else"]])
            elif m == "else":
                if not stack or stack[-1]["kind"] != "if" or stack[-1]["has_else"]:
                    raise AsmError("else without if", s.lineno, s.text)
                stack[-1]["has_else"] = True
                emit(s, "sync", [], s.labels)
                emit(s, ".label_only", [], [stack[-1]["else"]])
            elif m == "endif":
                if not stack or stack[-1]["kind"] != "if":
                    raise AsmError("endif without if", s.lineno, s.text)
                ctx = stack.pop()
                # every path into the join point passes through a sync
                emit(s, "sync", [], s.labels + ([] if ctx["has_else"] else [ctx["else"]]))
                emit(s, ".label_only", [], [ctx["join"]])
            elif m == "loop":
                ctx = {"kind": "loop", "top": self.fresh("top"), "end": self.fresh("end"), "stmt": s}
                stack.append(ctx)
                emit(s, "ssy", [ctx["end"]], s.labels)
                emit(s, ".label_only", [], [ctx["top"]])
            elif m in ("endloop.nz", "endloop.z"):
                if not stack or stack[-1]["kind"] != "loop":
                    raise AsmError(f"{m} without loop", s.lineno, s.text)
                if len(s.operands) != 1:
                    raise AsmError(f"{m} takes one register", s.lineno, s.text)
                ctx = stack.pop()
                emit(s, "bnz" if m == "endloop.nz" else "bz", [s.operands[0], ctx["top"]], s.labels)
                emit(s, "sync", [])
                emit(s, ".label_only", [], [ctx["end"]])
            else:
                out.append(s)
        if stack:
            st = stack[-1]["stmt"]
            raise AsmError(f"unterminated {st.mnemonic}", st.lineno, st.text)
        return out


# --------------------------------------------------------------- assembler


class Assembler:
    def __init__(self, defines: dict[str, int] | None = None, base: int = 0):
        self.defines = dict(defines or {})
        self.base = base

    # -- operands --------------------------------------------------------
    def _reg(self, tok: str, s: Stmt) -> int:
        t = tok.strip().lower()
        t = self.aliases.get(t, t)
        if t == "zero":
            return 0
        m = _REG_RE.match(t)
        if m:
            n = int(m.group(1))
        elif (m := _REGEXPR_RE.match(t)) is not None:
            n = self._value(m.group(1), s, final=True)
        else:
            raise AsmError(f"expected a register, got '{tok}'", s.lineno, s.text)
        if not 0 <= n < isa.NUM_REGS:
            raise AsmError(f"r{n} does not exist (r0..r15)", s.lineno, s.text)
        return n

    def _value(self, expr: str, s: Stmt, final: bool) -> int | None:
        try:
            return evaluate(expr, self.symbols)
        except _Unresolved as e:
            if final:
                raise AsmError(f"undefined symbol '{e.args[0]}'", s.lineno, s.text) from None
            return None
        except (ValueError, ZeroDivisionError) as e:
            raise AsmError(str(e), s.lineno, s.text) from None

    def _mem(self, tok: str, s: Stmt) -> tuple[int, int]:
        m = _MEM_RE.match(tok.strip())
        if not m:
            raise AsmError(f"expected a memory operand like [r1 + 4], got '{tok}'", s.lineno, s.text)
        inner = m.group(1).strip()
        mm = re.match(r"^(r\([^)]*\)|[A-Za-z_]\w*)\s*(?:([+-])\s*(.+))?$", inner)
        if mm:
            name = self.aliases.get(mm.group(1).lower(), mm.group(1).lower())
            if name == "zero" or _REG_RE.match(name) or _REGEXPR_RE.match(name):
                base = self._reg(name, s)
                # "[r1 - 4 + 8]" means r1 + (-4 + 8): evaluate sign and rest together
                off = self._value(mm.group(2) + mm.group(3), s, True) if mm.group(2) else 0
                return base, off
        return 0, self._value(inner, s, True)

    # -- tree walking ----------------------------------------------------
    def _walk(self, seq: list, visit) -> None:
        for node in seq:
            if isinstance(node, Rept):
                st = node.stmt
                for lab in st.labels:
                    visit(Stmt(st.lineno, st.text, ".label_only", [], [lab]))
                n = self._value(st.operands[0], st, final=True)
                var = st.operands[1].strip() if len(st.operands) > 1 else None
                for k in range(n):
                    if var:
                        self.symbols[var] = k
                    self._walk(node.body, visit)
            elif isinstance(node, Cond):
                st = node.stmt
                for lab in st.labels:
                    visit(Stmt(st.lineno, st.text, ".label_only", [], [lab]))
                self._walk(node.then if self._value(st.operands[0], st, True) else node.other, visit)
            else:
                visit(node)

    def _directive(self, s: Stmt) -> bool:
        m = s.mnemonic
        if m in (".equ", ".set"):
            if len(s.operands) != 2:
                raise AsmError(".equ NAME, value", s.lineno, s.text)
            self.symbols[s.operands[0].strip()] = self._value(s.operands[1], s, final=True)
            return True
        if m == ".alias":
            if len(s.operands) != 2:
                raise AsmError(".alias name, rN", s.lineno, s.text)
            self.aliases[s.operands[0].strip().lower()] = "r%d" % self._reg(s.operands[1], s)
            return True
        return m in (".label_only", ".global", ".text", ".kernel")

    def _size(self, s: Stmt) -> int:
        if s.mnemonic == ".word":
            return len(s.operands)
        if s.mnemonic != "li":
            return 1
        if len(s.operands) != 2:
            raise AsmError("li takes a register and a value", s.lineno, s.text)
        v = self._value(s.operands[1], s, final=False)
        if v is None:  # forward reference: reserve the worst case
            return 2
        v &= isa.MASK32
        # one word for a 22-bit signed value (li) or a value with zero low bits (lui)
        return 1 if isa.fits_signed(isa.sext(v, 32), isa.IMM22_BITS) or v & 0x3FF == 0 else 2

    def assemble(self, source: str) -> Program:
        tree = _Macros().expand(_group(_parse(source)))

        # pass 1: label addresses
        self.symbols, self.aliases = dict(self.defines), {}
        labels: dict[str, int] = {}
        sizes: list[int] = []
        addr = [self.base]

        def pass1(s: Stmt) -> None:
            for lab in s.labels:
                if lab in labels:
                    raise AsmError(f"label '{lab}' defined twice", s.lineno, s.text)
                labels[lab] = self.symbols[lab] = addr[0]
            if self._directive(s):
                return
            n = self._size(s)
            sizes.append(n)
            addr[0] += n

        self._walk(tree, pass1)

        # pass 2: encode
        self.symbols, self.aliases = dict(self.defines), {}
        self.symbols.update(labels)
        words: list[int] = []
        smap: list[tuple[int, int, str]] = []
        idx = [0]

        def pass2(s: Stmt) -> None:
            if self._directive(s):
                return
            n = sizes[idx[0]]
            idx[0] += 1
            start = len(words)
            if s.mnemonic == ".word":
                words.extend(self._value(op, s, True) & isa.MASK32 for op in s.operands)
            else:
                words.extend(self._encode(s, n))
            smap.extend((self.base + a, s.lineno, s.text) for a in range(start, len(words)))

        self._walk(tree, pass2)
        return Program(words, labels, smap, self.base)

    # -- encoding --------------------------------------------------------
    def _encode(self, s: Stmt, size: int) -> list[int]:
        m, ops = s.mnemonic, s.operands
        val = lambda e: self._value(e, s, final=True)  # noqa: E731
        reg = lambda t: self._reg(t, s)  # noqa: E731

        def need(n: int) -> None:
            if len(ops) != n:
                raise AsmError(f"'{m}' expects {n} operand(s), got {len(ops)}", s.lineno, s.text)

        def enc(*a, **k) -> int:
            try:
                return isa.encode(*a, **k)
            except ValueError as e:
                raise AsmError(str(e), s.lineno, s.text) from None

        # ---- pseudo-instructions
        if m == "nop":
            need(0)
            return [0]
        if m == "mov":
            need(2)
            return [enc(Fn.ADD, reg(ops[0]), reg(ops[1]), 0)]
        if m == "li":
            need(2)
            rd, v = reg(ops[0]), val(ops[1])
            if not -(1 << 31) <= v < (1 << 32):
                raise AsmError(f"value {v} does not fit in 32 bits", s.lineno, s.text)
            v &= isa.MASK32
            sv = isa.sext(v, 32)
            if isa.fits_signed(sv, isa.IMM22_BITS):
                out = [enc(Op.LI, rd, imm=sv)]
            else:
                out = [enc(Op.LUI, rd, imm=isa.sext(v >> 10, 22))]
                if v & 0x3FF:
                    out.append(enc(isa.ALU_I_BASE | Fn.OR, rd, rd, imm=v & 0x3FF))
            if len(out) > size:
                raise AsmError("li needs two words here; define its value before use", s.lineno, s.text)
            return out + [0] * (size - len(out))
        if m == "bra":
            need(1)
            return [enc(Op.BZ, 0, 0, imm=val(ops[0]))]
        if m == "neg":
            need(2)
            return [enc(Fn.SUB, reg(ops[0]), 0, reg(ops[1]))]
        if m == "not":
            need(2)
            return [enc(isa.ALU_I_BASE | Fn.XOR, reg(ops[0]), reg(ops[1]), imm=-1)]
        if m == "relu":
            need(2)
            return [enc(Fn.MAX, reg(ops[0]), reg(ops[1]), 0)]

        spec = BY_MNEMONIC.get(m)
        if spec is None:
            raise AsmError(f"unknown instruction '{m}'", s.lineno, s.text)
        f, op = spec.fmt, spec.opcode
        if f == "rrr":
            need(3)
            return [enc(op, reg(ops[0]), reg(ops[1]), reg(ops[2]))]
        if f == "rri":
            need(3)
            return [enc(op, reg(ops[0]), reg(ops[1]), imm=val(ops[2]))]
        if f == "rrrl":
            need(4)
            return [enc(op, reg(ops[0]), reg(ops[1]), reg(ops[2]), lane=val(ops[3]))]
        if f == "ld":
            need(2)
            base, off = self._mem(ops[1], s)
            return [enc(op, reg(ops[0]), base, imm=off)]
        if f == "st":
            need(2)
            base, off = self._mem(ops[0], s)
            return [enc(op, reg(ops[1]), base, imm=off)]
        if f == "ri":
            need(2)
            return [enc(op, reg(ops[0]), imm=val(ops[1]))]
        if f == "sr":
            need(2)
            name = ops[1].strip().lower()
            if not name.startswith("%") or name[1:].upper() not in SR.__members__:
                raise AsmError(f"unknown special register '{ops[1]}'", s.lineno, s.text)
            return [enc(op, reg(ops[0]), imm=int(SR[name[1:].upper()]))]
        if f == "br":
            need(2)
            return [enc(op, 0, reg(ops[0]), imm=val(ops[1]))]
        if f == "tgt":
            need(1)
            return [enc(op, imm=val(ops[0]))]
        need(0)
        return [enc(op)]


def assemble(source: str, defines: dict[str, int] | None = None, base: int = 0) -> Program:
    return Assembler(defines, base).assemble(source)


def assemble_file(path: str, defines: dict[str, int] | None = None, base: int = 0) -> Program:
    with open(path, encoding="utf-8") as f:
        return assemble(f.read(), defines, base)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Rosenblatt assembler")
    p.add_argument("source")
    p.add_argument("-o", "--output", help="hex output (one 32-bit word per line)")
    p.add_argument("-l", "--listing", action="store_true", help="print a listing")
    p.add_argument("-D", action="append", default=[], metavar="NAME=VALUE")
    a = p.parse_args(argv)
    defines = {}
    for d in a.D:
        k, _, v = d.partition("=")
        defines[k] = int(v, 0)
    try:
        prog = assemble_file(a.source, defines)
    except AsmError as e:
        print(f"{a.source}: {e}", file=sys.stderr)
        return 1
    if a.output:
        with open(a.output, "w", newline="\n") as f:
            f.write(prog.hex())
    if a.listing or not a.output:
        print(prog.listing())
    return 0


if __name__ == "__main__":
    sys.exit(main())

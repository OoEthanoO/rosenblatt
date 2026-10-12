"""Assembler and ISA encoding tests (no simulator needed)."""

import pytest

from rosenblatt import isa
from rosenblatt.asm import AsmError, assemble, link
from rosenblatt.device import Device, EmuBackend
from rosenblatt.isa import Fn, Op, SR


def words(src, **defines):
    return assemble(src, defines=defines).words


def test_encodings_roundtrip_through_the_disassembler():
    src = """
        add  r1, r2, r3
        subi r4, r5, -7
        mad  r6, r7, r8
        ldw  r9, [r10 + 12]
        stb  [r11 - 4], r12
        lds  r1, [r0 + 64]
        s2r  r13, %gtid
        lui  r14, -5
    """
    got = [isa.disassemble(w) for w in words(src)]
    assert got == ["add r1, r2, r3", "subi r4, r5, -7", "mad r6, r7, r8", "ldw r9, [r10 + 12]",
                   "stb [r11 - 4], r12", "lds r1, [r0 + 64]", "s2r r13, %gtid", "lui r14, -5"]


def test_field_layout():
    w = words("addi r3, r5, -2")[0]
    d = isa.decode(w)
    assert (d.opcode, d.rd, d.rs1, d.imm18) == (isa.ALU_I_BASE | Fn.ADD, 3, 5, -2)
    assert words("nop") == [0]          # add r0, r0, r0
    assert isa.decode(words("ret")[0]).opcode == Op.RET


@pytest.mark.parametrize("value,count", [(0, 1), (-1, 1), ((1 << 21) - 1, 1), (1 << 21, 1),
                                          (0x12345678, 2), (0xFFFF_FC00, 1), (-(1 << 31), 1)])
def test_li_uses_one_or_two_words_and_is_exact(value, count):
    prog = assemble(f"li r1, {value}\ns2r r2, %arg0\nstw [r2], r1\nret")
    li_words = len(prog.words) - 3
    assert li_words == count
    dev = Device(EmuBackend())
    out = dev.alloc(4)
    dev.load(prog)
    dev.launch(1, 1, [out])
    assert int(dev.download(out, 1)[0]) & isa.MASK32 == value & isa.MASK32


def test_forward_reference_li_keeps_addresses_stable():
    prog = assemble("li r1, target\nnop\ntarget: ret")
    # li to a forward label reserves two words, padded with a nop when one suffices
    assert prog.labels["target"] == 3


def test_rept_if_and_computed_registers():
    src = """
    .equ N, 3
    .rept N, k
        addi r(1 + k), r0, k * 10
    .endr
    .if N > 2
        mov r5, r3
    .else
        mov r5, r1
    .endif
    """
    got = [isa.disassemble(w) for w in words(src)]
    assert got == ["addi r1, r0, 0", "addi r2, r0, 10", "addi r3, r0, 20", "add r5, r3, r0"]


def test_defines_and_aliases():
    got = [isa.disassemble(w) for w in words(".alias acc, r7\naddi acc, acc, STEP", STEP=4)]
    assert got == ["addi r7, r7, 4"]


def test_structured_if_else_and_loop_expand_to_ssy_sync():
    prog = assemble("""
        if.nz r1
            addi r2, r2, 1
        else
            addi r2, r2, 2
        endif
        loop
            subi r3, r3, 1
        endloop.nz r3
        ret
    """)
    ops = [isa.decode(w).opcode for w in prog.words]
    assert ops.count(Op.SSY) == 2 and ops.count(Op.SYNC) == 3
    assert Op.BZ in ops and Op.BNZ in ops


def test_link_places_programs_and_exports_entries():
    a = assemble("top: bra top", base=0)
    b = assemble("start: ret", base=a.end)
    prog = link({"a": a, "b": b})
    assert prog.labels["b"] == 1 and prog.labels["b.start"] == 1 and prog.labels["a.top"] == 0
    with pytest.raises(AsmError, match="overlaps"):
        link({"a": a, "c": assemble("ret", base=0)})


@pytest.mark.parametrize("src,message", [
    ("add r1, r2", "expects 3"),
    ("addi r1, r2, 200000", "imm18 out of range"),
    ("ldw r1, r2", "memory operand"),
    ("foo r1", "unknown instruction"),
    ("s2r r1, %nope", "special register"),
    ("bra nowhere", "undefined symbol"),
    ("x: nop\nx: nop", "defined twice"),
    ("if.nz r1\nnop", "unterminated"),
    ("endif", "without if"),
    (".rept 2\nx: nop\n.endr", "not allowed inside .rept"),
    ("addi r(16), r0, 1", "r16 does not exist"),
])
def test_errors_point_at_the_problem(src, message):
    with pytest.raises(AsmError, match=message):
        assemble(src)


def test_special_register_numbers_match_the_hardware_header():
    header = isa.verilog_header()
    for sr in SR:
        assert f"`define RB_SR_{sr.name:<6} 5'd{int(sr)}" in header

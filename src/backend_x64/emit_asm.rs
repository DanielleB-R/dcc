use std::{collections::HashMap, sync::LazyLock};

use super::asm_ast::*;
use super::backend_table::BackendTable;
use super::platform::{emit_label, emit_local_label};
use crate::common::Identifier;
use crate::common::char_escape::escape;
use crate::common::symbol_table::StaticInit;

#[cfg(target_os = "linux")]
const LINUX_NX_STACK: &str = "\t.section .note.GNU-stack,\"\",@progbits";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bits {
    Eight,
    Sixteen,
    ThirtyTwo,
    SixtyFour,
}

impl From<AssemblyType> for Bits {
    fn from(value: AssemblyType) -> Self {
        match value {
            AssemblyType::Byte => Bits::Eight,
            AssemblyType::Word => Bits::Sixteen,
            AssemblyType::Longword => Bits::ThirtyTwo,
            AssemblyType::Quadword => Bits::SixtyFour,
            AssemblyType::DoublePrecision => Bits::SixtyFour,
            AssemblyType::ByteArray(..) => panic!("no registers for bytearray"),
        }
    }
}

static REGISTER_8_NAMES: LazyLock<HashMap<Register, &'static str>> = LazyLock::new(|| {
    [
        (Register::AX, "al"),
        (Register::BX, "bl"),
        (Register::CX, "cl"),
        (Register::DX, "dl"),
        (Register::DI, "dil"),
        (Register::SI, "sil"),
        (Register::R8, "r8b"),
        (Register::R9, "r9b"),
        (Register::R10, "r10b"),
        (Register::R11, "r11b"),
        (Register::R12, "r12b"),
        (Register::R13, "r13b"),
        (Register::R14, "r14b"),
        (Register::R15, "r15b"),
    ]
    .into()
});

static REGISTER_16_NAMES: LazyLock<HashMap<Register, &'static str>> = LazyLock::new(|| {
    [
        (Register::AX, "ax"),
        (Register::BX, "bx"),
        (Register::CX, "cx"),
        (Register::DX, "dx"),
        (Register::DI, "di"),
        (Register::SI, "si"),
        (Register::R8, "r8w"),
        (Register::R9, "r9w"),
        (Register::R10, "r10w"),
        (Register::R11, "r11w"),
        (Register::R12, "r12w"),
        (Register::R13, "r13w"),
        (Register::R14, "r14w"),
        (Register::R15, "r15w"),
    ]
    .into()
});

static REGISTER_32_NAMES: LazyLock<HashMap<Register, &'static str>> = LazyLock::new(|| {
    [
        (Register::AX, "eax"),
        (Register::BX, "ebx"),
        (Register::CX, "ecx"),
        (Register::DX, "edx"),
        (Register::DI, "edi"),
        (Register::SI, "esi"),
        (Register::R8, "r8d"),
        (Register::R9, "r9d"),
        (Register::R10, "r10d"),
        (Register::R11, "r11d"),
        (Register::R12, "r12d"),
        (Register::R13, "r13d"),
        (Register::R14, "r14d"),
        (Register::R15, "r15d"),
    ]
    .into()
});

static REGISTER_64_NAMES: LazyLock<HashMap<Register, &'static str>> = LazyLock::new(|| {
    [
        (Register::AX, "rax"),
        (Register::BX, "rbx"),
        (Register::CX, "rcx"),
        (Register::DX, "rdx"),
        (Register::DI, "rdi"),
        (Register::SI, "rsi"),
        (Register::R8, "r8"),
        (Register::R9, "r9"),
        (Register::R10, "r10"),
        (Register::R11, "r11"),
        (Register::R12, "r12"),
        (Register::R13, "r13"),
        (Register::R14, "r14"),
        (Register::R15, "r15"),
        (Register::SP, "rsp"),
        (Register::BP, "rbp"),
        (Register::XMM0, "xmm0"),
        (Register::XMM1, "xmm1"),
        (Register::XMM2, "xmm2"),
        (Register::XMM3, "xmm3"),
        (Register::XMM4, "xmm4"),
        (Register::XMM5, "xmm5"),
        (Register::XMM6, "xmm6"),
        (Register::XMM7, "xmm7"),
        (Register::XMM8, "xmm8"),
        (Register::XMM9, "xmm9"),
        (Register::XMM10, "xmm10"),
        (Register::XMM11, "xmm11"),
        (Register::XMM12, "xmm12"),
        (Register::XMM13, "xmm13"),
        (Register::XMM14, "xmm14"),
        (Register::XMM15, "xmm15"),
    ]
    .into()
});

fn register_name(reg: Register, bits: Bits) -> &'static str {
    (match bits {
        Bits::Eight => &REGISTER_8_NAMES,
        Bits::Sixteen => &REGISTER_16_NAMES,
        Bits::ThirtyTwo => &REGISTER_32_NAMES,
        Bits::SixtyFour => &REGISTER_64_NAMES,
    })
    .get(&reg)
    .unwrap()
}

fn emit_type_suffix(asm_type: AssemblyType) -> &'static str {
    match asm_type {
        AssemblyType::Byte => "b",
        AssemblyType::Word => "w",
        AssemblyType::Longword => "l",
        AssemblyType::Quadword => "q",
        AssemblyType::DoublePrecision => "sd",
        AssemblyType::ByteArray(_, _) => panic!("Instruction on ByteArray type"),
    }
}

fn emit_opcode(name: &str, asm_type: AssemblyType) -> String {
    format!("{}{}", name, emit_type_suffix(asm_type))
}

fn emit_data(name: Identifier, offset: usize) -> String {
    format!("{}+{}(%rip)", emit_label(name.value), offset)
}

fn emit_operand(code: Operand, bits: Bits) -> String {
    match code {
        Operand::Imm(n) => format!("${}", n),
        Operand::Reg(r) => format!("%{}", register_name(r, bits)),
        Operand::Pseudo(_) => panic!("pseudo register in final AST"),
        Operand::PseudoMem(_, _) => panic!("pseudo memory in final AST"),
        Operand::Memory(reg, offset) => {
            format!("{}(%{})", offset, register_name(reg, Bits::SixtyFour))
        }
        Operand::Data(name, offset) => emit_data(name, offset),
        Operand::Indexed(base, index, scale) => format!(
            "(%{}, %{}, {})",
            register_name(base, Bits::SixtyFour),
            register_name(index, Bits::SixtyFour),
            scale
        ),
        // _ => unimplemented!(),
    }
}

fn emit_unary(operator: UnaryOp, asm_type: AssemblyType) -> String {
    emit_opcode(
        match operator {
            UnaryOp::Neg => "neg",
            UnaryOp::Not => "not",
            UnaryOp::Shr => "shr",
        },
        asm_type,
    )
}

fn emit_binary(operator: BinOp, asm_type: AssemblyType) -> String {
    if operator == BinOp::Xor && asm_type == AssemblyType::DoublePrecision {
        return "xorpd".to_owned();
    }
    if operator == BinOp::Mult && asm_type == AssemblyType::DoublePrecision {
        return "mulsd".to_owned();
    }

    emit_opcode(
        match operator {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mult => "imul",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::Shr => "shr",
            BinOp::Sar => "sar",
            BinOp::DivDouble => "div",
        },
        asm_type,
    )
}

fn emit_condition_code(condition: ConditionCode) -> &'static str {
    match condition {
        ConditionCode::E => "e",
        ConditionCode::NE => "ne",
        ConditionCode::G => "g",
        ConditionCode::GE => "ge",
        ConditionCode::L => "l",
        ConditionCode::LE => "le",
        ConditionCode::A => "a",
        ConditionCode::AE => "ae",
        ConditionCode::B => "b",
        ConditionCode::BE => "be",
    }
}

#[cfg(target_os = "macos")]
fn emit_call(name: Identifier, _symbols: &BackendTable) -> String {
    format!("\tcall\t_{}", name.value)
}

#[cfg(target_os = "linux")]
fn emit_call(name: Identifier, symbols: &BackendTable) -> String {
    format!(
        "\tcall\t{}{}",
        name.value,
        match symbols.get(&name.value) {
            Some(_) => "",
            None => "@PLT",
        }
    )
}

fn emit_instruction(code: Inst, symbols: &BackendTable) -> String {
    use Bits::*;
    match code {
        Inst::Mov(asm_type, src, dest) => {
            format!(
                "\t{}\t{}, {}",
                emit_opcode("mov", asm_type),
                emit_operand(src, asm_type.into()),
                emit_operand(dest, asm_type.into())
            )
        }
        Inst::Movsx(src_type, dest_type, src, dest) => format!(
            "\tmovs{}{}\t{}, {}",
            emit_type_suffix(src_type),
            emit_type_suffix(dest_type),
            emit_operand(src, src_type.into()),
            emit_operand(dest, dest_type.into())
        ),
        Inst::MovZeroExtend(AssemblyType::Byte, dest_type, src, dest) => format!(
            "\t{}\t{}, {}",
            emit_opcode("movzb", dest_type),
            emit_operand(src, Eight),
            emit_operand(dest, dest_type.into()),
        ),
        Inst::MovZeroExtend(_, _, _, _) => unreachable!(),
        Inst::Lea(src, dest) => format!(
            "\tleaq\t{}, {}",
            emit_operand(src, SixtyFour),
            emit_operand(dest, SixtyFour)
        ),
        Inst::Cvttsd2si(dst_type, src, dest) => {
            format!(
                "\t{}\t{}, {}",
                emit_opcode("cvttsd2si", dst_type),
                emit_operand(src, SixtyFour),
                emit_operand(dest, dst_type.into()),
            )
        }
        Inst::Cvtsi2sd(src_type, src, dest) => {
            format!(
                "\t{}\t{}, {}",
                emit_opcode("cvtsi2sd", src_type),
                emit_operand(src, src_type.into()),
                emit_operand(dest, SixtyFour),
            )
        }
        Inst::Ret => "\tmovq\t%rbp, %rsp\n\tpopq\t%rbp\n\tret".to_owned(),
        Inst::Unary(asm_type, operator, operand) => {
            format!(
                "\t{}\t{}",
                emit_unary(operator, asm_type),
                emit_operand(operand, asm_type.into())
            )
        }
        Inst::Binary(
            asm_type,
            operator @ (BinOp::Shl | BinOp::Shr | BinOp::Sar),
            operand1,
            operand2,
        ) => {
            format!(
                "\t{}\t{}, {}",
                emit_binary(operator, asm_type),
                emit_operand(operand1, Eight),
                emit_operand(operand2, asm_type.into()),
            )
        }
        Inst::Binary(asm_type, operator, operand1, operand2, ..) => {
            format!(
                "\t{}\t{}, {}",
                emit_binary(operator, asm_type),
                emit_operand(operand1, asm_type.into()),
                emit_operand(operand2, asm_type.into())
            )
        }
        Inst::Cmp(AssemblyType::DoublePrecision, op1, op2) => format!(
            "\t{}\t{}, {}",
            emit_opcode("comi", AssemblyType::DoublePrecision),
            emit_operand(op1, SixtyFour),
            emit_operand(op2, SixtyFour)
        ),
        Inst::Cmp(asm_type, op1, op2) => format!(
            "\t{}\t{}, {}",
            emit_opcode("cmp", asm_type),
            emit_operand(op1, asm_type.into()),
            emit_operand(op2, asm_type.into())
        ),
        Inst::Idiv(asm_type, operand) => format!(
            "\t{}\t{}",
            emit_opcode("idiv", asm_type),
            emit_operand(operand, asm_type.into())
        ),
        Inst::Div(asm_type, operand) => format!(
            "\t{}\t{}",
            emit_opcode("div", asm_type),
            emit_operand(operand, asm_type.into())
        ),
        Inst::Cdq(AssemblyType::Longword) => "\tcdq".to_owned(),
        Inst::Cdq(AssemblyType::Quadword) => "\tcqo".to_owned(),
        Inst::Cdq(AssemblyType::DoublePrecision) => unreachable!(),
        Inst::Jmp(label) => format!("\tjmp\t{}", emit_local_label(label)),
        Inst::JmpCC(condition, label) => {
            format!(
                "\tj{}\t{}",
                emit_condition_code(condition),
                emit_local_label(label)
            )
        }
        Inst::SetCC(condition, operand) => format!(
            "\tset{}\t{}",
            emit_condition_code(condition),
            emit_operand(operand, Eight)
        ),
        Inst::Label(label) => format!("{}:", emit_local_label(label)),
        Inst::Push(operand) => format!("\tpushq\t{}", emit_operand(operand, SixtyFour)),
        Inst::Pop(reg) => format!("\tpopq\t%{}", register_name(reg, SixtyFour)),
        Inst::Call(name) => emit_call(name, symbols),
        _ => unimplemented!(),
    }
}

fn emit_function(code: Function, symbols: &BackendTable) -> Vec<String> {
    let mut lines = vec![];

    if code.global {
        lines.push(format!("\t.globl {}", emit_label(code.name.value)));
    }
    lines.push("\t.text".to_owned());

    lines.push(format!("{}:", emit_label(code.name.value)));

    lines.push("\tpushq\t%rbp".to_owned());
    lines.push("\tmovq\t%rsp, %rbp".to_owned());

    lines.extend(
        code.instructions
            .into_iter()
            .map(|instruction| emit_instruction(instruction, symbols)),
    );

    lines.push("".to_owned());

    lines
}

fn emit_align_directive(alignment: usize) -> String {
    format!("\t.balign {}", alignment)
}

fn emit_static_var(code: StaticVariable) -> Vec<String> {
    let mut lines = vec![];
    let use_bss = code.init_list.iter().all(|i| i.is_zero());

    if code.global {
        lines.push(format!("\t.globl {}", emit_label(code.identifier.value)));
    }
    if use_bss {
        lines.push("\t.bss".to_owned());
    } else {
        lines.push("\t.data".to_owned());
    }
    lines.push(emit_align_directive(code.alignment));
    lines.push(format!("{}:", emit_label(code.identifier.value)));
    if use_bss {
        lines.push(format!(
            "\t.zero {}",
            code.init_list.iter().map(|i| i.size()).sum::<usize>()
        ));
    } else {
        for init in code.init_list {
            lines.push(match init {
                StaticInit::IntInit(n) => format!("\t.long {}", n),
                StaticInit::LongInit(n) => format!("\t.quad {}", n),
                StaticInit::UIntInit(n) => format!("\t.long {}", n),
                StaticInit::ULongInit(n) => format!("\t.quad {}", n),
                StaticInit::DoubleInit(f) => format!("\t.quad {}", f.to_bits()),
                StaticInit::ZeroInit(n) => format!("\t.zero {}", n),
                StaticInit::CharInit(n) => format!("\t.byte {}", n),
                StaticInit::UCharInit(n) => format!("\t.byte {}", n),
                StaticInit::StringInit(s, true) => {
                    format!("\t.asciz \"{}\"", escape(&s))
                }
                StaticInit::StringInit(s, false) => {
                    format!("\t.ascii \"{}\"", escape(&s))
                }
                StaticInit::PointerInit(name) => format!("\t.quad {}", emit_label(name)),
                _ => unimplemented!(),
            })
        }
    }

    lines
}

#[cfg(target_os = "macos")]
fn emit_static_const(code: StaticConstant) -> Vec<String> {
    let mut result = vec![];

    result.push(format!("\t.literal{}", code.alignment));
    result.push(emit_align_directive(code.alignment));
    result.push(format!("_{}:", code.name.value));
    result.push(match code.init {
        StaticInit::DoubleInit(f) => format!("\t.quad {}", f.to_bits()),
        _ => unreachable!(),
    });

    if code.alignment == 16 {
        result.push("\t.quad 0".to_owned());
    }

    result
}

#[cfg(target_os = "linux")]
fn emit_static_const(code: StaticConstant) -> Vec<String> {
    vec![
        "\t.section .rodata".to_owned(),
        emit_align_directive(code.alignment),
        format!("{}:", code.name.value),
        match code.init {
            StaticInit::DoubleInit(f) => format!("\t.quad {}", f.to_bits()),
            StaticInit::StringInit(s, true) => {
                format!("\t.asciz \"{}\"", escape(&s))
            }
            StaticInit::StringInit(s, false) => {
                format!("\t.ascii \"{}\"", escape(&s))
            }
            _ => unreachable!(),
        },
    ]
}

fn emit_top_level(code: TopLevel, symbols: &BackendTable) -> Vec<String> {
    match code {
        TopLevel::Fn(f) => emit_function(f, symbols),
        TopLevel::Var(v) => emit_static_var(v),
        TopLevel::Const(c) => emit_static_const(c),
    }
}

pub fn emit_assembly(code: Program, symbols: &BackendTable) -> String {
    let mut fn_lines = code
        .top_level
        .into_iter()
        .flat_map(|tl| emit_top_level(tl, symbols))
        .collect::<Vec<_>>();

    #[cfg(target_os = "linux")]
    fn_lines.push(LINUX_NX_STACK.to_owned());

    fn_lines.push("".to_owned());
    fn_lines.join("\n")
}

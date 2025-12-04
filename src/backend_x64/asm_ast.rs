use std::collections::HashSet;

use derive_more::{Display, From, IsVariant, Unwrap};

use super::backend_table::BackendTable;
use crate::common::symbol_table::StaticInit;
use crate::common::type_table::TypeTable;
use crate::common::{CType, CodeLabel, Identifier, print_vec};
use crate::tacky::ir::{BinaryOp, UnaryOperator};

#[derive(Clone, Debug, Display, From)]
#[display("{}", print_vec(top_level, "\n\n"))]
pub struct Program {
    pub top_level: Vec<TopLevel>,
}

impl Program {
    pub fn map_fn(self, mut transform: impl FnMut(Function) -> Function) -> Self {
        Self {
            top_level: self
                .top_level
                .into_iter()
                .map(|tl| match tl {
                    TopLevel::Fn(f) => transform(f).into(),
                    tl => tl,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Display, From, IsVariant)]
pub enum TopLevel {
    Fn(Function),
    Var(StaticVariable),
    Const(StaticConstant),
}

#[derive(Clone, Debug, Display)]
#[display("{name}:\n{}\n", print_vec(instructions, "\n"))]
pub struct Function {
    pub name: Identifier,
    pub global: bool,
    pub stack_offset: isize,
    pub instructions: Vec<Inst>,
    pub callee_saved_regs: Vec<Register>,
    pub aliased_vars: HashSet<Identifier>,
}

#[derive(Clone, Debug, Display)]
#[display("{identifier}: {} align {alignment} {}", print_vec(init_list, ", "), match self.global {
                true => "global",
                false => "local",
            })]
pub struct StaticVariable {
    pub identifier: Identifier,
    pub global: bool,
    pub alignment: usize,
    pub init_list: Vec<StaticInit>,
}

#[derive(Clone, Debug, Display)]
#[display("{name}: {init} align {alignment}")]
pub struct StaticConstant {
    pub name: Identifier,
    pub alignment: usize,
    pub init: StaticInit,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Display)]
pub enum Inst {
    #[display("\tmov\t{_1}, {_2}\t{_0}")]
    Mov(AssemblyType, Operand, Operand),
    #[display("\tmovsx\t{_2}, {_3}\t{_0} {_1}")]
    Movsx(AssemblyType, AssemblyType, Operand, Operand),
    #[display("\tmovze\t{_2}, {_3}\t{_0} {_1}")]
    MovZeroExtend(AssemblyType, AssemblyType, Operand, Operand),
    #[display("\tlea\t{_0}, {_1}")]
    Lea(Operand, Operand),
    #[display("\tcvttsd2si\t{_1}, {_2}\t{_0}")]
    Cvttsd2si(AssemblyType, Operand, Operand),
    #[display("\tcvtsi2sd\t{_1}, {_2}\t{_0}")]
    Cvtsi2sd(AssemblyType, Operand, Operand),
    #[display("\t{_1}\t{_2}\t{_0}")]
    Unary(AssemblyType, UnaryOp, Operand),
    #[display("\t{_1}\t{_2}, {_3}\t{_0}")]
    Binary(AssemblyType, BinOp, Operand, Operand),
    #[display("\tcmp\t{_1}, {_2}\t{_0}")]
    Cmp(AssemblyType, Operand, Operand),
    #[display("\tidiv\t{_1}\t{_0}")]
    Idiv(AssemblyType, Operand),
    #[display("\tdiv\t{_1}\t{_0}")]
    Div(AssemblyType, Operand),
    #[display("\tcdq\t\t{_0}")]
    Cdq(AssemblyType),
    #[display("\tjmp\t{_0}")]
    Jmp(CodeLabel),
    #[display("\tj {_0}\t{_1}")]
    JmpCC(ConditionCode, CodeLabel),
    #[display("\tset {_0}\t{_1}")]
    SetCC(ConditionCode, Operand),
    #[display("{_0}:")]
    Label(CodeLabel),
    #[display("\tpush\t{_0}")]
    Push(Operand),
    #[display("\tpop\t{_0}")]
    Pop(Register),
    #[display("\tcall\t{_0}")]
    Call(Identifier),
    #[display("\tret")]
    Ret,
}

pub fn mov(asm_type: AssemblyType, src: impl Into<Operand>, dest: impl Into<Operand>) -> Inst {
    Inst::Mov(asm_type, src.into(), dest.into())
}

pub fn cmp(asm_type: AssemblyType, op1: impl Into<Operand>, op2: impl Into<Operand>) -> Inst {
    Inst::Cmp(asm_type, op1.into(), op2.into())
}

pub fn unary(asm_type: AssemblyType, operator: UnaryOp, operand: impl Into<Operand>) -> Inst {
    Inst::Unary(asm_type, operator, operand.into())
}

pub fn binary(
    operator: BinOp,
    asm_type: AssemblyType,
    operand1: impl Into<Operand>,
    operand2: impl Into<Operand>,
) -> Inst {
    Inst::Binary(asm_type, operator, operand1.into(), operand2.into())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Display)]
pub enum UnaryOp {
    #[display("neg")]
    Neg,
    #[display("not")]
    Not,
    #[display("shr")]
    Shr,
}

impl From<UnaryOperator> for UnaryOp {
    fn from(value: UnaryOperator) -> Self {
        match value {
            UnaryOperator::Complement => UnaryOp::Not,
            UnaryOperator::Negate => UnaryOp::Neg,
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Display)]
pub enum BinOp {
    #[display("add")]
    Add,
    #[display("sub")]
    Sub,
    #[display("mult")]
    Mult,
    #[display("ddiv")]
    DivDouble,
    #[display("and")]
    And,
    #[display("or")]
    Or,
    #[display("xor")]
    Xor,
    #[display("shl")]
    Shl,
    #[display("shr")]
    Shr,
    #[display("sar")]
    Sar,
}

#[derive(
    Clone, Copy, PartialEq, Eq, Debug, From, Display, Unwrap, IsVariant, Hash, PartialOrd, Ord,
)]
#[unwrap(ref)]
pub enum Operand {
    #[display("${_0}")]
    Imm(i64),
    Reg(Register),
    #[from(ignore)]
    #[display("{}", _0.value)]
    Pseudo(Identifier),
    #[display("{}[{_1}]", _0.value)]
    PseudoMem(Identifier, isize),
    #[display("{_1}({_0})")]
    #[from(ignore)]
    Memory(Register, isize),
    #[display("({_0} + {_1} * {_2})")]
    Indexed(Register, Register, usize),
    #[from(ignore)]
    #[display("{}+{_1}", _0.value)]
    Data(Identifier, usize),
}

impl Operand {
    pub fn is_mem(&self) -> bool {
        self.is_memory() || self.is_data() || self.is_indexed()
    }

    pub fn is_register(&self) -> bool {
        matches!(self, Self::Reg(_) | Self::Pseudo(_))
    }

    pub fn offset_clone(&self, offset: usize) -> Self {
        match self {
            Self::PseudoMem(base, old_offset) => {
                Self::PseudoMem(*base, old_offset + offset as isize)
            }
            Self::Memory(base, old_offset) => Self::Memory(*base, old_offset + offset as isize),
            Self::Data(base, old_offset) => Self::Data(*base, old_offset + offset),
            _ => panic!(),
        }
    }

    pub fn get_pseudo_type(&self, symbols: &BackendTable) -> AssemblyType {
        let name = match self {
            Self::Pseudo(name) => name,
            Self::PseudoMem(name, _) => name,
            _ => panic!(),
        };

        *symbols.get(&name.value).unwrap().unwrap_obj_ref().0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Display, PartialOrd, Ord)]
pub enum Register {
    AX,
    BX,
    CX,
    DX,
    DI,
    SI,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
    SP,
    BP,
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
    XMM8,
    XMM9,
    XMM10,
    XMM11,
    XMM12,
    XMM13,
    XMM14,
    XMM15,
}

impl Register {
    pub const INT_ALLOCATE_REGS: [Self; 12] = [
        Self::AX,
        Self::BX,
        Self::CX,
        Self::DX,
        Self::DI,
        Self::SI,
        Self::R8,
        Self::R9,
        Self::R12,
        Self::R13,
        Self::R14,
        Self::R15,
    ];

    pub const FP_ALLOCATE_REGS: [Self; 14] = [
        Self::XMM0,
        Self::XMM1,
        Self::XMM2,
        Self::XMM3,
        Self::XMM4,
        Self::XMM5,
        Self::XMM6,
        Self::XMM7,
        Self::XMM8,
        Self::XMM9,
        Self::XMM10,
        Self::XMM11,
        Self::XMM12,
        Self::XMM13,
    ];

    pub const INT_PARAM_REGS: [Self; 6] =
        [Self::DI, Self::SI, Self::DX, Self::CX, Self::R8, Self::R9];

    pub const FP_PARAM_REGS: [Self; 8] = [
        Self::XMM0,
        Self::XMM1,
        Self::XMM2,
        Self::XMM3,
        Self::XMM4,
        Self::XMM5,
        Self::XMM6,
        Self::XMM7,
    ];

    pub const INT_RET_REGS: [Self; 2] = [Self::AX, Self::DX];
    pub const FP_RET_REGS: [Self; 2] = [Self::XMM0, Self::XMM1];

    pub fn is_callee_saved(&self) -> bool {
        matches!(
            self,
            Self::BX | Self::R12 | Self::R13 | Self::R14 | Self::R15
        )
    }

    pub fn is_fp(&self) -> bool {
        use Register::*;
        matches!(
            self,
            XMM0 | XMM1
                | XMM2
                | XMM3
                | XMM4
                | XMM5
                | XMM6
                | XMM7
                | XMM8
                | XMM9
                | XMM10
                | XMM11
                | XMM12
                | XMM13
                | XMM14
                | XMM15
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Display)]
pub enum ConditionCode {
    E,
    NE,
    G,
    GE,
    L,
    LE,
    A,
    AE,
    B,
    BE,
}

impl ConditionCode {
    pub fn for_operator(operator: BinaryOp, is_signed: bool) -> Self {
        match (operator, is_signed) {
            (BinaryOp::Equal, _) => ConditionCode::E,
            (BinaryOp::NotEqual, _) => ConditionCode::NE,
            (BinaryOp::LessThan, true) => ConditionCode::L,
            (BinaryOp::LessOrEqual, true) => ConditionCode::LE,
            (BinaryOp::GreaterThan, true) => ConditionCode::G,
            (BinaryOp::GreaterOrEqual, true) => ConditionCode::GE,
            (BinaryOp::LessThan, false) => ConditionCode::B,
            (BinaryOp::LessOrEqual, false) => ConditionCode::BE,
            (BinaryOp::GreaterThan, false) => ConditionCode::A,
            (BinaryOp::GreaterOrEqual, false) => ConditionCode::AE,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub enum AssemblyType {
    Byte,
    Word,
    Longword,
    Quadword,
    DoublePrecision,
    #[display("ByteArray({_0}, {_1})")]
    ByteArray(usize, usize),
}

impl AssemblyType {
    pub fn from_c_type(c_type: CType, type_table: &TypeTable) -> Self {
        match c_type {
            CType::Void => Self::Byte,
            CType::Int => Self::Longword,
            CType::Long => Self::Quadword,
            CType::Short => Self::Word,
            CType::Unsigned => Self::Longword,
            CType::UnsignedLong => Self::Quadword,
            CType::UnsignedShort => Self::Word,
            CType::Double => Self::DoublePrecision,
            CType::Pointer(_) => Self::Quadword,
            CType::Char => Self::Byte,
            CType::SignedChar => Self::Byte,
            CType::UnsignedChar => Self::Byte,
            CType::Array(value_type, size) => Self::ByteArray(
                value_type.size(type_table) * size,
                if value_type.size(type_table) * size < 16 {
                    value_type.scalar_size(type_table)
                } else {
                    16
                },
            ),
            CType::Function(_) => panic!("cannot convert function type to assembly"),
            CType::Structure(tag) => {
                let struct_def = type_table.get(&tag.value).cloned().unwrap_or_default();
                Self::ByteArray(struct_def.size, struct_def.alignment)
            }
        }
    }

    pub fn alignment(&self) -> isize {
        match self {
            Self::Byte => 1,
            Self::Word => 2,
            Self::Longword => 4,
            Self::Quadword => 8,
            Self::DoublePrecision => 8,
            Self::ByteArray(_, alignment) => *alignment as isize,
        }
    }

    pub fn size(&self) -> isize {
        match self {
            Self::Byte => 1,
            Self::Word => 2,
            Self::Longword => 4,
            Self::Quadword => 8,
            Self::DoublePrecision => 8,
            Self::ByteArray(size, _) => *size as isize,
        }
    }

    pub fn is_scalar(&self) -> bool {
        !matches!(self, Self::ByteArray(..))
    }
}

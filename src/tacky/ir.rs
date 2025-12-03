use std::collections::HashSet;

use derive_more::{Display, From, Unwrap};
use serde::Serialize;

use crate::common::symbol_table::{STable, StaticInit, SymbolTable};
use crate::common::{CType, CodeLabel, Constant, Identifier, print_option, print_vec};
use crate::parser::ast::BinaryOperator;

pub use crate::parser::ast::UnaryOperator;

#[derive(Clone, Debug, Display, Serialize)]
#[display("{}\n", print_vec(top_level, "\n\n"))]
pub struct Program {
    pub top_level: Vec<TopLevel>,
}

#[derive(Clone, Debug, Display, From, Serialize)]
#[serde(tag = "type")]
pub enum TopLevel {
    Fn(Function),
    Var(StaticVariable),
    Const(StaticConstant),
}

#[derive(Clone, Debug, Display, Serialize)]
#[display("{name}: ({})\n{}", print_vec(params, ", "), print_vec(body, "\n"))]
pub struct Function {
    pub name: Identifier,
    pub global: bool,
    pub params: Vec<Value>,
    pub body: Vec<Instruction>,
}

#[derive(Clone, Debug, Display, Serialize)]
#[display("{name}: {var_type} {} {}", print_vec(init_list, ", "), match self.global {
                true => "global",
                false => "local",
            })]
pub struct StaticVariable {
    pub name: Identifier,
    pub global: bool,
    pub init_list: Vec<StaticInit>,
    pub var_type: CType,
}

#[derive(Clone, Debug, Display, Serialize)]
#[display("{name}: {var_type} const {init}")]
pub struct StaticConstant {
    pub name: Identifier,
    pub var_type: CType,
    pub init: StaticInit,
}

#[derive(Clone, Debug, Display, Serialize, PartialEq)]
pub enum Instruction {
    #[display("Return({})", print_option(_0))]
    Return(Option<Value>),
    #[display("{_1} = SignExtend({_0})")]
    SignExtend(Value, Value),
    #[display("{_1} = Truncate({_0})")]
    Truncate(Value, Value),
    #[display("{_1} = ZeroExtend({_0})")]
    ZeroExtend(Value, Value),
    #[display("{_1} = DoubleToInt({_0})")]
    DoubleToInt(Value, Value),
    #[display("{_1} = DoubleToUInt({_0})")]
    DoubleToUInt(Value, Value),
    #[display("({_1} = IntToDouble({_0}))")]
    IntToDouble(Value, Value),
    #[display("({_1} = UIntToDouble({_0}))")]
    UIntToDouble(Value, Value),
    #[display("{_2} = {_0}{_1}")]
    Unary(UnaryOperator, Value, Value),
    #[display("{_3} = {_1} {_0} {_2}")]
    Binary(BinaryOp, Value, Value, Value),
    #[display("{_1} = {_0}")]
    Copy(Value, Value),
    #[display("{_1} = GetAddress({_0})")]
    GetAddress(Value, Value),
    #[display("Load({_0}, {_1})")]
    Load(Value, Value),
    #[display("Store({_0}, {_1})")]
    Store(Value, Value),
    #[display("{_3} = AddPtr({_0}, {_1}, {_2})")]
    AddPtr(Value, Value, usize, Value),
    #[display("{_1}({_2}) = {_0}")]
    CopyToOffset(Value, Identifier, usize),
    #[display("{_2} = {_0}({_1})")]
    CopyFromOffset(Identifier, usize, Value),
    #[display("Jump({_0})")]
    Jump(CodeLabel),
    #[display("JumpIfZero({_0}, {_1})")]
    JumpIfZero(Value, CodeLabel),
    #[display("JumpIfNotZero({_0}, {_1})")]
    JumpIfNotZero(Value, CodeLabel),
    #[display("Label({_0})")]
    Label(CodeLabel),
    #[display("{} = {_0}({})", print_option(_2), print_vec(_1, ", "))]
    FunCall(Identifier, Vec<Value>, Option<Value>),
}

impl Instruction {
    pub fn is_jump(&self) -> bool {
        matches!(
            self,
            Self::Jump(_) | Self::JumpIfZero(..) | Self::JumpIfNotZero { .. }
        )
    }
}

#[derive(Clone, Copy, Debug, Unwrap)]
pub enum ExpressionResult {
    PlainOperand(Value),
    DereferencedPointer(Value),
    SubObject(Identifier, usize),
}

impl From<Value> for ExpressionResult {
    fn from(value: Value) -> Self {
        Self::PlainOperand(value)
    }
}

impl From<Constant> for ExpressionResult {
    fn from(value: Constant) -> Self {
        let v: Value = value.into();
        v.into()
    }
}

#[derive(Clone, Copy, Debug, Display, From, Unwrap, Serialize, PartialEq, Eq, Hash)]
#[serde(tag = "type")]
#[unwrap(ref)]
#[display("({})", _0)]
pub enum Value {
    Constant(Constant),
    Var(Identifier),
}

impl Value {
    pub fn get_type(&self, symbols: &SymbolTable) -> CType {
        match self {
            Self::Constant(c) => c.get_type(),
            Self::Var(name) => symbols.get_expected_type(name.value),
        }
    }

    pub fn is_static(&self, symbols: &SymbolTable) -> bool {
        match self {
            Self::Constant(_) => false,
            Self::Var(name) => symbols.get(&name.value).unwrap().attrs.is_static(),
        }
    }

    pub fn is_aliased(&self, aliased_vars: &HashSet<Identifier>) -> bool {
        match self {
            Self::Constant(_) => false,
            Self::Var(name) => aliased_vars.contains(name),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Display, Serialize)]
pub enum BinaryOp {
    #[display("+")]
    Add,
    #[display("-")]
    Subtract,
    #[display("*")]
    Multiply,
    #[display("/")]
    Divide,
    #[display("%")]
    Remainder,
    #[display("&")]
    BitwiseAnd,
    #[display("|")]
    BitwiseOr,
    #[display("^")]
    BitwiseXor,
    #[display("==")]
    Equal,
    #[display("!=")]
    NotEqual,
    #[display("<<")]
    LeftShift,
    #[display(">>")]
    RightShift,
    #[display("<")]
    LessThan,
    #[display("<=")]
    LessOrEqual,
    #[display(">")]
    GreaterThan,
    #[display(">=")]
    GreaterOrEqual,
}

impl TryFrom<BinaryOperator> for BinaryOp {
    type Error = ();
    fn try_from(value: BinaryOperator) -> Result<Self, Self::Error> {
        match value {
            BinaryOperator::Add => Ok(BinaryOp::Add),
            BinaryOperator::Subtract => Ok(BinaryOp::Subtract),
            BinaryOperator::Multiply => Ok(BinaryOp::Multiply),
            BinaryOperator::Divide => Ok(BinaryOp::Divide),
            BinaryOperator::Remainder => Ok(BinaryOp::Remainder),
            BinaryOperator::BitwiseAnd => Ok(BinaryOp::BitwiseAnd),
            BinaryOperator::BitwiseOr => Ok(BinaryOp::BitwiseOr),
            BinaryOperator::BitwiseXor => Ok(BinaryOp::BitwiseXor),
            BinaryOperator::Equal => Ok(BinaryOp::Equal),
            BinaryOperator::NotEqual => Ok(BinaryOp::NotEqual),
            BinaryOperator::LeftShift => Ok(BinaryOp::LeftShift),
            BinaryOperator::RightShift => Ok(BinaryOp::RightShift),
            BinaryOperator::LessThan => Ok(BinaryOp::LessThan),
            BinaryOperator::LessOrEqual => Ok(BinaryOp::LessOrEqual),
            BinaryOperator::GreaterThan => Ok(BinaryOp::GreaterThan),
            BinaryOperator::GreaterOrEqual => Ok(BinaryOp::GreaterOrEqual),
            _ => Err(()),
        }
    }
}

use serde::Serialize;

use crate::common::{CodeLabel, Identifier};

#[derive(Clone, Debug, Serialize)]
pub struct Program {
    pub functions: Vec<Function>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Function {
    pub name: Identifier,
    pub body: Vec<Inst>,
    pub params: Vec<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub enum Inst {
    Ret(Value),
    Negate(Value, Value),
    Complement(Value, Value),
    Binary(BinOp, Value, Value, Value),
    Copy(Value, Value),
    Jump(CodeLabel),
    Jnz(Value, CodeLabel, CodeLabel),
    Label(CodeLabel),
}

#[derive(Clone, Debug, Serialize)]
pub enum Value {
    Constant(i64),
    Temporary(&'static str),
}

#[derive(Clone, Debug, Serialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Sar,
    Ceq,
    Cne,
    Cslt,
    Csle,
    Csgt,
    Csge,
}

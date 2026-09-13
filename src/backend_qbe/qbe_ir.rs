use serde::Serialize;

use crate::common::Identifier;

#[derive(Clone, Debug, Serialize)]
pub struct Program {
    pub function: Function,
}

#[derive(Clone, Debug, Serialize)]
pub struct Function {
    pub name: Identifier,
    pub body: Vec<Inst>,
}

#[derive(Clone, Debug, Serialize)]
pub enum Inst {
    Ret(Value),
    Negate(Value, Value),
    Complement(Value, Value),
    Binary(BinOp, Value, Value, Value),
    Assign(Value, Value),
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
}

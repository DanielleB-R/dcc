use serde::Serialize;

use crate::common::Identifier;

#[derive(Clone, Debug, Serialize)]
pub struct Program {
    pub function: Function,
}

#[derive(Clone, Debug, Serialize)]
pub struct Function {
    pub name: Identifier,
    pub body: Inst,
}

#[derive(Clone, Debug, Serialize)]
pub enum Inst {
    Ret(i64),
}

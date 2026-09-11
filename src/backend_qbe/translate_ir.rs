use super::qbe_ir;
use crate::tacky::ir::{self, UnaryOperator};

fn translate_value(code: ir::Value) -> qbe_ir::Value {
    match code {
        ir::Value::Constant(c) => qbe_ir::Value::Constant(c.unwrap_integer()),
        ir::Value::Var(name) => qbe_ir::Value::Temporary(name.value),
    }
}

fn translate_instruction(code: ir::Instruction) -> qbe_ir::Inst {
    match code {
        ir::Instruction::Return(Some(val)) => qbe_ir::Inst::Ret(translate_value(val)),
        ir::Instruction::Unary(UnaryOperator::Negate, src, dest) => {
            qbe_ir::Inst::Negate(translate_value(src), translate_value(dest))
        }
        ir::Instruction::Unary(UnaryOperator::Complement, src, dest) => {
            qbe_ir::Inst::Complement(translate_value(src), translate_value(dest))
        }
        _ => unimplemented!(),
    }
}

fn translate_function(code: ir::Function) -> qbe_ir::Function {
    qbe_ir::Function {
        name: code.name,
        body: code.body.into_iter().map(translate_instruction).collect(),
    }
}

pub fn translate_ir(code: ir::Program) -> qbe_ir::Program {
    qbe_ir::Program {
        function: translate_function(match code.top_level[0].clone() {
            ir::TopLevel::Fn(f) => f,
            _ => unimplemented!(),
        }),
    }
}

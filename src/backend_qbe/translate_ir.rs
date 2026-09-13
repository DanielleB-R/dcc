use super::qbe_ir;
use crate::tacky::ir::{self, UnaryOperator};

fn translate_value(code: ir::Value) -> qbe_ir::Value {
    match code {
        ir::Value::Constant(c) => qbe_ir::Value::Constant(c.unwrap_integer()),
        ir::Value::Var(name) => qbe_ir::Value::Temporary(name.value),
    }
}

fn translate_binary(code: ir::BinaryOp) -> qbe_ir::BinOp {
    match code {
        ir::BinaryOp::Add => qbe_ir::BinOp::Add,
        ir::BinaryOp::Subtract => qbe_ir::BinOp::Sub,
        ir::BinaryOp::Multiply => qbe_ir::BinOp::Mul,
        ir::BinaryOp::Divide => qbe_ir::BinOp::Div,
        ir::BinaryOp::Remainder => qbe_ir::BinOp::Rem,
        _ => unimplemented!(),
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
        ir::Instruction::Binary(op, src1, src2, dest) => qbe_ir::Inst::Binary(
            translate_binary(op),
            translate_value(src1),
            translate_value(src2),
            translate_value(dest),
        ),
        ir::Instruction::Copy(src, dest) => {
            qbe_ir::Inst::Assign(translate_value(src), translate_value(dest))
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

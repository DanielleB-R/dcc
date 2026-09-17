use super::qbe_ir;
use crate::{
    common::CodeLabel,
    tacky::ir::{self, UnaryOperator},
};

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
        ir::BinaryOp::BitwiseAnd => qbe_ir::BinOp::And,
        ir::BinaryOp::BitwiseOr => qbe_ir::BinOp::Or,
        ir::BinaryOp::BitwiseXor => qbe_ir::BinOp::Xor,
        ir::BinaryOp::LeftShift => qbe_ir::BinOp::Shl,
        ir::BinaryOp::RightShift => qbe_ir::BinOp::Sar,
        ir::BinaryOp::Equal => qbe_ir::BinOp::Ceq,
        ir::BinaryOp::NotEqual => qbe_ir::BinOp::Cne,
        ir::BinaryOp::LessThan => qbe_ir::BinOp::Cslt,
        ir::BinaryOp::LessOrEqual => qbe_ir::BinOp::Csle,
        ir::BinaryOp::GreaterThan => qbe_ir::BinOp::Csgt,
        ir::BinaryOp::GreaterOrEqual => qbe_ir::BinOp::Csge,
    }
}

fn translate_instruction(code: ir::Instruction, body: &mut Vec<qbe_ir::Inst>) {
    match code {
        ir::Instruction::Return(Some(val)) => {
            body.push(qbe_ir::Inst::Ret(translate_value(val)));
            body.push(qbe_ir::Inst::Label(CodeLabel::from(".ret")))
        }
        ir::Instruction::Unary(UnaryOperator::Negate, src, dest) => body.push(
            qbe_ir::Inst::Negate(translate_value(src), translate_value(dest)),
        ),
        ir::Instruction::Unary(UnaryOperator::Complement, src, dest) => body.push(
            qbe_ir::Inst::Complement(translate_value(src), translate_value(dest)),
        ),
        ir::Instruction::Unary(UnaryOperator::Not, src, dest) => body.push(qbe_ir::Inst::Binary(
            qbe_ir::BinOp::Ceq,
            translate_value(src),
            qbe_ir::Value::Constant(0),
            translate_value(dest),
        )),
        ir::Instruction::Binary(op, src1, src2, dest) => body.push(qbe_ir::Inst::Binary(
            translate_binary(op),
            translate_value(src1),
            translate_value(src2),
            translate_value(dest),
        )),
        ir::Instruction::Copy(src, dest) => body.push(qbe_ir::Inst::Copy(
            translate_value(src),
            translate_value(dest),
        )),
        ir::Instruction::Jump(label) => {
            body.push(qbe_ir::Inst::Jump(label));
            body.push(qbe_ir::Inst::Label(CodeLabel::from(".jmp")));
        }
        ir::Instruction::JumpIfZero(val, target) => {
            let nz_label = CodeLabel::from(".jnz");

            body.push(qbe_ir::Inst::Jnz(translate_value(val), nz_label, target));
            body.push(qbe_ir::Inst::Label(nz_label));
        }
        ir::Instruction::JumpIfNotZero(val, target) => {
            let z_label = CodeLabel::from(".jnz");

            body.push(qbe_ir::Inst::Jnz(translate_value(val), target, z_label));
            body.push(qbe_ir::Inst::Label(z_label));
        }
        ir::Instruction::Label(label) => body.push(qbe_ir::Inst::Label(label)),
        ir::Instruction::FunCall(name, params, dest) => body.push(qbe_ir::Inst::Call(
            name,
            params.into_iter().map(translate_value).collect(),
            dest.map(translate_value)
                .unwrap_or_else(|| qbe_ir::Value::Temporary("%.nil")),
        )),
        _ => unimplemented!(),
    }
}

fn translate_function(code: ir::Function) -> qbe_ir::Function {
    let mut body = vec![];

    for instruction in code.body {
        translate_instruction(instruction, &mut body);
    }

    if matches!(body.last().unwrap(), qbe_ir::Inst::Label(_)) {
        body.pop();
    }

    qbe_ir::Function {
        name: code.name,
        exported: code.global,
        body,
        params: code.params.into_iter().map(translate_value).collect(),
    }
}

pub fn translate_ir(code: ir::Program) -> qbe_ir::Program {
    let mut top_level = vec![];

    for tl in code.top_level {
        top_level.push(qbe_ir::TopLevel::Fn(translate_function(match tl {
            ir::TopLevel::Fn(f) => f,
            _ => unimplemented!(),
        })))
    }

    qbe_ir::Program { top_level }
}

use super::qbe_ir;
use crate::tacky::ir;

fn translate_instruction(code: ir::Instruction) -> qbe_ir::Inst {
    match code {
        ir::Instruction::Return(Some(ir::Value::Constant(c))) => {
            qbe_ir::Inst::Ret(c.unwrap_integer())
        }
        _ => unimplemented!(),
    }
}

fn translate_function(code: ir::Function) -> qbe_ir::Function {
    qbe_ir::Function {
        name: code.name,
        body: translate_instruction(code.body[0].clone()),
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

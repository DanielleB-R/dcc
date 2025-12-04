use super::asm_ast::{AssemblyType::*, Register::*, *};

fn too_large(n: i64) -> bool {
    n > i32::MAX as i64 || n < i32::MIN as i64
}

fn too_large_for_byte(n: i64) -> bool {
    n > i8::MAX as i64 || n < i8::MIN as i64
}

fn fixup_instruction(i: Inst, callee_saved_regs: &[Register]) -> Vec<Inst> {
    match i {
        Inst::Mov(DoublePrecision, src, dest) if src.is_mem() && dest.is_mem() => {
            vec![
                mov(DoublePrecision, src, XMM14),
                mov(DoublePrecision, XMM14, dest),
            ]
        }
        Inst::Mov(asm_type, src, dest) if src.is_mem() && dest.is_mem() => {
            vec![mov(asm_type, src, R10), mov(asm_type, R10, dest)]
        }
        Inst::Mov(Quadword, Operand::Imm(n), dest) if too_large(n) && !dest.is_reg() => {
            vec![mov(Quadword, n, R10), mov(Quadword, R10, dest)]
        }
        Inst::Mov(Longword, Operand::Imm(n), dest) if too_large(n) => {
            vec![mov(Longword, n as i32 as i64, dest)]
        }
        Inst::Mov(Byte, Operand::Imm(n), dest) if too_large_for_byte(n) => {
            vec![mov(Byte, n as i8 as i64, dest)]
        }
        Inst::Movsx(src_type, dest_type, src @ Operand::Imm(_), dest) if dest.is_mem() => vec![
            mov(src_type, src, R10),
            Inst::Movsx(src_type, dest_type, R10.into(), R11.into()),
            mov(dest_type, R11, dest),
        ],
        Inst::Movsx(src_type, dest_type, src, dest) if dest.is_mem() => {
            vec![
                Inst::Movsx(src_type, dest_type, src, R11.into()),
                mov(dest_type, R11, dest),
            ]
        }
        Inst::Movsx(src_type, dest_type, src @ Operand::Imm(_), dest) => {
            vec![
                mov(src_type, src, R10),
                Inst::Movsx(src_type, dest_type, R10.into(), dest),
            ]
        }
        Inst::MovZeroExtend(Byte, dest_type, src, dest) => {
            let mut result = vec![];

            let source = if src.is_imm() {
                result.push(mov(Byte, src, R10));
                R10.into()
            } else {
                src
            };

            if dest.is_mem() {
                result.push(Inst::MovZeroExtend(Byte, dest_type, source, R11.into()));
                result.push(mov(dest_type, R11, dest));
            } else {
                result.push(Inst::MovZeroExtend(Byte, dest_type, source, dest));
            }

            result
        }
        Inst::MovZeroExtend(Longword, Quadword, src, dest @ Operand::Reg(_)) => {
            vec![mov(Longword, src, dest)]
        }
        Inst::MovZeroExtend(Longword, Quadword, src, dest) => {
            vec![mov(Longword, src, R11), mov(Quadword, R11, dest)]
        }
        Inst::Lea(src, dest) if dest.is_mem() => {
            vec![Inst::Lea(src, R11.into()), mov(Quadword, R11, dest)]
        }

        Inst::Cvttsd2si(dst_type, src, dest) if dest.is_mem() => {
            vec![
                Inst::Cvttsd2si(dst_type, src, R11.into()),
                mov(dst_type, R11, dest),
            ]
        }
        Inst::Cvtsi2sd(src_type, src, dest) => {
            let mut result = vec![];
            let inst_src = if src.is_imm() {
                result.push(mov(src_type, src, R10));
                R10.into()
            } else {
                src
            };
            if dest.is_mem() {
                result.push(Inst::Cvtsi2sd(src_type, inst_src, XMM15.into()));
                result.push(mov(DoublePrecision, XMM15, dest));
            } else {
                result.push(Inst::Cvtsi2sd(src_type, inst_src, dest));
            }
            result
        }
        Inst::Idiv(asm_type, operand @ Operand::Imm(_)) => {
            vec![
                mov(asm_type, operand, R10),
                Inst::Idiv(asm_type, R10.into()),
            ]
        }
        Inst::Div(asm_type, operand @ Operand::Imm(_)) => {
            vec![mov(asm_type, operand, R10), Inst::Div(asm_type, R10.into())]
        }
        Inst::Binary(DoublePrecision, operator, operand1, operand2) if operand2.is_mem() => {
            vec![
                mov(DoublePrecision, operand2, XMM15),
                binary(operator, DoublePrecision, operand1, XMM15),
                mov(DoublePrecision, XMM15, operand2),
            ]
        }

        Inst::Binary(asm_type, BinOp::Mult, operand1, operand2) => {
            let mut result = vec![];

            let src = if let Operand::Imm(n) = operand1
                && too_large(n)
            {
                result.push(mov(asm_type, n, R10));
                R10.into()
            } else {
                operand1
            };

            if operand2.is_mem() {
                result.push(mov(asm_type, operand2, R11));
                result.push(binary(BinOp::Mult, asm_type, src, R11));
                result.push(mov(asm_type, R11, operand2));
            } else {
                result.push(binary(BinOp::Mult, asm_type, src, operand2));
            }

            result
        }
        Inst::Binary(
            asm_type,
            operator @ (BinOp::Shl | BinOp::Shr | BinOp::Sar),
            operand1,
            operand2,
        ) if operand1.is_mem() || operand1.is_reg() => vec![
            mov(asm_type, operand1, CX),
            binary(operator, asm_type, CX, operand2),
        ],
        // this won't match Mult or the shift operators because of the above,
        // nor will it match floating point ops
        Inst::Binary(asm_type, operator, operand1, operand2)
            if operand1.is_mem() && operand2.is_mem() =>
        {
            vec![
                mov(asm_type, operand1, R10),
                binary(operator, asm_type, R10, operand2),
            ]
        }
        Inst::Binary(Quadword, operator, Operand::Imm(n), operand2) if too_large(n) => vec![
            mov(Quadword, n, R10),
            binary(operator, Quadword, R10, operand2),
        ],
        Inst::Cmp(DoublePrecision, operand1, operand2) if operand2.is_mem() => {
            vec![
                mov(DoublePrecision, operand2, XMM15),
                cmp(DoublePrecision, operand1, XMM15),
            ]
        }
        Inst::Cmp(asm_type, operand1, operand2) if operand1.is_mem() && operand2.is_mem() => {
            vec![mov(asm_type, operand1, R10), cmp(asm_type, R10, operand2)]
        }
        Inst::Cmp(Quadword, Operand::Imm(n1), Operand::Imm(n2))
            if too_large(n1) || too_large(n2) =>
        {
            vec![
                mov(Quadword, n1, R10),
                mov(Quadword, n2, R11),
                cmp(Quadword, R10, R11),
            ]
        }
        Inst::Cmp(Quadword, Operand::Imm(n), operand2) if too_large(n) => {
            vec![mov(Quadword, n, R10), cmp(Quadword, R10, operand2)]
        }
        Inst::Cmp(
            asm_type @ (AssemblyType::Longword | AssemblyType::Quadword),
            operand1,
            operand2 @ Operand::Imm(_),
        ) => {
            vec![mov(asm_type, operand2, R11), cmp(asm_type, operand1, R11)]
        }
        Inst::Push(Operand::Imm(n)) => {
            vec![mov(Quadword, n, R10), Inst::Push(R10.into())]
        }
        Inst::Push(Operand::Reg(r)) if r.is_fp() => {
            vec![
                binary(BinOp::Sub, Quadword, 8, Register::SP),
                mov(DoublePrecision, r, Operand::Memory(Register::SP, 0)),
            ]
        }
        Inst::Ret => {
            let mut result: Vec<_> = callee_saved_regs.iter().copied().map(Inst::Pop).collect();

            result.push(Inst::Ret);

            result
        }
        i => vec![i],
    }
}

pub fn fixup_instructions(code: Program) -> Program {
    code.map_fn(|f| Function {
        name: f.name,
        global: f.global,
        stack_offset: f.stack_offset,
        instructions: f
            .instructions
            .into_iter()
            .flat_map(|i| fixup_instruction(i, &f.callee_saved_regs))
            .collect(),
        callee_saved_regs: f.callee_saved_regs,
        aliased_vars: f.aliased_vars,
    })
}

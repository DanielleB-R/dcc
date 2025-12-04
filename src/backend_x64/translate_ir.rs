use std::mem;

use super::asm_ast::{
    AssemblyType::{self, *},
    *,
};
use crate::{
    common::{
        CType, CodeLabel, Constant, Identifier,
        symbol_table::{STable, StaticInit, SymbolTable},
        type_table::{EightbyteClass, TTable, TypeTable},
    },
    optimizer,
    tacky::ir::{self, BinaryOp, Instruction, Value},
};

type TypedOperand = (Operand, AssemblyType);

fn translate_operator(op: BinaryOp, is_signed: bool) -> BinOp {
    match op {
        BinaryOp::Add => BinOp::Add,
        BinaryOp::Subtract => BinOp::Sub,
        BinaryOp::Multiply => BinOp::Mult,
        BinaryOp::BitwiseAnd => BinOp::And,
        BinaryOp::BitwiseOr => BinOp::Or,
        BinaryOp::BitwiseXor => BinOp::Xor,
        BinaryOp::LeftShift => BinOp::Shl,
        BinaryOp::RightShift => {
            if is_signed {
                BinOp::Sar
            } else {
                BinOp::Shr
            }
        }
        _ => unreachable!(),
    }
}

fn copy_bytes(asm_type: AssemblyType, src: Operand, dest: Operand, instructions: &mut Vec<Inst>) {
    match asm_type {
        ByteArray(size, _) => {
            let mut offset = 0;
            while offset < size {
                offset += match size - offset {
                    8.. => {
                        instructions.push(mov(
                            Quadword,
                            src.offset_clone(offset),
                            dest.offset_clone(offset),
                        ));
                        8
                    }
                    4..8 => {
                        instructions.push(mov(
                            Longword,
                            src.offset_clone(offset),
                            dest.offset_clone(offset),
                        ));
                        4
                    }
                    _ => {
                        instructions.push(mov(
                            Byte,
                            src.offset_clone(offset),
                            dest.offset_clone(offset),
                        ));
                        1
                    }
                }
            }
        }
        t => instructions.push(mov(t, src, dest)),
    }
}

fn copy_bytes_to_reg(src: Operand, dest: Register, count: usize, instructions: &mut Vec<Inst>) {
    let mut offset = count - 1;
    loop {
        let byte_src = src.offset_clone(offset);
        instructions.push(mov(Byte, byte_src, dest));
        if offset > 0 {
            instructions.push(binary(BinOp::Shl, Quadword, 8, dest));
            offset -= 1;
        } else {
            break;
        }
    }
}

fn copy_bytes_from_reg(src: Register, dest: Operand, count: usize, instructions: &mut Vec<Inst>) {
    let mut offset = 0;
    while offset < count {
        let byte_dest = dest.offset_clone(offset);
        instructions.push(mov(Byte, src, byte_dest));
        if offset < count - 1 {
            instructions.push(binary(BinOp::Shr, Quadword, 8, src));
        }
        offset += 1;
    }
}

fn get_eightbyte_type(offset: usize, struct_size: usize) -> AssemblyType {
    match struct_size - offset {
        8.. => Quadword,
        4 => Longword,
        1 => Byte,
        bytes_from_end => ByteArray(bytes_from_end, 8),
    }
}

enum ReturnValue {
    Registers(Vec<TypedOperand>, Vec<Operand>),
    Memory,
}

struct AsmGenerator<'a> {
    symbols: &'a SymbolTable,
    types: &'a TypeTable,
    constant_index: usize,
    label_index: usize,
    static_consts: Vec<StaticConstant>,
}

impl<'a> AsmGenerator<'a> {
    fn new(symbols: &'a SymbolTable, types: &'a TypeTable) -> Self {
        Self {
            symbols,
            types,
            constant_index: 0,
            static_consts: vec![],
            label_index: 0,
        }
    }

    fn constant_label(&mut self) -> Identifier {
        self.constant_index += 1;
        format!(".L.double_const.{}", self.constant_index).into()
    }

    fn make_label(&mut self, tag: &'static str) -> CodeLabel {
        self.label_index += 1;
        CodeLabel {
            tag,
            counter: self.label_index,
        }
    }

    fn translate_value(&mut self, value: Value) -> Operand {
        match value {
            Value::Constant(c) => match c {
                Constant::Double(n) => {
                    let name = self.constant_label();
                    self.static_consts.push(StaticConstant {
                        name,
                        alignment: 8,
                        init: StaticInit::DoubleInit(n),
                    });

                    Operand::Data(name, 0)
                }
                // c is an integer
                c => c.unwrap_integer().into(),
            },
            Value::Var(name) => match self.symbols.get_expected_type(name.value) {
                CType::Array(_, _) => Operand::PseudoMem(name, 0),
                CType::Structure(_) => Operand::PseudoMem(name, 0),
                _ => Operand::Pseudo(name),
            },
        }
    }

    fn classify_params(
        &mut self,
        params: Vec<Value>,
        return_in_memory: bool,
    ) -> (Vec<TypedOperand>, Vec<Operand>, Vec<TypedOperand>) {
        use EightbyteClass::*;
        let mut int_reg_args = vec![];
        let mut double_reg_args = vec![];
        let mut stack_args = vec![];

        let int_regs_available = if return_in_memory { 5 } else { 6 };

        for p in params {
            let c_type = p.get_type(self.symbols);
            let struct_tag = if c_type.is_structure() {
                Some(*c_type.unwrap_structure_ref())
            } else {
                None
            };
            let t = AssemblyType::from_c_type(p.get_type(self.symbols), self.types);
            let operand = self.translate_value(p);

            if t == DoublePrecision {
                if double_reg_args.len() < 8 {
                    double_reg_args.push(operand);
                } else {
                    stack_args.push((operand, t));
                }
            } else if t.is_scalar() {
                if int_reg_args.len() < int_regs_available {
                    int_reg_args.push((operand, t));
                } else {
                    stack_args.push((operand, t));
                }
            } else {
                // we have a structure
                let classes = self
                    .types
                    .get_expected(&struct_tag.unwrap())
                    .classify(self.types);
                let mut use_stack = true;
                let struct_size = t.size() as usize;

                if classes[0] != MEMORY {
                    let mut tentative_ints = vec![];
                    let mut tentative_doubles = vec![];
                    let mut offset = 0;

                    for class in &classes {
                        let piece_operand = operand.offset_clone(offset);
                        if *class == SSE {
                            tentative_doubles.push(piece_operand);
                        } else {
                            tentative_ints
                                .push((piece_operand, get_eightbyte_type(offset, struct_size)));
                        }
                        offset += 8;
                    }

                    if tentative_doubles.len() + double_reg_args.len() <= 8
                        && tentative_ints.len() + int_reg_args.len() <= int_regs_available
                    {
                        double_reg_args.extend_from_slice(&tentative_doubles);
                        int_reg_args.extend_from_slice(&tentative_ints);
                        use_stack = false;
                    }
                }

                if use_stack {
                    let mut offset = 0;
                    for _ in classes {
                        stack_args.push((
                            operand.offset_clone(offset),
                            get_eightbyte_type(offset, struct_size),
                        ));
                        offset += 8;
                    }
                }
            }
        }

        (int_reg_args, double_reg_args, stack_args)
    }

    fn classify_return_value(&mut self, retval: Value) -> ReturnValue {
        let t = AssemblyType::from_c_type(retval.get_type(self.symbols), self.types);

        if t == DoublePrecision {
            ReturnValue::Registers(vec![], vec![self.translate_value(retval)])
        } else if t.is_scalar() {
            ReturnValue::Registers(vec![(self.translate_value(retval), t)], vec![])
        } else {
            let classes = self
                .types
                .get_expected(retval.get_type(self.symbols).unwrap_structure_ref())
                .classify(self.types);
            let struct_size = t.size() as usize;
            let operand = self.translate_value(retval);

            if classes[0] == EightbyteClass::MEMORY {
                ReturnValue::Memory
            } else {
                let mut int_retvals = vec![];
                let mut double_retvals = vec![];
                let mut offset = 0;

                for class in classes {
                    let piece_operand = operand.offset_clone(offset);
                    match class {
                        EightbyteClass::SSE => double_retvals.push(piece_operand),
                        EightbyteClass::INTEGER => int_retvals
                            .push((piece_operand, get_eightbyte_type(offset, struct_size))),
                        EightbyteClass::MEMORY => panic!(),
                    }
                    offset += 8;
                }
                ReturnValue::Registers(int_retvals, double_retvals)
            }
        }
    }

    fn translate_function_call(
        &mut self,
        name: Identifier,
        args: Vec<Value>,
        dest: Option<Value>,
        instructions: &mut Vec<Inst>,
    ) {
        let mut return_in_memory = false;
        let mut int_dests = vec![];
        let mut double_dests = vec![];

        if let Some(ref dest) = dest {
            match self.classify_return_value(*dest) {
                ReturnValue::Registers(int_ret_dests, double_ret_dests) => {
                    int_dests = int_ret_dests;
                    double_dests = double_ret_dests;
                }
                ReturnValue::Memory => {
                    return_in_memory = true;
                    instructions.push(Inst::Lea(self.translate_value(*dest), Register::DI.into()));
                }
            }
        }

        let (int_args, double_args, stack_args) = self.classify_params(args, return_in_memory);

        let stack_padding = if stack_args.len() % 2 == 0 { 0 } else { 8 };

        if stack_padding != 0 {
            instructions.push(binary(BinOp::Sub, Quadword, stack_padding, Register::SP));
        }

        for ((arg, asm_type), reg) in int_args.into_iter().zip(
            Register::INT_PARAM_REGS
                .iter()
                .skip(return_in_memory as usize),
        ) {
            if let ByteArray(size, _) = asm_type {
                copy_bytes_to_reg(arg, *reg, size, instructions);
            } else {
                instructions.push(mov(asm_type, arg, *reg));
            }
        }

        for (arg, reg) in double_args.into_iter().zip(Register::FP_PARAM_REGS.iter()) {
            instructions.push(mov(DoublePrecision, arg, *reg));
        }

        let stack_arg_length = stack_args.len();
        for (arg, asm_type) in stack_args.into_iter().rev() {
            if let ByteArray(..) = asm_type {
                instructions.push(binary(BinOp::Sub, Quadword, 8, Register::SP));
                copy_bytes(
                    asm_type,
                    arg,
                    Operand::Memory(Register::SP, 0),
                    instructions,
                );
            } else if asm_type == Quadword || asm_type == DoublePrecision {
                instructions.push(Inst::Push(arg));
            } else {
                match arg {
                    arg @ (Operand::Imm(_) | Operand::Reg(_)) => {
                        instructions.push(Inst::Push(arg));
                    }
                    arg => {
                        instructions.push(mov(asm_type, arg, Register::AX));
                        instructions.push(Inst::Push(Register::AX.into()));
                    }
                }
            }
        }

        instructions.push(Inst::Call(name));

        let bytes_to_remove = 8 * stack_arg_length as i64 + stack_padding;
        if bytes_to_remove != 0 {
            instructions.push(binary(BinOp::Add, Quadword, bytes_to_remove, Register::SP));
        }

        if dest.is_some() && !return_in_memory {
            for ((op, t), reg) in int_dests.into_iter().zip(Register::INT_RET_REGS.iter()) {
                if let ByteArray(size, _) = t {
                    copy_bytes_from_reg(*reg, op, size, instructions);
                } else {
                    instructions.push(mov(t, *reg, op));
                }
            }

            for (op, reg) in double_dests.into_iter().zip(Register::FP_RET_REGS.iter()) {
                instructions.push(mov(DoublePrecision, *reg, op));
            }
        }
    }

    fn translate_return_instruction(&mut self, value: Option<Value>, instructions: &mut Vec<Inst>) {
        if value.is_none() {
            instructions.push(Inst::Ret);
            return;
        }

        let retval = value.unwrap();

        match self.classify_return_value(retval) {
            ReturnValue::Memory => {
                instructions.push(mov(
                    Quadword,
                    Operand::Memory(Register::BP, -8),
                    Register::AX,
                ));
                copy_bytes(
                    AssemblyType::from_c_type(retval.get_type(self.symbols), self.types),
                    self.translate_value(retval),
                    Operand::Memory(Register::AX, 0),
                    instructions,
                );
            }
            ReturnValue::Registers(int_retvals, double_retvals) => {
                for ((operand, asm_type), register) in
                    int_retvals.into_iter().zip(Register::INT_RET_REGS.iter())
                {
                    if let ByteArray(size, _) = asm_type {
                        copy_bytes_to_reg(operand, *register, size, instructions);
                    } else {
                        instructions.push(mov(asm_type, operand, *register));
                    }
                }

                for (operand, register) in
                    double_retvals.into_iter().zip(Register::FP_RET_REGS.iter())
                {
                    instructions.push(mov(DoublePrecision, operand, *register));
                }
            }
        }
        instructions.push(Inst::Ret);
    }

    fn translate_instruction(&mut self, code: Instruction, instructions: &mut Vec<Inst>) {
        match code {
            Instruction::Return(val) => self.translate_return_instruction(val, instructions),
            Instruction::Copy(src, dst) => {
                copy_bytes(
                    AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                    self.translate_value(src),
                    self.translate_value(dst),
                    instructions,
                );
            }
            Instruction::Unary(ir::UnaryOperator::Not, src, dest) => {
                if src.get_type(self.symbols) == CType::Double {
                    instructions.push(binary(
                        BinOp::Xor,
                        DoublePrecision,
                        Register::XMM0,
                        Register::XMM0,
                    ));
                    instructions.push(cmp(
                        DoublePrecision,
                        self.translate_value(src),
                        Register::XMM0,
                    ))
                } else {
                    instructions.push(cmp(
                        AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                        0,
                        self.translate_value(src),
                    ));
                }
                instructions.push(mov(
                    AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                    0,
                    self.translate_value(dest),
                ));
                instructions.push(Inst::SetCC(ConditionCode::E, self.translate_value(dest)));
            }
            Instruction::Unary(operator, src, dest) => {
                if operator == ir::UnaryOperator::Negate
                    && (src.get_type(self.symbols) == CType::Double
                        || dest.get_type(self.symbols) == CType::Double)
                {
                    let const_name = self.constant_label();
                    self.static_consts.push(StaticConstant {
                        name: const_name,
                        alignment: 16,
                        init: StaticInit::DoubleInit(-0.0),
                    });

                    instructions.push(mov(
                        DoublePrecision,
                        self.translate_value(src),
                        self.translate_value(dest),
                    ));
                    instructions.push(binary(
                        BinOp::Xor,
                        DoublePrecision,
                        Operand::Data(const_name, 0),
                        self.translate_value(dest),
                    ));
                } else {
                    instructions.push(mov(
                        AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                        self.translate_value(src),
                        self.translate_value(dest),
                    ));
                    instructions.push(unary(
                        AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                        operator.into(),
                        self.translate_value(dest),
                    ));
                }
            }
            Instruction::Binary(
                operator @ (BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::GreaterThan
                | BinaryOp::GreaterOrEqual
                | BinaryOp::LessThan
                | BinaryOp::LessOrEqual),
                src1,
                src2,
                dest,
            ) => {
                let comparison_type = src1.get_type(self.symbols);
                let is_signed = comparison_type.is_signed();
                let comparison_type = AssemblyType::from_c_type(comparison_type, self.types);
                instructions.push(cmp(
                    comparison_type,
                    self.translate_value(src2),
                    self.translate_value(src1),
                ));
                instructions.push(mov(
                    AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                    0,
                    self.translate_value(dest),
                ));
                instructions.push(Inst::SetCC(
                    ConditionCode::for_operator(operator, is_signed),
                    self.translate_value(dest),
                ));
            }
            Instruction::Binary(
                operator @ (BinaryOp::Divide | BinaryOp::Remainder),
                src1,
                src2,
                dest,
            ) => {
                let src_type = src1.get_type(self.symbols);
                let is_signed = src_type.is_signed();
                let src_type = AssemblyType::from_c_type(src_type, self.types);

                if src_type == DoublePrecision {
                    instructions.push(mov(
                        DoublePrecision,
                        self.translate_value(src1),
                        self.translate_value(dest),
                    ));
                    instructions.push(binary(
                        BinOp::DivDouble,
                        DoublePrecision,
                        self.translate_value(src2),
                        self.translate_value(dest),
                    ));
                } else {
                    instructions.push(mov(src_type, self.translate_value(src1), Register::AX));

                    if is_signed {
                        instructions.push(Inst::Cdq(src_type));
                        instructions.push(Inst::Idiv(src_type, self.translate_value(src2)));
                    } else {
                        instructions.push(mov(src_type, 0, Register::DX));
                        instructions.push(Inst::Div(src_type, self.translate_value(src2)));
                    }

                    instructions.push(mov(
                        src_type,
                        match operator {
                            BinaryOp::Divide => Register::AX,
                            BinaryOp::Remainder => Register::DX,
                            _ => unreachable!(),
                        },
                        self.translate_value(dest),
                    ));
                }
            }
            Instruction::Binary(operator, src1, src2, dest) => {
                let src_type = src1.get_type(self.symbols);
                let is_signed = src_type.is_signed();
                let src_type = AssemblyType::from_c_type(src_type, self.types);
                instructions.push(mov(
                    src_type,
                    self.translate_value(src1),
                    self.translate_value(dest),
                ));
                instructions.push(binary(
                    translate_operator(operator, is_signed),
                    src_type,
                    self.translate_value(src2),
                    self.translate_value(dest),
                ));
            }
            Instruction::Jump(target) => instructions.push(Inst::Jmp(target)),
            Instruction::JumpIfZero(condition, target) => {
                if condition.get_type(self.symbols) == CType::Double {
                    instructions.push(binary(
                        BinOp::Xor,
                        DoublePrecision,
                        Register::XMM0,
                        Register::XMM0,
                    ));
                    instructions.push(cmp(
                        DoublePrecision,
                        self.translate_value(condition),
                        Register::XMM0,
                    ))
                } else {
                    instructions.push(cmp(
                        AssemblyType::from_c_type(condition.get_type(self.symbols), self.types),
                        0,
                        self.translate_value(condition),
                    ));
                }
                instructions.push(Inst::JmpCC(ConditionCode::E, target));
            }
            Instruction::JumpIfNotZero(condition, target) => {
                if condition.get_type(self.symbols) == CType::Double {
                    instructions.push(binary(
                        BinOp::Xor,
                        DoublePrecision,
                        Register::XMM0,
                        Register::XMM0,
                    ));
                    instructions.push(cmp(
                        DoublePrecision,
                        self.translate_value(condition),
                        Register::XMM0,
                    ))
                } else {
                    instructions.push(cmp(
                        AssemblyType::from_c_type(condition.get_type(self.symbols), self.types),
                        0,
                        self.translate_value(condition),
                    ));
                }
                instructions.push(Inst::JmpCC(ConditionCode::NE, target));
            }
            Instruction::Label(name) => instructions.push(Inst::Label(name)),
            Instruction::FunCall(name, args, dest) => {
                self.translate_function_call(name, args, dest, instructions);
            }
            Instruction::SignExtend(src, dest) => {
                instructions.push(Inst::Movsx(
                    AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                    AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                    self.translate_value(src),
                    self.translate_value(dest),
                ));
            }
            Instruction::Truncate(src, dest) => instructions.push(mov(
                AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                self.translate_value(src),
                self.translate_value(dest),
            )),
            Instruction::ZeroExtend(src, dest) => {
                instructions.push(Inst::MovZeroExtend(
                    AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                    AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                    self.translate_value(src),
                    self.translate_value(dest),
                ));
            }
            Instruction::DoubleToInt(src, dest) => {
                if dest.get_type(self.symbols).is_character() {
                    instructions.push(Inst::Cvttsd2si(
                        Longword,
                        self.translate_value(src),
                        Register::AX.into(),
                    ));
                    instructions.push(mov(Byte, Register::AX, self.translate_value(dest)));
                } else {
                    instructions.push(Inst::Cvttsd2si(
                        AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                        self.translate_value(src),
                        self.translate_value(dest),
                    ))
                }
            }
            Instruction::DoubleToUInt(src, dest) => match dest.get_type(self.symbols) {
                CType::UnsignedChar => {
                    instructions.push(Inst::Cvttsd2si(
                        Longword,
                        self.translate_value(src),
                        Register::AX.into(),
                    ));
                    instructions.push(mov(Byte, Register::AX, self.translate_value(dest)));
                }
                CType::Unsigned => {
                    instructions.push(Inst::Cvttsd2si(
                        Quadword,
                        self.translate_value(src),
                        Register::AX.into(),
                    ));
                    instructions.push(mov(Longword, Register::AX, self.translate_value(dest)));
                }
                _ => {
                    let const_name = self.constant_label();
                    self.static_consts.push(StaticConstant {
                        name: const_name,
                        alignment: 8,
                        init: StaticInit::DoubleInit(9223372036854775808.0),
                    });

                    instructions.push(cmp(
                        DoublePrecision,
                        Operand::Data(const_name, 0),
                        self.translate_value(src),
                    ));

                    let out_of_range_label = self.make_label("out_of_range");

                    instructions.push(Inst::JmpCC(ConditionCode::A, out_of_range_label));
                    instructions.push(Inst::Cvttsd2si(
                        Quadword,
                        self.translate_value(src),
                        self.translate_value(dest),
                    ));

                    let end_label = self.make_label("double_convert_end");
                    instructions.push(Inst::Jmp(end_label));
                    instructions.push(Inst::Label(out_of_range_label));
                    instructions.push(mov(
                        DoublePrecision,
                        self.translate_value(src),
                        Register::XMM1,
                    ));
                    instructions.push(binary(
                        BinOp::Sub,
                        DoublePrecision,
                        Operand::Data(const_name, 0),
                        Register::XMM1,
                    ));
                    instructions.push(Inst::Cvttsd2si(
                        Quadword,
                        Register::XMM1.into(),
                        self.translate_value(dest),
                    ));
                    instructions.push(binary(
                        BinOp::Add,
                        Quadword,
                        Operand::Imm(((i64::MAX as u64) + 1) as i64),
                        self.translate_value(dest),
                    ));
                    instructions.push(Inst::Label(end_label));
                }
            },
            Instruction::IntToDouble(src, dest) => {
                if src.get_type(self.symbols).is_character() {
                    instructions.push(Inst::Movsx(
                        Byte,
                        Longword,
                        self.translate_value(src),
                        Register::AX.into(),
                    ));
                    instructions.push(Inst::Cvtsi2sd(
                        Longword,
                        Register::AX.into(),
                        self.translate_value(dest),
                    ));
                } else {
                    instructions.push(Inst::Cvtsi2sd(
                        AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                        self.translate_value(src),
                        self.translate_value(dest),
                    ));
                }
            }
            Instruction::UIntToDouble(src, dest) => match src.get_type(self.symbols) {
                CType::UnsignedChar => {
                    instructions.push(Inst::MovZeroExtend(
                        Byte,
                        Longword,
                        self.translate_value(src),
                        Register::AX.into(),
                    ));
                    instructions.push(Inst::Cvtsi2sd(
                        Longword,
                        Register::AX.into(),
                        self.translate_value(dest),
                    ));
                }
                CType::Unsigned => {
                    instructions.push(Inst::MovZeroExtend(
                        Longword,
                        Quadword,
                        self.translate_value(src),
                        Register::AX.into(),
                    ));
                    instructions.push(Inst::Cvtsi2sd(
                        Quadword,
                        Register::AX.into(),
                        self.translate_value(dest),
                    ));
                }
                _ => {
                    instructions.push(cmp(Quadword, 0, self.translate_value(src)));
                    let out_of_range_label = self.make_label("out_of_range");
                    instructions.push(Inst::JmpCC(ConditionCode::L, out_of_range_label));
                    instructions.push(Inst::Cvtsi2sd(
                        Quadword,
                        self.translate_value(src),
                        self.translate_value(dest),
                    ));

                    let end_label = self.make_label("int_to_double_end");
                    instructions.push(Inst::Jmp(end_label));
                    instructions.push(Inst::Label(out_of_range_label));
                    instructions.push(mov(Quadword, self.translate_value(src), Register::AX));
                    instructions.push(mov(Quadword, Register::AX, Register::DX));
                    instructions.push(unary(Quadword, UnaryOp::Shr, Register::DX));
                    instructions.push(binary(BinOp::And, Quadword, 1, Register::AX));
                    instructions.push(binary(BinOp::Or, Quadword, Register::AX, Register::DX));
                    instructions.push(Inst::Cvtsi2sd(
                        Quadword,
                        Register::DX.into(),
                        Register::XMM0.into(),
                    ));
                    instructions.push(binary(
                        BinOp::Add,
                        DoublePrecision,
                        Register::XMM0,
                        Register::XMM0,
                    ));
                    instructions.push(mov(
                        DoublePrecision,
                        Register::XMM0,
                        self.translate_value(dest),
                    ));
                    instructions.push(Inst::Label(end_label));
                }
            },
            Instruction::GetAddress(src, dest) => {
                instructions.push(Inst::Lea(
                    self.translate_value(src),
                    self.translate_value(dest),
                ));
            }
            Instruction::Load(src, dest) => {
                instructions.push(mov(Quadword, self.translate_value(src), Register::DX));
                copy_bytes(
                    AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                    Operand::Memory(Register::DX, 0),
                    self.translate_value(dest),
                    instructions,
                );
            }
            Instruction::Store(src, dest) => {
                instructions.push(mov(Quadword, self.translate_value(dest), Register::DX));
                copy_bytes(
                    AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                    self.translate_value(src),
                    Operand::Memory(Register::DX, 0),
                    instructions,
                );
            }
            Instruction::CopyToOffset(src, dest, offset) => {
                copy_bytes(
                    AssemblyType::from_c_type(src.get_type(self.symbols), self.types),
                    self.translate_value(src),
                    Operand::PseudoMem(dest, offset as isize),
                    instructions,
                );
            }
            Instruction::CopyFromOffset(src, offset, dest) => copy_bytes(
                AssemblyType::from_c_type(dest.get_type(self.symbols), self.types),
                Operand::PseudoMem(src, offset as isize),
                self.translate_value(dest),
                instructions,
            ),
            Instruction::AddPtr(ptr, index, scale, dst) => {
                instructions.push(mov(Quadword, self.translate_value(ptr), Register::AX));
                match scale {
                    scale @ (1 | 2 | 4 | 8) => {
                        instructions.push(mov(Quadword, self.translate_value(index), Register::DX));
                        instructions.push(Inst::Lea(
                            Operand::Indexed(Register::AX, Register::DX, scale),
                            self.translate_value(dst),
                        ));
                    }
                    scale => match index {
                        Value::Constant(c) => instructions.push(Inst::Lea(
                            Operand::Memory(
                                Register::AX,
                                scale as isize * c.unwrap_integer() as isize,
                            ),
                            self.translate_value(dst),
                        )),
                        index @ Value::Var(_) => {
                            instructions.push(mov(
                                Quadword,
                                self.translate_value(index),
                                Register::DX,
                            ));
                            instructions.push(binary(
                                BinOp::Mult,
                                Quadword,
                                Operand::Imm(scale as i64),
                                Register::DX,
                            ));
                            instructions.push(Inst::Lea(
                                Operand::Indexed(Register::AX, Register::DX, 1),
                                self.translate_value(dst),
                            ))
                        }
                    },
                }
            }
        }
    }

    fn translate_params(
        &mut self,
        params: Vec<Value>,
        return_in_memory: bool,
        instructions: &mut Vec<Inst>,
    ) {
        let (int_reg_params, double_reg_params, stack_params) =
            self.classify_params(params, return_in_memory);

        if return_in_memory {
            instructions.push(mov(
                Quadword,
                Register::DI,
                Operand::Memory(Register::BP, -8),
            ));
        }

        for ((param, asm_type), reg) in int_reg_params.into_iter().zip(
            Register::INT_PARAM_REGS
                .iter()
                .skip(return_in_memory as usize),
        ) {
            if let ByteArray(size, _) = asm_type {
                copy_bytes_from_reg(*reg, param, size, instructions);
            } else {
                instructions.push(mov(asm_type, *reg, param));
            }
        }

        for (param, reg) in double_reg_params
            .into_iter()
            .zip(Register::FP_PARAM_REGS.iter())
        {
            instructions.push(mov(DoublePrecision, *reg, param));
        }

        let mut offset = 16;
        for (param, asm_type) in stack_params {
            copy_bytes(
                asm_type,
                Operand::Memory(Register::BP, offset),
                param,
                instructions,
            );
            offset += 8;
        }
    }

    fn translate_function(&mut self, code: ir::Function) -> Function {
        let aliased_vars = optimizer::address_taken_analysis(&code.body, self.symbols);

        let mut instructions = vec![];

        let return_in_memory = self
            .symbols
            .get(&code.name.value)
            .unwrap()
            .c_type
            .unwrap_function_ref()
            .ret
            .returns_in_memory(self.types);

        self.translate_params(code.params, return_in_memory, &mut instructions);

        for i in code.body {
            self.translate_instruction(i, &mut instructions);
        }

        Function {
            name: code.name,
            global: code.global,
            stack_offset: 0,
            instructions,
            callee_saved_regs: Default::default(),
            aliased_vars,
        }
    }

    fn translate_static_var(&mut self, code: ir::StaticVariable) -> StaticVariable {
        StaticVariable {
            identifier: code.name,
            global: code.global,
            alignment: code.var_type.alignment(self.types),
            init_list: code.init_list,
        }
    }

    fn translate_static_const(&mut self, code: ir::StaticConstant) -> StaticConstant {
        StaticConstant {
            name: code.name,
            alignment: code.var_type.alignment(self.types),
            init: code.init,
        }
    }

    fn translate_top_level(&mut self, code: ir::TopLevel) -> TopLevel {
        match code {
            ir::TopLevel::Fn(f) => self.translate_function(f).into(),
            ir::TopLevel::Var(v) => self.translate_static_var(v).into(),
            ir::TopLevel::Const(c) => self.translate_static_const(c).into(),
        }
    }

    pub fn translate_tacky(&mut self, code: ir::Program) -> Program {
        let mut top_level: Vec<_> = code
            .top_level
            .into_iter()
            .map(|t| self.translate_top_level(t))
            .collect();

        top_level.extend(
            mem::take(&mut self.static_consts)
                .into_iter()
                .map(|c| c.into()),
        );

        Program { top_level }
    }
}

pub fn translate(code: ir::Program, symbols: &SymbolTable, types: &TypeTable) -> Program {
    let mut generator = AsmGenerator::new(symbols, types);

    generator.translate_tacky(code)
}

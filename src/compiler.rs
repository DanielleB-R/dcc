use serde::Serialize;
use std::{fmt::Display, fs, process};

use crate::{
    backend_x64::{
        allocate_program, backend_table, emit_assembly, fixup_instructions,
        replace_pseudoregisters, translate_ir,
    },
    errors::CompilerError,
    lexer::lex_input,
    optimizer::optimize_program,
    parser::parse_tokens,
    semantic_analysis::{analyze_statements, resolve_variables, typecheck_program},
    tacky::tackify_program,
    OptimizationPasses,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Lexer,
    Parser,
    Validate,
    Tacky,
    Codegen,
    Complete,
}

fn write_debug_file(filename: &str, data: impl Serialize) {
    let _ = fs::write(filename, serde_json::to_vec_pretty(&data).unwrap());
}

fn write_debug_text_file(filename: &str, data: impl Display) {
    let _ = fs::write(filename, format!("{}", data));
}

fn preprocess_source(source_name: &str) -> std::io::Result<String> {
    let output = process::Command::new("gcc")
        .arg("-E")
        .arg("-P")
        .arg(source_name)
        .output()?;

    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        process::exit(-1);
    }

    Ok(String::from_utf8(output.stdout).unwrap())
}

pub fn compile(
    source_name: &str,
    stage: Stage,
    debug: bool,
    optimization_passes: OptimizationPasses,
) -> Result<(), CompilerError> {
    let source = preprocess_source(source_name)?;

    if debug {
        write_debug_text_file("preprocessed-source.i", &source);
    }

    let tokens = lex_input(&source)?;

    // println!(
    //     "[{}]",
    //     tokens
    //         .iter()
    //         .map(|t| format!("{}", t))
    //         .collect::<Vec<_>>()
    //         .join(", ")
    // );

    if stage == Stage::Lexer {
        process::exit(0);
    }

    let program = parse_tokens(tokens)?;

    if debug {
        write_debug_file("parsed-ast.json", &program);
        write_debug_text_file("parsed-ast.txt", &program);
    }

    if stage == Stage::Parser {
        process::exit(0);
    }

    let program = resolve_variables(program)?;

    if debug {
        write_debug_file("resolved-ast.json", &program);
        write_debug_text_file("resolved-ast.txt", &program);
    }

    let program = analyze_statements(program)?;

    if debug {
        write_debug_file("statement-analyzed-ast.json", &program);
    }

    let (program, mut symbol_table, type_table) = typecheck_program(program)?;

    if debug {
        write_debug_file("typechecked-ast.json", &program);
        write_debug_file("typechecked-symbol-table.json", &symbol_table);
        write_debug_file("type-table.json", &type_table);
        write_debug_text_file("typechecked-ast.txt", &program);
    }

    if stage == Stage::Validate {
        process::exit(0);
    }

    let tacky_program = tackify_program(program, &mut symbol_table, &type_table);

    if debug {
        write_debug_file("tacky.json", &tacky_program);
        write_debug_file("tacky-symbol-table.json", &symbol_table);
        write_debug_text_file("tacky.txt", &tacky_program);
    }

    if stage == Stage::Tacky {
        process::exit(0);
    }

    let tacky_program = optimize_program(tacky_program, &symbol_table, optimization_passes);

    if debug {
        write_debug_file("optimized-tacky.json", &tacky_program);
        write_debug_text_file("optimized-tacky.txt", &tacky_program);
    }

    let asm_program = translate_ir::translate(tacky_program, &symbol_table, &type_table);

    let symbols = backend_table::convert_table(symbol_table, &type_table);

    if debug {
        write_debug_text_file("raw_asm_ast.txt", &asm_program);
    }

    let asm_program = allocate_program(asm_program, &symbols);

    if debug {
        write_debug_text_file("allocated_asm_ast.txt", &asm_program);
    }

    let asm_program = fixup_instructions(replace_pseudoregisters(asm_program, &symbols));

    if debug {
        write_debug_text_file("processed_asm_ast.txt", &asm_program);
    }

    if stage == Stage::Codegen {
        process::exit(0);
    }

    let output = emit_assembly(asm_program, &symbols);

    let asm_name = source_name.replace(".c", ".s");

    fs::write(asm_name, output)?;

    Ok(())
}

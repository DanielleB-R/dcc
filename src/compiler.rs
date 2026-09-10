use std::process;

use crate::{
    backend_qbe::QbeBackend,
    backend_x64::X64Backend,
    common::{backend::Backend, write_debug_file, write_debug_text_file},
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum)]
pub enum BackendKind {
    #[default]
    X64,
    Qbe,
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
    backend_kind: BackendKind,
) -> Result<String, CompilerError> {
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

    // At this point, we have identified any C language errors in the source
    // that may be present. Any errors in the following parts represent
    // compiler bugs and will panic.

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

    match backend_kind {
        BackendKind::X64 => {
            let backend = X64Backend::new(debug, stage, source_name.to_owned());
            backend.emit(tacky_program, symbol_table, &type_table)
        }
        BackendKind::Qbe => {
            let backend = QbeBackend::new(source_name.to_owned());
            backend.emit(tacky_program, symbol_table, &type_table)
        }
    }
}

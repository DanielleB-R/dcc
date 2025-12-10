use clap::{Args, Parser};
use std::{env, process};

use dcc::{compile, errors::CompilerError, OptimizationPasses, Stage};

#[derive(Args, Debug)]
#[group(required = false, multiple = false)]
struct StageArgs {
    #[arg(long)]
    lex: bool,

    #[arg(long)]
    parse: bool,

    #[arg(long)]
    validate: bool,

    #[arg(long)]
    tacky: bool,

    #[arg(long)]
    codegen: bool,

    #[arg(long, short = 's')]
    skip: bool,

    #[arg(long, short)]
    compile: bool,
}

fn compiler_stage(args: &StageArgs) -> Stage {
    if args.lex {
        Stage::Lexer
    } else if args.parse {
        Stage::Parser
    } else if args.validate {
        Stage::Validate
    } else if args.tacky {
        Stage::Tacky
    } else if args.codegen {
        Stage::Codegen
    } else {
        Stage::Complete
    }
}

#[derive(Debug, Args)]
#[group(required = false, multiple = true)]
struct OptimizationArgs {
    #[arg(long)]
    fold_constants: bool,

    #[arg(long)]
    propagate_copies: bool,

    #[arg(long)]
    eliminate_unreachable_code: bool,

    #[arg(long)]
    eliminate_dead_stores: bool,

    #[arg(long, short = 'O')]
    optimize: bool,
}

fn compiler_optimizations(args: &OptimizationArgs) -> OptimizationPasses {
    let mut passes = OptimizationPasses::default();

    if args.fold_constants {
        passes.constant_folding = true;
    }
    if args.propagate_copies {
        passes.copy_propagation = true;
    }
    if args.eliminate_unreachable_code {
        passes.unreachable_code_elimination = true;
    }
    if args.eliminate_dead_stores {
        passes.dead_store_elimination = true;
    }
    if args.optimize {
        passes.constant_folding = true;
        passes.copy_propagation = true;
        passes.unreachable_code_elimination = true;
        passes.dead_store_elimination = true;
    }

    passes
}

fn assemble_source(asm_name: &str) -> std::io::Result<String> {
    let object_name = asm_name.replace(".s", ".o");

    let output = process::Command::new("gcc")
        .arg("-g")
        .arg("-c")
        .arg(asm_name)
        .arg("-o")
        .arg(&object_name)
        .output()?;

    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        process::exit(-1);
    }

    Ok(object_name)
}

fn compile_source(asm_name: &str, libraries: &[String]) -> std::io::Result<String> {
    let output_name = asm_name.replace(".s", "");

    let mut command = process::Command::new("gcc");

    command.arg("-g").arg(asm_name).arg("-o").arg(&output_name);

    for lib in libraries {
        command.arg(format!("-l{}", lib));
    }

    let output = command.output()?;

    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        process::exit(-1);
    }

    Ok(output_name)
}

#[derive(Debug, Parser)]
struct Options {
    #[command(flatten)]
    stage_args: StageArgs,

    source_name: String,

    #[arg(long)]
    debug: bool,

    #[command(flatten)]
    optimize_args: OptimizationArgs,

    #[arg(long, short)]
    library: Vec<String>,
}

fn main() -> Result<(), CompilerError> {
    let args = Options::parse();

    let source_name = args.source_name;
    let stage = compiler_stage(&args.stage_args);
    let mut debug = args.debug;
    let optimization_passes = compiler_optimizations(&args.optimize_args);

    let debug_var = env::var("DEBUG");
    if debug_var.is_ok_and(|s| s != "0") {
        debug = true;
    }

    let asm_name = compile(&source_name, stage, debug, optimization_passes).unwrap_or_else(|e| {
        eprintln!("{}", e);
        process::exit(1);
    });

    if args.stage_args.skip || stage != Stage::Complete {
        process::exit(0);
    }

    if args.stage_args.compile {
        assemble_source(&asm_name)?;
    } else {
        compile_source(&asm_name, &args.library)?;
    }

    Ok(())
}

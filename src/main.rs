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

#[derive(Debug, Parser)]
struct Options {
    #[command(flatten)]
    stage_args: StageArgs,

    source_name: String,

    #[arg(long)]
    debug: bool,

    #[arg(long)]
    optimize: Option<String>,
}

fn main() -> Result<(), CompilerError> {
    let args = Options::parse();

    let source_name = args.source_name;
    let stage = compiler_stage(&args.stage_args);
    let mut debug = args.debug;
    let mut optimization_passes: OptimizationPasses = Default::default();

    if let Some(o) = args.optimize {
        if o.contains("fold") {
            optimization_passes.constant_folding = true;
        }
        if o.contains("unreachable") {
            optimization_passes.unreachable_code_elimination = true;
        }
        if o.contains("propagate") {
            optimization_passes.copy_propagation = true;
        }
        if o.contains("dead-stores") {
            optimization_passes.dead_store_elimination = true;
        }
    }

    let debug_var = env::var("DEBUG");
    if debug_var.is_ok_and(|s| s != "0") {
        debug = true;
    }

    compile(&source_name, stage, debug, optimization_passes).unwrap_or_else(|e| {
        eprintln!("{}", e);
        process::exit(1);
    });

    Ok(())
}

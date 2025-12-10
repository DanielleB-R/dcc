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

#[derive(Debug, Parser)]
struct Options {
    #[command(flatten)]
    stage_args: StageArgs,

    source_name: String,

    #[arg(long)]
    debug: bool,

    #[command(flatten)]
    optimize_args: OptimizationArgs,
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

    compile(&source_name, stage, debug, optimization_passes).unwrap_or_else(|e| {
        eprintln!("{}", e);
        process::exit(1);
    });

    Ok(())
}

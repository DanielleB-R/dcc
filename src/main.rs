use std::{env, fs, process};

use my_c_compiler::{OptimizationPasses, Stage, compile, errors::CompilerError};

fn main() -> Result<(), CompilerError> {
    let args: Vec<_> = env::args().collect();

    let source_name: String;
    let stage: Stage;
    let mut debug = false;
    let mut optimization_passes: OptimizationPasses = Default::default();

    if args.len() == 1 {
        eprintln!("No source file provided");
        process::exit(2);
    } else if args.len() == 2 {
        let second = args.get(1).unwrap();
        if second.starts_with("--") {
            eprintln!("No source file provided");
            process::exit(2);
        }
        source_name = second.clone();
        stage = Stage::Complete;
    } else if args.len() == 3 {
        stage = match args.get(1).unwrap().as_str() {
            "--lex" => Stage::Lexer,
            "--parse" => Stage::Parser,
            "--validate" => Stage::Validate,
            "--tacky" => Stage::Tacky,
            "--codegen" => Stage::Codegen,
            "--debug" => {
                debug = true;
                Stage::Complete
            }
            o => {
                if o.starts_with("--optimize") {
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
                    Stage::Complete
                } else {
                    eprintln!("Unrecognized option {}", o);
                    process::exit(2);
                }
            }
        };
        source_name = args.get(2).unwrap().clone();
    } else {
        eprintln!("Too many arguments");
        process::exit(2);
    }

    let debug_var = env::var("DEBUG");
    if debug_var.is_ok_and(|s| s != "0") {
        debug = true;
    }

    let output = compile(&source_name, stage, debug, optimization_passes).unwrap_or_else(|e| {
        eprintln!("{}", e);
        process::exit(1);
    });

    let asm_name = source_name.replace(".i", ".s");

    if let Err(e) = fs::write(asm_name, output) {
        eprintln!("Error writing assembly file {}", e);
        process::exit(1);
    }

    Ok(())
}

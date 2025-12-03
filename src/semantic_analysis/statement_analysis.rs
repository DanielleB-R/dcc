use std::collections::HashMap;
use std::mem;

use super::visitor::StatementVisitor;
use crate::common::CodeLabel;
use crate::errors::SemanticAnalysisError;
use crate::parser::ast::*;

type Result<T> = std::result::Result<T, SemanticAnalysisError>;

struct LoopLabeller {
    label_count: usize,
}

impl LoopLabeller {
    fn new() -> Self {
        Self { label_count: 0 }
    }

    fn make_label(&mut self, tag: &'static str) -> CodeLabel {
        self.label_count += 1;
        CodeLabel {
            tag,
            counter: self.label_count,
        }
    }

    fn label_program(mut self, code: Program) -> Result<Program> {
        code.map(|decl| {
            decl.fn_map(|func| func.map_block(|block| self.visit_block(block, &mut (None, None))))
        })
    }
}

impl StatementVisitor for LoopLabeller {
    type ExtraParam = (Option<CodeLabel>, Option<CodeLabel>);
    type Error = SemanticAnalysisError;

    fn visit_break(
        &mut self,
        _: Option<CodeLabel>,
        (break_label, _): &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        match break_label {
            label @ Some(_) => Ok(Stmt::Break(*label)),
            None => Err(SemanticAnalysisError::BareBreak),
        }
    }

    fn visit_continue(
        &mut self,
        _: Option<CodeLabel>,
        (_, continue_label): &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        match continue_label {
            label @ Some(_) => Ok(Stmt::Continue(*label)),
            None => Err(SemanticAnalysisError::BareContinue),
        }
    }

    fn visit_while(
        &mut self,
        condition: Expression,
        body: Statement,
        _: Option<CodeLabel>,
        _: &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        let new_label = self.make_label(".while");
        Ok(Stmt::While(
            condition,
            self.visit_statement(body, &mut (Some(new_label), Some(new_label)))?,
            Some(new_label),
        ))
    }

    fn visit_do_while(
        &mut self,
        body: Statement,
        condition: Expression,
        _: Option<CodeLabel>,
        _: &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        let new_label = self.make_label(".dowhile");
        Ok(Stmt::DoWhile(
            self.visit_statement(body, &mut (Some(new_label), Some(new_label)))?,
            condition,
            Some(new_label),
        ))
    }

    fn visit_for(
        &mut self,
        init: Box<ForInit>,
        cond: Option<Expression>,
        incr: Option<Expression>,
        body: Statement,
        _: Option<CodeLabel>,
        _: &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        let new_label = self.make_label(".for");
        Ok(Stmt::For(
            init,
            cond,
            incr,
            self.visit_statement(body, &mut (Some(new_label), Some(new_label)))?,
            Some(new_label),
        ))
    }

    fn visit_switch(
        &mut self,
        expr: Expression,
        body: Statement,
        cases: Vec<CaseInfo>,
        default_label: Option<CodeLabel>,
        _: Option<CodeLabel>,
        (_, continue_label): &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        let switch_label = self.make_label(".switch");
        Ok(Stmt::Switch(
            expr,
            self.visit_statement(body, &mut (Some(switch_label), *continue_label))?,
            cases,
            default_label,
            Some(switch_label),
        ))
    }
}

fn label_loops(code: Program) -> Result<Program> {
    let labeller = LoopLabeller::new();

    labeller.label_program(code)
}

struct LabelUniquifier {
    label_index: usize,
}

impl LabelUniquifier {
    fn new() -> Self {
        Self { label_index: 0 }
    }

    fn uniquify_label(&mut self, label: CodeLabel) -> CodeLabel {
        self.label_index += 1;
        CodeLabel {
            tag: label.tag,
            counter: self.label_index,
        }
    }

    fn uniquify_program(&mut self, code: Program) -> Result<Program> {
        code.map(|decl| {
            decl.fn_map(|function| {
                let mut label_map = HashMap::new();
                function
                    .map_block(|block| self.visit_block(block, &mut label_map))?
                    .map_block(|block| GotoUniquifier.visit_block(block, &mut label_map))
            })
        })
    }
}

impl StatementVisitor for LabelUniquifier {
    type ExtraParam = HashMap<&'static str, CodeLabel>;
    type Error = SemanticAnalysisError;

    fn visit_statement(
        &mut self,
        mut statement: Statement,
        label_map: &mut Self::ExtraParam,
    ) -> std::result::Result<Statement, Self::Error> {
        let labels = mem::take(&mut statement.labels);

        statement.labels = labels
            .into_iter()
            .map(|label| {
                if let Label::Plain(label) = label {
                    if label_map.contains_key(label.tag) {
                        Err(SemanticAnalysisError::DuplicateLabel)
                    } else {
                        let unique_label = self.uniquify_label(label);
                        label_map.insert(label.tag, unique_label);
                        Ok(unique_label.into())
                    }
                } else {
                    Ok(label)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        statement.map(|stmt| self.visit_stmt(stmt, label_map))
    }
}

struct GotoUniquifier;

impl StatementVisitor for GotoUniquifier {
    type ExtraParam = HashMap<&'static str, CodeLabel>;
    type Error = SemanticAnalysisError;

    fn visit_goto(
        &mut self,
        label: CodeLabel,
        label_map: &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        if let Some(unique_label) = label_map.get(label.tag) {
            Ok(Stmt::Goto(*unique_label))
        } else {
            Err(SemanticAnalysisError::UnknownLabel)
        }
    }
}

fn uniquify_labels(code: Program) -> Result<Program> {
    let mut uniquifier = LabelUniquifier::new();

    uniquifier.uniquify_program(code)
}

struct SwitchCaseGatherer {
    label_index: usize,
}

impl SwitchCaseGatherer {
    fn new() -> Self {
        Self { label_index: 0 }
    }

    fn get_case_label(&mut self) -> CodeLabel {
        self.label_index += 1;
        CodeLabel {
            tag: ".switch.case",
            counter: self.label_index,
        }
    }

    fn get_default_label(&mut self) -> CodeLabel {
        self.label_index += 1;
        CodeLabel {
            tag: ".switch.default",
            counter: self.label_index,
        }
    }

    fn gather_program(mut self, code: Program) -> Result<Program> {
        code.map(|decl| {
            decl.fn_map(|func| func.map_block(|block| self.visit_block(block, &mut None)))
        })
    }
}

impl StatementVisitor for SwitchCaseGatherer {
    type ExtraParam = Option<(Vec<CaseInfo>, Option<CodeLabel>)>;
    type Error = SemanticAnalysisError;

    fn visit_switch(
        &mut self,
        expr: Expression,
        body: Statement,
        _: Vec<CaseInfo>,
        _: Option<CodeLabel>,
        label: Option<CodeLabel>,
        _: &mut Self::ExtraParam,
    ) -> std::result::Result<Stmt, Self::Error> {
        let cases = vec![];
        let default_label = None;
        let mut extra = Some((cases, default_label));

        let body = self.visit_statement(body, &mut extra)?;

        let extra = extra.unwrap();
        Ok(Stmt::Switch(expr, body, extra.0, extra.1, label))
    }

    fn visit_statement(
        &mut self,
        mut statement: Statement,
        extra: &mut Self::ExtraParam,
    ) -> std::result::Result<Statement, Self::Error> {
        let labels = mem::take(&mut statement.labels);

        statement.labels = labels
            .into_iter()
            .map(|label| match label {
                Label::Plain(l) => Ok(Label::Plain(l)),
                Label::Case(expr) => {
                    let line = expr.get_line();
                    match extra {
                        Some(extra) => {
                            if let Expr::Constant(c) = expr.unwrap() {
                                let new_label = self.get_case_label();
                                extra.0.push(CaseInfo {
                                    value: c,
                                    label: new_label,
                                });
                                Ok(Label::Plain(new_label))
                            } else {
                                Err(SemanticAnalysisError::NonConstantCase(line))
                            }
                        }
                        None => Err(SemanticAnalysisError::BareSwitchLabel),
                    }
                }
                Label::Default => match extra {
                    Some(extra) => {
                        if extra.1.is_some() {
                            Err(SemanticAnalysisError::DuplicateDefault)
                        } else {
                            let new_label = self.get_default_label();
                            extra.1 = Some(new_label);
                            Ok(Label::Plain(new_label))
                        }
                    }
                    None => Err(SemanticAnalysisError::BareSwitchLabel),
                },
            })
            .collect::<Result<Vec<_>>>()?;

        statement.map(|stmt| self.visit_stmt(stmt, extra))
    }
}

fn gather_switch_cases(code: Program) -> Result<Program> {
    SwitchCaseGatherer::new().gather_program(code)
}

pub fn analyze_statements(code: Program) -> Result<Program> {
    gather_switch_cases(uniquify_labels(label_loops(code)?)?)
}

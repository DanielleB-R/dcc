use std::error::Error;

use crate::common::CodeLabel;
use crate::parser::ast::{Block, CaseInfo, Expression, ForInit, Statement, Stmt};

pub trait StatementVisitor {
    type ExtraParam;
    type Error: Error;

    fn visit_return(
        &mut self,
        value: Option<Expression>,
        _extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::Return(value))
    }

    fn visit_expression_stmt(
        &mut self,
        expr: Expression,
        _extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::Expression(expr))
    }

    fn visit_if(
        &mut self,
        condition: Expression,
        then_body: Statement,
        else_body: Option<Statement>,
        extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::If(
            condition,
            self.visit_statement(then_body, extra)?,
            else_body
                .map(|body| self.visit_statement(body, extra))
                .transpose()?,
        ))
    }

    fn visit_compound(
        &mut self,
        body: Block,
        extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::Compound(self.visit_block(body, extra)?))
    }

    fn visit_break(
        &mut self,
        label: Option<CodeLabel>,
        _extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::Break(label))
    }

    fn visit_continue(
        &mut self,
        label: Option<CodeLabel>,
        _extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::Continue(label))
    }

    fn visit_while(
        &mut self,
        condition: Expression,
        body: Statement,
        label: Option<CodeLabel>,
        extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::While(
            condition,
            self.visit_statement(body, extra)?,
            label,
        ))
    }

    fn visit_do_while(
        &mut self,
        body: Statement,
        condition: Expression,
        label: Option<CodeLabel>,
        extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::DoWhile(
            self.visit_statement(body, extra)?,
            condition,
            label,
        ))
    }

    fn visit_for(
        &mut self,
        init: Box<ForInit>,
        cond: Option<Expression>,
        incr: Option<Expression>,
        body: Statement,
        label: Option<CodeLabel>,
        extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::For(
            init,
            cond,
            incr,
            self.visit_statement(body, extra)?,
            label,
        ))
    }

    fn visit_goto(
        &mut self,
        label: CodeLabel,
        _extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::Goto(label))
    }

    fn visit_switch(
        &mut self,
        expr: Expression,
        body: Statement,
        cases: Vec<CaseInfo>,
        default_label: Option<CodeLabel>,
        label: Option<CodeLabel>,
        extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        Ok(Stmt::Switch(
            expr,
            self.visit_statement(body, extra)?,
            cases,
            default_label,
            label,
        ))
    }

    fn visit_stmt(
        &mut self,
        stmt: Stmt,
        extra: &mut Self::ExtraParam,
    ) -> Result<Stmt, Self::Error> {
        match stmt {
            Stmt::Return(value) => self.visit_return(value, extra),
            Stmt::Expression(expr) => self.visit_expression_stmt(expr, extra),
            Stmt::If(cond, then_body, else_body) => {
                self.visit_if(cond, then_body, else_body, extra)
            }
            Stmt::Compound(body) => self.visit_compound(body, extra),
            Stmt::Break(label) => self.visit_break(label, extra),
            Stmt::Continue(label) => self.visit_continue(label, extra),
            Stmt::While(cond, body, label) => self.visit_while(cond, body, label, extra),
            Stmt::DoWhile(body, cond, label) => self.visit_do_while(body, cond, label, extra),
            Stmt::For(init, cond, incr, body, label) => {
                self.visit_for(init, cond, incr, body, label, extra)
            }
            Stmt::Goto(label) => self.visit_goto(label, extra),
            Stmt::Switch(expr, body, cases, default_label, label) => {
                self.visit_switch(expr, body, cases, default_label, label, extra)
            }
            Stmt::Null => Ok(Stmt::Null),
        }
    }

    fn visit_statement(
        &mut self,
        statement: Statement,
        extra: &mut Self::ExtraParam,
    ) -> Result<Statement, Self::Error> {
        statement.map(|s| self.visit_stmt(s, extra))
    }

    fn visit_block(
        &mut self,
        block: Block,
        extra: &mut Self::ExtraParam,
    ) -> Result<Block, Self::Error> {
        block.map_stmt(|stmt| self.visit_statement(stmt, extra))
    }
}

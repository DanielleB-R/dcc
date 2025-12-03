use std::collections::HashMap;

use crate::common::{CType, Identifier, ctype::FunctionType};
use crate::errors::SemanticAnalysisError;
use crate::parser::ast::*;

pub type Result<T> = std::result::Result<T, SemanticAnalysisError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Linkage {
    None,
    External,
}

trait Shadowable {
    fn shadow(&self) -> Self;
}

#[derive(Clone, Copy, Debug)]
struct IdentifierEntry {
    name: Identifier,
    from_current_scope: bool,
    linkage: Linkage,
}

impl Shadowable for IdentifierEntry {
    fn shadow(&self) -> Self {
        Self {
            name: self.name,
            from_current_scope: false,
            linkage: self.linkage,
        }
    }
}

type IdentifierMap = HashMap<&'static str, IdentifierEntry>;

#[derive(Clone, Copy, Debug)]
struct StructTagEntry {
    tag: Identifier,
    from_current_scope: bool,
}

impl Shadowable for StructTagEntry {
    fn shadow(&self) -> Self {
        Self {
            tag: self.tag,
            from_current_scope: false,
        }
    }
}

type StructTagMap = HashMap<&'static str, StructTagEntry>;

fn copy_shadowable_map<T: Shadowable>(map: &HashMap<&'static str, T>) -> HashMap<&'static str, T> {
    map.iter()
        .map(|(name, entry)| (*name, entry.shadow()))
        .collect()
}

fn resolve_type(c_type: CType, structure_map: &StructTagMap) -> Result<CType> {
    match c_type {
        CType::Structure(tag) => {
            if let Some(entry) = structure_map.get(&tag.value) {
                Ok(CType::Structure(entry.tag))
            } else {
                Err(SemanticAnalysisError::UndeclaredStruct)
            }
        }
        CType::Pointer(referenced) => {
            Ok(CType::pointer_to(resolve_type(*referenced, structure_map)?))
        }
        CType::Array(element, size) => Ok(CType::Array(
            Box::new(resolve_type(*element, structure_map)?),
            size,
        )),
        CType::Function(fn_type) => Ok(CType::Function(Box::new(FunctionType {
            params: fn_type
                .params
                .into_iter()
                .map(|t| resolve_type(t, structure_map))
                .collect::<Result<Vec<_>>>()?,
            ret: resolve_type(fn_type.ret, structure_map)?,
        }))),
        t => Ok(t),
    }
}

struct IdentifierResolver {
    name_counter: usize,
}

impl IdentifierResolver {
    fn new() -> Self {
        Self { name_counter: 0 }
    }

    fn uniquify_name(&mut self, name: &Identifier) -> Identifier {
        self.name_counter += 1;

        Identifier {
            value: format!("{}.{}", name.value, self.name_counter).leak(),
            line: name.line,
            location: name.location,
        }
    }

    fn resolve_optional_exp(
        &mut self,
        expr: Option<Expression>,
        identifier_map: &mut IdentifierMap,
        structure_map: &StructTagMap,
    ) -> Result<Option<Expression>> {
        expr.map(|e| self.resolve_exp(e, identifier_map, structure_map))
            .transpose()
    }

    fn resolve_exp(
        &mut self,
        expr: Expression,
        identifier_map: &mut IdentifierMap,
        structure_map: &StructTagMap,
    ) -> Result<Expression> {
        expr.map(|e, line| self.resolve_expr(e, identifier_map, structure_map, line))
    }

    fn resolve_expr(
        &mut self,
        expr: Expr,
        identifier_map: &mut IdentifierMap,
        structure_map: &StructTagMap,
        line: usize,
    ) -> Result<Expr> {
        match expr {
            Expr::Assignment(left, right) => Ok(Expr::Assignment(
                self.resolve_exp(left, identifier_map, structure_map)?,
                self.resolve_exp(right, identifier_map, structure_map)?,
            )),
            Expr::CompoundAssignment(operator, left, right, _) => Ok(Expr::CompoundAssignment(
                operator,
                self.resolve_exp(left, identifier_map, structure_map)?,
                self.resolve_exp(right, identifier_map, structure_map)?,
                None,
            )),
            Expr::Var(var) => {
                if let Some(IdentifierEntry {
                    name: unique_name, ..
                }) = identifier_map.get(&var.value)
                {
                    Ok(Expr::Var((*unique_name).relocate(&var)))
                } else {
                    Err(SemanticAnalysisError::UndeclaredVariable(
                        var.line,
                        var.location,
                    ))
                }
            }
            Expr::Unary(
                operator @ (UnaryOperator::PreIncrement | UnaryOperator::PreDecrement),
                operand,
            ) => Ok(Expr::Unary(
                operator,
                self.resolve_exp(operand, identifier_map, structure_map)?,
            )),
            Expr::Unary(operator, operand) => Ok(Expr::Unary(
                operator,
                self.resolve_exp(operand, identifier_map, structure_map)?,
            )),
            Expr::Binary(operator, left, right) => Ok(Expr::Binary(
                operator,
                self.resolve_exp(left, identifier_map, structure_map)?,
                self.resolve_exp(right, identifier_map, structure_map)?,
            )),
            Expr::Postfix(operator, operand) => Ok(Expr::Postfix(
                operator,
                self.resolve_exp(operand, identifier_map, structure_map)?,
            )),
            Expr::Conditional(cond, then_expr, else_expr) => Ok(Expr::Conditional(
                self.resolve_exp(cond, identifier_map, structure_map)?,
                self.resolve_exp(then_expr, identifier_map, structure_map)?,
                self.resolve_exp(else_expr, identifier_map, structure_map)?,
            )),
            Expr::FunctionCall(name, args) => {
                if let Some(IdentifierEntry { name: new_name, .. }) =
                    identifier_map.get(&name.value)
                {
                    Ok(Expr::FunctionCall(
                        *new_name,
                        args.into_iter()
                            .map(|arg| self.resolve_exp(arg, identifier_map, structure_map))
                            .collect::<Result<Vec<_>>>()?,
                    ))
                } else {
                    Err(SemanticAnalysisError::UndeclaredFunction(line))
                }
            }
            Expr::Cast(cast_type, cast_expr) => Ok(Expr::Cast(
                resolve_type(cast_type, structure_map)?,
                self.resolve_exp(cast_expr, identifier_map, structure_map)?,
            )),
            Expr::Dereference(inner) => Ok(Expr::Dereference(self.resolve_exp(
                inner,
                identifier_map,
                structure_map,
            )?)),
            Expr::AddrOf(inner) => Ok(Expr::AddrOf(self.resolve_exp(
                inner,
                identifier_map,
                structure_map,
            )?)),
            Expr::Subscript(arr, subscript) => Ok(Expr::Subscript(
                self.resolve_exp(arr, identifier_map, structure_map)?,
                self.resolve_exp(subscript, identifier_map, structure_map)?,
            )),
            Expr::SizeOf(inner) => Ok(Expr::SizeOf(self.resolve_exp(
                inner,
                identifier_map,
                structure_map,
            )?)),
            Expr::SizeOfT(t) => Ok(Expr::SizeOfT(resolve_type(t, structure_map)?)),
            Expr::Dot(expr, name) => Ok(Expr::Dot(
                self.resolve_exp(expr, identifier_map, structure_map)?,
                name,
            )),
            Expr::Arrow(expr, name) => Ok(Expr::Arrow(
                self.resolve_exp(expr, identifier_map, structure_map)?,
                name,
            )),
            e @ (Expr::Constant(_) | Expr::String(_)) => Ok(e),
        }
    }

    fn resolve_for_init(
        &mut self,
        init: ForInit,
        identifier_map: &mut IdentifierMap,
        structure_map: &StructTagMap,
    ) -> Result<Box<ForInit>> {
        Ok(Box::new(match init {
            ForInit::InitDecl(decl) => self
                .resolve_var_declaration(decl, identifier_map, structure_map)?
                .into(),
            ForInit::InitExp(expr) => self
                .resolve_optional_exp(expr, identifier_map, structure_map)?
                .into(),
        }))
    }

    fn resolve_statement(
        &mut self,
        statement: Statement,
        identifier_map: &mut IdentifierMap,
        structure_map: &mut StructTagMap,
    ) -> Result<Statement> {
        statement.map(|s| self.resolve_stmt(s, identifier_map, structure_map))
    }

    fn resolve_stmt(
        &mut self,
        statement: Stmt,
        identifier_map: &mut IdentifierMap,
        structure_map: &mut StructTagMap,
    ) -> Result<Stmt> {
        Ok(match statement {
            Stmt::Return(expr) => {
                Stmt::Return(self.resolve_optional_exp(expr, identifier_map, structure_map)?)
            }
            Stmt::Expression(expr) => {
                Stmt::Expression(self.resolve_exp(expr, identifier_map, structure_map)?)
            }
            Stmt::Null => Stmt::Null,
            Stmt::If(condition, then_stmt, else_stmt) => Stmt::If(
                self.resolve_exp(condition, identifier_map, structure_map)?,
                self.resolve_statement(then_stmt, identifier_map, structure_map)?,
                else_stmt
                    .map(|stmt| self.resolve_statement(stmt, identifier_map, structure_map))
                    .transpose()?,
            ),
            Stmt::Compound(block) => Stmt::Compound(self.resolve_block(
                block,
                &mut copy_shadowable_map(identifier_map),
                &mut copy_shadowable_map(structure_map),
            )?),
            Stmt::While(condition, body, ..) => Stmt::While(
                self.resolve_exp(condition, identifier_map, structure_map)?,
                self.resolve_statement(body, identifier_map, structure_map)?,
                None,
            ),
            Stmt::DoWhile(body, condition, ..) => Stmt::DoWhile(
                self.resolve_statement(body, identifier_map, structure_map)?,
                self.resolve_exp(condition, identifier_map, structure_map)?,
                None,
            ),
            Stmt::For(init, cond, incr, body, ..) => {
                let mut inner_identifier_map = copy_shadowable_map(identifier_map);
                let mut inner_structure_map = copy_shadowable_map(structure_map);
                Stmt::For(
                    self.resolve_for_init(*init, &mut inner_identifier_map, &inner_structure_map)?,
                    self.resolve_optional_exp(
                        cond,
                        &mut inner_identifier_map,
                        &inner_structure_map,
                    )?,
                    self.resolve_optional_exp(
                        incr,
                        &mut inner_identifier_map,
                        &inner_structure_map,
                    )?,
                    self.resolve_statement(
                        body,
                        &mut inner_identifier_map,
                        &mut inner_structure_map,
                    )?,
                    None,
                )
            }
            Stmt::Switch(cond, body, cases, default_label, label) => Stmt::Switch(
                self.resolve_exp(cond, identifier_map, structure_map)?,
                self.resolve_statement(body, identifier_map, structure_map)?,
                cases,
                default_label,
                label,
            ),
            s => s,
        })
    }

    fn resolve_struct_decl(
        &mut self,
        declaration: StructDeclaration,
        structure_map: &mut StructTagMap,
    ) -> Result<StructDeclaration> {
        let unique_tag = if let Some(prev_entry) = structure_map.get(&declaration.tag.value)
            && prev_entry.from_current_scope
        {
            prev_entry.tag
        } else {
            let unique_tag = self.uniquify_name(&declaration.tag);
            structure_map.insert(
                declaration.tag.value,
                StructTagEntry {
                    tag: unique_tag,
                    from_current_scope: true,
                },
            );
            unique_tag
        };

        let processed_members = declaration
            .members
            .into_iter()
            .map(|m| {
                Ok(MemberDeclaration {
                    name: m.name,
                    member_type: resolve_type(m.member_type, structure_map)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(StructDeclaration {
            tag: unique_tag,
            members: processed_members,
        })
    }

    fn resolve_file_scope_var_decl(
        &mut self,
        declaration: VarDeclaration,
        identifier_map: &mut IdentifierMap,
        structure_map: &mut StructTagMap,
    ) -> Result<VarDeclaration> {
        let new_type = resolve_type(declaration.var_type, structure_map)?;

        identifier_map.insert(
            declaration.name.value,
            IdentifierEntry {
                name: declaration.name,
                from_current_scope: true,
                linkage: Linkage::External,
            },
        );

        Ok(VarDeclaration {
            name: declaration.name,
            init: declaration.init,
            var_type: new_type,
            storage_class: declaration.storage_class,
        })
    }

    fn resolve_initializer(
        &mut self,
        initializer: Initializer,
        identifier_map: &mut IdentifierMap,
        structure_map: &StructTagMap,
    ) -> Result<Initializer> {
        if initializer.is_single() {
            Ok(self
                .resolve_exp(
                    initializer.unwrap().unwrap_single_init(),
                    identifier_map,
                    structure_map,
                )?
                .into())
        } else {
            Ok(initializer
                .unwrap()
                .unwrap_compound_init()
                .into_iter()
                .map(|i| self.resolve_initializer(i, identifier_map, structure_map))
                .collect::<Result<Vec<_>>>()?
                .into())
        }
    }

    fn resolve_var_declaration(
        &mut self,
        declaration: VarDeclaration,
        identifier_map: &mut IdentifierMap,
        structure_map: &StructTagMap,
    ) -> Result<VarDeclaration> {
        if let Some(entry) = identifier_map.get(&declaration.name.value)
            && entry.from_current_scope
            && !(entry.linkage == Linkage::External
                && declaration.storage_class == StorageClass::Extern)
        {
            return Err(SemanticAnalysisError::DuplicateDeclaration);
        }

        match declaration.storage_class {
            StorageClass::Extern => {
                let new_type = resolve_type(declaration.var_type, structure_map)?;

                identifier_map.insert(
                    declaration.name.value,
                    IdentifierEntry {
                        name: declaration.name,
                        from_current_scope: true,
                        linkage: Linkage::External,
                    },
                );

                Ok(VarDeclaration {
                    name: declaration.name,
                    init: declaration.init,
                    storage_class: declaration.storage_class,
                    var_type: new_type,
                })
            }
            _ => {
                let unique_name = self.uniquify_name(&declaration.name);
                identifier_map.insert(
                    declaration.name.value,
                    IdentifierEntry {
                        name: unique_name,
                        from_current_scope: true,
                        linkage: Linkage::None,
                    },
                );

                let new_type = resolve_type(declaration.var_type, structure_map)?;

                let init = declaration
                    .init
                    .map(|i| self.resolve_initializer(i, identifier_map, structure_map))
                    .transpose()?;

                Ok(VarDeclaration {
                    name: unique_name,
                    init,
                    var_type: new_type,
                    storage_class: declaration.storage_class,
                })
            }
        }
    }

    fn resolve_block(
        &mut self,
        code: Block,
        identifier_map: &mut IdentifierMap,
        structure_map: &mut StructTagMap,
    ) -> Result<Block> {
        code.map(|item| match item {
            BlockItem::S(stmt) => Ok(self
                .resolve_statement(stmt, identifier_map, structure_map)?
                .into()),
            BlockItem::D(decl) => Ok(self
                .resolve_declaration(decl, identifier_map, structure_map, false)?
                .into()),
        })
    }

    fn resolve_param(
        &mut self,
        code: Identifier,
        identifier_map: &mut IdentifierMap,
    ) -> Result<Identifier> {
        if let Some(IdentifierEntry {
            from_current_scope: true,
            ..
        }) = identifier_map.get(&code.value)
        {
            return Err(SemanticAnalysisError::DuplicateDeclaration);
        }

        let unique_name = self.uniquify_name(&code);
        identifier_map.insert(
            code.value,
            IdentifierEntry {
                name: unique_name,
                from_current_scope: true,
                linkage: Linkage::None,
            },
        );

        Ok(unique_name)
    }

    fn resolve_function(
        &mut self,
        code: FunctionDeclaration,
        identifier_map: &mut IdentifierMap,
        structure_map: &mut StructTagMap,
        top_level: bool,
    ) -> Result<FunctionDeclaration> {
        if let Some(IdentifierEntry {
            from_current_scope: true,
            linkage: Linkage::None,
            ..
        }) = identifier_map.get(&code.name.value)
        {
            return Err(SemanticAnalysisError::DuplicateDeclaration);
        }

        if !top_level && code.storage_class == StorageClass::Static {
            return Err(SemanticAnalysisError::StaticFunctionInBlock);
        }

        identifier_map.insert(
            code.name.value,
            IdentifierEntry {
                name: code.name,
                from_current_scope: true,
                linkage: Linkage::External,
            },
        );

        let mut inner_map = copy_shadowable_map(identifier_map);
        let mut inner_structure_map = copy_shadowable_map(structure_map);

        let new_type = resolve_type(code.fun_type.clone().into(), &inner_structure_map)?;

        let new_params = code
            .params
            .into_iter()
            .map(|param| self.resolve_param(param, &mut inner_map))
            .collect::<Result<Vec<_>>>()?;

        Ok(FunctionDeclaration {
            name: code.name,
            params: new_params,
            fun_type: *new_type.unwrap_function(),
            body: code
                .body
                .map(|block| self.resolve_block(block, &mut inner_map, &mut inner_structure_map))
                .transpose()?,
            storage_class: code.storage_class,
        })
    }

    fn resolve_declaration(
        &mut self,
        code: Declaration,
        identifier_map: &mut IdentifierMap,
        structure_map: &mut StructTagMap,
        top_level: bool,
    ) -> Result<Declaration> {
        match code {
            Declaration::Var(decl) => Ok(if top_level {
                self.resolve_file_scope_var_decl(decl, identifier_map, structure_map)?
            } else {
                self.resolve_var_declaration(decl, identifier_map, structure_map)?
            }
            .into()),
            Declaration::Fn(decl) => Ok(self
                .resolve_function(decl, identifier_map, structure_map, top_level)?
                .into()),
            Declaration::Struct(decl) => Ok(self.resolve_struct_decl(decl, structure_map)?.into()),
        }
    }

    fn resolve_program(mut self, code: Program) -> Result<Program> {
        let mut identifier_map = HashMap::new();
        let mut structure_map = HashMap::new();

        code.map(|decl| {
            self.resolve_declaration(decl, &mut identifier_map, &mut structure_map, true)
        })
    }
}

pub fn resolve_variables(code: Program) -> Result<Program> {
    let resolver = IdentifierResolver::new();

    resolver.resolve_program(code)
}

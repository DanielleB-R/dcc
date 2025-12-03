use super::parser::Parser;
use crate::{
    common::{CType, Identifier, ctype::FunctionType},
    errors::ParserError,
    lexer::token::TokenType::*,
};

pub type ParamList = Vec<(CType, Declarator)>;

/// Represents a C declarator (used in variable and function declarations)
#[derive(Debug, Clone)]
pub enum Declarator {
    Ident(Identifier),
    Pointer(Box<Declarator>),
    Array(Box<Declarator>, usize),
    Function(ParamList, Box<Declarator>),
}

impl Declarator {
    /// Parse a function parameter into a base type/declarator pair
    fn parse_param(parser: &mut Parser) -> Result<(CType, Declarator), ParserError> {
        if parser.peek().token_type.is_type() {
            Ok((parser.consume_and_parse_type()?, Self::parse(parser)?))
        } else {
            Err(ParserError::InvalidFunctionArgument(
                parser.peek().token_type,
                parser.peek().location,
            ))
        }
    }

    /// Parse the parameter list for a function into a list of base/declarator pairs
    fn parse_param_list(parser: &mut Parser) -> Result<ParamList, ParserError> {
        parser.expect(OpenParen)?;

        if parser.seeing(VoidKeyword) && parser.peek_next().token_type == CloseParen {
            parser.take();
            parser.take();
            Ok(vec![])
        } else {
            parser.list_of(Self::parse_param, Comma, CloseParen)
        }
    }

    fn parse_simple(parser: &mut Parser) -> Result<Self, ParserError> {
        match parser.peek().token_type {
            Identifier => Ok(Declarator::Ident(parser.take().into())),
            OpenParen => {
                parser.take();
                let declarator = Self::parse(parser)?;
                parser.expect(CloseParen)?;
                Ok(declarator)
            }
            _ => Err(ParserError::BadDeclarator(parser.peek().line)),
        }
    }

    fn parse_direct(parser: &mut Parser) -> Result<Self, ParserError> {
        let simple_declarator = Self::parse_simple(parser)?;

        match parser.peek().token_type {
            OpenParen => {
                let param_list = Self::parse_param_list(parser)?;
                Ok(Declarator::Function(
                    param_list,
                    Box::new(simple_declarator),
                ))
            }
            OpenBracket => {
                let mut declarator = simple_declarator;

                while parser.seeing(OpenBracket) {
                    parser.take();
                    if !parser.peek().token_type.is_integer_constant() {
                        return Err(ParserError::BadDeclarator(parser.peek().line));
                    }
                    let size = parser.constant()?.unwrap_integer();
                    parser.expect(CloseBracket)?;

                    declarator = Declarator::Array(Box::new(declarator), size as usize);
                }

                Ok(declarator)
            }
            _ => Ok(simple_declarator),
        }
    }

    pub fn parse(parser: &mut Parser) -> Result<Self, ParserError> {
        if parser.seeing(Star) {
            parser.take();
            Ok(Declarator::Pointer(Box::new(Self::parse(parser)?)))
        } else {
            Self::parse_direct(parser)
        }
    }

    // Convert the declarator to the concrete C type and identifier
    pub fn process(
        self,
        base_type: CType,
    ) -> Result<(Identifier, CType, Vec<Identifier>), ParserError> {
        match self {
            Declarator::Ident(name) => Ok((name, base_type, vec![])),
            Declarator::Pointer(d) => d.process(CType::pointer_to(base_type)),
            Declarator::Array(d, size) => d.process(CType::array_of(base_type, size)),
            Declarator::Function(params, inner) => {
                if let Declarator::Ident(name) = *inner {
                    let mut param_names = vec![];
                    let mut param_types = vec![];

                    for (p_base_type, p_declarator) in params {
                        let (param_name, param_type, _) = p_declarator.process(p_base_type)?;
                        if param_type.is_function() {
                            return Err(ParserError::NoFunctionParams);
                        }

                        param_names.push(param_name);
                        param_types.push(param_type);
                    }
                    let fun_type = FunctionType {
                        params: param_types,
                        ret: base_type,
                    };

                    Ok((name, fun_type.into(), param_names))
                } else {
                    Err(ParserError::NoFunctionDerivations)
                }
            }
        }
    }
}

/// Represents an abstract C declarator (used in casts)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractDeclarator {
    Pointer(Box<AbstractDeclarator>),
    Array(Box<AbstractDeclarator>, usize),
    Base,
}

impl AbstractDeclarator {
    fn parse_direct(parser: &mut Parser) -> Result<Self, ParserError> {
        match parser.peek().token_type {
            OpenParen => {
                parser.take();
                let mut declarator = Self::parse(parser)?;
                parser.expect(CloseParen)?;

                while parser.seeing(OpenBracket) {
                    parser.take();
                    if !parser.peek().token_type.is_integer_constant() {
                        return Err(ParserError::BadDeclarator(parser.peek().line));
                    }
                    let size = parser.constant()?.unwrap_integer();
                    parser.expect(CloseBracket)?;
                    declarator = AbstractDeclarator::Array(Box::new(declarator), size as usize);
                }

                Ok(declarator)
            }
            OpenBracket => {
                let mut declarator = AbstractDeclarator::Base;
                while parser.seeing(OpenBracket) {
                    parser.take();
                    if !parser.peek().token_type.is_integer_constant() {
                        return Err(ParserError::BadDeclarator(parser.peek().line));
                    }
                    let size = parser.constant()?.unwrap_integer();
                    parser.expect(CloseBracket)?;

                    declarator = AbstractDeclarator::Array(Box::new(declarator), size as usize)
                }
                Ok(declarator)
            }
            _ => Ok(AbstractDeclarator::Base),
        }
    }

    pub fn parse(parser: &mut Parser) -> Result<Self, ParserError> {
        if parser.seeing(Star) {
            parser.take();

            Ok(AbstractDeclarator::Pointer(Box::from(Self::parse(parser)?)))
        } else {
            Self::parse_direct(parser)
        }
    }

    /// Convert the abstract declarator to a concrete C type
    pub fn process(self, base_type: CType) -> CType {
        match self {
            Self::Base => base_type,
            Self::Pointer(p) => p.process(CType::pointer_to(base_type)),
            Self::Array(base, size) => base.process(CType::array_of(base_type, size)),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::lexer::logos_lexer::lex_input;

    fn parser_of(input: &str) -> Parser {
        Parser::new(lex_input(input).unwrap())
    }

    #[test]
    fn test_abstract_declarator_blank() {
        let input = "";

        let mut parser = parser_of(input);

        let abstract_declarator = AbstractDeclarator::parse(&mut parser).unwrap();

        let final_type = abstract_declarator.process(CType::Int);

        assert_eq!(final_type, CType::Int);
    }

    #[test]
    fn test_abstract_declarator_pointer() {
        let input = "*";

        let mut parser = parser_of(input);

        let abstract_declarator = AbstractDeclarator::parse(&mut parser).unwrap();

        assert_eq!(
            abstract_declarator,
            AbstractDeclarator::Pointer(Box::new(AbstractDeclarator::Base))
        );

        let final_type = abstract_declarator.process(CType::Int);

        assert_eq!(final_type, CType::pointer_to(CType::Int));
    }

    #[test]
    fn test_abstract_declarator_bracketed_pointer() {
        let input = "(*)";

        let mut parser = parser_of(input);

        let abstract_declarator = AbstractDeclarator::parse(&mut parser).unwrap();

        assert_eq!(
            abstract_declarator,
            AbstractDeclarator::Pointer(Box::new(AbstractDeclarator::Base))
        );

        let final_type = abstract_declarator.process(CType::Int);

        assert_eq!(final_type, CType::pointer_to(CType::Int));
    }

    #[test]
    fn test_abstract_declarator_array() {
        let input = "[4]";

        let mut parser = parser_of(input);

        let abstract_declarator = AbstractDeclarator::parse(&mut parser).unwrap();

        assert_eq!(
            abstract_declarator,
            AbstractDeclarator::Array(Box::new(AbstractDeclarator::Base), 4)
        );

        let final_type = abstract_declarator.process(CType::Int);

        assert_eq!(final_type, CType::array_of(CType::Int, 4));
    }

    #[test]
    fn test_abstract_declarator_array_of_pointers() {
        let input = "*[4]";

        let mut parser = parser_of(input);

        let abstract_declarator = AbstractDeclarator::parse(&mut parser).unwrap();

        assert_eq!(
            abstract_declarator,
            AbstractDeclarator::Pointer(Box::new(AbstractDeclarator::Array(
                Box::new(AbstractDeclarator::Base),
                4
            )),)
        );

        let final_type = abstract_declarator.process(CType::Int);

        assert_eq!(
            final_type,
            CType::array_of(CType::pointer_to(CType::Int), 4)
        );
    }

    #[test]
    fn test_abstract_declarator_pointer_to_array() {
        let input = "(*)[4]";

        let mut parser = parser_of(input);

        let abstract_declarator = AbstractDeclarator::parse(&mut parser).unwrap();

        assert_eq!(
            abstract_declarator,
            AbstractDeclarator::Array(
                Box::new(AbstractDeclarator::Pointer(Box::new(
                    AbstractDeclarator::Base
                ))),
                4
            )
        );

        let final_type = abstract_declarator.process(CType::Int);

        assert_eq!(
            final_type,
            CType::pointer_to(CType::array_of(CType::Int, 4))
        );
    }
}

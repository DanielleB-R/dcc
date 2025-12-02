use logos::{Lexer, Logos, Skip};

use super::token::{Token, TokenType};
use crate::errors::LexerError;

// This lexer deviates from the book by using the `logos` crate rather
// than directly using a regex engine. This is much faster than the approach
// in the book, as the `regex` crate does full runtime compilation of the
// regular expressions which can be expensive.
//
// The limited regex engine in logos does require some external logic
// to handle anything that requires lookahead, hence the following functions.

/// Update the line count and the char index when we hit a newline.
fn newline_callback(lex: &mut Lexer<LogosToken>) -> Skip {
    lex.extras.0 += 1;
    lex.extras.1 = lex.span().end;
    Skip
}

fn is_ascii_word_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

// Ensure that we only accept a token when it's followed by a word break.
fn word_break_callback(lex: &mut Lexer<LogosToken>) -> Option<String> {
    let next_char = lex.remainder().as_bytes().first()?;
    if is_ascii_word_char(*next_char) {
        None
    } else {
        Some(lex.slice().to_owned())
    }
}

fn number_break_callback(lex: &mut Lexer<LogosToken>) -> Option<String> {
    let next_char = lex.remainder().as_bytes().first()?;
    if is_ascii_word_char(*next_char) || *next_char == b'.' {
        None
    } else {
        Some(lex.slice().to_owned())
    }
}

fn number_break_one_callback(lex: &mut Lexer<LogosToken>) -> Option<String> {
    let next_char = lex.remainder().as_bytes().first()?;
    if is_ascii_word_char(*next_char) || *next_char == b'.' {
        None
    } else {
        let s = lex.slice();
        Some(s[..(s.len() - 1)].to_owned())
    }
}

fn number_break_two_callback(lex: &mut Lexer<LogosToken>) -> Option<String> {
    let next_char = lex.remainder().as_bytes().first()?;
    if is_ascii_word_char(*next_char) || *next_char == b'.' {
        None
    } else {
        let s = lex.slice();
        Some(s[..(s.len() - 2)].to_owned())
    }
}

// Remove the surrounding characters from the found token (e.g. for
// string literals)
fn wrapped_callback(lex: &mut Lexer<LogosToken>) -> String {
    let s = lex.slice();
    s[1..(s.len() - 1)].to_owned()
}

#[derive(Logos, Debug)]
#[logos(extras = (usize, usize))]
#[logos(skip(r"\n", newline_callback))]
#[logos(skip(r"[ \t]+"))]
enum LogosToken {
    #[regex(r"[a-zA-Z_][0-9a-zA-Z_]*", word_break_callback)]
    Identifier(String),
    #[regex(r"([0-9]+)", number_break_callback)]
    IntConstant(String),
    #[regex(r"([0-9]+[lL])", number_break_one_callback)]
    LongConstant(String),
    #[regex(r"([0-9]+[uU])", number_break_one_callback)]
    UnsignedConstant(String),
    #[regex(r"[0-9]+([lL][uU]|[uU][lL])", number_break_two_callback)]
    UnsignedLongConstant(String),
    #[regex(
        r"(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)",
        number_break_callback
    )]
    FloatingPointConstant(String),
    #[regex(r#"'([^'\\\n]|\\['"?\\abfnrtv])'"#, wrapped_callback)]
    CharConstant(String),
    #[regex(r#""([^"\\\n]|\\['"\\?abfnrtv])*""#, wrapped_callback)]
    StringLiteral(String),

    #[token("break")]
    BreakKeyword,
    #[token("case")]
    CaseKeyword,
    #[token("char")]
    CharKeyword,
    #[token("continue")]
    ContinueKeyword,
    #[token("default")]
    DefaultKeyword,
    #[token("do")]
    DoKeyword,
    #[token("double")]
    DoubleKeyword,
    #[token("else")]
    ElseKeyword,
    #[token("extern")]
    ExternKeyword,
    #[token("for")]
    ForKeyword,
    #[token("goto")]
    GotoKeyword,
    #[token("if")]
    IfKeyword,
    #[token("int")]
    IntKeyword,
    #[token("long")]
    LongKeyword,
    #[token("return")]
    ReturnKeyword,
    #[token("short")]
    ShortKeyword,
    #[token("signed")]
    SignedKeyword,
    #[token("sizeof")]
    SizeofKeyword,
    #[token("static")]
    StaticKeyword,
    #[token("struct")]
    StructKeyword,
    #[token("switch")]
    SwitchKeyword,
    #[token("unsigned")]
    UnsignedKeyword,
    #[token("void")]
    VoidKeyword,
    #[token("while")]
    WhileKeyword,

    #[token("(")]
    OpenParen,
    #[token(")")]
    CloseParen,
    #[token("{")]
    OpenBrace,
    #[token("}")]
    CloseBrace,
    #[token("[")]
    OpenBracket,
    #[token("]")]
    CloseBracket,

    #[token(";")]
    Semicolon,

    #[token("&")]
    And,
    #[token("&&")]
    AndAnd,
    #[token("&=")]
    AndEqual,
    #[token("->")]
    Arrow,
    #[token("!")]
    Bang,
    #[token("!=")]
    BangEqual,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("=")]
    Equal,
    #[token("==")]
    EqualEqual,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    #[token(">>")]
    GreaterGreater,
    #[token(">>=")]
    GreaterGreaterEqual,
    #[token("^")]
    Hat,
    #[token("^=")]
    HatEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,
    #[token("<<")]
    LessLess,
    #[token("<<=")]
    LessLessEqual,
    #[token("-")]
    Minus,
    #[token("-=")]
    MinusEqual,
    #[token("--")]
    MinusMinus,
    #[token("%")]
    Percent,
    #[token("%=")]
    PercentEqual,
    #[token("|")]
    Pipe,
    #[token("|=")]
    PipeEqual,
    #[token("||")]
    PipePipe,
    #[token("+")]
    Plus,
    #[token("+=")]
    PlusEqual,
    #[token("++")]
    PlusPlus,
    #[token("?")]
    Question,
    #[token("/")]
    Slash,
    #[token("/=")]
    SlashEqual,
    #[token("*")]
    Star,
    #[token("*=")]
    StarEqual,
    #[token("~")]
    Tilde,
}

impl LogosToken {
    fn get_string(self) -> Option<String> {
        match self {
            Self::Identifier(s)
            | Self::IntConstant(s)
            | Self::LongConstant(s)
            | Self::UnsignedConstant(s)
            | Self::UnsignedLongConstant(s)
            | Self::FloatingPointConstant(s)
            | Self::CharConstant(s)
            | Self::StringLiteral(s) => Some(s),
            _ => None,
        }
    }

    fn get_token_type(&self) -> TokenType {
        match self {
            Self::Identifier(_) => TokenType::Identifier,
            Self::IntConstant(_) => TokenType::IntConstant,
            Self::LongConstant(_) => TokenType::LongConstant,
            Self::UnsignedConstant(_) => TokenType::UnsignedConstant,
            Self::UnsignedLongConstant(_) => TokenType::UnsignedLongConstant,
            Self::FloatingPointConstant(_) => TokenType::FloatingPointConstant,
            Self::CharConstant(_) => TokenType::CharConstant,
            Self::StringLiteral(_) => TokenType::StringLiteral,

            Self::BreakKeyword => TokenType::BreakKeyword,
            Self::CaseKeyword => TokenType::CaseKeyword,
            Self::CharKeyword => TokenType::CharKeyword,
            Self::ContinueKeyword => TokenType::ContinueKeyword,
            Self::DefaultKeyword => TokenType::DefaultKeyword,
            Self::DoKeyword => TokenType::DoKeyword,
            Self::DoubleKeyword => TokenType::DoubleKeyword,
            Self::ElseKeyword => TokenType::ElseKeyword,
            Self::ExternKeyword => TokenType::ExternKeyword,
            Self::ForKeyword => TokenType::ForKeyword,
            Self::GotoKeyword => TokenType::GotoKeyword,
            Self::IfKeyword => TokenType::IfKeyword,
            Self::IntKeyword => TokenType::IntKeyword,
            Self::LongKeyword => TokenType::LongKeyword,
            Self::ReturnKeyword => TokenType::ReturnKeyword,
            Self::ShortKeyword => TokenType::ShortKeyword,
            Self::SignedKeyword => TokenType::SignedKeyword,
            Self::SizeofKeyword => TokenType::SizeofKeyword,
            Self::StaticKeyword => TokenType::StaticKeyword,
            Self::StructKeyword => TokenType::StructKeyword,
            Self::SwitchKeyword => TokenType::SwitchKeyword,
            Self::UnsignedKeyword => TokenType::UnsignedKeyword,
            Self::VoidKeyword => TokenType::VoidKeyword,
            Self::WhileKeyword => TokenType::WhileKeyword,

            Self::OpenParen => TokenType::OpenParen,
            Self::CloseParen => TokenType::CloseParen,
            Self::OpenBrace => TokenType::OpenBrace,
            Self::CloseBrace => TokenType::CloseBrace,
            Self::OpenBracket => TokenType::OpenBracket,
            Self::CloseBracket => TokenType::CloseBracket,

            Self::Semicolon => TokenType::Semicolon,

            Self::And => TokenType::And,
            Self::AndAnd => TokenType::AndAnd,
            Self::AndEqual => TokenType::AndEqual,
            Self::Arrow => TokenType::Arrow,
            Self::Bang => TokenType::Bang,
            Self::BangEqual => TokenType::BangEqual,
            Self::Colon => TokenType::Colon,
            Self::Comma => TokenType::Comma,
            Self::Dot => TokenType::Dot,
            Self::Equal => TokenType::Equal,
            Self::EqualEqual => TokenType::EqualEqual,
            Self::Greater => TokenType::Greater,
            Self::GreaterEqual => TokenType::GreaterEqual,
            Self::GreaterGreater => TokenType::GreaterGreater,
            Self::GreaterGreaterEqual => TokenType::GreaterGreaterEqual,
            Self::Hat => TokenType::Hat,
            Self::HatEqual => TokenType::HatEqual,
            Self::Less => TokenType::Less,
            Self::LessEqual => TokenType::LessEqual,
            Self::LessLess => TokenType::LessLess,
            Self::LessLessEqual => TokenType::LessLessEqual,
            Self::Minus => TokenType::Minus,
            Self::MinusEqual => TokenType::MinusEqual,
            Self::MinusMinus => TokenType::MinusMinus,
            Self::Percent => TokenType::Percent,
            Self::PercentEqual => TokenType::PercentEqual,
            Self::Pipe => TokenType::Pipe,
            Self::PipeEqual => TokenType::PipeEqual,
            Self::PipePipe => TokenType::PipePipe,
            Self::Plus => TokenType::Plus,
            Self::PlusEqual => TokenType::PlusEqual,
            Self::PlusPlus => TokenType::PlusPlus,
            Self::Question => TokenType::Question,
            Self::Slash => TokenType::Slash,
            Self::SlashEqual => TokenType::SlashEqual,
            Self::Star => TokenType::Star,
            Self::StarEqual => TokenType::StarEqual,
            Self::Tilde => TokenType::Tilde,
        }
    }
}

pub fn lex_input(input: &str) -> Result<Vec<Token>, LexerError> {
    let mut lex = LogosToken::lexer(input);

    let mut result = vec![];

    while let Some(token) = lex.next() {
        let line = lex.extras.0 + 1;
        let location = lex.span().start - lex.extras.1;
        match token {
            Ok(token) => {
                result.push(Token::new(
                    token.get_token_type(),
                    token.get_string(),
                    line,
                    location,
                ));
            }
            Err(_) => {
                return Err(LexerError::NoToken(
                    line,
                    lex.remainder().as_bytes()[0] as char,
                ));
            }
        }
    }

    result.push(Token::new(TokenType::EOF, None, 0, 0));

    Ok(result)
}

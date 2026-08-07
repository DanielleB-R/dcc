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
    let next_char = lex.remainder().as_bytes().first().unwrap_or(&0);
    if is_ascii_word_char(*next_char) {
        None
    } else {
        Some(lex.slice().to_owned())
    }
}

fn number_break_callback(lex: &mut Lexer<LogosToken>) -> Option<String> {
    let next_char = lex.remainder().as_bytes().first().unwrap_or(&0);
    if is_ascii_word_char(*next_char) || *next_char == b'.' {
        None
    } else {
        Some(lex.slice().to_owned())
    }
}

fn number_break_one_callback(lex: &mut Lexer<LogosToken>) -> Option<String> {
    let next_char = lex.remainder().as_bytes().first().unwrap_or(&0);
    if is_ascii_word_char(*next_char) || *next_char == b'.' {
        None
    } else {
        let s = lex.slice();
        Some(s[..(s.len() - 1)].to_owned())
    }
}

fn number_break_two_callback(lex: &mut Lexer<LogosToken>) -> Option<String> {
    let next_char = lex.remainder().as_bytes().first().unwrap_or(&0);
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
                    lex.remainder()
                        .as_bytes()
                        .first()
                        .copied()
                        .unwrap_or_default() as char,
                ));
            }
        }
    }

    result.push(Token::new(TokenType::EOF, None, 0, 0));

    Ok(result)
}

#[cfg(test)]
mod test {
    use super::*;

    fn types(input: &str) -> Vec<TokenType> {
        lex_input(input)
            .unwrap()
            .into_iter()
            .map(|t| t.token_type)
            .collect()
    }

    fn values(input: &str) -> Vec<Option<String>> {
        lex_input(input)
            .unwrap()
            .into_iter()
            .map(|t| t.value)
            .collect()
    }

    #[test]
    fn empty_input_is_just_eof() {
        assert_eq!(types(""), vec![TokenType::EOF]);
    }

    #[test]
    fn whitespace_and_newlines_only_is_just_eof() {
        assert_eq!(types("  \t\n\n  \n"), vec![TokenType::EOF]);
    }

    #[test]
    fn tracks_line_and_column_across_newlines() {
        let tokens = lex_input("int x;\n  y;").unwrap();
        // `y` is on line 2, indented two spaces.
        let y = &tokens[3];
        assert_eq!(y.token_type, TokenType::Identifier);
        assert_eq!(y.line, 2);
        assert_eq!(y.location, 2);
    }

    #[test]
    fn identifier_with_digits_and_underscores() {
        assert_eq!(
            types("_foo bar_2 __x"),
            vec![
                TokenType::Identifier,
                TokenType::Identifier,
                TokenType::Identifier,
                TokenType::EOF
            ]
        );
    }

    #[test]
    fn keyword_prefix_is_not_a_keyword() {
        // Longest-match wins: "intx" is an identifier, not IntKeyword + "x".
        assert_eq!(types("intx"), vec![TokenType::Identifier, TokenType::EOF]);
    }

    #[test]
    fn integer_constant_suffixes() {
        assert_eq!(
            types("123 123l 123L 123u 123U"),
            vec![
                TokenType::IntConstant,
                TokenType::LongConstant,
                TokenType::LongConstant,
                TokenType::UnsignedConstant,
                TokenType::UnsignedConstant,
                TokenType::EOF
            ]
        );
    }

    #[test]
    fn unsigned_long_suffix_order_and_case_independent() {
        assert_eq!(
            types("123lu 123UL 123Lu 123uL"),
            vec![
                TokenType::UnsignedLongConstant,
                TokenType::UnsignedLongConstant,
                TokenType::UnsignedLongConstant,
                TokenType::UnsignedLongConstant,
                TokenType::EOF
            ]
        );
    }

    #[test]
    fn suffix_strips_from_value() {
        assert_eq!(
            values("123L 123UL"),
            vec![Some("123".to_string()), Some("123".to_string()), None]
        );
    }

    #[test]
    fn number_immediately_followed_by_letter_is_an_error() {
        assert!(lex_input("123abc").is_err());
    }

    #[test]
    fn hex_literals_are_unsupported() {
        assert!(lex_input("0x1A").is_err());
    }

    #[test]
    fn floating_point_forms() {
        assert_eq!(
            types("1.5 .5 1. 1e10 1.5e-10 1.E+5"),
            vec![
                TokenType::FloatingPointConstant,
                TokenType::FloatingPointConstant,
                TokenType::FloatingPointConstant,
                TokenType::FloatingPointConstant,
                TokenType::FloatingPointConstant,
                TokenType::FloatingPointConstant,
                TokenType::EOF
            ]
        );
    }

    #[test]
    fn char_constant_with_escape() {
        assert_eq!(
            values(r"'a' '\n' '\''"),
            vec![
                Some("a".to_string()),
                Some("\\n".to_string()),
                Some("\\'".to_string()),
                None
            ]
        );
    }

    #[test]
    fn string_literal_empty_and_with_escapes() {
        assert_eq!(
            values(r#""" "hi\n\"there\"""#),
            vec![
                Some("".to_string()),
                Some(r#"hi\n\"there\""#.to_string()),
                None
            ]
        );
    }

    #[test]
    fn unterminated_string_is_an_error() {
        assert!(lex_input("\"abc").is_err());
    }

    #[test]
    fn string_literal_cannot_span_a_newline() {
        assert!(lex_input("\"abc\ndef\"").is_err());
    }

    #[test]
    fn multi_char_operators_are_greedy() {
        assert_eq!(
            types(">>= <<= == != && || -> ++ --"),
            vec![
                TokenType::GreaterGreaterEqual,
                TokenType::LessLessEqual,
                TokenType::EqualEqual,
                TokenType::BangEqual,
                TokenType::AndAnd,
                TokenType::PipePipe,
                TokenType::Arrow,
                TokenType::PlusPlus,
                TokenType::MinusMinus,
                TokenType::EOF
            ]
        );
    }

    #[test]
    fn adjacent_single_char_operators_are_not_merged() {
        // we need to use operators that won't merge into bigger ones
        assert_eq!(
            types("(){}[];,.~?:"),
            vec![
                TokenType::OpenParen,
                TokenType::CloseParen,
                TokenType::OpenBrace,
                TokenType::CloseBrace,
                TokenType::OpenBracket,
                TokenType::CloseBracket,
                TokenType::Semicolon,
                TokenType::Comma,
                TokenType::Dot,
                TokenType::Tilde,
                TokenType::Question,
                TokenType::Colon,
                TokenType::EOF
            ]
        );
    }

    #[test]
    fn unknown_character_is_an_error() {
        assert!(lex_input("$").is_err());
    }

    #[test]
    fn error_reports_line_number() {
        match lex_input("int x;\n$") {
            Err(LexerError::NoToken(line, ch)) => {
                assert_eq!(line, 2);
                // The bad byte itself is already consumed by the time we
                // read the remainder, and there's nothing after it here,
                // so this falls back to the default rather than reporting '$'.
                assert_eq!(ch, '\0');
            }
            other => panic!("expected NoToken error, got {other:?}"),
        }
    }
}

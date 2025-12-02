use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum LexerError {
    #[error("No matching token at char {0} ({1})")]
    NoToken(usize, char),
}

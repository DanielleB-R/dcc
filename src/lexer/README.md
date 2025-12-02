# lexer

This directory has the lexer for the C compiler.

This lexer deviates from the book by using the `logos` crate rather than directly using a regex engine. This is much faster than the approach in the book, as the `regex` crate does full runtime compilation of the regular expressions which can be expensive.

## Representation

While the `logos` crate used for lexing represents tokens with content as tuple variants, it is more convenient for parsing to have them represented as a token type with separate content. This leads to some repetitive boilerplate code in the lexer but allows us to write convenient utilities in the parser.

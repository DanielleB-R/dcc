# dcc

A C compiler based on Nora Sandler's book [Writing a C Compiler](https://norasandler.com/book/). It implements all 20 chapters of the book and all of the extra credit in Part I.

## Design Notes

### Overall

I have used the `thiserror` library to create error enums for the various fallible stages of the compiler. They include line information where that's available. The type checker has enough distinct error conditions that I stopped creating variants; instead creating a `MiscError` variant with a supplied error message. The `derive_more` crate is also used to cut down on boilerplate for handling enums.

In lieu of a proper interning framework, identifiers are converted to `&'static str` with the `.leak` method; since we tear down the compiler for each source file this will not leak excessive amounts of memory. As a result many more data types can be `Copy` and not require string cloning or reference counting to be referred to in multiple locations.

For debugging, all source reprensentations implement both `Display` (for a compact custom debug representation) and `serde::Serialize` (to examine a full 1:1 JSON dump of the data structures). Debug information is dumped to a collection of files if the compiler is invoked with the `DEBUG` environment variable set to a non-zero value.

Symbol tables are passed into the various passes explicitly to avoid difficulties with Rust's support for global data.

### Lexer

Rather than using a regex engine (e.g. the `regex` crate) directly, I rewrote the lexer to use the `logos` library. In addition to being faster to tokenize the input, it has a much smaller initialization cost as the `regex` crate does all regex compilation and processing at runtime.

Unfortunately `logos` is not directly compatible with the token representation used for the parser, so there is the need for some boilerplate code to convert between enums. This might be simplified by macros but I have not investigate it deeply.

### Parser

The parser is reasonably close to the book, including precedence climbing for expression parsing. Rather than the book's integers I used a Rust enum for precedence that allows me to make the precedence levels self-describing.

I have made some effort to preserve source line information in the AST for error reporting; this is not complete as of yet but informed my design for incorporating types in the AST. Both the Statement and Expression node types were split into a wrapper struct (`Statement` or `Expression`) containing metadata such as types, and a content enum (`Stmt` or `Expr`) that contained the actual node contents. This structure also permitted me uniform boxing of AST nodes to prevent unbounded type sizes; the outer struct holds a `Box<T>` of its content and the content enum then refers to the outer struct for recursive children.

Additional metadata for statements was required for implementing `goto` and `switch` statements as extra credit; this took the form of a list of `Label` enums that represent statement labels and the `case` and `default` from inside `switch` statements).

### Semantic Analysis

Identifier and struct tag resolution closely resembles the book. It introduces a pattern used for a number of automatic name generation uses of building a Rust struct to contain a counter variable and then making the AST traversal code methods of that struct.

The remainder of the pre-typecheck code has required extensive modification for supporting `goto` and `switch` statements. To clean up repetitive code for traversing the AST at statement depth I have written a `StatementVisitor` trait that will by default traverse all of the statements in the AST, with overridable methods for each of the statement types we might want to add behaviour to. There are four different implementation of this trait:

- a loop labelling pass like that in the book
- a pass that generates unique labels in the same way the identifier resolution pass generates unique identifiers
- a pass that alters `goto` statements to use the unique labels; this is a separate pass because the `goto` may refer to labels yet to be defined in the source
- a pass that gathers the cases of a `switch` statement and replaces the cases with plain labels that can be used in TACKY generation

Type checking is similar to the book but extended with `short` and `unsigned short` integer types, which are needed to compile code with real-world headers.

### TACKY IR and conversion logic

The TACKY IR has not required much in extensions beyond the book to implement the extra credit items. There are additional operators for the `Unary` and `Binary` instructions but no new instructions.

### TACKY optimizations

The optimizations from Chapter 19 of the book are implemented in full. Many of the data structures have set semantics and have been implemented as `HashSet`s. This has required a custom implementation of `Hash` and `Eq` for the representation of numeric constants so that we can store floating point constants, as `f64` implements neither of those traits. Binary operations on constants are delegated to the (many) relevant traits to keep the constant folding code simple.

### Assembly generation

The backend as a whole resembles that in the book fairly closely.

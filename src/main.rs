//! CandleCLI binary entry point.
//!
//! Responsibility: initialize the interactive chat CLI and hand control to
//! the chat loop in [`candlecli::cli`]. No inference logic lives here — the
//! binary is a thin front-end over the library crate.

fn main() {
    candlecli::cli::run();
}

package oksl

// Tokens of both dialects. The lexer marks the first token of each line, so
// the preprocessor finds directives, and tokens that follow whitespace, so
// `#define F(x)` (a function-like macro) differs from `#define F (x)`.

Token_Kind :: enum u8 {
	EOF,
	Ident,
	Int_Lit,
	Uint_Lit,
	Float_Lit,
	// punctuation
	Plus,
	Minus,
	Star,
	Slash,
	Percent,
	Inc,
	Dec,
	Plus_Eq,
	Minus_Eq,
	Star_Eq,
	Slash_Eq,
	Percent_Eq,
	Eq_Eq,
	Not_Eq,
	Lt,
	Gt,
	Lt_Eq,
	Gt_Eq,
	And_And,
	Or_Or,
	Xor_Xor,
	Bang,
	Tilde,
	Amp,
	Pipe,
	Caret,
	Shl,
	Shr,
	Amp_Eq,
	Pipe_Eq,
	Caret_Eq,
	Shl_Eq,
	Shr_Eq,
	Eq,
	Question,
	Colon,
	Semicolon,
	Comma,
	Dot,
	L_Paren,
	R_Paren,
	L_Bracket,
	R_Bracket,
	L_Brace,
	R_Brace,
	Hash,
	Hash_Hash,
	// Anything else; the parser reports it.
	Invalid,
}

Token_Flag :: enum u8 {
	// First token on its source line.
	Line_Start,
	// Whitespace or a comment came before the token.
	Space_Before,
	// An identifier the preprocessor must not expand again (it named a
	// macro inside that macro's own expansion).
	No_Expand,
}
Token_Flags :: bit_set[Token_Flag;u8]

Pos :: struct {
	line, col: i32,
	// The source of a program the position is in (Main for one compile).
	src:       Source,
}

Token :: struct {
	kind:  Token_Kind,
	flags: Token_Flags,
	pos:   Pos,
	text:  string,
}

@(private)
Lexer :: struct {
	src:  string,
	i:    int,
	line: i32,
	col:  i32,
}

// Splits `src` into tokens. Malformed input becomes `Invalid` tokens, which
// the parser reports with their position.
tokenize :: proc(src: string, allocator := context.allocator) -> [dynamic]Token {
	l := Lexer{src = src, line = 1, col = 1}
	tokens := make([dynamic]Token, allocator)
	line_start := true
	space := false
	for {
		// Whitespace, comments and line continuations.
		for l.i < len(l.src) {
			c := l.src[l.i]
			if c == '\n' {
				advance(&l)
				line_start = true
				space = true
			} else if c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' {
				advance(&l)
				space = true
			} else if c == '\\' && l.i + 1 < len(l.src) && (l.src[l.i + 1] == '\n' || l.src[l.i + 1] == '\r') {
				advance(&l)
				if l.src[l.i] == '\r' {advance(&l)}
				if l.i < len(l.src) && l.src[l.i] == '\n' {advance(&l)}
				space = true
			} else if c == '/' && l.i + 1 < len(l.src) && l.src[l.i + 1] == '/' {
				for l.i < len(l.src) && l.src[l.i] != '\n' {
					advance(&l)
				}
				space = true
			} else if c == '/' && l.i + 1 < len(l.src) && l.src[l.i + 1] == '*' {
				advance(&l)
				advance(&l)
				for l.i < len(l.src) && !(l.src[l.i] == '*' && l.i + 1 < len(l.src) && l.src[l.i + 1] == '/') {
					if l.src[l.i] == '\n' {line_start = true}
					advance(&l)
				}
				if l.i < len(l.src) {
					advance(&l)
					advance(&l)
				}
				space = true
			} else {
				break
			}
		}
		flags: Token_Flags
		if line_start {flags += {.Line_Start}}
		if space {flags += {.Space_Before}}
		line_start = false
		space = false
		start := l.i
		pos := Pos{line = l.line, col = l.col}
		if l.i >= len(l.src) {
			append(&tokens, Token{.EOF, flags + {.Line_Start}, pos, ""})
			return tokens
		}
		kind := lex_one(&l)
		append(&tokens, Token{kind, flags, pos, l.src[start:l.i]})
	}
}

@(private)
advance :: proc(l: ^Lexer) {
	if l.src[l.i] == '\n' {
		l.line += 1
		l.col = 1
	} else {
		l.col += 1
	}
	l.i += 1
}

@(private)
is_ident_start :: proc(c: u8) -> bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
}

@(private)
is_digit :: proc(c: u8) -> bool {
	return c >= '0' && c <= '9'
}

@(private)
is_hex :: proc(c: u8) -> bool {
	return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

@(private)
lex_one :: proc(l: ^Lexer) -> Token_Kind {
	c := l.src[l.i]
	peek :: proc(l: ^Lexer, off: int) -> u8 {
		return l.src[l.i + off] if l.i + off < len(l.src) else 0
	}
	if is_ident_start(c) {
		for l.i < len(l.src) && (is_ident_start(l.src[l.i]) || is_digit(l.src[l.i])) {
			advance(l)
		}
		return .Ident
	}
	if is_digit(c) || (c == '.' && is_digit(peek(l, 1))) {
		return lex_number(l)
	}
	three := [?]struct {
		s: string,
		k: Token_Kind,
	}{{"<<=", .Shl_Eq}, {">>=", .Shr_Eq}}
	for t in three {
		if l.i + 3 <= len(l.src) && l.src[l.i:l.i + 3] == t.s {
			for _ in 0 ..< 3 {advance(l)}
			return t.k
		}
	}
	two := [?]struct {
		s: string,
		k: Token_Kind,
	} {
		{"++", .Inc},
		{"--", .Dec},
		{"+=", .Plus_Eq},
		{"-=", .Minus_Eq},
		{"*=", .Star_Eq},
		{"/=", .Slash_Eq},
		{"%=", .Percent_Eq},
		{"==", .Eq_Eq},
		{"!=", .Not_Eq},
		{"<=", .Lt_Eq},
		{">=", .Gt_Eq},
		{"&&", .And_And},
		{"||", .Or_Or},
		{"^^", .Xor_Xor},
		{"<<", .Shl},
		{">>", .Shr},
		{"&=", .Amp_Eq},
		{"|=", .Pipe_Eq},
		{"^=", .Caret_Eq},
		{"##", .Hash_Hash},
	}
	for t in two {
		if l.i + 2 <= len(l.src) && l.src[l.i:l.i + 2] == t.s {
			advance(l)
			advance(l)
			return t.k
		}
	}
	advance(l)
	switch c {
	case '+':
		return .Plus
	case '-':
		return .Minus
	case '*':
		return .Star
	case '/':
		return .Slash
	case '%':
		return .Percent
	case '<':
		return .Lt
	case '>':
		return .Gt
	case '!':
		return .Bang
	case '~':
		return .Tilde
	case '&':
		return .Amp
	case '|':
		return .Pipe
	case '^':
		return .Caret
	case '=':
		return .Eq
	case '?':
		return .Question
	case ':':
		return .Colon
	case ';':
		return .Semicolon
	case ',':
		return .Comma
	case '.':
		return .Dot
	case '(':
		return .L_Paren
	case ')':
		return .R_Paren
	case '[':
		return .L_Bracket
	case ']':
		return .R_Bracket
	case '{':
		return .L_Brace
	case '}':
		return .R_Brace
	case '#':
		return .Hash
	}
	return .Invalid
}

@(private)
lex_number :: proc(l: ^Lexer) -> Token_Kind {
	src := l.src
	if src[l.i] == '0' && l.i + 1 < len(src) && (src[l.i + 1] == 'x' || src[l.i + 1] == 'X') {
		advance(l)
		advance(l)
		for l.i < len(src) && is_hex(src[l.i]) {
			advance(l)
		}
		if l.i < len(src) && (src[l.i] == 'u' || src[l.i] == 'U') {
			advance(l)
			return .Uint_Lit
		}
		return .Int_Lit
	}
	is_float := false
	for l.i < len(src) && is_digit(src[l.i]) {
		advance(l)
	}
	if l.i < len(src) && src[l.i] == '.' {
		is_float = true
		advance(l)
		for l.i < len(src) && is_digit(src[l.i]) {
			advance(l)
		}
	}
	if l.i < len(src) && (src[l.i] == 'e' || src[l.i] == 'E') {
		j := l.i + 1
		if j < len(src) && (src[j] == '+' || src[j] == '-') {
			j += 1
		}
		if j < len(src) && is_digit(src[j]) {
			is_float = true
			for l.i < j {
				advance(l)
			}
			for l.i < len(src) && is_digit(src[l.i]) {
				advance(l)
			}
		}
	}
	if l.i < len(src) && (src[l.i] == 'f' || src[l.i] == 'F') {
		advance(l)
		return .Float_Lit
	}
	if !is_float && l.i < len(src) && (src[l.i] == 'u' || src[l.i] == 'U') {
		advance(l)
		return .Uint_Lit
	}
	return .Float_Lit if is_float else .Int_Lit
}

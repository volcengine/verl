def parse_messages(messages, strip_sys_prompt=True):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    if messages is None: return ''

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)
    
    chat = "\n".join(
        f"**{m.role.capitalize()}**: {m.content}" for m in messages
    )

    return chat

def strip_system_prompt(messages):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    return [msg for msg in messages if msg.role != 'system']


def extract_json(s):

    def convert_value(value):
        true_values = {'true': True, 'false': False, 'null': None}
        value_lower = value.lower()
        if value_lower in true_values:
            return true_values[value_lower]
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # Return as string if not a number

    def parse_number(s, pos):
        start = pos
        while pos < len(s) and s[pos] in '-+0123456789.eE':
            pos +=1
        num_str = s[start:pos]
        try:
            if '.' in num_str or 'e' in num_str.lower():
                return float(num_str), pos
            else:
                return int(num_str), pos
        except ValueError:
            raise ValueError(f'Invalid number at position {start}: {num_str}')

    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos] in ' \t\n\r':
            pos +=1
        return pos

    def parse_string(s, pos):
        quote_char = s[pos]
        assert quote_char in ('"', "'")
        pos += 1
        result = ''
        while pos < len(s):
            c = s[pos]
            if c == '\\':
                pos += 1
                if pos >= len(s):
                    raise ValueError('Invalid escape sequence')
                c = s[pos]
                escape_sequences = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', quote_char: quote_char}
                result += escape_sequences.get(c, c)
            elif c == quote_char:
                pos += 1
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += c
            pos += 1
        raise ValueError('Unterminated string')
            
    def parse_key(s, pos):
        pos = skip_whitespace(s, pos)
        if s[pos] in ('"', "'"):
            key, pos = parse_string(s, pos)
            return key, pos
        else:
            raise ValueError(f'Expected string for key at position {pos}')

    def parse_object(s, pos):
        obj = {}
        assert s[pos] == '{'
        pos +=1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != '}':
            pos = skip_whitespace(s, pos)
            key, pos = parse_key(s, pos)
            pos = skip_whitespace(s, pos)
            if pos >= len(s) or s[pos] != ':':
                raise ValueError(f'Expected ":" at position {pos}')
            pos +=1
            pos = skip_whitespace(s, pos)
            value, pos = parse_value(s, pos)
            obj[key] = value
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ',':
                pos +=1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == '}':
                break
            elif pos < len(s) and s[pos] != '}':
                raise ValueError(f'Expected "," or "}}" at position {pos}')
        if pos >= len(s) or s[pos] != '}':
            raise ValueError(f'Expected "}}" at position {pos}')
        pos +=1
        return obj, pos

    def parse_array(s, pos):
        lst = []
        assert s[pos] == '['
        pos +=1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != ']':
            value, pos = parse_value(s, pos)
            lst.append(value)
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ',':
                pos +=1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == ']':
                break
            elif pos < len(s) and s[pos] != ']':
                raise ValueError(f'Expected "," or "]" at position {pos}')
        if pos >= len(s) or s[pos] != ']':
            raise ValueError(f'Expected "]" at position {pos}')
        pos +=1
        return lst, pos

    def parse_triple_quoted_string(s, pos):
        if s[pos:pos+3] == "'''":
            quote_str = "'''"
        elif s[pos:pos+3] == '"""':
            quote_str = '"""'
        else:
            raise ValueError(f'Expected triple quotes at position {pos}')
        pos += 3
        result = ''
        while pos < len(s):
            if s[pos:pos+3] == quote_str:
                pos += 3
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += s[pos]
                pos +=1
        raise ValueError('Unterminated triple-quoted string')

    def parse_value(s, pos):
        pos = skip_whitespace(s, pos)
        if pos >= len(s):
            raise ValueError('Unexpected end of input')
        if s[pos] == '{':
            return parse_object(s, pos)
        elif s[pos] == '[':
            return parse_array(s, pos)
        elif s[pos:pos+3] in ("'''", '"""'):
            return parse_triple_quoted_string(s, pos)
        elif s[pos] in ('"', "'"):
            return parse_string(s, pos)
        elif s[pos:pos+4].lower() == 'true':
            return True, pos+4
        elif s[pos:pos+5].lower() == 'false':
            return False, pos+5
        elif s[pos:pos+4].lower() == 'null':
            return None, pos+4
        elif s[pos] in '-+0123456789.':
            return parse_number(s, pos)
        else:
            raise ValueError(f'Unexpected character at position {pos}: {s[pos]}')

    json_start = s.index("{")
    json_end = s.rfind("}")
    s = s[json_start:json_end + 1]

    s = s.strip()
    result, pos = parse_value(s, 0)
    pos = skip_whitespace(s, pos)
    if pos != len(s):
        raise ValueError(f'Unexpected content at position {pos}')
    return result

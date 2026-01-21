import torch
import numpy as np

def parse_with_types(s):
    if not s:
        return []

    def is_digit(c):
        return '0' <= c <= '9'

    result = []
    n = len(s)
    i = 0

    while i < n:
        c = s[i]

        # -------------------------------------------------
        # 1) ЧИСЛО, НАЧИНАЮЩЕЕСЯ С '-'
        #   Разбираем ТОЛЬКО если это реально число:
        #   - "-<digit>"
        #   - "-.<digit>"
        #   Всё остальное с '-' пусть идёт в строковый сканер.
        # -------------------------------------------------
        if c == '-' and i + 1 < n:
            c1 = s[i + 1]

            # "-.<digit>" → float
            if c1 == '.' and i + 2 < n and is_digit(s[i + 2]):
                j = i + 3
                while j < n and is_digit(s[j]):
                    j += 1
                token = s[i:j]
                result.append((float(token), 'float', token))
                i = j
                continue

            # "-<digit>" → int/float
            if is_digit(c1):
                j = i + 2
                while j < n and is_digit(s[j]):
                    j += 1

                # возможная дробная часть
                if j < n and s[j] == '.':
                    if j + 1 < n and is_digit(s[j + 1]):
                        k = j + 2
                        while k < n and is_digit(s[k]):
                            k += 1
                        token = s[i:k]
                        result.append((float(token), 'float', token))
                        i = k
                        continue
                    else:
                        token = s[i:j + 1]
                        result.append((float(token), 'float', token))
                        i = j + 1
                        continue

                token = s[i:j]
                result.append((int(token), 'int', token))
                i = j
                continue

            # сюда НЕ попадаем, если '-' начинает число;
            # любые другие варианты '-' (типа "-+", "--", "-.") пойдут
            # ниже в строковый сканер.

        # -------------------------------------------------
        # 2) ЧИСЛО, НАЧИНАЮЩЕЕСЯ С '.'
        #   Обрабатываем ТОЛЬКО ".<digit>"
        #   Остальное с '.' — в строковый сканер.
        # -------------------------------------------------
        if c == '.' and i + 1 < n and is_digit(s[i + 1]):
            j = i + 2
            while j < n and is_digit(s[j]):
                j += 1
            token = s[i:j]
            result.append((float('0' + token), 'float', token))
            i = j
            continue

        # -------------------------------------------------
        # 3) ЧИСЛО, НАЧИНАЮЩЕЕСЯ С ЦИФРЫ
        # -------------------------------------------------
        if is_digit(c):
            j = i + 1
            while j < n and is_digit(s[j]):
                j += 1

            # дробная часть
            if j < n and s[j] == '.':
                if j + 1 < n and is_digit(s[j + 1]):
                    k = j + 2
                    while k < n and is_digit(s[k]):
                        k += 1
                    token = s[i:k]
                    result.append((float(token), 'float', token))
                    i = k
                    continue
                else:
                    token = s[i:j + 1]
                    result.append((float(token), 'float', token))
                    i = j + 1
                    continue

            token = s[i:j]
            result.append((int(token), 'int', token))
            i = j
            continue

        # -------------------------------------------------
        # 4) СТРОКА — собираем до потенциального начала числа
        #   Тут ключевая правка: правильно определять старт числа
        #   для '-' (только "-<digit>" или "-.<digit>") и для '.'
        # -------------------------------------------------
        j = i
        while j < n:
            cc = s[j]

            if cc == '-':
                if j + 1 < n:
                    c1 = s[j + 1]
                    # число только если "-<digit>" или "-.<digit>"
                    if is_digit(c1) or (c1 == '.' and j + 2 < n and is_digit(s[j + 2])):
                        break

            elif cc == '.':
                # число только если ".<digit>"
                if j + 1 < n and is_digit(s[j + 1]):
                    break

            elif is_digit(cc):
                break

            j += 1

        token = s[i:j]
        result.append((token, 'string', token))
        i = j

    return result

def parse_with_types_with_auto_check(s):
    result = parse_with_types(s)
    reconstructed = ''.join(substr for _, _, substr in result)
    if reconstructed != s:
        raise ValueError(
            f"Broken integrity\n"
            f"Original:      '{s}'\n"
            f"Reconstructed: '{reconstructed}'\n"
            f"Result: {result}"
        )
    return result

def apply_transform(x, method="log"):
    if method == "log":
        return np.sign(x) * np.log1p(np.abs(x))
    else: return x # TODO

def re_transform(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))

def process_single_svg_str(s, max_len, tokenizer):
    parsed = parse_with_types_with_auto_check(s)
    num_id = tokenizer.convert_tokens_to_ids("[NUM]")

    ids = []
    is_num = []
    values = []
    
    for v, t, _ in parsed:
        if len(ids) >= max_len: break
            
        if t == "string":
            toks = tokenizer.tokenize(v)
            for tok in toks:
                if len(ids) >= max_len: break 
                ids.append(tokenizer.convert_tokens_to_ids(tok))
                is_num.append(0.0)
                values.append(0.0)
        else:
            if len(ids) >= max_len: break 
            ids.append(num_id)
            is_num.append(1.0)
            values.append(apply_transform(float(v), method="original")) 

    return {
        "input_ids": ids,
        "is_number": is_num,
        "num_values": values,
        "attention_mask": [1] * len(ids)
    }
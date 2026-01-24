import math
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

def decompose_signed_mantissa(x, eps=1e-12):
    # [FIX 1] 强制处理 NaN/Inf，防止污染下游
    if not math.isfinite(x):
        return 0.0, 0
    if abs(x) < eps:
        return 0.0, 0  # 特殊处理零（可选）
    
    # [FIX 2] 防止 log10 报错（虽然上面过滤了 0，但为了稳健）
    try:
        exp = math.floor(math.log10(abs(x)))
    except ValueError:
        return 0.0, 0

    mantissa = x / (10 ** exp)
    
    # 确保 |mantissa| ∈ [1, 10)
    if abs(mantissa) < 1:
        mantissa *= 10
        exp -= 1
    elif abs(mantissa) >= 10:
        mantissa /= 10
        exp += 1
    
    # [FIX 4] 强制截断 Exponent，适配模型定义的 [-10, 10]
    exp = max(min(exp, 10), -10)
    return mantissa, exp

def process_single_svg_str(s, max_len, tokenizer):
    try:
        parsed = parse_with_types_with_auto_check(s)
    except Exception as e:
        print(f"Parse error: {e}")
        return {} # 返回空或者跳过

    num_id = tokenizer.convert_tokens_to_ids("[NUM]")

    # CLS TOKEN
    ids = [tokenizer.cls_token_id]
    is_num = [0.0]
    values = [0.0]

    # [ADD]
    mantissa = [0.0]
    exponent = [0]
    
    for v, t, _ in parsed:
        if len(ids) >= max_len - 1: break # Leave room for [SEP]
            
        if t == "string":
            toks = tokenizer.tokenize(v)
            for tok in toks:
                if len(ids) >= max_len: break 
                ids.append(tokenizer.convert_tokens_to_ids(tok))
                is_num.append(0.0)
                values.append(0.0)
                # [ADD]
                mantissa.append(0.0)
                exponent.append(0)
        else:
            if len(ids) >= max_len: break 
            # [FIX 5] 安全转换 Float，拦截 NaN/Inf/Overflow
            try:
                val_float = float(v)
                # 检查无穷大或非数值
                if math.isnan(val_float) or math.isinf(val_float):
                    val_float = 0.0
                
                # [FIX 6] 极端数值截断 (防止 sin(x) 数值不稳定)
                val_float = max(min(val_float, 1e10), -1e10)
                
            except (ValueError, OverflowError):
                val_float = 0.0

            ids.append(num_id)
            is_num.append(1.0)
            
            value = apply_transform(val_float, method="original")
            values.append(value) # 采用原始空间的值
            # [ADD]
            m, e = decompose_signed_mantissa(value)
            mantissa.append(m)
            exponent.append(e)
            

    # SEP TOKEN
    ids.append(tokenizer.sep_token_id)
    is_num.append(0.0)
    values.append(0.0)

    # [ADD]
    mantissa.append(0.0)
    exponent.append(0)

    assert len(ids) == len(values) == len(mantissa) == len(exponent)

    return {
        "input_ids": ids,
        "is_number": is_num,
        "num_values": values,
        "mantissa": mantissa,
        "exponent": exponent,
        "attention_mask": [1] * len(ids)
    }
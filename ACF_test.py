import Levenshtein
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else \
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "ai-forever/sage-fredt5-large", torch_dtype=torch.float16
).to(device)

toreplace = {
    'b': 'в',
    'a': 'а',
    'c': 'с',
    'e': 'е',
    'o': 'о',
    'p': 'р',
    'y': 'у',
    'x': 'х',
    'fl': 'я',
    'g': 'г',
    'll': 'п',
    'bl': 'ы',
    'n': 'п',
    'u': 'и',
    'd': 'д',
    'k': 'к',
    'h': 'н',
    't': 'т',
    'r': 'г',
}  # '': '',

similarity_weights = {
    'а': {'а': 1.0, 'о': 0.4, 'у': 0.1, 'я': 0.7, 'е': 0.2, 'с': 0.1, 'д': 0.1},
    'б': {'б': 1.0, 'в': 0.7, 'д': 0.5, 'з': 0.4, 'п': 0.3, 'ф': 0.2, 'ь': 0.1},
    'в': {'в': 1.0, 'б': 0.7, 'д': 0.6, 'з': 0.5, 'у': 0.4, 'ф': 0.3, 'ь': 0.2},
    'г': {'г': 1.0, 'д': 0.6, 'ж': 0.4, 'з': 0.3, 'т': 0.5, 'р': 0.2, 'п': 0.1},
    'д': {'д': 1.0, 'г': 0.6, 'б': 0.5, 'з': 0.7, 'л': 0.4, 'т': 0.6, 'ц': 0.3},
    'е': {'е': 1.0, 'ё': 0.95, 'э': 0.5, 'и': 0.3, 'з': 0.2, 'с': 0.4, 'о': 0.35},
    'ё': {'ё': 1.0, 'е': 0.95, 'о': 0.6, 'ж': 0.1, 'э': 0.3, 'ю': 0.2},
    'ж': {'ж': 1.0, 'к': 0.3, 'х': 0.4, 'з': 0.2, 'г': 0.4, 'ш': 0.8, 'щ': 0.7},
    'з': {'з': 1.0, 'д': 0.7, 'в': 0.5, 'б': 0.4, 'э': 0.3, 'с': 0.8, 'ж': 0.2},
    'и': {'и': 1.0, 'й': 0.6, 'н': 0.4, 'ш': 0.1, 'ц': 0.3, 'у': 0.2, 'л': 0.2},
    'й': {'й': 1.0, 'и': 0.6, 'ц': 0.4, 'у': 0.3, 'н': 0.2, 'к': 0.1},
    'к': {'к': 1.0, 'н': 0.7, 'ж': 0.3, 'х': 0.6, 'т': 0.4, 'р': 0.5},
    'л': {'л': 1.0, 'д': 0.4, 'п': 0.3, 'м': 0.5, 'ь': 0.6, 'и': 0.2},
    'м': {'м': 1.0, 'н': 0.8, 'л': 0.5, 'т': 0.3, 'ш': 0.1, 'щ': 0.1},
    'н': {'н': 1.0, 'п': 0.9, 'к': 0.7, 'и': 0.4, 'т': 0.5, 'м': 0.8},
    'о': {'о': 1.0, 'а': 0.4, 'у': 0.3, 'ё': 0.6, 'е': 0.35, 'с': 0.5, 'р': 0.1},
    'п': {'п': 1.0, 'н': 0.9, 'р': 0.4, 'т': 0.6, 'г': 0.1, 'л': 0.3},
    'р': {'р': 1.0, 'п': 0.4, 'к': 0.5, 'г': 0.2, 'т': 0.7, 'ж': 0.3},
    'с': {'с': 1.0, 'з': 0.8, 'э': 0.5, 'ц': 0.6, 'е': 0.4, 'о': 0.5},
    'т': {'т': 1.0, 'г': 0.5, 'п': 0.6, 'р': 0.7, 'н': 0.5, 'м': 0.3},
    'у': {'у': 1.0, 'о': 0.3, 'и': 0.2, 'в': 0.4, 'ц': 0.3, 'ю': 0.6},
    'ф': {'ф': 1.0, 'в': 0.3, 'б': 0.2, 'х': 0.4, 'ж': 0.1, 'к': 0.1},
    'х': {'х': 1.0, 'ж': 0.4, 'к': 0.6, 'ф': 0.4, 'н': 0.2, 'м': 0.1},
    'ц': {'ц': 1.0, 'ч': 0.7, 'у': 0.5, 'щ': 0.4, 'й': 0.4, 'н': 0.2},
    'ч': {'ч': 1.0, 'ц': 0.7, 'щ': 0.8, 'г': 0.2, 'ш': 0.5, 'ж': 0.3},
    'ш': {'ш': 1.0, 'щ': 0.9, 'ч': 0.5, 'ж': 0.8, 'м': 0.1, 'и': 0.1},
    'щ': {'щ': 1.0, 'ш': 0.9, 'ч': 0.8, 'ц': 0.4, 'м': 0.2, 'н': 0.1},
    'ъ': {'ъ': 1.0, 'ь': 0.8, 'ы': 0.4, 'э': 0.1, 'о': 0.1},
    'ы': {'ы': 1.0, 'ь': 0.3, 'ъ': 0.4, 'о': 0.2, 'а': 0.1, 'н': 0.1},
    'ь': {'ь': 1.0, 'ъ': 0.8, 'ы': 0.3, 'б': 0.1, 'в': 0.2, 'л': 0.6},
    'э': {'э': 1.0, 'е': 0.5, 'з': 0.3, 'с': 0.5, 'ё': 0.3, 'ц': 0.1},
    'ю': {'ю': 1.0, 'у': 0.6, 'о': 0.4, 'ё': 0.2, 'г': 0.1, 'ж': 0.1},
    'я': {'я': 1.0, 'а': 0.7, 'е': 0.4, 'ю': 0.3, 'з': 0.1, 'н': 0.1}
}


def rep(text):
    result = ""
    i = 0
    while i < len(text):
        longest_match = ""
        for key in toreplace:
            if text[i:].startswith(key) and len(key) > len(longest_match):
                longest_match = key
        if longest_match:
            result += toreplace[longest_match]
            i += len(longest_match)
        else:
            result += text[i]
            i += 1
    return result


def norm(text, mode=0):
    if mode == 1:
        text = (
            text.replace('...', ' … ').replace('?!', ' ‽ ')
            .replace('!', ' ! ').replace('?', ' ? ')
            .replace('.', ' . ').replace(',', ' , ')
            .replace(';', ' ; ').replace(':', ' : ')
        )
        sentences = []
        current_sentence = []
        for word in text.split():
            if word in '.!?…‽':
                if current_sentence:
                    current_sentence.append(word)
                    sentences.append(" ".join(current_sentence))
                    current_sentence = []
            else:
                current_sentence.append(word)
        if current_sentence: sentences.append(" ".join(current_sentence))
        return sentences
    text = rep(text.lower())
    return text.split()


def denorm(sentences):
    if not sentences: return ""
    if isinstance(sentences[0], list): sentences = [" ".join(sentence).capitalize() for sentence in sentences]
    text = " ".join(sentences)
    for p in ',;:!?.…‽': text = text.replace(f' {p} ', f'{p} ').replace(f' {p}', f'{p}')
    text = text.replace(' .', '.').replace(' !', '!').replace(' ?', '?')
    text = text.replace('…', '...').replace('‽', '?!')
    return text


dictionary = set()
with open('dict.opcorpora.txt', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            word = line.split(maxsplit=1)[0].lower()
            if not word.isdigit(): dictionary.add(word)
dictionary = list(dictionary)


def build_ngram_index(words, n=3):
    index = defaultdict(list)
    for word in words:
        for i in range(len(word) - n + 1):
            ngram = word[i:i + n]
            index[ngram].append(word)
    return index


index = build_ngram_index(dictionary)


def del_value(w1, w2, p1, p2):
    res = 1
    if p1 == 0 or p1 == len(w1) - 1: res *= 0.5
    if w1[p1 - 1] == w1[p1] and p1 != 0: res *= 0.1
    return res


def wLev(w1, w2):
    dist = 0
    for i in Levenshtein.editops(w1, w2):
        if i[0] == 'replace':
            try:
                dist += (1 - similarity_weights[w1[i[1]]][w2[i[2]]])
            except KeyError:
                dist += 1
        elif i[0] == 'delete':
            dist += del_value(w1, w2, i[1], i[2])
        else:
            dist += 1
    return dist


def guess(word, n=3, border=5):
    global index
    candidates = set()
    for i in range(len(word) - n + 1):
        ngram = word[i:i + n]
        candidates.update(index[ngram])
    closest_words = []
    for candidate in candidates:
        score = wLev(word, candidate)
        if score <= border:
            closest_words.append((candidate, score))
    closest_words.sort(key=lambda x: x[1])
    return closest_words[:5]


def correction(text):
    text = norm(text, 1)
    for i in range(len(text)): text[i] = norm(text[i])
    for i in range(len(text)):
        for j in range(len(text[i])):
            if text[i][j] not in dictionary:
                var = guess(text[i][j])
                if var != []: text[i][j] = var[0][0]
    return denorm(text)


def correction_ai(text, dnm=True):
    text = norm(text, mode=1)
    for i in range(len(text)):
        inputs = tokenizer(text[i], max_length=None, padding="longest", truncation=False, return_tensors="pt").to(
            device)
        outputs = model.generate(**inputs.to(model.device), max_length=inputs["input_ids"].size(1) * 1.5)
        text[i] = norm(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    if dnm:
        return denorm(text)
    else:
        return text


def report(text):
    res = []
    text = norm(text, 1)
    for i in range(len(text)): text[i] = norm(text[i])
    for i in range(len(text)):
        for j in range(len(text[i])):
            if text[i][j] not in dictionary:
                var = guess(text[i][j])
                if var != []:
                    st = f"{text[i][j]} => {var[0]}"
                else:
                    st = f"{text[i][j]} => None"
                print(st)
                res.append(st)
    return res


def report_ai(text):
    res = []
    ctext = correction_ai(text, dnm=False)
    text = norm(text, mode=1)
    for i in range(len(text)): text[i] = norm(text[i])
    for i in range(len(text)):
        for j in range(len(text[i])):
            if text[i][j] not in ctext[i]:
                c = ctext[i][ctext[i].index(sorted(ctext[i], key=lambda x: wLev(x, text[i][j]))[0])]
                st = f"{text[i][j]} => {c}"
                print(st)
                res.append(st)
    return res


if __name__ == '__main__':
    text = ''
    with open('[e]file_1.txt', encoding="utf-8") as f:
        c = 0
        fr, t = 52, 81
        for l in f:
            if l == '\n':
                c += 1
                continue
            if c >= fr and c <= t: text += f" {l}"
            c += 1
    report = report_ai(text)

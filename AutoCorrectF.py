from spylls.hunspell import Dictionary

Corrector = Dictionary.from_files('ru')

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
    'bl': 'ы'
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
    if mode == 1: return text.replace(',',' ,').replace(';',' ;').replace(':',' :').split('. ')
    text = rep(text.lower())
    return (text.replace('.','')).split()

def correctinon(text):
    text = norm(text, 1)
    for i in range(len(text)): text[i] = norm(text[i])
    for i in range(len(text)):
        for j in range(len(text[i])):
            if not Corrector.lookup(text[i][j]):
                var = list(Corrector.suggest(text[i][j]))
                if var != []: text[i][j] = var[0]
    return ('. '.join([' '.join(text[i]).capitalize() for i in range(len(text))])).replace(' ,',',').replace(' ;',';').replace(' :',':')+'.'
#!/usr/bin/env python3
"""Постобработка OCR паспорта РФ.

Использование:
    python postprocess.py <predictions.jsonl> <answers.jsonl>

Скрипт решает задачу бинарной классификации (accept / reject) и валидации/коррекции
распознанных полей паспорта РФ.
"""

import sys
import os
import json
import csv
import re
from datetime import datetime, timedelta

# Эталонная дата действительности паспорта РФ по условию задачи
REF_DATE = datetime.strptime('01.10.2026', '%d.%m.%Y')

# Официальная таблица транслитерации МСЗ (MRZ) паспорта РФ (Приказ ФМС РФ № 294)
MRZ_TO_RU = {
    'A': 'А', 'B': 'Б', 'V': 'В', 'G': 'Г', 'D': 'Д', 'E': 'Е', '2': 'Ё', 'J': 'Ж',
    'Z': 'З', 'I': 'И', 'Q': 'Й', 'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н', 'O': 'О',
    'P': 'П', 'R': 'Р', 'S': 'С', 'T': 'Т', 'U': 'У', 'F': 'Ф', 'H': 'Х', 'C': 'Ц',
    '3': 'Ч', '4': 'Ш', 'W': 'Щ', 'X': 'Ъ', 'Y': 'Ы', '9': 'Ь', '6': 'Э', '7': 'Ю',
    '8': 'Я'
}

VALID_ORG_STARTS = (
    'ОТДЕЛ', 'ОТДЕЛЕНИЕ', 'ОТДЕЛОМ', 'ОТДЕЛЕНИЕМ', 'ТП', 'УФМС', 'ОФМС',
    'МВД', 'ГУ', 'УМВД', 'МИГРАЦИОННЫЙ', 'ТЕРРИТОРИАЛЬНЫЙ', 'ОВД', 'РОВД',
    'УВД', 'ГОМ', 'МРО', 'ПУНКТ', 'МЕЖРАЙОННЫЙ'
)

INVALID_ORG_ENDS = ('ПО', 'В', 'И', 'С', 'НА', 'ОТ')
INVALID_BP_STARTS = ('Р-НА', 'РАЙОНА', 'ОБЛ.', 'ОБЛАСТИ', 'КРАЯ', 'РЕСПУБЛИКИ', 'АО')


def norm(s):
    """Нормализация строки в верхний регистр с удалением лишних пробелов."""
    return " ".join(str(s or "").split()).upper()


def calc_check_digit(s):
    """Вычисление контрольного символа по стандарту ICAO 9303 (веса 7-3-1)."""
    weights = [7, 3, 1]
    total = 0
    for i, c in enumerate(s):
        if c.isdigit():
            val = int(c)
        elif c.isalpha():
            val = ord(c.upper()) - ord('A') + 10
        elif c == '<':
            val = 0
        else:
            return None
        total += val * weights[i % 3]
    return str(total % 10)


def decode_mrz_name(s):
    """Декодирование транслитерированного имени из MRZ в кириллицу."""
    res = []
    for c in s:
        if c == '<':
            res.append(' ')
        else:
            res.append(MRZ_TO_RU.get(c, c))
    return ' '.join(''.join(res).split())


def add_years(d, years):
    """Добавление лет с корректной обработкой високосного 29 февраля."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def is_valid_passport(b_date_str, i_date_str):
    """Проверка действительности паспорта РФ на дату 01.10.2026.
    
    Правила:
    1. Паспорт выдается с 14 лет (Issue_date >= Birth_date + 14 лет).
    2. Паспорт не может быть выдан в будущем (Issue_date <= 01.10.2026).
    3. При достижении 20 лет паспорт подлежит замене (выдан в возрасте >= 20 лет).
    4. При достижении 45 лет паспорт подлежит замене (выдан в возрасте >= 45 лет).
    """
    try:
        b_date = datetime.strptime(b_date_str, '%d.%m.%Y')
        i_date = datetime.strptime(i_date_str, '%d.%m.%Y')
    except Exception:
        return False
        
    b14 = add_years(b_date, 14)
    b20 = add_years(b_date, 20)
    b45 = add_years(b_date, 45)
    
    if i_date < b14 or i_date > REF_DATE:
        return False
    if REF_DATE >= b20 and i_date < b20:
        return False
    if REF_DATE >= b45 and i_date < b45:
        return False
    return True


def load_vocabularies(base_dir):
    """Загрузка словаря подразделений из subdivisions.csv и train_labels.jsonl."""
    known_org_words = set()
    
    sub_paths = [
        os.path.join(base_dir, 'reference', 'subdivisions.csv'),
        os.path.join(base_dir, 'subdivisions.csv'),
        'reference/subdivisions.csv'
    ]
    for sp in sub_paths:
        if os.path.exists(sp):
            with open(sp, encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) > 1:
                        for w in norm(row[1]).split():
                            known_org_words.add(w)
            break
            
    train_paths = [
        os.path.join(base_dir, 'data', 'train_labels.jsonl'),
        os.path.join(base_dir, 'train_labels.jsonl'),
        'data/train_labels.jsonl'
    ]
    for tp in train_paths:
        if os.path.exists(tp):
            with open(tp, encoding='utf-8') as f:
                for line in f:
                    try:
                        d = json.loads(line).get('fields', {})
                        for w in norm(d.get('Issue_organization_ru', '')).split():
                            known_org_words.add(w)
                    except Exception:
                        pass
            break
            
    return known_org_words


def process_document(doc, known_org_words, min_char_c=0.985):
    """Принятие решения по документу и извлечение полей."""
    doc_id = doc.get('doc_id')
    viz = doc.get('viz', {})
    mrz = doc.get('mrz')
    
    # 1. Требуем наличие и валидность машиночитаемой зоны (MRZ)
    if not mrz or not mrz.get('lines') or len(mrz['lines']) < 2:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    l1 = mrz['lines'][0].get('text', '')
    l2 = mrz['lines'][1].get('text', '')
    
    if not l1.startswith('PNRUS') or len(l2) != 44:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Проверка контрольных чисел ICAO в строке 2
    cd1_ok = (calc_check_digit(l2[0:9]) == l2[9])
    cd2_ok = (calc_check_digit(l2[13:19]) == l2[19])
    cd3_ok = (calc_check_digit(l2[28:41]) == l2[42])
    
    if not (cd1_ok and cd2_ok and cd3_ok):
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Извлечение полей из MRZ
    m_lic = f'{l2[0:2]} {l2[2]}{l2[28]} {l2[3:9]}'
    
    if not (l2[13:15].isdigit() and l2[29:31].isdigit()):
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    byy = int(l2[13:15])
    byear = 2000 + byy if byy <= 26 else 1900 + byy
    m_bdate = f'{l2[17:19]}.{l2[15:17]}.{byear}'
    
    m_sex = 'МУЖ.' if l2[20] == 'M' else ('ЖЕН.' if l2[20] == 'F' else '')
    if not m_sex:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    iyy = int(l2[29:31])
    iyear = 2000 + iyy if iyy <= 26 else 1900 + iyy
    m_idate = f'{l2[33:35]}.{l2[31:33]}.{iyear}'
    
    m_code = f'{l2[35:38]}-{l2[38:41]}'
    
    # 2. Проверка срока действия паспорта на 01.10.2026
    if not is_valid_passport(m_bdate, m_idate):
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Извлечение ФИО из MRZ
    body = l1[5:].rstrip('<')
    parts = body.split('<<')
    if len(parts) < 2:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    m_last = decode_mrz_name(parts[0])
    rest = parts[1].split('<')
    m_first = decode_mrz_name(rest[0])
    m_mid = decode_mrz_name(rest[1]) if len(rest) > 1 and rest[1] else ''
    
    # 3. Извлечение полей из визуальной зоны (VIZ)
    vf_last = viz.get('Last_name_ru')
    vf_first = viz.get('First_name_ru')
    vf_mid = viz.get('Middle_name_ru')
    vf_sex = viz.get('Sex_ru')
    vf_bdate = viz.get('Birth_date')
    vf_idate = viz.get('Issue_date')
    vf_code = viz.get('Issue_organisation_code')
    vf_lic = viz.get('Licence_number')
    vf_bp = viz.get('Birth_place_ru')
    vf_org = viz.get('Issue_organization_ru')
    
    v_last = norm(' '.join(l['text'] for l in vf_last['lines'])) if vf_last else ''
    v_first = norm(' '.join(l['text'] for l in vf_first['lines'])) if vf_first else ''
    v_mid = norm(' '.join(l['text'] for l in vf_mid['lines'])) if vf_mid else ''
    v_sex = norm(' '.join(l['text'] for l in vf_sex['lines'])) if vf_sex else ''
    v_bdate = norm(' '.join(l['text'] for l in vf_bdate['lines'])) if vf_bdate else ''
    v_idate = norm(' '.join(l['text'] for l in vf_idate['lines'])) if vf_idate else ''
    v_code = norm(' '.join(l['text'] for l in vf_code['lines'])) if vf_code else ''
    v_lic = norm(' '.join(l['text'] for l in vf_lic['lines'])) if vf_lic else ''
    
    # 4. Строгая перекрёстная сверка 8 полей между MRZ и VIZ
    if not (m_lic == v_lic and m_bdate == v_bdate and m_sex == v_sex and
            m_idate == v_idate and m_code == v_code and
            norm(m_last) == v_last and norm(m_first) == v_first and norm(m_mid) == v_mid):
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # 5. Верификация двух полей, отсутствующих в MRZ (Birth_place_ru и Issue_organization_ru)
    if not vf_bp or not vf_org:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    if vf_bp.get('conf', 0) < 0.98 or vf_org.get('conf', 0) < 0.98:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    bp_confs = [c for l in vf_bp.get('lines', []) for c in l.get('char_conf', [])]
    org_confs = [c for l in vf_org.get('lines', []) for c in l.get('char_conf', [])]
    if not bp_confs or not org_confs:
        return {'doc_id': doc_id, 'decision': 'reject'}
    if min(bp_confs) < min_char_c or min(org_confs) < min_char_c:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    v_bp = norm(' '.join(l['text'] for l in vf_bp.get('lines', [])))
    v_org = norm(' '.join(l['text'] for l in vf_org.get('lines', [])))
    
    # Проверка структуры и полноты Issue_organization_ru
    org_words = v_org.split()
    if len(org_words) < 4:
        return {'doc_id': doc_id, 'decision': 'reject'}
    if not (any(org_words[0].startswith(s) for s in VALID_ORG_STARTS) or org_words[0][0].isdigit()):
        return {'doc_id': doc_id, 'decision': 'reject'}
    if org_words[-1] in INVALID_ORG_ENDS:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Геометрический контроль детектора (пропущенные строки в высоком боксе)
    org_box = vf_org.get('box', [0, 0, 0, 0])
    if org_box[3] > 45 and len(vf_org.get('lines', [])) < 2:
        return {'doc_id': doc_id, 'decision': 'reject'}
    if org_box[3] > 75 and len(vf_org.get('lines', [])) < 3:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Проверка структуры Birth_place_ru
    if any(v_bp.startswith(s) for s in INVALID_BP_STARTS):
        return {'doc_id': doc_id, 'decision': 'reject'}
    bp_words = v_bp.split()
    if len(bp_words) <= 2 and any(obl in v_bp for obl in ['ОБЛ', 'ОБЛАСТ', 'КРАЙ', 'РЕСПУБЛИК']):
        return {'doc_id': doc_id, 'decision': 'reject'}
    bp_box = vf_bp.get('box', [0, 0, 0, 0])
    if bp_box[3] > 45 and len(vf_bp.get('lines', [])) < 2:
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Контроль невозможных символьных кластеров (артефакты OCR)
    if any(w.startswith('ЧС') or w.startswith('Ь') or w.startswith('Ъ') for w in bp_words):
        return {'doc_id': doc_id, 'decision': 'reject'}
    if any(w.startswith('ЧС') or w.startswith('Ь') or w.startswith('Ъ') for w in org_words):
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Словарный контроль Issue_organization_ru
    if known_org_words and not all(w in known_org_words for w in org_words):
        return {'doc_id': doc_id, 'decision': 'reject'}
        
    # Все проверки успешно пройдены: документ принимается
    fields = {
        'Last_name_ru': v_last,
        'First_name_ru': v_first,
        'Middle_name_ru': v_mid,
        'Sex_ru': v_sex,
        'Birth_date': v_bdate,
        'Birth_place_ru': v_bp,
        'Issue_date': v_idate,
        'Issue_organisation_code': v_code,
        'Licence_number': v_lic,
        'Issue_organization_ru': v_org
    }
    return {'doc_id': doc_id, 'decision': 'accept', 'fields': fields}


def main():
    if len(sys.argv) < 3:
        sys.exit("Использование: python postprocess.py <predictions.jsonl> <answers.jsonl>")
        
    pred_path = sys.argv[1]
    ans_path = sys.argv[2]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    known_org_words = load_vocabularies(script_dir)
    
    accepted_count = 0
    total_count = 0
    
    with open(pred_path, 'r', encoding='utf-8') as fin, \
         open(ans_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            res = process_document(doc, known_org_words)
            if res.get('decision') == 'accept':
                accepted_count += 1
            total_count += 1
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    print(f"Обработано документов: {total_count}")
    print(f"Принято:               {accepted_count} ({accepted_count / (total_count or 1):.1%})")
    print(f"Отклонено:             {total_count - accepted_count} ({(total_count - accepted_count) / (total_count or 1):.1%})")
    print(f"Результаты записаны в: {ans_path}")


if __name__ == '__main__':
    main()

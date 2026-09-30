import os
import csv
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "3_sentences_selected")
ANNOTATIONS_FILE = os.path.join(SCRIPT_FOLDER, "3_manual_annotations", "manual_annotations.csv")
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "4_sentences_with_sentiment")
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "sentences_with_sentiment.csv")

if not os.path.exists(INPUT_FOLDER):
    print("Input folder does not exist.")
    exit(1)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

MODEL = "gemma4:e4b"

PROMPT = """
DEFINIÇÃO DE OTIMISMO:
Ocorre quando as projeções indicam que a inflação ficará abaixo da meta ou dentro do intervalo de tolerância com folga. 
Isso pode sinalizar que o Banco Central vê espaço para reduzir juros ou manter uma política monetária mais acomodatícia. 

DEFINIÇÃO DE PESSIMISMO:
Ocorre quando as projeções apontam para inflação acima da meta ou próxima do teto do intervalo de tolerância. 
Isso sugere preocupação com pressões inflacionárias e pode justificar uma política monetária mais restritiva.

AVALIE A FRASE COMO
O para OTIMISTA
N para NEUTRA
P para PESSIMISTA
SUA RESPOSTA DEVE SER APENAS UMA LETRA, SEM QUALQUER OUTRO TEXTO

A FRASE É:
"""

GRADE_MAP = {'O': 1, 'N': 0, 'P': -1}
REVERSE_GRADE_MAP = {1: 'O', 0: 'N', -1: 'P'}

REQUEST_DELAY = 0.2

def _meeting_key(filename):
    stem = os.path.splitext(filename)[0]
    return int(stem.split('_', 1)[0])

def _parse_filename_date(filename):
    stem = os.path.splitext(filename)[0]
    ddmmyyyy = stem.split('_', 1)[1]
    return f"{ddmmyyyy[0:2]}/{ddmmyyyy[2:4]}/{ddmmyyyy[4:8]}"

def _date_sort_key(date_str):
    d, m, y = date_str.split('/')
    return y + m + d

def load_manual_annotations():
    existing = {}
    if not os.path.exists(ANNOTATIONS_FILE):
        return existing
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            key = (row['Date'].strip(), row['Sentence'].strip())
            existing[key] = {
                'model': row['Model'].strip(),
                'grade': int(row['Grade'].strip()),
            }
    print(f"Loaded {len(existing)} existing annotations.")
    return existing

def load_output_checkpoint():
    done = set()
    if not os.path.exists(OUTPUT_FILE):
        return done
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            done.add((row['Date'].strip(), row['Sentence'].strip()))
    print(f"Resuming: {len(done)} sentences already in output file.")
    return done

def build_few_shot_messages(existing):
    messages = []
    seen_sentences = set()
    for (date, sentence), ann in existing.items():
        if ann['model'] == 'human' and sentence not in seen_sentences:
            seen_sentences.add(sentence)
            letter = REVERSE_GRADE_MAP[ann['grade']]
            messages.append({"role": "user", "content": PROMPT + sentence})
            messages.append({"role": "assistant", "content": letter})
    return messages

def annotate_sentence(client, sentence, few_shot_messages):
    messages = few_shot_messages + [{"role": "user", "content": PROMPT + sentence}]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=5,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().upper()
    for char in answer:
        if char in GRADE_MAP:
            return GRADE_MAP[char]
    print(f"  Warning: unexpected LLM response '{answer}', defaulting to Neutral.")
    return 0

def collect_sentences():
    txt_files = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')],
        key=_meeting_key,
    )

    rows = []
    for filename in txt_files:
        date_str = _parse_filename_date(filename)
        filepath = os.path.join(INPUT_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    rows.append((date_str, sentence))

    print(f"Found {len(rows)} sentences across {len(txt_files)} meeting files.")
    return rows

def main():
    existing = load_manual_annotations()
    already_done = load_output_checkpoint()

    sentences = collect_sentences()
    if not sentences:
        print("No sentences found in input folder. Run c_selectPhrases.py first.")
        return

    #client = OpenAI(api_key=api_key, base_url=base_url)

    few_shot_messages = build_few_shot_messages(existing)
    print(f"Built {len(few_shot_messages) // 2} few-shot examples from human annotations.")
    #print(few_shot_messages)

    sys.exit()
    write_header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    out_f = open(OUTPUT_FILE, 'a', encoding='utf-8', newline='')
    writer = csv.DictWriter(out_f, fieldnames=['Date', 'Model', 'Grade', 'Sentence'], delimiter='|')
    if write_header:
        writer.writeheader()

    for date_str, sentence in sentences:
        if (date_str, sentence) not in already_done and (date_str, sentence) in existing:
            ann = existing[(date_str, sentence)]
            writer.writerow({'Date': date_str, 'Model': ann['model'], 'Grade': ann['grade'], 'Sentence': sentence})
            out_f.flush()
            already_done.add((date_str, sentence))

    unique_pairs = set((d, s) for d, s in sentences)
    to_annotate_list = list(dict.fromkeys((d, s) for d, s in sentences if (d, s) not in already_done))
    print(f"Skipping {len(unique_pairs) - len(to_annotate_list)} already-annotated sentences, calling LLM for {len(to_annotate_list)} new sentences.")

    for date_str, sentence in tqdm(to_annotate_list, desc="Annotating", unit="sentence"):
        while True:
            try:
                grade = annotate_sentence(client, sentence, few_shot_messages)
                writer.writerow({'Date': date_str, 'Model': MODEL, 'Grade': grade, 'Sentence': sentence})
                out_f.flush()
                time.sleep(REQUEST_DELAY)
                break
            except Exception as e:
                tqdm.write(f"  Error annotating sentence: {e}. Retrying in 5s...")
                time.sleep(5)

    out_f.close()

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        all_rows = list(reader)
    all_rows.sort(key=lambda r: _date_sort_key(r['Date']))
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Date', 'Model', 'Grade', 'Sentence'], delimiter='|')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} total annotations saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

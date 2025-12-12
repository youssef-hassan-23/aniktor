import os
# تعطيل كل الـGPU واستخدام CPU فقط
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
from math import log2
from sentence_transformers import SentenceTransformer, util

# ==============================
# Load CSV
# ==============================
def load_csv(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    expected = ['name','gender','country','occupation','birth_date','death_date','image_url','description']
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    df['alive'] = df['death_date'].astype(str).str.strip() == ""
    df['score'] = 1.0
    return df

# ==============================
# Input handling
# ==============================
def yes_no_idk(prompt):
    while True:
        ans = input(prompt + " (yes/no/idk): ").strip().lower()
        if ans in ['yes','y']:
            return 'yes'
        elif ans in ['no','n']:
            return 'no'
        elif ans in ['idk','i dont know','dont know','unknown']:
            return 'idk'
        else:
            print("Please answer yes / no / idk.")

# ==============================
# Entropy & Question Selection
# ==============================
def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * log2(p)
    return ent

def _best_question(possible, columns, asked):
    best = None
    best_gain = 0.0
    total = len(possible)
    if total <= 1:
        return None

    for col in columns:
        if col == 'alive':
            if ('alive', None) in asked:
                continue
            yes_count = int((possible['alive'] == True).sum())
            no_count = total - yes_count
            parent_entropy = _entropy([yes_count, no_count])
            e_yes = _entropy([yes_count, 0])
            e_no = _entropy([no_count, 0])
            expected = (yes_count / total) * e_yes + (no_count / total) * e_no
            gain = parent_entropy - expected
            if gain > best_gain:
                best_gain = gain
                best = (col, None, gain)
            continue

        vals = [v for v in possible[col].dropna().unique() if str(v).strip()]
        for val in vals:
            key = (col, str(val))
            if key in asked:
                continue
            yes_count = int((possible[col] == val).sum())
            no_count = total - yes_count
            parent_entropy = _entropy([yes_count, no_count])
            e_yes = _entropy([yes_count, 0])
            e_no = _entropy([no_count, 0])
            expected = (yes_count / total) * e_yes + (no_count / total) * e_no
            gain = parent_entropy - expected
            if gain > best_gain:
                best_gain = gain
                best = (col, val, gain)
    return best

# ==============================
# Core System (combined filtering + scoring)
# ==============================
def akinator_probabilistic(df):
    possible = df.copy()
    print("Welcome to the Expert System! Answer yes / no / idk only.\n")

    columns_to_probe = ['gender', 'country', 'occupation', 'alive']
    asked = set()

    while True:
        if len(possible) == 0:
            print("No candidates remain.")
            return

        best_guess = possible.sort_values('score', ascending=False).iloc[0]
        confidence = best_guess['score'] / possible['score'].mean()

        if confidence >= 2.0 and best_guess['score'] > 1.5:
            ans = yes_no_idk(f"Are you thinking of {best_guess['name']}?")
            if ans == 'yes':
                print("\n🎯 Great! I guessed it right!")
                print_person(best_guess)
                return
            else:
                possible.loc[possible['name'] == best_guess['name'], 'score'] *= 0.5
                possible['score'] = possible['score'] / possible['score'].mean()

        if len(possible) <= 3:
            goto_final(possible)
            return

        best = _best_question(possible, columns_to_probe, asked)
        if not best:
            goto_final(possible)
            return

        col, val, gain = best

        if col == 'alive':
            q = "Is the character still alive?"
        else:
            q = f"Is the character's {col} '{val}'?"

        ans = yes_no_idk(q)

        if col == 'alive':
            asked.add(('alive', None))
        else:
            asked.add((col, str(val)))

        if ans == 'idk':
            continue

        if col == 'alive':
            want_alive = (ans == 'yes')
            possible = possible[possible['alive'] == want_alive].copy()
            possible['score'] *= 1.2
        else:
            if ans == 'yes':
                kept = possible[possible[col] == val].copy()
                if kept.empty:
                    possible.loc[possible[col] == val, 'score'] *= 1.2
                    possible.loc[possible[col] != val, 'score'] *= 0.8
                else:
                    kept['score'] *= 1.2
                    possible = kept
            else:
                possible = possible[possible[col] != val].copy()
                possible['score'] *= 1.1

        if len(possible) > 0:
            possible['score'] = possible['score'] / possible['score'].mean()

# ==============================
# Final Stage with NLP-based description matching (CPU-only)
# ==============================
nlp_model = SentenceTransformer('all-MiniLM-L6-v2')  # CPU

def goto_final(possible, previous_hint=None, excluded_names=None):
    if excluded_names is None:
        excluded_names = set()
    if len(possible) == 0:
        print("No candidates remain.")
        return
    if len(possible) == 1:
        print("Found one candidate:")
        print_person(possible.iloc[0])
        confirm_final(possible.iloc[0], possible, previous_hint, excluded_names)
        return

    while True:
        candidates = possible[~possible['name'].isin(excluded_names)].copy()
        if len(candidates) == 0:
            print("No remaining candidates after exclusion.")
            return

        print(f"{len(candidates)} candidates remain.")
        hint = input("Enter a short description/hint or type 'idk': ").strip().lower()
        if not hint or hint in ['idk','i dont know','i don\'t know']:
            print("Remaining candidates:")
            for _, r in candidates.sort_values('score', ascending=False).iterrows():
                print("-", r['name'], "|", r['occupation'], "|", "Alive" if r['alive'] else "Deceased")
            return

        combined_hint = (previous_hint + " " + hint) if previous_hint else hint

        hint_embedding = nlp_model.encode(combined_hint, convert_to_tensor=True)
        desc_embeddings = nlp_model.encode(candidates['description'].tolist(), convert_to_tensor=True)
        cos_scores = util.cos_sim(hint_embedding, desc_embeddings)[0]
        best_idx = cos_scores.argmax().item()
        best_row = candidates.iloc[best_idx]

        print("Best match based on your hint (using NLP similarity on CPU):")
        print_person(best_row)

        confirm = yes_no_idk(f"Is {best_row['name']} the character you are thinking of?")
        if confirm == 'yes':
            print("\n🎯 Great! I guessed it right!")
            return
        else:
            print("Okay, let's try again with a new hint.")
            excluded_names.add(best_row['name'])
            previous_hint = combined_hint
            continue

def confirm_final(best_row, possible, previous_hint, excluded_names):
    confirm = yes_no_idk(f"Is {best_row['name']} the character you are thinking of?")
    if confirm == 'yes':
        print("\n🎯 Great! I guessed it right!")
        return
    else:
        print("Okay, let's try again with a new hint.")
        excluded_names.add(best_row['name'])
        goto_final(possible, previous_hint, excluded_names)

# ==============================
# Display Person Info
# ==============================
def print_person(row):
    print("------------------------")
    print("Name:", row['name'])
    print("Gender:", row['gender'])
    print("Country:", row['country'])
    print("Occupation:", row['occupation'])
    print("Birth Year:", row['birth_date'])
    print("Death Year:", row['death_date'])
    print("Alive:", "Yes" if row['alive'] else "No")
    if row['description']:
        print("Description:", row['description'][:300])
    print("Score:", round(row['score'], 3))
    print("------------------------")


# ==============================
# Entrypoint
# ==============================

class AkinatorEngine:
    def __init__(self, df):
        self.df = df
        self.possible = df.copy()
        self.columns_to_probe = ['gender','country','occupation','alive']
        self.asked = set()

    def next_question(self):
        from main import _best_question
        best = _best_question(self.possible, self.columns_to_probe, self.asked)
        if not best:
            return None
        col, val, gain = best
        if col == "alive":
            q = "Is the character still alive?"
        else:
            q = f"Is the character's {col} '{val}'?"
        return (col, val, q)

    def apply_answer(self, col, val, ans):
        if col == 'alive':
            want_alive = (ans == 'yes')
            self.possible = self.possible[self.possible['alive'] == want_alive].copy()
        else:
            if ans == 'yes':
                self.possible = self.possible[self.possible[col] == val].copy()
            else:
                self.possible = self.possible[self.possible[col] != val].copy()

        if len(self.possible) > 0:
            self.possible['score'] = self.possible['score'] / self.possible['score'].mean()

    def best_guess(self):
        if len(self.possible) == 0:
            return None
        return self.possible.sort_values('score', ascending=False).iloc[0]

    def ask(self):
        q = self.next_question()
        if q is None:
            return None
        col, val, q_text = q
        self._last_question = (col, val)
        remaining = len(self.possible)
        return (q_text, remaining)

    def answer(self, user_answer):
        # apply the stored last question's answer; return True when finished
        if not hasattr(self, '_last_question') or self._last_question is None:
            return False
        col, val = self._last_question
        # if user answered idk, do not filter
        if user_answer != 'idk':
            self.apply_answer(col, val, user_answer)
        # clear last question
        self._last_question = None
        # consider finished when 0 or 1 candidates remain
        return len(self.possible) <= 1

    def get_best(self):
        return self.best_guess()

def akinator_probabilistic_step(df):
    return AkinatorEngine(df)
if __name__ == "__main__":
    df = load_csv("/mnt/youssef/python_projects/akinator/data/arabic_personalities.csv")
    akinator_probabilistic(df)

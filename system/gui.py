# Revised gui.py — compatible with AkinatorEngine in main.py
import sys
import requests
from io import BytesIO
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QHBoxLayout, QMessageBox, QProgressBar, QLineEdit, QSizePolicy, QScrollArea
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from main import load_csv, akinator_probabilistic_step


class HintWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, candidates_df, combined_hint, excluded_names):
        super().__init__()
        self.candidates_df = candidates_df
        self.combined_hint = combined_hint
        self.excluded_names = excluded_names

    def run(self):
        try:
            import main
            nlp_model = main.nlp_model
            util = main.util
            candidates = self.candidates_df[~self.candidates_df['name'].isin(self.excluded_names)].copy()
            if len(candidates) == 0:
                self.finished.emit(None)
                return

            hint_embedding = nlp_model.encode(self.combined_hint, convert_to_tensor=True)
            desc_embeddings = nlp_model.encode(candidates['description'].tolist(), convert_to_tensor=True)
            cos_scores = util.cos_sim(hint_embedding, desc_embeddings)[0]
            best_idx = int(cos_scores.argmax().item())
            best_row = candidates.iloc[best_idx]
            self.finished.emit(best_row)
        except Exception:
            self.finished.emit(None)


class QuestionWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.setWindowTitle("Anikator")
        self.resize(800, 600)

        self.layout = QVBoxLayout()

        # Question area
        self.question_label = QLabel("")
        self.question_label.setWordWrap(True)
        self.question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.question_label)

        # Yes/No/IDK buttons
        btns = QHBoxLayout()
        self.yes_btn = QPushButton("YES")
        self.no_btn = QPushButton("NO")
        self.idk_btn = QPushButton("IDK")
        btns.addWidget(self.yes_btn)
        btns.addWidget(self.no_btn)
        btns.addWidget(self.idk_btn)
        self.layout.addLayout(btns)

        # Progress
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)

        # Hint input area
        hint_layout = QHBoxLayout()
        self.hint_input = QLineEdit()
        self.hint_input.setPlaceholderText("Enter a short hint/description (or type 'idk')...")
        self.submit_hint_btn = QPushButton("Analyze Hint")
        hint_layout.addWidget(self.hint_input)
        hint_layout.addWidget(self.submit_hint_btn)
        self.layout.addLayout(hint_layout)

        # Result display area (with image)
        result_area = QHBoxLayout()

        # left: image
        self.image_label = QLabel()
        self.image_label.setFixedSize(240, 320)
        self.image_label.setStyleSheet("border: 1px solid #888;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_area.addWidget(self.image_label)

        # right: details (scrollable)
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.details_label)
        result_area.addWidget(scroll)

        self.layout.addLayout(result_area)

        # Confirm buttons
        confirm_layout = QHBoxLayout()
        self.confirm_yes_btn = QPushButton("Yes, this is correct")
        self.confirm_no_btn = QPushButton("No, try again")
        confirm_layout.addWidget(self.confirm_yes_btn)
        confirm_layout.addWidget(self.confirm_no_btn)
        self.layout.addLayout(confirm_layout)

        # hide confirm buttons initially
        self.confirm_yes_btn.setVisible(False)
        self.confirm_no_btn.setVisible(False)

        self.setLayout(self.layout)

        # connections
        self.yes_btn.clicked.connect(lambda: self.answer("yes"))
        self.no_btn.clicked.connect(lambda: self.answer("no"))
        self.idk_btn.clicked.connect(lambda: self.answer("idk"))
        self.submit_hint_btn.clicked.connect(self.submit_hint)
        self.confirm_yes_btn.clicked.connect(self.confirm_yes)
        self.confirm_no_btn.clicked.connect(self.confirm_no)

        # hint-workflow state
        self.combined_hint = ""
        self.excluded_names = set()
        self.last_guess_name = None
        self.hint_worker = None

    def start(self):
        q = self.controller.next_question()
        self.update_ui(q)

    def update_ui(self, question):
        if question is None:
            # enter final / hint stage
            self.question_label.setText("Final stage — enter a hint to help me guess the character.")
            self.progress.setValue(0)
            return

        # question is (q_text, remaining_count)
        q_text, remaining = question
        self.question_label.setText(q_text)
        try:
            self.progress.setRange(0, 100)
            val = int(100 - (remaining * 2))
            val = max(0, min(100, val))
            self.progress.setValue(val)
        except Exception:
            self.progress.setValue(0)

    def answer(self, user_answer):
        done = self.controller.process_answer(user_answer)
        if done:
            # switch to final/hint stage (GUI will request hints)
            self.update_ui(None)
            return
        q = self.controller.next_question()
        self.update_ui(q)

    def show_person_full(self, row):
        # row is a pandas Series or similar mapping
        if row is None:
            self.details_label.setText("No candidates remain.")
            self.image_label.clear()
            self.confirm_yes_btn.setVisible(False)
            self.confirm_no_btn.setVisible(False)
            return

        name = row.get('name', '')
        gender = row.get('gender', '')
        country = row.get('country', '')
        occupation = row.get('occupation', '')
        birth = row.get('birth_date', '')
        death = row.get('death_date', '')
        alive = "Yes" if row.get('alive', False) else "No"
        desc = row.get('description', '') or ''
        img_url = row.get('image_url', '') or ''

        details = (
            f"<b>Name:</b> {name}<br>"
            f"<b>Gender:</b> {gender}<br>"
            f"<b>Country:</b> {country}<br>"
            f"<b>Occupation:</b> {occupation}<br>"
            f"<b>Birth Year:</b> {birth}<br>"
            f"<b>Death Year:</b> {death}<br>"
            f"<b>Alive:</b> {alive}<br><br>"
            f"<b>Description:</b><br>{desc}"
        )
        self.details_label.setText(details)

        # load image (non-blocking would be nicer, but keep simple)
        self.image_label.setText("Loading image...")
        if img_url:
            try:
                resp = requests.get(img_url, timeout=6)
                if resp.status_code == 200:
                    data = resp.content
                    pix = QPixmap()
                    if pix.loadFromData(data):
                        scaled = pix.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        self.image_label.setPixmap(scaled)
                    else:
                        self.image_label.setText("Image load failed")
                else:
                    self.image_label.setText("Image not found")
            except Exception:
                self.image_label.setText("Image error")
        else:
            self.image_label.setText("No image")

        # show confirm buttons
        self.confirm_yes_btn.setVisible(True)
        self.confirm_no_btn.setVisible(True)

    # ------------------ Hint workflow ------------------
    def submit_hint(self):
        raw = self.hint_input.text().strip()
        if not raw:
            return

        # handle 'idk' / empty hint -> list remaining candidates
        if raw.lower() in ['idk', "i don't know", "i dont know", "dont know"]:
            # show remaining candidates
            remaining = self.controller.get_all_candidates(exclude=self.excluded_names)
            if remaining is None or len(remaining) == 0:
                self.details_label.setText("No remaining candidates.")
            else:
                lines = []
                for _, r in remaining.sort_values('score', ascending=False).iterrows():
                    lines.append(f"{r['name']} — {r.get('occupation','')} — {'Alive' if r['alive'] else 'Deceased'}")
                text = "<br>".join(lines[:200])  # limit
                self.details_label.setText("<b>Remaining candidates:</b><br>" + text)
            return

        # append to combined hint
        if self.combined_hint:
            self.combined_hint = self.combined_hint + " " + raw
        else:
            self.combined_hint = raw

        # disable UI while processing
        self.submit_hint_btn.setEnabled(False)
        self.hint_input.setEnabled(False)
        self.progress.setRange(0, 0)  # busy indicator

        candidates_df = self.controller.df.copy()
        self.hint_worker = HintWorker(candidates_df, self.combined_hint, self.excluded_names.copy())
        self.hint_worker.finished.connect(self.on_hint_result)
        self.hint_worker.start()

    def on_hint_result(self, best_row):
        # re-enable UI
        self.progress.setRange(0, 100)
        self.submit_hint_btn.setEnabled(True)
        self.hint_input.setEnabled(True)

        if best_row is None:
            self.details_label.setText("No guess could be made from that hint. Please try another hint.")
            self.confirm_yes_btn.setVisible(False)
            self.confirm_no_btn.setVisible(False)
            return

        # show full person info (image + details)
        self.show_person_full(best_row)
        self.last_guess_name = best_row.get('name')

    def confirm_yes(self):
        QMessageBox.information(self, "Success", "Great! I guessed the character!")
        # disable further hinting
        self.submit_hint_btn.setEnabled(False)
        self.hint_input.setEnabled(False)
        self.confirm_yes_btn.setEnabled(False)
        self.confirm_no_btn.setEnabled(False)

    def confirm_no(self):
        if self.last_guess_name:
            self.excluded_names.add(self.last_guess_name)
        # clear result area to allow new hint
        self.details_label.setText("")
        self.image_label.clear()
        self.hint_input.clear()
        self.hint_input.setFocus()
        self.confirm_yes_btn.setVisible(False)
        self.confirm_no_btn.setVisible(False)
        # combined_hint remains so hints accumulate like CLI


class Controller:
    """
    Adapter between GUI and main.AkinatorEngine.
    The AkinatorEngine in main.py provides:
      - next_question() -> (col, val, q) or None
      - apply_answer(col, val, ans)
      - best_guess() -> pandas Series
      - possible attribute (DataFrame)
    This Controller normalizes to GUI expectations.
    """
    def __init__(self, df):
        self.df = df
        self.engine = akinator_probabilistic_step(df)
        self._last_q = None  # stores (col, val)

    def next_question(self):
        nxt = self.engine.next_question()
        if nxt is None:
            return None
        col, val, q_text = nxt
        self._last_q = (col, val)
        remaining = len(self.engine.possible) if hasattr(self.engine, "possible") else len(self.df)
        return (q_text, remaining)

    def process_answer(self, ans):
        """
        Apply answer to last question. Return True if we should enter final stage (no more probing),
        else False to continue asking.
        """
        if self._last_q is None:
            return True  # nothing to process -> final stage
        col, val = self._last_q
        # map 'yes'/'no'/'idk' behavior
        if ans == 'idk':
            # treat as no change; continue probing
            return False
        self.engine.apply_answer(col, val, ans)
        # reset last question
        self._last_q = None
        # if few candidates left or no best question next -> final stage
        if len(self.engine.possible) <= 3:
            return True
        # otherwise there may still be questions
        next_q = self.engine.next_question()
        if next_q is None:
            return True
        return False

    def best_guess(self):
        return self.engine.best_guess()

    def get_all_candidates(self, exclude=None):
        if exclude is None:
            exclude = set()
        df = self.engine.possible if hasattr(self.engine, "possible") else self.df
        return df[~df['name'].isin(exclude)].copy()


def main():
    df = load_csv("/mnt/youssef/python_projects/akinator/data/arabic_personalities.csv")
    app = QApplication(sys.argv)

    controller = Controller(df)
    win = QuestionWindow(controller)
    win.show()
    win.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

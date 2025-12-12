# aniktor

Overview
This project implements a probabilistic character-guessing system inspired by Akinator.
The program takes user answers to a sequence of questions and progressively filters characters
using a scoring and probability-based model until it reaches the most likely character.

How the Engine Works

1. Loading the Dataset
Characters and their attributes are loaded from a CSV file.
Each row represents a character, and each column represents a feature.

2. User Answers
Every question produces an answer: yes or no.
The system interprets it as a filter condition.

3. Probabilistic Step
Each answer updates the candidate list:
If answer is yes → keep characters that match the attribute
If answer is no → remove characters that match the attribute
A score is maintained for each character to represent how strongly it matches previous answers.

4. Final Guess
When candidates are below a threshold (e.g., 3), the system ranks them by score and suggests the top result.


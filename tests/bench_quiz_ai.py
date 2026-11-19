"""
Speed and click accuracy of the quiz AI backend on the saved quiz screenshots
(makes real model calls; not part of the unit tests).

    venv/Scripts/python.exe -m tests.bench_quiz_ai [--runs 3]

The right answer for each screenshot comes from the built-in solver, which
solves all of them: a click counts as a hit when it lands inside a tile the
solver would click. Answers are drawn to logs/letter_quiz/bench/.
"""

import argparse
import glob
import os
import statistics
import time

import cv2

from src.detection import letter_quiz as lq
from src.detection import quiz_ai

SHOTS = sorted(glob.glob('assets/pop-up-3-quiz-[2-6].png')) + [
    'assets/pop-up-3-quiz.png', 'assets/pop-up-7-quiz.png',
    'logs/letter_quiz/miss-20261117-223929/029.7s-title0.85.jpg',
    'logs/letter_quiz/miss-20261117-102904/030.6s-title0.75.jpg',
]
OUT = 'logs/letter_quiz/bench'


def inside(point, box, pad=4):
    x, y = point
    x0, y0, x1, y1 = box
    return x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=2, help='Asks per screenshot.')
    args = parser.parse_args()
    ai = quiz_ai.QuizAI()
    if not ai.on:
        print(f'AI fallback is off: {ai.why_off}')
        return
    print(f'Backend {ai.backend} ({ai.model}); timeout {ai.timeout:.0f} s.')
    if ai.backend == 'ollama':
        ai._ollama_warm()                       # time the first load separately
    renderer, glyphs, items = lq._TitleRenderer(), lq.Glyphs(), lq.Items()
    os.makedirs(OUT, exist_ok=True)
    times, hits, total = [], 0, 0
    for path in SHOTS:
        frame = cv2.imread(path)
        if frame is None:
            continue
        truth = lq.solve(frame, renderer, glyphs, items=items)
        frame = frame.copy()
        frame[0, 0] = (int(time.time() * 1000) % 255, 7, 7)   # a new image: no cache help
        footer = lq.find_footer(frame, renderer)
        crop = lq.panel_box(frame, footer) if footer else None
        for run in range(args.runs):
            total += 1
            try:
                reply = ai.solve(frame, crop)
            except Exception as error:
                print(f'  {os.path.basename(path)}: FAILED ({error})')
                continue
            times.append(reply.seconds)
            ok = None
            if truth is not None:
                want = [truth.tiles[i] for i in truth.chosen]
                ok = (len(reply.points) == len(want) and
                      all(any(inside(p[:2], box) for box in want) for p in reply.points))
                hits += bool(ok)
            mark = '?' if ok is None else 'HIT ' if ok else 'MISS'
            print(f'  {mark} {reply.seconds:4.1f} s  {os.path.basename(path)}: {reply.summary()}')
            if run == 0:
                cv2.imwrite(os.path.join(OUT, os.path.basename(path).rsplit('.', 1)[0] + '.png'),
                            quiz_ai.annotate(frame, reply))
    if times:
        print(f'\n{hits}/{total} right; reply time median {statistics.median(times):.1f} s, '
              f'slowest {max(times):.1f} s (the quiz gives 10 s; the AI is asked at ~10 s left).')


if __name__ == '__main__':
    main()

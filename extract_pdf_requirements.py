import sys
import io
from pypdf import PdfReader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

pdf = PdfReader('RLI_22_A0 - ASSIGNMENT - GROUP ASSIGNMENT -Tinder for RL.pdf')
text = '\n'.join([p.extract_text() for p in pdf.pages])

print("=" * 80)
print("SEARCHING FOR DQN AND REINFORCE REQUIREMENTS")
print("=" * 80)

lines = text.split('\n')
matches = []
for i, line in enumerate(lines):
    if any(kw in line for kw in ['DQN', 'REINFORCE', 'algorithm', 'policy gradient', 'actor-critic', 'Q-learning']):
        matches.append(f'Line {i}: {line.strip()[:100]}')

for m in matches[:50]:
    print(m)

# Save a summary to file instead of printing all
with open('pdf_requirements_summary.txt', 'w', encoding='utf-8') as f:
    f.write("DQN and REINFORCE requirements from assignment PDF\n")
    f.write("=" * 80 + "\n\n")
    for m in matches:
        f.write(m + "\n")

print("\nResults saved to pdf_requirements_summary.txt")

import sys

longest = 0
current = 0
for line in sys.stdin:
	if line.strip() == 'OUT_E':
		current += 1
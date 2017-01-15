import sys

counter = 0
prev_line = False
for line in sys.stdin:
	if len(line) == 0 or line == ' ':
		prev_line = True
	elif line[:3] == "[][":
		prev_line = False
		if line[3] == "]" and prev_line:
			print counter

	counter += 1
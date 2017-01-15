import sys

for line in sys.stdin:
	halves = line.strip().split('|||')
	print halves[1].lower() + ' ||| ' + halves[0].lower()
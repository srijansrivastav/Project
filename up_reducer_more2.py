#!/usr/bin/env python
""" reducer.py """
import sys

country = ""
current_country = ""
current_count = 0

for line in sys.stdin:
	country, count = line.split('\t', 1)
	try:
		count = int(count)
	except ValueError:
		continue

	if current_country == country:
		current_count += count
	else:
		if current_country:
			print '%s\t%d' % (current_country, current_count)
			current_count = count
			current_country = country

if current_country == country:
	print '%s\t%d' % (current_country, current_count)

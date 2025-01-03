#!/usr/bin/env python
""" mapper.py """
import sys
import re
import math

for line in sys.stdin:
	cols = line.split('\t')
	if re.match('^[-+]?[0-9]*\\.?[0-9]+$', cols[5]):
		try:
			lat = float(cols[5])
		except ValueError:
			break

		lower = math.floor(lat/10)*10
		upper = math.ceil(lat/10)*10
		if lower == upper:
			upper += 10
		band = str(lower) + '-' + str(upper)
		print '%s\t%d' % (band, 1)

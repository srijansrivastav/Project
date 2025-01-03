#!/usr/bin/env python

""" mapper.py """

import sys
import re

for line in sys.stdin:
	cols = line.split('\t')

	if re.match('^\\d{1,9}$', cols[4]):
		try:
			population = int(cols[4])
		except ValueError:
			break
		if population > 100000:
			print '%s\t%d' % (cols[2], 1)
